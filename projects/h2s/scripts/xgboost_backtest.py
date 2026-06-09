"""Backtest the operational XGBoost H2S model against historical observations.

Loads the single NESTOR-BES XGBoost model (43-feature set, 3 classes) and
replays it against modeldata_h2s_nofill.parquet, evaluating predictions
against observed H2S using the three-tier alert system:

  Green  : H2S < 5 ppb
  Yellow : 5 ≤ H2S < 30 ppb
  Orange : H2S ≥ 30 ppb

Produces:
  records.parquet          — raw prediction vs actual rows
  monthly/<YYYY-MM>.html   — per-month HTML report with charts
  index.html               — landing page linking all monthly reports

Usage (S3 model — requires env vars from .env):
    cd projects/h2s
    source .env    # or export S3_ACCESS_KEY / S3_SECRET_KEY manually
    uv run python scripts/xgboost_backtest.py \\
        --data ../../data/modeldata_h2s_nofill.parquet \\
        --output ./output/xgboost_backtest/

Usage (local model files):
    uv run python scripts/xgboost_backtest.py \\
        --data ../../data/modeldata_h2s_nofill.parquet \\
        --model /path/to/nestor_xgboost_weighted_model.json \\
        --prep-info /path/to/nestor_preprocessing_info.json \\
        --output ./output/xgboost_backtest/

Usage (report-only from saved records):
    uv run python scripts/xgboost_backtest.py \\
        --report-only ./output/xgboost_backtest/records.parquet \\
        --output ./output/xgboost_backtest/
"""

import argparse
import base64
import io
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


# ── constants ─────────────────────────────────────────────────────────────────

_FALLBACK_PARQUET_URL = (
    "https://oss.resilientservice.mooo.com/resilentpublic/"
    "latest/tijuana/forecast_data/modeldata_h2s_nofill.parquet"
)

_SITE = "NESTOR - BES"

_GREEN_MAX = 5.0
_ORANGE_MIN = 30.0


# ── helpers ───────────────────────────────────────────────────────────────────

def _categorize(h2s: float) -> str:
    if h2s < _GREEN_MAX:
        return "green"
    if h2s < _ORANGE_MIN:
        return "yellow"
    return "orange"


def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=110)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def _pct(v: float) -> str:
    return f"{v * 100:.1f}%"


def _color_cell(val: float, low: float = 0.5, high: float = 0.75) -> str:
    if val >= high:
        return "pass"
    if val >= low:
        return "warn"
    return "fail"


# ── data loading ──────────────────────────────────────────────────────────────

def load_data(path: str | None) -> pd.DataFrame:
    if path and Path(path).exists():
        df = pd.read_parquet(path)
    else:
        print(f"Local path not found — loading from {_FALLBACK_PARQUET_URL}")
        df = pd.read_parquet(_FALLBACK_PARQUET_URL)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    return df


_PUBLIC_MODEL_URL   = "https://oss.resilientservice.mooo.com/resilentpublic/tijuana/forecast/models/nestor_xgboost_weighted_model.json"
_PUBLIC_PREPINFO_URL = "https://oss.resilientservice.mooo.com/resilentpublic/tijuana/forecast/models/nestor_preprocessing_info.json"



def load_predictor(model_path: str | None, prep_info_path: str | None):
    """Load predictor: local files → authenticated S3 → public S3 fallback."""
    import json, tempfile
    from h2s.predictor.h2s_predictor import H2SPredictor

    if model_path and prep_info_path:
        print(f"Loading model from local files: {model_path}")
        return H2SPredictor.from_local(model_path, prep_info_path)

    # Try authenticated S3 if credentials are available
    if os.environ.get("S3_ACCESS_KEY") and os.environ.get("S3_SECRET_KEY"):
        from h2s.resources.minio import S3Resource
        from h2s.constants import MODEL_PATH
        s3 = S3Resource(
            S3_BUCKET=os.environ.get("S3_BUCKET", "test"),
            S3_ADDRESS=os.environ.get("S3_ADDRESS", "oss.resilientservice.mooo.com"),
            S3_PORT=os.environ.get("S3_PORT", "443"),
            S3_USE_SSL=os.environ.get("S3_USE_SSL", "true").lower() == "true",
            S3_ACCESS_KEY=os.environ["S3_ACCESS_KEY"],
            S3_SECRET_KEY=os.environ["S3_SECRET_KEY"],
        )
        print(f"Loading model from S3 (authenticated): {MODEL_PATH}/nestor_xgboost_weighted_model.json")
        return H2SPredictor.from_s3(
            s3,
            f"{MODEL_PATH}/nestor_xgboost_weighted_model.json",
            f"{MODEL_PATH}/nestor_preprocessing_info.json",
            model_name="nestor_xgboost",
        )

    # Public S3 fallback (no credentials needed)
    print("No S3 credentials found — loading model from public URL")
    import io, urllib.request, xgboost as xgb, joblib
    with urllib.request.urlopen(_PUBLIC_MODEL_URL, timeout=120) as resp:
        model_bytes = resp.read()
    print(f"  Downloaded model: {len(model_bytes):,} bytes")
    if model_bytes[:1] == b'\x80':
        model = joblib.load(io.BytesIO(model_bytes))
    else:
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            f.write(model_bytes)
            tmp = f.name
        try:
            model = xgb.XGBClassifier()
            model.load_model(tmp)
        finally:
            os.unlink(tmp)
    with urllib.request.urlopen(_PUBLIC_PREPINFO_URL, timeout=30) as resp:
        prep_info = json.loads(resp.read())
    return H2SPredictor(model, prep_info, model_name="nestor_xgboost")


# ── prediction ────────────────────────────────────────────────────────────────

def run_predictions(df: pd.DataFrame, predictor) -> pd.DataFrame:
    """Run predictor over all NESTOR-BES rows with valid H2S observations.

    Uses observed H2S for lag features (oracle lags) — this matches how the
    pipeline behaves when run hourly against live sensor data.
    """
    nestor = df[df["site_name"] == _SITE].copy()
    nestor = nestor.sort_values("time").reset_index(drop=True)

    if "H2S" not in nestor.columns:
        raise ValueError("Data must contain H2S column for actual labels")

    valid = nestor[nestor["H2S"].notna()].copy()
    print(f"  NESTOR-BES rows: {len(nestor):,}  with valid H2S: {len(valid):,}")

    preprocessed = predictor.preprocess_data(valid)
    result = predictor.predict(preprocessed)

    # Merge back time + actual H2S
    result["time"] = valid["time"].values
    result["actual_h2s"] = valid["H2S"].values
    result["actual_category"] = result["actual_h2s"].apply(_categorize)
    result["month"] = pd.to_datetime(result["time"]).dt.to_period("M").astype(str)

    # Binary thresholds for tier reporting
    result["actual_above_5"]  = result["actual_h2s"] >= _GREEN_MAX
    result["actual_above_30"] = result["actual_h2s"] >= _ORANGE_MIN
    result["pred_above_5"]    = result["predicted_category"].isin(["yellow", "orange"])
    result["pred_above_30"]   = result["predicted_category"] == "orange"

    return result


# ── metrics ───────────────────────────────────────────────────────────────────

def _metrics_3class(df: pd.DataFrame) -> dict:
    """Compute per-class precision / recall / F1 and overall balanced accuracy."""
    from sklearn.metrics import balanced_accuracy_score, confusion_matrix

    y_true = df["actual_category"]
    y_pred = df["predicted_category"]
    classes = ["green", "yellow", "orange"]

    # confusion matrix rows=actual, cols=predicted
    cm = confusion_matrix(y_true, y_pred, labels=classes).tolist()

    ba = float(balanced_accuracy_score(y_true, y_pred))

    per_class = {}
    for cls in classes:
        yt = (y_true == cls).astype(int)
        yp = (y_pred == cls).astype(int)
        tp = int((yt & yp).sum())
        fp = int((~yt.astype(bool) & yp.astype(bool)).sum())
        fn = int((yt.astype(bool) & ~yp.astype(bool)).sum())
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec  = tp / (tp + fn) if (tp + fn) else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        per_class[cls] = {"precision": prec, "recall": rec, "f1": f1, "n": int(yt.sum())}

    # False alarm rate: FP orange / total non-orange observed
    orange_yt = (y_true == "orange").astype(int)
    orange_yp = (y_pred == "orange").astype(int)
    fp_orange = int(((orange_yp == 1) & (orange_yt == 0)).sum())
    n_non_orange = int((orange_yt == 0).sum())
    far = fp_orange / n_non_orange if n_non_orange else 0.0

    # Tier-specific binary metrics (5ppb and 30ppb thresholds)
    def _binary(yt_col: str, yp_col: str) -> dict:
        yt = df[yt_col].astype(bool)
        yp = df[yp_col].astype(bool)
        tp = int((yt & yp).sum())
        fp = int((~yt & yp).sum())
        fn = int((yt & ~yp).sum())
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec  = tp / (tp + fn) if (tp + fn) else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        return {"precision": prec, "recall": rec, "f1": f1, "tp": tp, "fp": fp, "fn": fn,
                "n_events": tp + fn}

    return {
        "n": len(df),
        "balanced_accuracy": ba,
        "false_alarm_rate": far,
        "per_class": per_class,
        "confusion_matrix": cm,
        "tier_5ppb":  _binary("actual_above_5",  "pred_above_5"),
        "tier_30ppb": _binary("actual_above_30", "pred_above_30"),
    }


def compute_monthly_metrics(records: pd.DataFrame) -> dict[str, dict]:
    months = sorted(records["month"].unique())
    result: dict[str, dict] = {}
    for m in months:
        mdf = records[records["month"] == m]
        result[m] = _metrics_3class(mdf)
    result["ALL"] = _metrics_3class(records)
    return result


# ── charts ────────────────────────────────────────────────────────────────────

_GREEN_COLOR  = "#27ae60"
_YELLOW_COLOR = "#f39c12"
_ORANGE_COLOR = "#e74c3c"
_BLUE_COLOR   = "#2980b9"


def _chart_monthly_orange(monthly: dict[str, dict]) -> str:
    """Monthly orange recall and false alarm rate."""
    months = [m for m in sorted(monthly) if m != "ALL"]
    recall = [monthly[m]["per_class"]["orange"]["recall"] for m in months]
    far    = [monthly[m]["false_alarm_rate"] for m in months]

    fig, ax = plt.subplots(figsize=(11, 4))
    xs = range(len(months))
    ax.bar(xs, recall, color=_ORANGE_COLOR, alpha=0.8, label="Orange recall (detection rate)")
    ax.plot(xs, far, color=_BLUE_COLOR, marker="D", markersize=5,
            linewidth=1.8, label="False alarm rate", zorder=5)
    ax.axhline(0.613, color=_ORANGE_COLOR, linewidth=1.0, linestyle="--",
               alpha=0.6, label="Reported baseline 61.3%")
    ax.axhline(0.054, color=_BLUE_COLOR, linewidth=1.0, linestyle="--",
               alpha=0.6, label="Reported baseline 5.4%")
    ax.set_xticks(list(xs))
    ax.set_xticklabels(months, rotation=45, ha="right", fontsize=8)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
    ax.set_ylim(0, 1.05)
    ax.set_title("Monthly Orange (≥30 ppb) Detection Rate vs False Alarm Rate", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(axis="y", linewidth=0.4, alpha=0.5)
    fig.tight_layout()
    return _fig_to_b64(fig)


def _chart_precision_recall_by_class(monthly: dict[str, dict]) -> str:
    """Monthly precision and recall for each class."""
    months = [m for m in sorted(monthly) if m != "ALL"]
    classes = [("green", _GREEN_COLOR), ("yellow", _YELLOW_COLOR), ("orange", _ORANGE_COLOR)]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    fig.suptitle("Monthly Precision & Recall by Class", fontsize=11, fontweight="bold")

    for ax, (cls, color) in zip(axes, classes):
        prec = [monthly[m]["per_class"][cls]["precision"] for m in months]
        rec  = [monthly[m]["per_class"][cls]["recall"] for m in months]
        xs = range(len(months))
        ax.plot(xs, prec, color=color, marker="o", markersize=4,
                linewidth=1.8, label="Precision")
        ax.plot(xs, rec, color=color, marker="s", markersize=4,
                linewidth=1.8, linestyle="--", alpha=0.6, label="Recall")
        ax.set_title(f"{cls.capitalize()} (<{5 if cls=='green' else 30} ppb{'↑' if cls=='orange' else ''})",
                     fontsize=10)
        ax.set_xticks(list(xs))
        ax.set_xticklabels(months, rotation=45, ha="right", fontsize=7)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8)
        ax.grid(axis="y", linewidth=0.4, alpha=0.5)

    fig.tight_layout()
    return _fig_to_b64(fig)


def _chart_tier_thresholds(monthly: dict[str, dict]) -> str:
    """Monthly recall for ≥5 ppb and ≥30 ppb binary thresholds."""
    months = [m for m in sorted(monthly) if m != "ALL"]
    rec5  = [monthly[m]["tier_5ppb"]["recall"]  for m in months]
    rec30 = [monthly[m]["tier_30ppb"]["recall"] for m in months]
    prec5  = [monthly[m]["tier_5ppb"]["precision"]  for m in months]
    prec30 = [monthly[m]["tier_30ppb"]["precision"] for m in months]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4), sharey=True)
    xs = range(len(months))

    for ax, rec, prec, label, color in [
        (ax1, rec5,  prec5,  "≥5 ppb (Yellow+Orange)", _YELLOW_COLOR),
        (ax2, rec30, prec30, "≥30 ppb (Orange only)",  _ORANGE_COLOR),
    ]:
        ax.bar([x - 0.2 for x in xs], prec, width=0.38, color=color,
               alpha=0.7, label="Precision")
        ax.bar([x + 0.2 for x in xs], rec,  width=0.38, color=color,
               alpha=0.4, label="Recall")
        ax.set_title(f"Did the model catch {label}?", fontsize=10)
        ax.set_xticks(list(xs))
        ax.set_xticklabels(months, rotation=45, ha="right", fontsize=7)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8)
        ax.grid(axis="y", linewidth=0.4, alpha=0.5)

    fig.suptitle("Tier Threshold Detection — Did the Model Get It Right?",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    return _fig_to_b64(fig)


def _chart_confusion_matrix(m: dict, title: str) -> str:
    """Confusion matrix heatmap for a single period."""
    cm = np.array(m["confusion_matrix"], dtype=float)
    labels = ["Green", "Yellow", "Orange"]

    fig, ax = plt.subplots(figsize=(5, 4))
    row_totals = cm.sum(axis=1, keepdims=True)
    cm_norm = np.where(row_totals > 0, np.divide(cm, row_totals, where=row_totals > 0), 0.0)

    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks([0, 1, 2])
    ax.set_yticks([0, 1, 2])
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Predicted", fontsize=9)
    ax.set_ylabel("Actual", fontsize=9)
    ax.set_title(title, fontsize=10)

    for i in range(3):
        for j in range(3):
            count = int(cm[i, j])
            pct = cm_norm[i, j]
            color = "white" if pct > 0.6 else "black"
            ax.text(j, i, f"{count}\n({pct:.0%})",
                    ha="center", va="center", fontsize=8, color=color)

    fig.colorbar(im, ax=ax, shrink=0.85)
    fig.tight_layout()
    return _fig_to_b64(fig)


def _chart_event_counts(monthly: dict[str, dict]) -> str:
    """Monthly counts of observed H2S tier events."""
    months = [m for m in sorted(monthly) if m != "ALL"]
    n_green  = [monthly[m]["per_class"]["green"]["n"]  for m in months]
    n_yellow = [monthly[m]["per_class"]["yellow"]["n"] for m in months]
    n_orange = [monthly[m]["per_class"]["orange"]["n"] for m in months]

    xs = np.arange(len(months))
    w = 0.26
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.bar(xs - w, n_green,  width=w, color=_GREEN_COLOR,  label="Green (<5 ppb)")
    ax.bar(xs,     n_yellow, width=w, color=_YELLOW_COLOR, label="Yellow (5–30 ppb)")
    ax.bar(xs + w, n_orange, width=w, color=_ORANGE_COLOR, label="Orange (≥30 ppb)")
    ax.set_xticks(xs)
    ax.set_xticklabels(months, rotation=45, ha="right", fontsize=8)
    ax.set_title("Observed H2S Event Counts by Tier and Month", fontsize=11)
    ax.set_ylabel("Hours")
    ax.legend(fontsize=8)
    ax.grid(axis="y", linewidth=0.4, alpha=0.5)
    fig.tight_layout()
    return _fig_to_b64(fig)


# ── HTML generation ───────────────────────────────────────────────────────────

_CSS = """
body { font-family: system-ui, sans-serif; margin: 0; background: #f5f6fa; color: #333; }
.header { background: #1a2233; color: #fff; padding: 18px 32px; }
.header h1 { margin: 0; font-size: 1.4em; }
.header p { margin: 4px 0 0; font-size: 0.85em; color: #aab; }
.content { max-width: 1100px; margin: 28px auto; padding: 0 24px; }
.section { background: #fff; border-radius: 8px; padding: 20px 24px; margin-bottom: 20px;
           box-shadow: 0 1px 4px rgba(0,0,0,.07); }
h2 { margin: 0 0 14px; font-size: 1.1em; color: #1a2233; border-bottom: 2px solid #eee; padding-bottom: 8px; }
h3 { font-size: 0.98em; color: #444; margin: 14px 0 8px; }
table { border-collapse: collapse; width: 100%; font-size: 0.84em; }
th { background: #1a2233; color: #fff; padding: 7px 10px; text-align: left; }
td { padding: 6px 10px; border-bottom: 1px solid #eee; }
tr:hover td { background: #f0f4ff; }
.pass { background: #d5f5e3; color: #196f3d; font-weight: 600; }
.warn { background: #fef9e7; color: #7d6608; }
.fail { background: #fadbd8; color: #922b21; font-weight: 600; }
.metric-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 14px; margin-bottom: 18px; }
.metric-card { background: #f8fafc; border: 1px solid #e0e0e0; border-radius: 6px;
               padding: 14px 16px; text-align: center; }
.metric-card .val { font-size: 1.8em; font-weight: 700; color: #1a2233; }
.metric-card .lbl { font-size: 0.78em; color: #777; margin-top: 4px; }
.chart-row { display: flex; gap: 14px; flex-wrap: wrap; margin-bottom: 14px; }
.chart-row img { max-width: 100%; border-radius: 6px; box-shadow: 0 1px 3px rgba(0,0,0,.1); }
.nav { background: #fff; padding: 10px 24px; display: flex; gap: 14px; flex-wrap: wrap;
       border-bottom: 1px solid #ddd; font-size: 0.85em; }
.nav a { color: #2563eb; text-decoration: none; }
.nav a:hover { text-decoration: underline; }
.tier-green  { color: #196f3d; font-weight: 600; }
.tier-yellow { color: #7d6608; font-weight: 600; }
.tier-orange { color: #922b21; font-weight: 600; }
"""


def _overall_summary_cards(overall: dict) -> str:
    pc = overall["per_class"]
    t30 = overall["tier_30ppb"]
    t5  = overall["tier_5ppb"]
    cards = [
        (f"{_pct(pc['orange']['recall'])}", "Orange Detection Rate"),
        (f"{_pct(overall['false_alarm_rate'])}", "False Alarm Rate"),
        (f"{_pct(overall['balanced_accuracy'])}", "Balanced Accuracy"),
        (f"{_pct(t5['recall'])}", "≥5 ppb Recall"),
        (f"{_pct(t30['recall'])}", "≥30 ppb Recall"),
        (f"{overall['n']:,}", "Total Predictions"),
    ]
    html = '<div class="metric-grid">'
    for val, lbl in cards:
        html += f'<div class="metric-card"><div class="val">{val}</div><div class="lbl">{lbl}</div></div>'
    html += "</div>"
    return html


def _metrics_table_rows(m: dict) -> str:
    pc = m["per_class"]
    rows = ""
    for cls, color_cls in [("green", "tier-green"), ("yellow", "tier-yellow"), ("orange", "tier-orange")]:
        p = pc[cls]
        cell_prec = f'class="{_color_cell(p["precision"])}"'
        cell_rec  = f'class="{_color_cell(p["recall"])}"'
        cell_f1   = f'class="{_color_cell(p["f1"])}"'
        rows += (
            f"<tr>"
            f"<td class='{color_cls}'>{cls.capitalize()}</td>"
            f"<td {cell_prec}>{_pct(p['precision'])}</td>"
            f"<td {cell_rec}>{_pct(p['recall'])}</td>"
            f"<td {cell_f1}>{_pct(p['f1'])}</td>"
            f"<td>{p['n']:,}</td>"
            f"</tr>"
        )
    return rows


def _tier_table_html(m: dict) -> str:
    rows = ""
    for label, key in [("≥5 ppb (Yellow+Orange)", "tier_5ppb"), ("≥30 ppb (Orange)", "tier_30ppb")]:
        t = m[key]
        rows += (
            f"<tr>"
            f"<td>{label}</td>"
            f"<td class='{_color_cell(t['precision'])}'>{_pct(t['precision'])}</td>"
            f"<td class='{_color_cell(t['recall'])}'>{_pct(t['recall'])}</td>"
            f"<td class='{_color_cell(t['f1'])}'>{_pct(t['f1'])}</td>"
            f"<td>{t['n_events']:,}</td>"
            f"<td>{t['tp']:,} TP / {t['fp']:,} FP / {t['fn']:,} FN</td>"
            f"</tr>"
        )
    return f"""
<table>
  <thead><tr><th>Threshold</th><th>Precision</th><th>Recall</th><th>F1</th><th>Events</th><th>Breakdown</th></tr></thead>
  <tbody>{rows}</tbody>
</table>"""


def _monthly_summary_table(monthly: dict[str, dict], months: list[str]) -> str:
    header = (
        "<tr><th>Month</th><th>n</th>"
        "<th>Orange Recall</th><th>Orange Precision</th>"
        "<th>Yellow Recall</th><th>FAR</th>"
        "<th>≥5ppb Recall</th><th>≥30ppb Recall</th>"
        "<th>Balanced Acc.</th></tr>"
    )
    rows = ""
    for m in months:
        mm = monthly[m]
        pc = mm["per_class"]
        rows += (
            f"<tr>"
            f"<td><b>{m}</b></td>"
            f"<td>{mm['n']:,}</td>"
            f"<td class='{_color_cell(pc['orange']['recall'])}'>{_pct(pc['orange']['recall'])}</td>"
            f"<td class='{_color_cell(pc['orange']['precision'])}'>{_pct(pc['orange']['precision'])}</td>"
            f"<td class='{_color_cell(pc['yellow']['recall'])}'>{_pct(pc['yellow']['recall'])}</td>"
            f"<td class='{_color_cell(1 - mm['false_alarm_rate'], low=0.9, high=0.96)}'>{_pct(mm['false_alarm_rate'])}</td>"
            f"<td class='{_color_cell(mm['tier_5ppb']['recall'])}'>{_pct(mm['tier_5ppb']['recall'])}</td>"
            f"<td class='{_color_cell(mm['tier_30ppb']['recall'])}'>{_pct(mm['tier_30ppb']['recall'])}</td>"
            f"<td class='{_color_cell(mm['balanced_accuracy'])}'>{_pct(mm['balanced_accuracy'])}</td>"
            f"</tr>"
        )
    return f"<table><thead>{header}</thead><tbody>{rows}</tbody></table>"


def build_index_html(monthly: dict[str, dict], records: pd.DataFrame, months: list[str]) -> str:
    overall = monthly["ALL"]
    chart_orange = _chart_monthly_orange(monthly)
    chart_tiers  = _chart_tier_thresholds(monthly)
    chart_prec_rec = _chart_precision_recall_by_class(monthly)
    chart_counts = _chart_event_counts(monthly)
    chart_cm_all = _chart_confusion_matrix(overall, "Overall Confusion Matrix")

    time_range = f"{records['time'].min().strftime('%Y-%m-%d')} – {records['time'].max().strftime('%Y-%m-%d')}"

    # Month navigation links
    nav_links = "".join(
        f'<a href="monthly/{m}.html">{m}</a>' for m in months
    )

    month_table = _monthly_summary_table(monthly, months)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>XGBoost H2S Backtest — NESTOR-BES</title>
  <style>{_CSS}</style>
</head>
<body>
  <div class="header">
    <h1>XGBoost H2S Model Backtest — NESTOR-BES</h1>
    <p>43-feature model · Three-tier alert system (Green &lt;5 ppb · Yellow 5–30 ppb · Orange ≥30 ppb) · {time_range}</p>
  </div>
  <div class="nav">
    <strong>Monthly:</strong> {nav_links}
  </div>
  <div class="content">

    <div class="section">
      <h2>Overall Performance ({records['time'].min().strftime('%b %Y')} – {records['time'].max().strftime('%b %Y')})</h2>
      {_overall_summary_cards(overall)}
      <div class="chart-row">
        <img src="data:image/png;base64,{chart_cm_all}" style="max-width:360px">
      </div>
      <h3>Per-Class Metrics</h3>
      <table>
        <thead><tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1</th><th>Observed Hours</th></tr></thead>
        <tbody>{_metrics_table_rows(overall)}</tbody>
      </table>
      <h3>Tier Threshold Detection</h3>
      {_tier_table_html(overall)}
    </div>

    <div class="section">
      <h2>Orange Event Detection vs False Alarm Rate</h2>
      <img src="data:image/png;base64,{chart_orange}" style="width:100%">
    </div>

    <div class="section">
      <h2>Tier Thresholds — Did the Model Get It Right?</h2>
      <img src="data:image/png;base64,{chart_tiers}" style="width:100%">
    </div>

    <div class="section">
      <h2>Precision &amp; Recall by Class</h2>
      <img src="data:image/png;base64,{chart_prec_rec}" style="width:100%">
    </div>

    <div class="section">
      <h2>Observed Event Counts by Month</h2>
      <img src="data:image/png;base64,{chart_counts}" style="width:100%">
    </div>

    <div class="section">
      <h2>Monthly Summary Table</h2>
      {month_table}
    </div>

  </div>
</body>
</html>"""


def build_month_html(month: str, m: dict, records: pd.DataFrame,
                     all_months: list[str]) -> str:
    """Build HTML page for a single month."""
    mdf = records[records["month"] == month]
    chart_cm = _chart_confusion_matrix(m, f"Confusion Matrix — {month}")

    # Simple bar chart: predicted vs actual distribution
    pred_counts  = mdf["predicted_category"].value_counts().reindex(["green", "yellow", "orange"], fill_value=0)
    actual_counts = mdf["actual_category"].value_counts().reindex(["green", "yellow", "orange"], fill_value=0)
    fig, ax = plt.subplots(figsize=(7, 3.5))
    xs = np.arange(3)
    ax.bar(xs - 0.2, actual_counts.values,  width=0.38, label="Actual",    color=[_GREEN_COLOR, _YELLOW_COLOR, _ORANGE_COLOR], alpha=0.8)
    ax.bar(xs + 0.2, pred_counts.values, width=0.38, label="Predicted", color=[_GREEN_COLOR, _YELLOW_COLOR, _ORANGE_COLOR], alpha=0.4)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["Green (<5 ppb)", "Yellow (5–30 ppb)", "Orange (≥30 ppb)"])
    ax.set_title(f"Actual vs Predicted Distribution — {month}", fontsize=10)
    ax.set_ylabel("Hours")
    ax.legend()
    ax.grid(axis="y", linewidth=0.4, alpha=0.5)
    fig.tight_layout()
    chart_dist = _fig_to_b64(fig)

    prev_idx = all_months.index(month) - 1
    next_idx = all_months.index(month) + 1
    prev_link = f'<a href="{all_months[prev_idx]}.html">← {all_months[prev_idx]}</a>' if prev_idx >= 0 else ""
    next_link = f'<a href="{all_months[next_idx]}.html">{all_months[next_idx]} →</a>' if next_idx < len(all_months) else ""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>XGBoost Backtest — {month}</title>
  <style>{_CSS}</style>
</head>
<body>
  <div class="header">
    <h1>XGBoost H2S Backtest — {month}</h1>
    <p>NESTOR-BES · {m['n']:,} hourly predictions</p>
  </div>
  <div class="nav">
    <a href="../index.html">↑ All months</a>
    {prev_link}
    {next_link}
  </div>
  <div class="content">

    <div class="section">
      <h2>Month Summary</h2>
      {_overall_summary_cards(m)}
    </div>

    <div class="section">
      <h2>Three-Class Performance</h2>
      <div class="chart-row">
        <img src="data:image/png;base64,{chart_cm}" style="max-width:400px">
        <img src="data:image/png;base64,{chart_dist}" style="max-width:500px">
      </div>
      <table>
        <thead><tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1</th><th>Observed Hours</th></tr></thead>
        <tbody>{_metrics_table_rows(m)}</tbody>
      </table>
    </div>

    <div class="section">
      <h2>Tier Threshold Detection</h2>
      <p style="font-size:0.85em;color:#555;">
        Did the model correctly flag hours when H2S crossed the 5 ppb (yellow+orange) and 30 ppb (orange) thresholds?
      </p>
      {_tier_table_html(m)}
    </div>

  </div>
</body>
</html>"""


# ── main ──────────────────────────────────────────────────────────────────────

def generate_reports(records: pd.DataFrame, output_dir: Path) -> None:
    monthly = compute_monthly_metrics(records)
    months  = [m for m in sorted(monthly) if m != "ALL"]

    output_dir.mkdir(parents=True, exist_ok=True)
    monthly_dir = output_dir / "monthly"
    monthly_dir.mkdir(exist_ok=True)

    # Index page
    index_html = build_index_html(monthly, records, months)
    (output_dir / "index.html").write_text(index_html, encoding="utf-8")
    print(f"  index.html → {output_dir / 'index.html'}")

    # Per-month pages
    for month in months:
        html = build_month_html(month, monthly[month], records, months)
        out = monthly_dir / f"{month}.html"
        out.write_text(html, encoding="utf-8")
        print(f"  {month}.html → {out}")

    print(f"\nDone. Open: {output_dir / 'index.html'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest XGBoost H2S model against historical observations")
    parser.add_argument("--data", help="Path to modeldata_h2s_nofill.parquet (falls back to public S3 URL)")
    parser.add_argument("--model", help="Path to local XGBoost model JSON (skips S3 load)")
    parser.add_argument("--prep-info", help="Path to local preprocessing JSON (skips S3 load)")
    parser.add_argument("--output", default="./output/xgboost_backtest", help="Output directory")
    parser.add_argument("--report-only", metavar="RECORDS_PARQUET",
                        help="Skip prediction; load saved records parquet and regenerate reports")
    parser.add_argument("--start", help="Filter data from this date (YYYY-MM-DD)")
    parser.add_argument("--end",   help="Filter data up to this date (YYYY-MM-DD, inclusive)")
    args = parser.parse_args()

    out_dir = Path(args.output)

    if args.report_only:
        print(f"Loading records from {args.report_only} ...")
        records = pd.read_parquet(args.report_only)
        records["time"] = pd.to_datetime(records["time"], utc=True)
        print(f"  {len(records):,} records")
        generate_reports(records, out_dir)
        return

    print("Loading observation data...")
    df = load_data(args.data)
    print(f"  {len(df):,} rows, {df['time'].min().date()} – {df['time'].max().date()}")

    if args.start:
        df = df[df["time"] >= pd.Timestamp(args.start, tz="UTC")]
    if args.end:
        df = df[df["time"] <= pd.Timestamp(args.end, tz="UTC") + pd.Timedelta(days=1)]

    print("Loading model...")
    predictor = load_predictor(args.model, args.prep_info)
    print(f"  Model loaded: {len(predictor.feature_cols)} features, classes={predictor.class_names}")

    print("Running predictions...")
    records = run_predictions(df, predictor)
    print(f"  {len(records):,} predictions generated")

    out_dir.mkdir(parents=True, exist_ok=True)
    records_path = out_dir / "records.parquet"
    records.to_parquet(records_path, index=False)
    print(f"  Records saved → {records_path}")

    print("Generating reports...")
    generate_reports(records, out_dir)


if __name__ == "__main__":
    main()
