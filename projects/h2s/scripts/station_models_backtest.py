"""Backtest the deployed per-station H2S models against historical observations.

The analog of scripts/xgboost_backtest.py for the *present* models. Instead of
the single hourly 3-class NESTOR classifier, this replays the per-station daily
models — regression + clf_5ppb / clf_10ppb / clf_30ppb, for both the Evidence
(33-feature, production default) and Lean (19-feature) variants — over
modeldata_h2s_nofill.parquet for all three stations.

Honest scope: features (including the H2S lags/rolling windows) are built from
*observed* H2S via `prepare_multi_station_features`, the same routine the models
were trained on — i.e. these are oracle (one-step) inputs, not the recursive
multi-step forecast the products pipeline runs. So this measures the model's
skill given good inputs (directly comparable to the training/calibration
metrics); the recursive skill-decay-with-lead-hour is what the Phase-5
validation store measures instead.

Metrics per (station, variant), monthly + overall:
  - regression : Spearman(actual, pred), MAE, recall/precision@{5,10,30,100}
                 (magnitude — calibration harness, h2s_pred cut at k)
  - classifier : prob-call recall/precision at 5/10/30 (p_k > 0.5 vs actual ≥ k)
  - 3-class    : per-class precision/recall/F1, balanced accuracy, false-alarm
                 rate, confusion matrix (category derived from the regression)

Produces:
  records.parquet                       — raw prediction vs actual rows
  index.html                            — all stations × variants overview
  <STATION>_<variant>.html              — per (station, variant) detail + monthly
  <STATION>_<variant>_monthly/<YYYY-MM>.html  (optional drill-down)

Usage (S3 deployed models — requires env vars from .env):
    cd projects/h2s
    set -a; source .env; set +a
    uv run python scripts/station_models_backtest.py \\
        --data ../../data/modeldata_h2s_nofill.parquet \\
        --output ./output/station_backtest/

Usage (report-only from saved records):
    uv run python scripts/station_models_backtest.py \\
        --report-only ./output/station_backtest/records.parquet \\
        --output ./output/station_backtest/
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from h2s.constants import STATIONS  # noqa: E402
from h2s.training.calibration_eval import recall_at_threshold, spearman_rank  # noqa: E402
from h2s.training.multi_station_trainer import prepare_multi_station_features  # noqa: E402

# ── constants ─────────────────────────────────────────────────────────────────

_FALLBACK_PARQUET_URL = (
    "https://oss.resilientservice.mooo.com/resilentpublic/"
    "latest/tijuana/forecast_data/modeldata_h2s_nofill.parquet"
)
_STATION_MODELS_BASE = "tijuana/forecast/models/stations"
_VARIANTS = ("evidence", "lean")
_TASKS = ("regression", "clf_5ppb", "clf_10ppb", "clf_30ppb")
_THRESHOLDS = (5, 10, 30, 100)
_PROB_FOR_THRESHOLD = {5: "p5", 10: "p10", 30: "p30"}

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


def _pct(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"{v * 100:.1f}%"


def _num(v, fmt="{:.3f}") -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return fmt.format(v)


def _color_cell(val, low: float = 0.5, high: float = 0.75) -> str:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return ""
    if val >= high:
        return "pass"
    if val >= low:
        return "warn"
    return "fail"


# ── data + model loading ──────────────────────────────────────────────────────

def load_data(path: str | None) -> pd.DataFrame:
    if path and Path(path).exists():
        print(f"Loading observations from {path}")
        df = pd.read_parquet(path)
    else:
        print(f"Local path not found — loading from {_FALLBACK_PARQUET_URL}")
        df = pd.read_parquet(_FALLBACK_PARQUET_URL)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    return df


def _s3_resource():
    from h2s.resources.minio import S3Resource
    return S3Resource(
        S3_BUCKET=os.environ.get("S3_BUCKET", "test"),
        S3_ADDRESS=os.environ.get("S3_ADDRESS", "oss.resilientservice.mooo.com"),
        S3_PORT=os.environ.get("S3_PORT", "443"),
        S3_USE_SSL=os.environ.get("S3_USE_SSL", "true").lower() == "true",
        S3_ACCESS_KEY=os.environ["S3_ACCESS_KEY"],
        S3_SECRET_KEY=os.environ["S3_SECRET_KEY"],
    )


def load_station_models(s3, station_key: str, variant: str) -> dict | None:
    """Load one station's variant model set + feature schema from S3.

    Tries the suffixed names first (`regression_evidence.pkl`, current per-station
    deployment). For the ``evidence`` variant it falls back to the legacy
    un-suffixed names (`regression.pkl`) so the backtest also works against older
    single-variant deployments — those have no Lean and may lack clf_30ppb, which
    degrade gracefully (Lean skipped, p30 → NaN).

    Returns {task: model, '_features': [...], '_legacy': bool} or None if no
    regression model is found.
    """
    base = f"{_STATION_MODELS_BASE}/{station_key}"

    def _load(name):
        try:
            return pickle.loads(s3.getFile(path=f"{base}/{name}", bucket=s3.S3_BUCKET))
        except Exception:
            return None

    models: dict = {}
    legacy = False
    for task in _TASKS:
        m = _load(f"{task}_{variant}.pkl")
        if m is None and variant == "evidence":
            m = _load(f"{task}.pkl")  # legacy un-suffixed deployment
            if m is not None:
                legacy = True
        models[task] = m
    if models.get("regression") is None:
        return None

    feats = None
    candidates = [f"features_{variant}.json"]
    if variant == "evidence":
        candidates.append("features.json")
    for fname in candidates:
        try:
            feats = json.loads(s3.getFile(path=f"{base}/{fname}", bucket=s3.S3_BUCKET).decode("utf-8"))
            break
        except Exception:
            continue
    if feats is None:
        from h2s.constants import MODEL_FEATURES, MODEL_FEATURES_LEAN
        feats = MODEL_FEATURES if variant == "evidence" else MODEL_FEATURES_LEAN

    models["_features"] = feats
    models["_legacy"] = legacy
    return models


# ── prediction ────────────────────────────────────────────────────────────────

def run_predictions(frame: pd.DataFrame, models: dict, station: str, variant: str) -> pd.DataFrame:
    """Score one station's feature frame with one variant's model set."""
    feats = models["_features"]
    for col in feats:
        if col not in frame.columns:
            frame[col] = 0.0
    X = frame[feats].astype(float).to_numpy()

    h2s_pred = np.clip(models["regression"].predict(X), 0.0, None)

    def proba(clf):
        if clf is None:
            return np.full(len(X), np.nan)
        return clf.predict_proba(X)[:, 1]

    out = pd.DataFrame({
        "time": frame["time"].values,
        "station": station,
        "variant": variant,
        "actual_h2s": frame["H2S"].values,
        "h2s_pred": h2s_pred,
        "p5": proba(models.get("clf_5ppb")),
        "p10": proba(models.get("clf_10ppb")),
        "p30": proba(models.get("clf_30ppb")),
    })
    out["month"] = pd.to_datetime(out["time"]).dt.to_period("M").astype(str)
    out["actual_category"] = out["actual_h2s"].apply(_categorize)
    out["pred_category"] = out["h2s_pred"].apply(_categorize)
    return out


# ── metrics ───────────────────────────────────────────────────────────────────

def _threeclass(df: pd.DataFrame) -> dict:
    from sklearn.metrics import balanced_accuracy_score, confusion_matrix
    classes = ["green", "yellow", "orange"]
    y_true, y_pred = df["actual_category"], df["pred_category"]
    cm = confusion_matrix(y_true, y_pred, labels=classes).tolist()
    ba = float(balanced_accuracy_score(y_true, y_pred)) if df["actual_category"].nunique() > 1 else float("nan")
    per_class = {}
    for cls in classes:
        yt = (y_true == cls); yp = (y_pred == cls)
        tp = int((yt & yp).sum()); fp = int((~yt & yp).sum()); fn = int((yt & ~yp).sum())
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        per_class[cls] = {"precision": prec, "recall": rec, "f1": f1, "n": int(yt.sum())}
    orange_yt = (y_true == "orange"); orange_yp = (y_pred == "orange")
    fp_orange = int((orange_yp & ~orange_yt).sum())
    n_non_orange = int((~orange_yt).sum())
    far = fp_orange / n_non_orange if n_non_orange else 0.0
    return {"per_class": per_class, "confusion_matrix": cm,
            "balanced_accuracy": ba, "false_alarm_rate": far}


def _prob_call(df: pd.DataFrame, k: int) -> dict:
    col = _PROB_FOR_THRESHOLD.get(k)
    out = {"recall": None, "precision": None, "n": 0}
    if col is None or col not in df.columns:
        return out
    mask = df[col].notna()
    if not mask.any():
        return out
    pred = df.loc[mask, col] > 0.5
    act = df.loc[mask, "actual_h2s"] >= k
    tp = int((pred & act).sum())
    out["n"] = int(mask.sum())
    out["recall"] = (tp / int(act.sum())) if act.sum() else None
    out["precision"] = (tp / int(pred.sum())) if pred.sum() else None
    return out


def compute_metrics(df: pd.DataFrame) -> dict:
    m = {"n": int(len(df)),
         "spearman": spearman_rank(df["actual_h2s"], df["h2s_pred"]),
         "mae": float((df["h2s_pred"] - df["actual_h2s"]).abs().mean()),
         "mag": {}, "prob": {}}
    for k in _THRESHOLDS:
        r = recall_at_threshold(df["actual_h2s"], df["h2s_pred"], k)
        m["mag"][k] = {"recall": r["recall"], "precision": r["precision"],
                       "n_pos": r["n_positives"]}
        m["prob"][k] = _prob_call(df, k)
    m.update(_threeclass(df))
    return m


def monthly_metrics(records: pd.DataFrame) -> dict[str, dict]:
    out = {m: compute_metrics(records[records["month"] == m])
           for m in sorted(records["month"].unique())}
    out["ALL"] = compute_metrics(records)
    return out


# ── charts ────────────────────────────────────────────────────────────────────

_C = {"green": "#27ae60", "yellow": "#f39c12", "orange": "#e74c3c",
      "blue": "#2980b9", "purple": "#8e44ad"}


def _chart_spearman_recall_by_month(monthly: dict) -> str:
    months = [m for m in sorted(monthly) if m != "ALL"]
    sp = [monthly[m]["spearman"] for m in months]
    r30 = [monthly[m]["mag"][30]["recall"] for m in months]
    r100 = [monthly[m]["mag"][100]["recall"] for m in months]
    xs = range(len(months))
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(xs, sp, color=_C["blue"], marker="o", lw=1.8, label="Spearman (rank skill)")
    ax.plot(xs, r30, color=_C["orange"], marker="s", lw=1.8, label="recall@30 (magnitude)")
    ax.plot(xs, r100, color=_C["purple"], marker="^", lw=1.8, label="recall@100 (magnitude)")
    ax.set_xticks(list(xs)); ax.set_xticklabels(months, rotation=45, ha="right", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
    ax.set_title("Monthly rank skill & exceedance recall", fontsize=11)
    ax.legend(fontsize=8); ax.grid(axis="y", lw=0.4, alpha=0.5)
    fig.tight_layout()
    return _fig_to_b64(fig)


def _chart_mae_by_month(monthly: dict) -> str:
    months = [m for m in sorted(monthly) if m != "ALL"]
    mae = [monthly[m]["mae"] for m in months]
    n_orange = [monthly[m]["per_class"]["orange"]["n"] for m in months]
    xs = np.arange(len(months))
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.bar(xs, mae, color=_C["blue"], alpha=0.75, label="MAE (ppb)")
    ax2 = ax.twinx()
    ax2.plot(xs, n_orange, color=_C["orange"], marker="D", lw=1.6, label="Observed orange hours")
    ax.set_xticks(list(xs)); ax.set_xticklabels(months, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("MAE (ppb)"); ax2.set_ylabel("Orange hours")
    ax.set_title("Monthly MAE vs observed orange-event volume", fontsize=11)
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8); ax.grid(axis="y", lw=0.4, alpha=0.5)
    fig.tight_layout()
    return _fig_to_b64(fig)


def _chart_confusion(m: dict, title: str) -> str:
    cm = np.array(m["confusion_matrix"], dtype=float)
    labels = ["Green", "Yellow", "Orange"]
    fig, ax = plt.subplots(figsize=(5, 4))
    totals = cm.sum(axis=1, keepdims=True)
    norm = np.where(totals > 0, np.divide(cm, totals, where=totals > 0), 0.0)
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks([0, 1, 2]); ax.set_yticks([0, 1, 2])
    ax.set_xticklabels(labels, fontsize=9); ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Predicted", fontsize=9); ax.set_ylabel("Actual", fontsize=9)
    ax.set_title(title, fontsize=10)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{int(cm[i, j])}\n({norm[i, j]:.0%})", ha="center", va="center",
                    fontsize=8, color="white" if norm[i, j] > 0.6 else "black")
    fig.colorbar(im, ax=ax, shrink=0.85); fig.tight_layout()
    return _fig_to_b64(fig)


def _chart_scatter(records: pd.DataFrame, title: str) -> str:
    fig, ax = plt.subplots(figsize=(5, 4.5))
    a = records["actual_h2s"].to_numpy(); p = records["h2s_pred"].to_numpy()
    ax.scatter(a, p, s=6, alpha=0.25, color=_C["blue"], edgecolors="none")
    hi = float(np.nanpercentile(np.concatenate([a, p]), 99.5)) if len(a) else 1.0
    hi = max(hi, 35.0)
    ax.plot([0, hi], [0, hi], color="#888", lw=1.0, ls="--")
    for thr, col in [(5, _C["yellow"]), (30, _C["orange"])]:
        ax.axvline(thr, color=col, lw=0.8, alpha=0.6); ax.axhline(thr, color=col, lw=0.8, alpha=0.6)
    ax.set_xlim(0, hi); ax.set_ylim(0, hi)
    ax.set_xlabel("Actual H2S (ppb)"); ax.set_ylabel("Predicted H2S (ppb)")
    ax.set_title(title, fontsize=10); ax.grid(lw=0.3, alpha=0.4)
    fig.tight_layout()
    return _fig_to_b64(fig)


# ── HTML ──────────────────────────────────────────────────────────────────────

_CSS = """
body { font-family: system-ui, sans-serif; margin: 0; background: #f5f6fa; color: #333; }
.header { background: #1a2233; color: #fff; padding: 18px 32px; }
.header h1 { margin: 0; font-size: 1.4em; }
.header p { margin: 4px 0 0; font-size: 0.85em; color: #aab; }
.content { max-width: 1150px; margin: 28px auto; padding: 0 24px; }
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
.metric-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 14px; margin-bottom: 18px; }
.metric-card { background: #f8fafc; border: 1px solid #e0e0e0; border-radius: 6px; padding: 14px 16px; text-align: center; }
.metric-card .val { font-size: 1.7em; font-weight: 700; color: #1a2233; }
.metric-card .lbl { font-size: 0.76em; color: #777; margin-top: 4px; }
.chart-row { display: flex; gap: 14px; flex-wrap: wrap; margin-bottom: 14px; align-items: flex-start; }
.chart-row img { max-width: 100%; border-radius: 6px; box-shadow: 0 1px 3px rgba(0,0,0,.1); }
.nav { background: #fff; padding: 10px 24px; display: flex; gap: 14px; flex-wrap: wrap; border-bottom: 1px solid #ddd; font-size: 0.85em; }
.nav a { color: #2563eb; text-decoration: none; }
.nav a:hover { text-decoration: underline; }
.badge { display:inline-block; padding:2px 8px; border-radius:10px; font-size:0.72em; font-weight:600; }
.ev { background:#dbeafe; color:#1e40af; } .ln { background:#ede9fe; color:#5b21b6; }
"""


def _cards(m: dict) -> str:
    cards = [
        (_num(m["spearman"]), "Spearman"),
        (_num(m["mae"], "{:.1f}"), "MAE (ppb)"),
        (_pct(m["mag"][30]["recall"]), "recall@30 (mag)"),
        (_pct(m["mag"][100]["recall"]), "recall@100 (mag)"),
        (_pct(m["prob"][30]["recall"]), "P(>30)>.5 recall"),
        (_pct(m["per_class"]["orange"]["recall"]), "Orange detect (3-cls)"),
        (_pct(m["false_alarm_rate"]), "False alarm rate"),
        (f"{m['n']:,}", "Predictions"),
    ]
    html = '<div class="metric-grid">'
    for val, lbl in cards:
        html += f'<div class="metric-card"><div class="val">{val}</div><div class="lbl">{lbl}</div></div>'
    return html + "</div>"


def _threshold_table(m: dict) -> str:
    rows = ""
    for k in _THRESHOLDS:
        mag = m["mag"][k]; prob = m["prob"][k]
        rows += (
            f"<tr><td>≥{k} ppb</td>"
            f"<td>{mag['n_pos']:,}</td>"
            f"<td class='{_color_cell(mag['recall'])}'>{_pct(mag['recall'])}</td>"
            f"<td class='{_color_cell(mag['precision'])}'>{_pct(mag['precision'])}</td>"
            f"<td class='{_color_cell(prob['recall'])}'>{_pct(prob['recall'])}</td>"
            f"<td class='{_color_cell(prob['precision'])}'>{_pct(prob['precision'])}</td></tr>"
        )
    return f"""<table><thead><tr>
      <th>Threshold</th><th>Observed events</th>
      <th>Recall (magnitude)</th><th>Precision (magnitude)</th>
      <th>Recall (P&gt;0.5)</th><th>Precision (P&gt;0.5)</th></tr></thead>
      <tbody>{rows}</tbody></table>"""


def _monthly_table(monthly: dict, months: list[str]) -> str:
    head = ("<tr><th>Month</th><th>n</th><th>Spearman</th><th>MAE</th>"
            "<th>recall@30 (mag)</th><th>recall@100 (mag)</th>"
            "<th>P(&gt;30)&gt;.5 recall</th><th>Orange detect</th><th>FAR</th></tr>")
    rows = ""
    for m in months:
        mm = monthly[m]
        rows += (
            f"<tr><td><b>{m}</b></td><td>{mm['n']:,}</td>"
            f"<td class='{_color_cell(mm['spearman'])}'>{_num(mm['spearman'])}</td>"
            f"<td>{_num(mm['mae'], '{:.1f}')}</td>"
            f"<td class='{_color_cell(mm['mag'][30]['recall'])}'>{_pct(mm['mag'][30]['recall'])}</td>"
            f"<td class='{_color_cell(mm['mag'][100]['recall'])}'>{_pct(mm['mag'][100]['recall'])}</td>"
            f"<td class='{_color_cell(mm['prob'][30]['recall'])}'>{_pct(mm['prob'][30]['recall'])}</td>"
            f"<td class='{_color_cell(mm['per_class']['orange']['recall'])}'>{_pct(mm['per_class']['orange']['recall'])}</td>"
            f"<td class='{_color_cell(1 - mm['false_alarm_rate'], low=0.9, high=0.96)}'>{_pct(mm['false_alarm_rate'])}</td></tr>"
        )
    return f"<table><thead>{head}</thead><tbody>{rows}</tbody></table>"


def build_station_page(station: str, variant: str, records: pd.DataFrame, monthly: dict) -> str:
    overall = monthly["ALL"]
    months = [m for m in sorted(monthly) if m != "ALL"]
    rng = f"{records['time'].min().strftime('%Y-%m-%d')} – {records['time'].max().strftime('%Y-%m-%d')}"
    vlabel = "Evidence (33 feat)" if variant == "evidence" else "Lean (19 feat)"
    return f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Backtest — {station} / {variant}</title><style>{_CSS}</style></head><body>
<div class="header"><h1>{station} — {vlabel}</h1>
<p>Per-station model backtest (oracle one-step inputs) · {rng} · {overall['n']:,} hours</p></div>
<div class="nav"><a href="index.html">↑ All stations</a></div>
<div class="content">
  <div class="section"><h2>Overall</h2>{_cards(overall)}
    <div class="chart-row">
      <img src="data:image/png;base64,{_chart_confusion(overall, 'Confusion matrix (category from regression)')}" style="max-width:380px">
      <img src="data:image/png;base64,{_chart_scatter(records, 'Predicted vs actual H2S')}" style="max-width:430px">
    </div>
    <h3>Threshold detection — magnitude cut vs classifier probability</h3>
    {_threshold_table(overall)}
  </div>
  <div class="section"><h2>Monthly skill</h2>
    <img src="data:image/png;base64,{_chart_spearman_recall_by_month(monthly)}" style="width:100%">
    <img src="data:image/png;base64,{_chart_mae_by_month(monthly)}" style="width:100%">
  </div>
  <div class="section"><h2>Monthly summary</h2>{_monthly_table(monthly, months)}</div>
</div></body></html>"""


def build_index(summaries: list[dict], records: pd.DataFrame) -> str:
    rng = f"{records['time'].min().strftime('%Y-%m-%d')} – {records['time'].max().strftime('%Y-%m-%d')}"
    head = ("<tr><th>Station</th><th>Variant</th><th>n</th><th>Spearman</th><th>MAE</th>"
            "<th>recall@30</th><th>recall@100</th><th>P(&gt;30)&gt;.5 recall</th>"
            "<th>Orange detect</th><th>FAR</th></tr>")
    rows = ""
    for s in summaries:
        m = s["overall"]; badge = "ev" if s["variant"] == "evidence" else "ln"
        rows += (
            f"<tr><td><a href='{s['page']}'>{s['station']}</a></td>"
            f"<td><span class='badge {badge}'>{s['variant']}</span></td>"
            f"<td>{m['n']:,}</td>"
            f"<td class='{_color_cell(m['spearman'])}'>{_num(m['spearman'])}</td>"
            f"<td>{_num(m['mae'], '{:.1f}')}</td>"
            f"<td class='{_color_cell(m['mag'][30]['recall'])}'>{_pct(m['mag'][30]['recall'])}</td>"
            f"<td class='{_color_cell(m['mag'][100]['recall'])}'>{_pct(m['mag'][100]['recall'])}</td>"
            f"<td class='{_color_cell(m['prob'][30]['recall'])}'>{_pct(m['prob'][30]['recall'])}</td>"
            f"<td class='{_color_cell(m['per_class']['orange']['recall'])}'>{_pct(m['per_class']['orange']['recall'])}</td>"
            f"<td class='{_color_cell(1 - m['false_alarm_rate'], low=0.9, high=0.96)}'>{_pct(m['false_alarm_rate'])}</td></tr>"
        )
    return f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Per-station model backtest</title><style>{_CSS}</style></head><body>
<div class="header"><h1>Per-Station H2S Model Backtest</h1>
<p>Evidence + Lean · regression + clf_5/10/30 · oracle one-step inputs · {rng}</p></div>
<div class="content">
  <div class="section"><h2>All stations × variants</h2>
  <p style="font-size:0.85em;color:#555;">Magnitude metrics cut the regression prediction at the
  threshold; <i>P(&gt;30)&gt;.5 recall</i> is the classifier's exceedance call. Features use observed
  H2S lags (one-step / oracle), so these are upper-bound skill given good inputs — not the recursive
  multi-step forecast (see the Phase-5 validation store for skill-vs-lead-hour).</p>
  <table><thead>{head}</thead><tbody>{rows}</tbody></table></div>
</div></body></html>"""


# ── orchestration ─────────────────────────────────────────────────────────────

def generate_reports(records: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for (station, variant), sub in records.groupby(["station", "variant"]):
        monthly = monthly_metrics(sub)
        page = f"{station.replace(' ', '_').replace('-', '')}_{variant}.html"
        (output_dir / page).write_text(
            build_station_page(station, variant, sub, monthly), encoding="utf-8")
        print(f"  {page}")
        summaries.append({"station": station, "variant": variant,
                          "page": page, "overall": monthly["ALL"]})
    summaries.sort(key=lambda s: (s["station"], s["variant"]))
    (output_dir / "index.html").write_text(build_index(summaries, records), encoding="utf-8")
    print(f"\nDone. Open: {output_dir / 'index.html'}")


def main() -> None:
    p = argparse.ArgumentParser(description="Backtest deployed per-station H2S models")
    p.add_argument("--data", help="Path to modeldata_h2s_nofill.parquet (falls back to public URL)")
    p.add_argument("--output", default="./output/station_backtest", help="Output directory")
    p.add_argument("--report-only", metavar="RECORDS_PARQUET",
                   help="Skip prediction; rebuild reports from saved records parquet")
    p.add_argument("--stations", nargs="*", help="Station names (default: all three)")
    p.add_argument("--variants", nargs="*", default=list(_VARIANTS), help="evidence / lean")
    p.add_argument("--start", help="Filter from date (YYYY-MM-DD)")
    p.add_argument("--end", help="Filter to date (YYYY-MM-DD inclusive)")
    args = p.parse_args()
    out_dir = Path(args.output)

    if args.report_only:
        print(f"Loading records from {args.report_only} ...")
        records = pd.read_parquet(args.report_only)
        records["time"] = pd.to_datetime(records["time"], utc=True)
        generate_reports(records, out_dir)
        return

    print("Loading observation data...")
    df = load_data(args.data)
    if args.start:
        df = df[df["time"] >= pd.Timestamp(args.start, tz="UTC")]
    if args.end:
        df = df[df["time"] <= pd.Timestamp(args.end, tz="UTC") + pd.Timedelta(days=1)]
    print(f"  {len(df):,} rows, {df['time'].min().date()} – {df['time'].max().date()}")

    print("Engineering features (prepare_multi_station_features)...")
    feat = prepare_multi_station_features(df)
    print(f"  {len(feat):,} feature rows across {feat['site_name'].nunique()} stations")

    stations = args.stations or [info_name for info_name in STATIONS]
    s3 = _s3_resource()
    print(f"Loading models from s3://{s3.S3_BUCKET}/{_STATION_MODELS_BASE} ...")

    all_records = []
    for station in stations:
        if station not in STATIONS:
            print(f"  ! unknown station '{station}' — skipping"); continue
        key = STATIONS[station]["key"]
        sframe = feat[feat["site_name"] == station].copy()
        if sframe.empty:
            print(f"  ! {station}: no rows — skipping"); continue
        for variant in args.variants:
            models = load_station_models(s3, key, variant)
            if models is None:
                print(f"  ! {station}/{variant}: no deployed model — skipping"); continue
            rec = run_predictions(sframe, models, station, variant)
            all_records.append(rec)
            tag = " [legacy un-suffixed]" if models.get("_legacy") else ""
            has30 = "p30" if models.get("clf_30ppb") is not None else "no-clf30"
            print(f"  ✓ {station}/{variant}: {len(rec):,} predictions, "
                  f"{len(models['_features'])} feat, {has30}{tag} "
                  f"(Spearman={spearman_rank(rec['actual_h2s'], rec['h2s_pred']):.3f})")

    if not all_records:
        print("No predictions generated — check model deployment / station names.")
        sys.exit(1)

    records = pd.concat(all_records, ignore_index=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    records.to_parquet(out_dir / "records.parquet", index=False)
    print(f"  Records saved → {out_dir / 'records.parquet'}")

    print("Generating reports...")
    generate_reports(records, out_dir)


if __name__ == "__main__":
    main()
