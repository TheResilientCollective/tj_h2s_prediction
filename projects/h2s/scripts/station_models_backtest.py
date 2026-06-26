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
  <STATION>_<variant>.html              — per (station, variant) detail + monthly summary + month tabs
  <STATION>_<variant>_monthly/<YYYY-MM>.html  — individual month detail pages (clickable from summary)

Usage (S3 deployed models — requires env vars from .env):
    cd projects/h2s
    set -a; source .env; set +a
    uv run python scripts/station_models_backtest.py \\
        --data ../../data/modeldata_h2s_nofill.parquet \\
        --output ./output/station_backtest/

Usage (also upload the rendered report tree to S3):
    # models load from S3_BUCKET; reports upload to --s3-bucket under
    # tijuana/forecast/backtest/station_models/ (override with --s3-prefix)
    S3_BUCKET=resilentpublic uv run python scripts/station_models_backtest.py \\
        --s3-bucket resilentpublic

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

from h2s.constants import BACKTEST_RESULTS_BASE, STATIONS  # noqa: E402
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
# 8 ppb is the resident-complaint trigger (the 5–10 "yellow-low" band), so it
# sits in the ladder as a magnitude-only threshold — there is no clf_8ppb.
_THRESHOLDS = (5, 8, 10, 30, 100)
_PROB_FOR_THRESHOLD = {5: "p5", 10: "p10", 30: "p30"}
_COMPLAINT_PPB = 8

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


# Default S3 destination for the rendered backtest (kept under a `station_models`
# subprefix so it never collides with other reports written to BACKTEST_RESULTS_BASE).
_DEFAULT_S3_PREFIX = f"{BACKTEST_RESULTS_BASE}/station_models"

_CONTENT_TYPES = {
    ".html": "text/html",
    ".json": "application/json",
    ".parquet": "application/octet-stream",
    ".png": "image/png",
    ".csv": "text/csv",
}


def upload_reports_to_s3(out_dir: Path, bucket: str, prefix: str) -> int:
    """Upload every file under *out_dir* to ``s3://{bucket}/{prefix}/`` preserving
    the relative tree. Returns the number of objects written."""
    s3 = _s3_resource()
    prefix = prefix.rstrip("/")
    n = 0
    for f in sorted(out_dir.rglob("*")):
        if not f.is_file():
            continue
        key = f"{prefix}/{f.relative_to(out_dir).as_posix()}"
        s3.putFile(
            f.read_bytes(), key, bucket=bucket,
            content_type=_CONTENT_TYPES.get(f.suffix, "application/octet-stream"),
        )
        n += 1
    print(f"  Uploaded {n} files → s3://{bucket}/{prefix}/")
    print(f"  Index: {s3.publicUrl(path=f'{prefix}/index.html', bucket=bucket)}")
    return n


def load_station_models(s3, station_key: str, variant: str) -> dict | None:
    """Load one station's variant model set + feature schema from S3.

    Pins a single variant: loads only the suffixed names (`regression_evidence.pkl`,
    `features_evidence.json`). There is deliberately NO un-suffixed legacy fallback
    — the stale 2026-05-20 `{task}.pkl` / `features.json` pickles are 44-feature and
    would silently corrupt inference if loaded against the 33-feature Evidence (or
    19-feature Lean) schema. Mirrors the production loader and the hardened
    `backfill_validation.py`. A missing pickle stays None (clf_30ppb may be absent
    before a station's first post-Phase-1 retrain → p30 NaN).

    Returns {task: model, '_features': [...], '_legacy': bool, '_metadata': {...}} or None if no
    regression model is found.
    """
    base = f"{_STATION_MODELS_BASE}/{station_key}"

    def _load(name):
        try:
            return pickle.loads(s3.getFile(path=f"{base}/{name}", bucket=s3.S3_BUCKET))
        except Exception:
            return None

    models: dict = {}
    for task in _TASKS:
        models[task] = _load(f"{task}_{variant}.pkl")
    if models.get("regression") is None:
        return None

    feats = None
    try:
        feats = json.loads(
            s3.getFile(path=f"{base}/features_{variant}.json", bucket=s3.S3_BUCKET).decode("utf-8")
        )
    except Exception:
        pass
    if feats is None:
        from h2s.constants import MODEL_FEATURES, MODEL_FEATURES_LEAN
        feats = MODEL_FEATURES if variant == "evidence" else MODEL_FEATURES_LEAN

    # Try to load training metadata (training_report.json has date ranges)
    metadata = {}
    for bucket in [s3.S3_BUCKET, "resilientpublic"]:
        try:
            # Try training_report.json first (has training data info)
            meta_bytes = s3.getFile(path=f"{base}/training_report.json", bucket=bucket)
            metadata = json.loads(meta_bytes.decode("utf-8"))
            break
        except Exception:
            pass
        try:
            # Fallback to deployment_metadata.json
            meta_bytes = s3.getFile(path=f"{base}/deployment_metadata.json", bucket=bucket)
            metadata = json.loads(meta_bytes.decode("utf-8"))
            break
        except Exception:
            pass

    models["_features"] = feats
    models["_legacy"] = False  # legacy un-suffixed fallback removed (variant pinned)
    models["_metadata"] = metadata
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


def _get_training_end_month(metadata: dict = None, records: pd.DataFrame = None) -> str | None:
    """Extract training end month from metadata or estimate from training data split."""
    if not metadata:
        return None

    # Try different metadata field names for explicit date range
    candidates = [
        ('training_data_range', 'training_data_range'),  # deployment_metadata format: "2024-01 to 2024-09"
        ('training_date_range', 'training_date_range'),   # alternate format
        ('training_end_date', 'training_end_date'),       # single date format
        ('training_period', 'training_period'),           # another alternate
        ('training_dates', 'training_dates'),
        ('data_range', 'data_range'),
    ]

    for field_name, desc in candidates:
        if field_name not in metadata:
            continue
        try:
            value = metadata[field_name]
            if not isinstance(value, str):
                continue

            # Parse "2024-01 to 2024-09" format
            if ' to ' in value:
                end_date = value.split(' to ')[1].strip()
            else:
                # Single date like "2024-09" or "2024-09-30"
                end_date = value

            # Extract YYYY-MM from various date formats
            if '-' in end_date:
                parts = end_date.split('-')
                return '-'.join(parts[:2])
            return end_date
        except Exception:
            continue

    # Fallback: if we have training snapshot with timestamp, use that as approximate training end
    try:
        if 'training_snapshot' in metadata and isinstance(metadata['training_snapshot'], dict):
            timestamp = metadata['training_snapshot'].get('timestamp', '')
            if timestamp and 'T' in timestamp:
                # Format like "2026-06-16T185116Z" -> "2026-06"
                date_part = timestamp.split('T')[0]
                return '-'.join(date_part.split('-')[:2])
    except Exception:
        pass

    return None


def _chart_spearman_recall_by_month(monthly: dict, metadata: dict = None) -> str:
    months = [m for m in sorted(monthly) if m != "ALL"]
    sp = [monthly[m]["spearman"] for m in months]
    r8 = [monthly[m]["mag"][8]["recall"] for m in months]
    r10 = [monthly[m]["mag"][10]["recall"] for m in months]
    r30 = [monthly[m]["mag"][30]["recall"] for m in months]
    xs = range(len(months))
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(xs, sp, color=_C["blue"], marker="o", lw=1.8, label="Spearman (rank skill)")
    ax.plot(xs, r8, color="#d4ac0d", marker="*", ms=8, lw=1.8, label="recall@8 (complaint)")
    ax.plot(xs, r10, color=_C["yellow"], marker="s", lw=1.6, label="recall@10")
    ax.plot(xs, r30, color=_C["orange"], marker="^", lw=1.6, label="recall@30")

    # Add training/test split line if metadata available
    training_end = _get_training_end_month(metadata)
    if training_end:
        if training_end in months:
            split_idx = months.index(training_end) + 0.5
            ax.axvline(split_idx, color="#e74c3c", lw=2, ls="--", alpha=0.7, label="Training data end")
        elif training_end > months[-1] if months else False:
            # Training end is beyond backtest data - show at the right edge
            # Position line slightly past the last data point to make it visible
            split_idx = len(months) + 0.3
            ax.axvline(split_idx, color="#e74c3c", lw=2.5, ls="--", alpha=0.8, label=f"Training data end ({training_end})")

    ax.set_xticks(list(xs)); ax.set_xticklabels(months, rotation=45, ha="right", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
    ax.set_title("Monthly rank skill & alert detection (complaint band → watch)", fontsize=11)
    ax.legend(fontsize=8, ncol=2); ax.grid(axis="y", lw=0.4, alpha=0.5)
    fig.tight_layout()
    return _fig_to_b64(fig)


def _chart_mae_by_month(monthly: dict, metadata: dict = None) -> str:
    months = [m for m in sorted(monthly) if m != "ALL"]
    mae = [monthly[m]["mae"] for m in months]
    n_orange = [monthly[m]["per_class"]["orange"]["n"] for m in months]
    xs = np.arange(len(months))
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.bar(xs, mae, color=_C["blue"], alpha=0.75, label="MAE (ppb)")
    ax2 = ax.twinx()
    ax2.plot(xs, n_orange, color=_C["orange"], marker="D", lw=1.6, label="Observed orange hours")

    # Add training/test split line if metadata available
    training_end = _get_training_end_month(metadata)
    if training_end:
        if training_end in months:
            split_idx = months.index(training_end) + 0.5
            ax.axvline(split_idx, color="#e74c3c", lw=2, ls="--", alpha=0.7, label="Training data end")
        elif training_end > months[-1] if months else False:
            # Training end is beyond backtest data - show at the right edge
            # Position line slightly past the last data point to make it visible
            split_idx = len(months) + 0.3
            ax.axvline(split_idx, color="#e74c3c", lw=2.5, ls="--", alpha=0.8, label=f"Training data end ({training_end})")

    ax.set_xticks(list(xs)); ax.set_xticklabels(months, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("MAE (ppb)"); ax2.set_ylabel("Orange hours")
    ax.set_title("Monthly MAE vs observed orange-event volume", fontsize=11)
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8); ax.grid(axis="y", lw=0.4, alpha=0.5)
    fig.tight_layout()
    return _fig_to_b64(fig)


def _chart_category_recall_by_month(monthly: dict, metadata: dict = None) -> str:
    months = [m for m in sorted(monthly) if m != "ALL"]
    green_recall = [monthly[m]["per_class"]["green"]["recall"] for m in months]
    yellow_recall = [monthly[m]["per_class"]["yellow"]["recall"] for m in months]
    orange_recall = [monthly[m]["per_class"]["orange"]["recall"] for m in months]
    xs = np.arange(len(months))
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(xs, green_recall, color=_C["green"], marker="o", lw=1.8, label="Green recall")
    ax.plot(xs, yellow_recall, color=_C["yellow"], marker="s", lw=1.8, label="Yellow recall")
    ax.plot(xs, orange_recall, color=_C["orange"], marker="^", lw=1.8, label="Orange recall")

    # Add training/test split line if metadata available
    training_end = _get_training_end_month(metadata)
    if training_end:
        if training_end in months:
            split_idx = months.index(training_end) + 0.5
            ax.axvline(split_idx, color="#e74c3c", lw=2, ls="--", alpha=0.7, label="Training data end")
        elif training_end > months[-1] if months else False:
            # Training end is beyond backtest data - show at the right edge
            # Position line slightly past the last data point to make it visible
            split_idx = len(months) + 0.3
            ax.axvline(split_idx, color="#e74c3c", lw=2.5, ls="--", alpha=0.8, label=f"Training data end ({training_end})")

    ax.set_xticks(list(xs)); ax.set_xticklabels(months, rotation=45, ha="right", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
    ax.set_title("Monthly category detection rates (3-class classification)", fontsize=11)
    ax.legend(fontsize=8, ncol=3); ax.grid(axis="y", lw=0.4, alpha=0.5)
    fig.tight_layout()
    return _fig_to_b64(fig)


def _chart_category_precision_by_month(monthly: dict, metadata: dict = None) -> str:
    months = [m for m in sorted(monthly) if m != "ALL"]
    green_prec = [monthly[m]["per_class"]["green"]["precision"] for m in months]
    yellow_prec = [monthly[m]["per_class"]["yellow"]["precision"] for m in months]
    orange_prec = [monthly[m]["per_class"]["orange"]["precision"] for m in months]
    xs = np.arange(len(months))
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(xs, green_prec, color=_C["green"], marker="o", lw=1.8, label="Green precision")
    ax.plot(xs, yellow_prec, color=_C["yellow"], marker="s", lw=1.8, label="Yellow precision")
    ax.plot(xs, orange_prec, color=_C["orange"], marker="^", lw=1.8, label="Orange precision")

    # Add training/test split line if metadata available
    training_end = _get_training_end_month(metadata)
    if training_end:
        if training_end in months:
            split_idx = months.index(training_end) + 0.5
            ax.axvline(split_idx, color="#e74c3c", lw=2, ls="--", alpha=0.7, label="Training data end")
        elif training_end > months[-1] if months else False:
            # Training end is beyond backtest data - show at the right edge
            # Position line slightly past the last data point to make it visible
            split_idx = len(months) + 0.3
            ax.axvline(split_idx, color="#e74c3c", lw=2.5, ls="--", alpha=0.8, label=f"Training data end ({training_end})")

    ax.set_xticks(list(xs)); ax.set_xticklabels(months, rotation=45, ha="right", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
    ax.set_title("Monthly false alarm rates by category", fontsize=11)
    ax.legend(fontsize=8, ncol=3); ax.grid(axis="y", lw=0.4, alpha=0.5)
    fig.tight_layout()
    return _fig_to_b64(fig)


def _chart_category_volume_by_month(monthly: dict, metadata: dict = None) -> str:
    months = [m for m in sorted(monthly) if m != "ALL"]
    green_n = [monthly[m]["per_class"]["green"]["n"] for m in months]
    yellow_n = [monthly[m]["per_class"]["yellow"]["n"] for m in months]
    orange_n = [monthly[m]["per_class"]["orange"]["n"] for m in months]
    xs = np.arange(len(months))
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.bar(xs - 0.25, green_n, width=0.25, color=_C["green"], alpha=0.75, label="Green hours")
    ax.bar(xs, yellow_n, width=0.25, color=_C["yellow"], alpha=0.75, label="Yellow hours")
    ax.bar(xs + 0.25, orange_n, width=0.25, color=_C["orange"], alpha=0.75, label="Orange hours")

    # Add training/test split line if metadata available
    training_end = _get_training_end_month(metadata)
    if training_end:
        if training_end in months:
            split_idx = months.index(training_end) + 0.5
            ax.axvline(split_idx, color="#e74c3c", lw=2, ls="--", alpha=0.7, label="Training data end")
        elif training_end > months[-1] if months else False:
            # Training end is beyond backtest data - show at the right edge
            # Position line slightly past the last data point to make it visible
            split_idx = len(months) + 0.3
            ax.axvline(split_idx, color="#e74c3c", lw=2.5, ls="--", alpha=0.8, label=f"Training data end ({training_end})")

    ax.set_xticks(list(xs)); ax.set_xticklabels(months, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Hours observed")
    ax.set_title("Monthly distribution of observed H2S categories", fontsize=11)
    ax.legend(fontsize=8, ncol=3); ax.grid(axis="y", lw=0.4, alpha=0.5)
    fig.tight_layout()
    return _fig_to_b64(fig)


def _chart_confusion(m: dict, title: str) -> str:
    cm = np.array(m["confusion_matrix"], dtype=float)
    labels = ["Green", "Yellow", "Orange"]
    fig, ax = plt.subplots(figsize=(5, 4))
    totals = cm.sum(axis=1, keepdims=True)
    norm = np.zeros_like(cm)
    mask = totals.ravel() > 0
    norm[mask] = cm[mask] / totals[mask]
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
.month-tabs { background: #fff; padding: 0 24px; display: flex; gap: 4px; border-bottom: 1px solid #ddd; font-size: 0.85em; margin-bottom: 12px; }
.month-tab { padding: 10px 14px; cursor: pointer; text-decoration: none; color: #2563eb; border-bottom: 2px solid transparent; margin-bottom: -1px; }
.month-tab:hover { background: #f8fafc; }
.month-tab.active { color: #fff; background: #1a2233; border-bottom-color: #1a2233; }
.badge { display:inline-block; padding:2px 8px; border-radius:10px; font-size:0.72em; font-weight:600; }
.ev { background:#dbeafe; color:#1e40af; } .ln { background:#ede9fe; color:#5b21b6; }
.month-link { color: #2563eb; text-decoration: none; cursor: pointer; }
.month-link:hover { text-decoration: underline; }
.metadata-box { background: #f0f4ff; border-left: 4px solid #2563eb; padding: 12px 14px; margin-bottom: 16px; font-size: 0.84em; border-radius: 4px; }
.metadata-box > div { margin: 5px 0; }
.metadata-box b { color: #1a2233; }
"""


def _metadata_section(metadata: dict = None, test_data_range: tuple = None) -> str:
    """Format model and test data metadata into HTML."""
    html = '<div class="metadata-box">'

    # Model training info
    if metadata and metadata.get('model_version'):
        html += f"<div><b>Model version:</b> {metadata['model_version']}</div>"
    if metadata and metadata.get('training_date'):
        html += f"<div><b>Trained:</b> {metadata['training_date']}</div>"
    if metadata and metadata.get('training_data_range'):
        html += f"<div><b>Training data:</b> {metadata['training_data_range']}</div>"

    # Test data info
    if test_data_range:
        start, end = test_data_range
        html += f"<div><b>Backtest data (holdout):</b> {start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}</div>"
        html += f"<div style='font-size:0.85em;color:#666;margin-top:6px;'>Features use observed H2S values (oracle one-step inputs) — not recursive forecasts</div>"

    html += '</div>'
    return html


def _cards(m: dict) -> str:
    cards = [
        (_num(m["spearman"]), "Spearman"),
        (_num(m["mae"], "{:.1f}"), "MAE (ppb)"),
        (_pct(m["mag"][5]["recall"]), "recall@5"),
        (_pct(m["mag"][8]["recall"]), "recall@8 (complaint)"),
        (_pct(m["mag"][10]["recall"]), "recall@10"),
        (_pct(m["mag"][30]["recall"]), "recall@30"),
        (_pct(m["mag"][100]["recall"]), "recall@100"),
        (_pct(m["false_alarm_rate"]), "False alarm rate"),
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


def _monthly_table(monthly: dict, months: list[str], station: str = None, variant: str = None) -> str:
    head = ("<tr><th>Month</th><th>n</th><th>Spearman</th><th>MAE</th>"
            "<th>recall@5</th><th>recall@8</th><th>recall@10</th>"
            "<th>recall@30</th><th>recall@100</th><th>FAR</th></tr>")
    rows = ""
    for m in months:
        mm = monthly[m]
        if station and variant:
            station_clean = station.replace(' ', '_').replace('-', '')
            month_link = f"<a href='{station_clean}_{variant}_monthly/{m}.html' class='month-link'>{m}</a>"
        else:
            month_link = m
        rows += (
            f"<tr><td><b>{month_link}</b></td><td>{mm['n']:,}</td>"
            f"<td class='{_color_cell(mm['spearman'])}'>{_num(mm['spearman'])}</td>"
            f"<td>{_num(mm['mae'], '{:.1f}')}</td>"
            f"<td class='{_color_cell(mm['mag'][5]['recall'])}'>{_pct(mm['mag'][5]['recall'])}</td>"
            f"<td class='{_color_cell(mm['mag'][8]['recall'])}'>{_pct(mm['mag'][8]['recall'])}</td>"
            f"<td class='{_color_cell(mm['mag'][10]['recall'])}'>{_pct(mm['mag'][10]['recall'])}</td>"
            f"<td class='{_color_cell(mm['mag'][30]['recall'])}'>{_pct(mm['mag'][30]['recall'])}</td>"
            f"<td class='{_color_cell(mm['mag'][100]['recall'])}'>{_pct(mm['mag'][100]['recall'])}</td>"
            f"<td class='{_color_cell(1 - mm['false_alarm_rate'], low=0.9, high=0.96)}'>{_pct(mm['false_alarm_rate'])}</td></tr>"
        )
    return f"<table><thead>{head}</thead><tbody>{rows}</tbody></table>"


def build_month_page(station: str, variant: str, month: str, month_data: pd.DataFrame, metrics: dict, metadata: dict = None) -> str:
    """Build detail page for a single month."""
    vlabel = "Evidence (33 feat)" if variant == "evidence" else "Lean (19 feat)"
    page_title = f"{station} — {vlabel} — {month}"
    rng = f"{month_data['time'].min().strftime('%Y-%m-%d')} – {month_data['time'].max().strftime('%Y-%m-%d')}"
    station_clean = station.replace(' ', '_').replace('-', '')

    metadata_html = ""
    if metadata and metadata.get('model_version'):
        metadata_html = f"""<div class="metadata-box">
    <div><b>Model version:</b> {metadata.get('model_version')}</div>"""
        if metadata.get('training_date'):
            metadata_html += f"<div><b>Trained:</b> {metadata['training_date']}</div>"
        metadata_html += "</div>"

    return f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Backtest — {page_title}</title><style>{_CSS}</style></head><body>
<div class="header"><h1>{page_title}</h1>
<p>{rng} · {metrics['n']:,} hours</p></div>
<div class="nav"><a href="../{station_clean}_{variant}.html">← Back to {station}</a></div>
<div class="content">
  {f'<div class="section"><h2>Model Info</h2>{metadata_html}</div>' if metadata_html else ''}
  <div class="section"><h2>Detailed metrics for {month}</h2>{_cards(metrics)}
    <div class="chart-row">
      <img src="data:image/png;base64,{_chart_confusion(metrics, 'Confusion matrix')}" style="max-width:380px">
      <img src="data:image/png;base64,{_chart_scatter(month_data, 'Predicted vs actual H2S')}" style="max-width:430px">
    </div>
    <h3>Threshold detection — magnitude cut vs classifier probability</h3>
    {_threshold_table(metrics)}
  </div>
</div></body></html>"""


def build_station_page(station: str, variant: str, records: pd.DataFrame, monthly: dict, metadata: dict = None) -> str:
    overall = monthly["ALL"]
    months = [m for m in sorted(monthly) if m != "ALL"]
    rng = f"{records['time'].min().strftime('%Y-%m-%d')} – {records['time'].max().strftime('%Y-%m-%d')}"
    vlabel = "Evidence (33 feat)" if variant == "evidence" else "Lean (19 feat)"

    month_tabs = '<div class="month-tabs">'
    month_tabs += '<a href="#overall" class="month-tab active">Overall</a>'
    for m in months:
        month_tabs += f'<a href="#{m}" class="month-tab">{m}</a>'
    month_tabs += '</div>'

    test_data_range = (records['time'].min(), records['time'].max())
    metadata_html = _metadata_section(metadata, test_data_range)

    return f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Backtest — {station} / {variant}</title><style>{_CSS}</style></head><body>
<div class="header"><h1>{station} — {vlabel}</h1>
<p>Per-station model backtest (oracle one-step inputs) · {rng} · {overall['n']:,} hours</p></div>
<div class="nav"><a href="index.html">↑ All stations</a></div>
{month_tabs}
<div class="content">
  <div class="section"><h2 id="overall">Model & Data Info</h2>{metadata_html}</div>
  <div class="section"><h2 id="overall">Overall</h2>{_cards(overall)}
    <div class="chart-row">
      <img src="data:image/png;base64,{_chart_confusion(overall, 'Confusion matrix (category from regression)')}" style="max-width:380px">
      <img src="data:image/png;base64,{_chart_scatter(records, 'Predicted vs actual H2S')}" style="max-width:430px">
    </div>
    <h3>Threshold detection — magnitude cut vs classifier probability</h3>
    {_threshold_table(overall)}
  </div>
  <div class="section"><h2>Monthly skill trends</h2>
    <img src="data:image/png;base64,{_chart_spearman_recall_by_month(monthly, metadata)}" style="width:100%">
    <img src="data:image/png;base64,{_chart_mae_by_month(monthly, metadata)}" style="width:100%">
    <img src="data:image/png;base64,{_chart_category_recall_by_month(monthly, metadata)}" style="width:100%">
    <img src="data:image/png;base64,{_chart_category_precision_by_month(monthly, metadata)}" style="width:100%">
    <img src="data:image/png;base64,{_chart_category_volume_by_month(monthly, metadata)}" style="width:100%">
  </div>
  <div class="section"><h2>Monthly summary</h2>
    <p style="font-size:0.85em;color:#555;">Click on a month to see detailed metrics for that month.</p>
    {_monthly_table(monthly, months, station, variant)}</div>
</div></body></html>"""


def build_index(summaries: list[dict], records: pd.DataFrame) -> str:
    rng = f"{records['time'].min().strftime('%Y-%m-%d')} – {records['time'].max().strftime('%Y-%m-%d')}"
    head = ("<tr><th>Station</th><th>Variant</th><th>n</th><th>Spearman</th>"
            "<th>recall@5</th><th>recall@8</th><th>recall@10</th>"
            "<th>recall@30</th><th>recall@100</th><th>FAR</th></tr>")
    rows = ""
    for s in summaries:
        m = s["overall"]; badge = "ev" if s["variant"] == "evidence" else "ln"
        rows += (
            f"<tr><td><a href='{s['page']}'>{s['station']}</a></td>"
            f"<td><span class='badge {badge}'>{s['variant']}</span></td>"
            f"<td>{m['n']:,}</td>"
            f"<td class='{_color_cell(m['spearman'])}'>{_num(m['spearman'])}</td>"
            f"<td class='{_color_cell(m['mag'][5]['recall'])}'>{_pct(m['mag'][5]['recall'])}</td>"
            f"<td class='{_color_cell(m['mag'][8]['recall'])}'>{_pct(m['mag'][8]['recall'])}</td>"
            f"<td class='{_color_cell(m['mag'][10]['recall'])}'>{_pct(m['mag'][10]['recall'])}</td>"
            f"<td class='{_color_cell(m['mag'][30]['recall'])}'>{_pct(m['mag'][30]['recall'])}</td>"
            f"<td class='{_color_cell(m['mag'][100]['recall'])}'>{_pct(m['mag'][100]['recall'])}</td>"
            f"<td class='{_color_cell(1 - m['false_alarm_rate'], low=0.9, high=0.96)}'>{_pct(m['false_alarm_rate'])}</td></tr>"
        )
    return f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Per-station model backtest</title><style>{_CSS}</style></head><body>
<div class="header"><h1>Per-Station H2S Model Backtest</h1>
<p>Evidence + Lean · regression + clf_5/10/30 · oracle one-step inputs · {rng}</p></div>
<div class="content">
  <div class="section"><h2>All stations × variants</h2>
  <p style="font-size:0.85em;color:#555;"><b>Alerting focus:</b> residents report H2S complaints
  around <b>~8 ppb</b> (the 5–10 "yellow-low" band), so <b>recall@5 / @8 / @10</b> are the
  alerting-relevant detection rates; recall@30 / @100 cover the watch / extreme tiers. Each number
  is the magnitude recall — the share of hours that actually crossed the threshold that the model's
  predicted ppb also crossed. The per-station pages add the classifier's P(&gt;k)&gt;0.5 call and
  precision. Features use observed H2S lags (one-step / oracle), so these are upper-bound skill given
  good inputs — not the recursive multi-step forecast (see the Phase-5 validation store for
  skill-vs-lead-hour).</p>
  <table><thead>{head}</thead><tbody>{rows}</tbody></table></div>
</div></body></html>"""


# ── orchestration ─────────────────────────────────────────────────────────────

def generate_reports(records: pd.DataFrame, output_dir: Path, models_metadata: dict = None) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for (station, variant), sub in records.groupby(["station", "variant"]):
        monthly = monthly_metrics(sub)
        station_clean = station.replace(' ', '_').replace('-', '')
        page = f"{station_clean}_{variant}.html"

        # Get metadata for this station/variant
        metadata = models_metadata.get((station, variant), {}) if models_metadata else {}

        # Generate individual month detail pages
        month_dir = output_dir / f"{station_clean}_{variant}_monthly"
        month_dir.mkdir(parents=True, exist_ok=True)
        months = [m for m in sorted(monthly) if m != "ALL"]
        for month in months:
            month_records = sub[sub["month"] == month]
            month_metrics = compute_metrics(month_records)
            month_page = f"{month}.html"
            (month_dir / month_page).write_text(
                build_month_page(station, variant, month, month_records, month_metrics, metadata),
                encoding="utf-8")

        # Generate main station page
        (output_dir / page).write_text(
            build_station_page(station, variant, sub, monthly, metadata), encoding="utf-8")
        print(f"  {page}")
        for month in months:
            print(f"    → {month}.html")
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
    p.add_argument("--s3-bucket", metavar="BUCKET",
                   help="Upload the rendered report tree to this S3 bucket "
                        "(e.g. test / resilentpublic). Models still load from S3_BUCKET.")
    p.add_argument("--s3-prefix", default=_DEFAULT_S3_PREFIX,
                   help=f"Destination key prefix for --s3-bucket (default: {_DEFAULT_S3_PREFIX})")
    args = p.parse_args()
    out_dir = Path(args.output)

    if args.report_only:
        print(f"Loading records from {args.report_only} ...")
        records = pd.read_parquet(args.report_only)
        records["time"] = pd.to_datetime(records["time"], utc=True)

        # Try to load metadata from JSON file if available
        models_metadata = {}
        metadata_file = out_dir / "models_metadata.json"
        if metadata_file.exists():
            try:
                metadata_json = json.loads(metadata_file.read_text(encoding="utf-8"))
                # Convert back to (station, variant) tuple keys
                for key_str, meta in metadata_json.items():
                    parts = key_str.split("/")
                    if len(parts) == 2:
                        models_metadata[(parts[0], parts[1])] = meta
                print(f"  Loaded metadata for {len(models_metadata)} station/variant pairs")
            except Exception as e:
                print(f"  Warning: Could not load metadata: {e}")

        generate_reports(records, out_dir, models_metadata)
        if args.s3_bucket:
            upload_reports_to_s3(out_dir, args.s3_bucket, args.s3_prefix)
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
    models_metadata = {}
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
            # Store metadata for this station/variant
            models_metadata[(station, variant)] = models.get("_metadata", {})

    if not all_records:
        print("No predictions generated — check model deployment / station names.")
        sys.exit(1)

    records = pd.concat(all_records, ignore_index=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    records.to_parquet(out_dir / "records.parquet", index=False)
    print(f"  Records saved → {out_dir / 'records.parquet'}")

    # Save metadata as JSON for report-only mode
    if models_metadata:
        metadata_file = out_dir / "models_metadata.json"
        metadata_json = {k[0] + "/" + k[1]: v for k, v in models_metadata.items()}
        metadata_file.write_text(json.dumps(metadata_json, indent=2), encoding="utf-8")
        print(f"  Metadata saved → {metadata_file}")

    print("Generating reports...")
    generate_reports(records, out_dir, models_metadata)

    if args.s3_bucket:
        print(f"Uploading reports to s3://{args.s3_bucket}/{args.s3_prefix.rstrip('/')}/ ...")
        upload_reports_to_s3(out_dir, args.s3_bucket, args.s3_prefix)


if __name__ == "__main__":
    main()
