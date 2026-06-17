"""Monthly walk-forward backfill training + backtesting pipeline.

Each monthly partition trains one set of per-station models using only data
BEFORE the partition start date (cutoff), then evaluates out-of-sample on data
FROM that date onwards. This provides honest walk-forward validation.

Feature engineering is done on the FULL dataset (pre + post cutoff) so that
H2S lag features at the boundary are computed from real observations, not NaN.
Only the pre-cutoff rows are used for model training.

S3 paths:
  Backfill model archives (same root as production archive, backfill_ prefix):
    STATION_MODELS_ARCHIVE_BASE/{station_key}/backfill_{month_key}_{version_tag}/
      {task}_{variant}.pkl          — model pickle (regression/clf_5ppb/clf_10ppb/clf_30ppb)
      training_report.json          — in-sample val metrics + feature lists per variant
      archive_metadata.json         — version tag, backfill flags, algorithm_choices, artifacts
  Backtest results:
    tijuana/forecast/backtest/{month_key}/backtest_results.json
    tijuana/forecast/backtest/{month_key}/{station_key}_{variant}.html
    tijuana/forecast/backtest/{month_key}/index.html
  Comparison index:
    tijuana/forecast/backtest/index.html

Jobs:
  station_backfill_training_job  -- monthly partitioned, trains + evaluates
  station_backtest_index_job     -- unpartitioned, rebuilds comparison index
"""

import base64
import io
import json
import os
import pickle
from datetime import datetime, timezone

import dagster as dg
from dagster import AssetExecutionContext
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from h2s.constants import (
    BACKTEST_RESULTS_BASE,
    MODEL_FEATURES,
    MODEL_FEATURES_LEAN,
    OBS_DATA_PATH,
    STATION_MODELS_ARCHIVE_BASE,
    STATIONS,
    TRAINING_SNAPSHOTS_PATH,
)
from h2s.training.calibration_eval import recall_at_threshold, spearman_rank
from h2s.training.multi_station_trainer import (
    TRAIN_FRACTION,
    eval_classifier,
    eval_regressor,
    prepare_multi_station_features,
    train_and_select,
)

# ============================================================================
# Partition definition
# ============================================================================

BACKFILL_MONTHLY_PARTITIONS = dg.MonthlyPartitionsDefinition(
    start_date="2026-01-01",
    timezone="UTC",
)

_KEY = lambda name: dg.AssetKey(["h2s", name])

_VARIANTS: dict[str, list[str]] = {
    "evidence": MODEL_FEATURES,
    "lean": MODEL_FEATURES_LEAN,
}

_TASKS = ("regression", "clf_5ppb", "clf_10ppb", "clf_30ppb")

# Magnitude thresholds (8 ppb = resident complaint boundary)
_THRESHOLDS = (5, 8, 10, 30, 100)
_PROB_THRESHOLDS = (5, 10, 30)
_PROB_COL = {5: "p5", 10: "p10", 30: "p30"}

_GREEN_MAX = 5.0
_ORANGE_MIN = 30.0


# ============================================================================
# Version-tag + importance helpers
# ============================================================================

def _git_short_sha() -> str:
    import subprocess
    try:
        return (subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL, text=True,
        ).strip()[:7] or "unknown")
    except (subprocess.CalledProcessError, FileNotFoundError):
        return os.environ.get("GIT_SHA", "unknown")[:7] or "unknown"


def _importance_for_features(model, feature_names: list, top_n: int = 10) -> dict:
    """Feature importance dict keyed by the variant's own column list."""
    imp = getattr(model, "feature_importances_", None)
    if imp is None:
        return {}
    imp = np.asarray(imp)
    idx = np.argsort(imp)[::-1][:top_n]
    return {feature_names[i]: round(float(imp[i]), 4) for i in idx if i < len(feature_names)}


# ============================================================================
# Shared metric helpers (adapted from station_models_backtest.py)
# ============================================================================

def _categorize(h2s: float) -> str:
    if h2s < _GREEN_MAX:
        return "green"
    if h2s < _ORANGE_MIN:
        return "yellow"
    return "orange"


def _prob_call(df: pd.DataFrame, k: int) -> dict:
    col = _PROB_COL.get(k)
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
    out["recall"] = float(tp / int(act.sum())) if act.sum() else None
    out["precision"] = float(tp / int(pred.sum())) if pred.sum() else None
    return out


def _threeclass(df: pd.DataFrame) -> dict:
    from sklearn.metrics import balanced_accuracy_score, confusion_matrix as sk_cm
    classes = ["green", "yellow", "orange"]
    y_true, y_pred = df["actual_category"], df["pred_category"]
    cm = sk_cm(y_true, y_pred, labels=classes).tolist()
    ba = (float(balanced_accuracy_score(y_true, y_pred))
          if df["actual_category"].nunique() > 1 else float("nan"))
    per_class = {}
    for cls in classes:
        yt = (y_true == cls)
        yp = (y_pred == cls)
        tp = int((yt & yp).sum())
        fp = int((~yt & yp).sum())
        fn = int((yt & ~yp).sum())
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        per_class[cls] = {"precision": prec, "recall": rec, "f1": f1,
                          "n": int(yt.sum())}
    orange_yt = (y_true == "orange")
    orange_yp = (y_pred == "orange")
    fp_orange = int((orange_yp & ~orange_yt).sum())
    n_non_orange = int((~orange_yt).sum())
    far = fp_orange / n_non_orange if n_non_orange else 0.0
    return {"per_class": per_class, "confusion_matrix": cm,
            "balanced_accuracy": ba, "false_alarm_rate": far}


def _compute_metrics(df: pd.DataFrame) -> dict:
    """Compute full metric set for one (station, variant) slice."""
    m: dict = {
        "n": int(len(df)),
        "spearman": float(spearman_rank(df["actual_h2s"], df["h2s_pred"])),
        "mae": float((df["h2s_pred"] - df["actual_h2s"]).abs().mean()),
        "mag": {},
        "prob": {},
    }
    for k in _THRESHOLDS:
        r = recall_at_threshold(df["actual_h2s"], df["h2s_pred"], k)
        m["mag"][str(k)] = {
            "recall": float(r["recall"]) if r["recall"] is not None else None,
            "precision": float(r["precision"]) if r["precision"] is not None else None,
            "n_pos": int(r["n_positives"]),
        }
    for k in _PROB_THRESHOLDS:
        m["prob"][str(k)] = _prob_call(df, k)
    m.update(_threeclass(df))
    return m


def _monthly_metrics(records: pd.DataFrame) -> dict[str, dict]:
    out = {m: _compute_metrics(records[records["month"] == m])
           for m in sorted(records["month"].unique())}
    out["ALL"] = _compute_metrics(records)
    return out


# ============================================================================
# Training helper (station-level, all tasks for one variant)
# ============================================================================

def _train_one_variant_local(
    log,
    sdf: pd.DataFrame,
    features: list[str],
    split: int,
    ensemble_margin: float,
    variant_label: str,
) -> dict[str, tuple]:
    """Train regression + clf_5ppb + clf_10ppb + clf_30ppb for one variant."""
    X = sdf[features].values
    y_cont = sdf["H2S"].values
    y_5 = sdf["exceed_5"].values
    y_10 = sdf["exceed_10"].values
    y_30 = sdf["exceed_30"].values

    Xtr, Xte = X[:split], X[split:]
    result: dict[str, tuple] = {}
    for task, ytr, yte in [
        ("regression", y_cont[:split], y_cont[split:]),
        ("clf_5ppb",   y_5[:split],    y_5[split:]),
        ("clf_10ppb",  y_10[:split],   y_10[split:]),
        ("clf_30ppb",  y_30[:split],   y_30[split:]),
    ]:
        log.info(f"    [{variant_label}] {task}…")
        model, choice, _ = train_and_select(
            Xtr, Xte, ytr, yte, task, ensemble_margin=ensemble_margin
        )
        log.info(f"      → {choice}")
        result[task] = (model, choice)
    return result


# ============================================================================
# Asset 1: Load + trim training data (monthly partitioned)
# ============================================================================

@dg.asset(
    key_prefix="h2s",
    group_name="h2s_backfill",
    required_resource_keys={"s3"},
    kinds={"python", "s3"},
    partitions_def=BACKFILL_MONTHLY_PARTITIONS,
    description=(
        "Load full obs data from S3, engineer features on the FULL dataset "
        "(so H2S lags at the boundary are correct), then snapshot the "
        "pre-cutoff slice. Returns the FULL feature DataFrame — callers split "
        "at is_pretrain == True for training and False for OOS eval."
    ),
    config_schema={
        "s3_bucket": dg.Field(
            str,
            default_value="resilentpublic",
            description="S3 bucket for obs data (resilentpublic or test)",
        ),
    },
)
def backfill_training_data(context: AssetExecutionContext) -> pd.DataFrame:
    """Load full obs data, engineer features, mark pre/post cutoff."""
    s3 = context.resources.s3
    bucket = context.op_config["s3_bucket"]

    cutoff = pd.Timestamp(context.partition_key, tz="UTC")
    month_key = cutoff.strftime("%Y-%m")
    context.log.info(f"Backfill partition {month_key} — cutoff {cutoff.date()}")

    raw_bytes = s3.getFile(path=OBS_DATA_PATH, bucket=bucket)
    raw_df = pd.read_parquet(io.BytesIO(raw_bytes))
    context.log.info(f"Loaded {len(raw_df):,} raw rows from S3")

    # Feature engineering on the FULL dataset — ensures H2S lags at boundary
    # are computed from real pre-cutoff observations, not NaN.
    full_df = prepare_multi_station_features(raw_df)
    full_df["is_pretrain"] = full_df["time"] < cutoff

    # Snapshot the pre-cutoff training slice (for audit trail)
    pretrain_df = full_df[full_df["is_pretrain"]]
    snapshot_ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    snap_path = (f"{TRAINING_SNAPSHOTS_PATH}/backfill/{month_key}/"
                 f"{snapshot_ts}/training_data.parquet")
    buf = io.BytesIO()
    pretrain_df.to_parquet(buf, index=False)
    s3.putFile(buf.getvalue(), snap_path, bucket=s3.S3_BUCKET,
               content_type="application/octet-stream")
    context.log.info(f"Snapshotted pre-cutoff data ({len(pretrain_df):,} rows) → {snap_path}")

    context.add_output_metadata({
        "month_key": month_key,
        "cutoff_date": str(cutoff.date()),
        "n_total": int(len(full_df)),
        "n_pretrain": int(len(pretrain_df)),
        "n_oos": int((~full_df["is_pretrain"]).sum()),
        "stations": sorted(full_df["site_name"].unique().tolist()),
        "snapshot_path": snap_path,
    })
    return full_df


# ============================================================================
# Asset 2: Train backfill models (monthly partitioned, all 3 stations)
# ============================================================================

@dg.asset(
    key_prefix="h2s",
    group_name="h2s_backfill",
    required_resource_keys={"s3"},
    kinds={"python", "ml", "s3"},
    partitions_def=BACKFILL_MONTHLY_PARTITIONS,
    ins={"backfill_training_data": dg.AssetIn(key=_KEY("backfill_training_data"))},
    description=(
        "Train Evidence + Lean models for all 3 stations using only pre-cutoff data. "
        "Archives under STATION_MODELS_ARCHIVE_BASE with backfill_{month_key}_{version_tag} "
        "prefix — same root as production, distinguished by the backfill_ prefix. "
        "Writes training_report.json (val metrics + feature lists) and a merged "
        "archive_metadata.json (algorithm_choices + artifacts). No features_{variant}.json — "
        "features are already inside training_report.json."
    ),
    config_schema={
        "ensemble_margin": dg.Field(float, default_value=0.01),
    },
)
def backfill_station_models(
    context: AssetExecutionContext,
    backfill_training_data: pd.DataFrame,
) -> dict:
    """Train all stations × variants × tasks on pre-cutoff data; archive to S3.

    Returns metadata dict (not the models themselves) with S3 paths so that
    station_backtest_results can load them directly.
    """
    s3 = context.resources.s3
    cutoff = pd.Timestamp(context.partition_key, tz="UTC")
    month_key = cutoff.strftime("%Y-%m")
    ensemble_margin = context.op_config["ensemble_margin"]

    pretrain_df = backfill_training_data[backfill_training_data["is_pretrain"]].copy()
    context.log.info(
        f"Training backfill models for {month_key} "
        f"({len(pretrain_df):,} pre-cutoff rows across "
        f"{pretrain_df['site_name'].nunique()} stations)"
    )

    now = datetime.now(timezone.utc)
    now_str = now.strftime("%Y%m%dT%H%M%SZ")
    sha = _git_short_sha()

    station_meta: dict[str, dict] = {}

    for station_name, info in STATIONS.items():
        station_key = info["key"]
        sdf = (pretrain_df[pretrain_df["site_name"] == station_name]
               .sort_values("time")
               .reset_index(drop=True))

        if len(sdf) < 50:
            context.log.warning(f"  {station_name}: only {len(sdf)} rows — skipping")
            continue

        split = int(len(sdf) * TRAIN_FRACTION)
        context.log.info(
            f"  {station_name}: {len(sdf):,} rows (train: {split:,}, val: {len(sdf)-split:,})"
        )

        # One version tag per station (distinct timestamp-sha per station run)
        version_tag = f"backfill_{month_key}_{now_str}-{sha}"
        s3_base = f"{STATION_MODELS_ARCHIVE_BASE}/{station_key}/{version_tag}"

        tasks_metrics: dict[str, dict] = {}
        algorithm_choices: dict[str, dict] = {}

        # Shared label arrays for val scoring
        y_cont = sdf["H2S"].values
        y_5 = sdf["exceed_5"].values
        y_10 = sdf["exceed_10"].values
        y_30 = sdf["exceed_30"].values

        for variant, features in _VARIANTS.items():
            X = sdf[features].values
            Xte = X[split:]

            variant_results = _train_one_variant_local(
                context.log, sdf, features, split, ensemble_margin, variant
            )

            variant_metrics: dict[str, dict] = {}
            variant_choices: dict[str, str] = {}
            for task_base, yte in [
                ("regression", y_cont[split:]),
                ("clf_5ppb",   y_5[split:]),
                ("clf_10ppb",  y_10[split:]),
                ("clf_30ppb",  y_30[split:]),
            ]:
                model, choice = variant_results[task_base]
                m = (eval_regressor(model, Xte, yte) if task_base == "regression"
                     else eval_classifier(model, Xte, yte))
                variant_metrics[task_base] = {
                    **m,
                    "feature_importance": _importance_for_features(model, features),
                }
                variant_choices[task_base] = choice

                s3.putFile(
                    pickle.dumps(model),
                    f"{s3_base}/{task_base}_{variant}.pkl",
                    bucket=s3.S3_BUCKET,
                    content_type="application/octet-stream",
                )

            tasks_metrics[variant] = variant_metrics
            algorithm_choices[variant] = variant_choices

        # training_report.json — matches production format; adds backfill extras
        training_report = {
            "generated_at": now.isoformat(),
            "station": station_name,
            "station_key": station_key,
            "is_backfill": True,
            "month_key": month_key,
            "cutoff_date": cutoff.isoformat(),
            "n_records": int(len(sdf)),
            "n_train": int(split),
            "n_val": int(len(sdf) - split),
            "features": {v: feats for v, feats in _VARIANTS.items()},
            "ensemble_margin": ensemble_margin,
            "tasks": tasks_metrics,
        }
        s3.putFile(
            json.dumps(training_report, indent=2, default=str).encode(),
            f"{s3_base}/training_report.json",
            bucket=s3.S3_BUCKET,
            content_type="application/json",
        )

        # archive_metadata.json — merged: artifacts + algorithm_choices + backfill fields
        artifacts = (
            [f"{task}_{variant}.pkl" for variant in _VARIANTS for task in _TASKS]
            + ["training_report.json", "archive_metadata.json"]
        )
        archive_meta = {
            "model_version": version_tag,
            "is_backfill": True,
            "month_key": month_key,
            "cutoff_date": cutoff.isoformat(),
            "archived_at": now.isoformat(),
            "git_sha": sha,
            "station": station_name,
            "station_key": station_key,
            "n_pretrain": int(len(sdf)),
            "n_train": int(split),
            "n_val": int(len(sdf) - split),
            "algorithm_choices": algorithm_choices,
            "artifacts": artifacts,
        }
        s3.putFile(
            json.dumps(archive_meta, indent=2).encode(),
            f"{s3_base}/archive_metadata.json",
            bucket=s3.S3_BUCKET,
            content_type="application/json",
        )

        station_meta[station_key] = {
            "station_name": station_name,
            "s3_base": s3_base,
            "version_tag": version_tag,
            "n_pretrain": int(len(sdf)),
            "n_train": int(split),
            "n_val": int(len(sdf) - split),
            "algorithm_choices": algorithm_choices,
        }
        context.log.info(f"  ✓ {station_name} → {s3_base}")

    context.add_output_metadata({
        "month_key": month_key,
        "cutoff_date": str(cutoff.date()),
        "stations_trained": list(station_meta.keys()),
        "variants": list(_VARIANTS.keys()),
    })
    return {"month_key": month_key, "cutoff": cutoff.isoformat(),
            "stations": station_meta}


# ============================================================================
# Shared chart helpers (adapted from station_models_backtest.py)
# ============================================================================

_C = {
    "green": "#27ae60", "yellow": "#f39c12", "orange": "#e74c3c",
    "blue": "#2980b9", "purple": "#8e44ad",
}


def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def _chart_spearman_recall(monthly: dict) -> str:
    months = [m for m in sorted(monthly) if m != "ALL"]
    if not months:
        return ""
    xs = range(len(months))
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(xs, [monthly[m]["spearman"] for m in months],
            color=_C["blue"], marker="o", lw=1.8, label="Spearman")
    ax.plot(xs, [monthly[m]["mag"]["8"]["recall"] for m in months],
            color="#d4ac0d", marker="*", ms=8, lw=1.8, label="recall@8 (complaint)")
    ax.plot(xs, [monthly[m]["mag"]["10"]["recall"] for m in months],
            color=_C["yellow"], marker="s", lw=1.6, label="recall@10")
    ax.plot(xs, [monthly[m]["mag"]["30"]["recall"] for m in months],
            color=_C["orange"], marker="^", lw=1.6, label="recall@30")
    ax.set_xticks(list(xs))
    ax.set_xticklabels(months, rotation=45, ha="right", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
    ax.set_title("Monthly rank skill & alert detection", fontsize=11)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(axis="y", lw=0.4, alpha=0.5)
    fig.tight_layout()
    return _fig_to_b64(fig)


def _chart_mae(monthly: dict) -> str:
    months = [m for m in sorted(monthly) if m != "ALL"]
    if not months:
        return ""
    xs = np.arange(len(months))
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.bar(xs, [monthly[m]["mae"] for m in months],
           color=_C["blue"], alpha=0.75, label="MAE (ppb)")
    ax2 = ax.twinx()
    ax2.plot(xs, [monthly[m]["per_class"]["orange"]["n"] for m in months],
             color=_C["orange"], marker="D", lw=1.6, label="Observed orange hours")
    ax.set_xticks(list(xs))
    ax.set_xticklabels(months, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("MAE (ppb)")
    ax2.set_ylabel("Orange hours")
    ax.set_title("Monthly MAE vs observed orange-event volume", fontsize=11)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8)
    ax.grid(axis="y", lw=0.4, alpha=0.5)
    fig.tight_layout()
    return _fig_to_b64(fig)


def _chart_category_recall(monthly: dict) -> str:
    months = [m for m in sorted(monthly) if m != "ALL"]
    if not months:
        return ""
    xs = np.arange(len(months))
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(xs, [monthly[m]["per_class"]["green"]["recall"] for m in months],
            color=_C["green"], marker="o", lw=1.8, label="Green recall")
    ax.plot(xs, [monthly[m]["per_class"]["yellow"]["recall"] for m in months],
            color=_C["yellow"], marker="s", lw=1.8, label="Yellow recall")
    ax.plot(xs, [monthly[m]["per_class"]["orange"]["recall"] for m in months],
            color=_C["orange"], marker="^", lw=1.8, label="Orange recall")
    ax.set_xticks(list(xs))
    ax.set_xticklabels(months, rotation=45, ha="right", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
    ax.set_title("Monthly category detection rates", fontsize=11)
    ax.legend(fontsize=8, ncol=3)
    ax.grid(axis="y", lw=0.4, alpha=0.5)
    fig.tight_layout()
    return _fig_to_b64(fig)


def _chart_confusion(m: dict, title: str) -> str:
    cm = np.array(m["confusion_matrix"], dtype=float)
    labels = ["Green", "Yellow", "Orange"]
    fig, ax = plt.subplots(figsize=(5, 4))
    totals = cm.sum(axis=1, keepdims=True)
    norm = np.where(totals > 0, cm / totals, 0.0)
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks([0, 1, 2])
    ax.set_yticks([0, 1, 2])
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Predicted", fontsize=9)
    ax.set_ylabel("Actual", fontsize=9)
    ax.set_title(title, fontsize=10)
    for i in range(3):
        for j in range(3):
            ax.text(j, i,
                    f"{int(cm[i, j])}\n({norm[i, j]:.0%})",
                    ha="center", va="center", fontsize=8,
                    color="white" if norm[i, j] > 0.6 else "black")
    fig.colorbar(im, ax=ax, shrink=0.85)
    fig.tight_layout()
    return _fig_to_b64(fig)


def _chart_scatter(df: pd.DataFrame, title: str) -> str:
    fig, ax = plt.subplots(figsize=(5, 4.5))
    a = df["actual_h2s"].to_numpy()
    p = df["h2s_pred"].to_numpy()
    ax.scatter(a, p, s=6, alpha=0.25, color=_C["blue"], edgecolors="none")
    hi = float(np.nanpercentile(np.concatenate([a, p]), 99.5)) if len(a) else 1.0
    hi = max(hi, 35.0)
    ax.plot([0, hi], [0, hi], color="#888", lw=1.0, ls="--")
    for thr, col in [(5, _C["yellow"]), (30, _C["orange"])]:
        ax.axvline(thr, color=col, lw=0.8, alpha=0.6)
        ax.axhline(thr, color=col, lw=0.8, alpha=0.6)
    ax.set_xlim(0, hi)
    ax.set_ylim(0, hi)
    ax.set_xlabel("Actual H2S (ppb)")
    ax.set_ylabel("Predicted H2S (ppb)")
    ax.set_title(title, fontsize=10)
    ax.grid(lw=0.3, alpha=0.4)
    fig.tight_layout()
    return _fig_to_b64(fig)


# ============================================================================
# HTML helpers (adapted from station_models_backtest.py)
# ============================================================================

_CSS = """
body { font-family: system-ui, sans-serif; margin: 0; background: #f5f6fa; color: #333; }
.header { background: #1a2233; color: #fff; padding: 18px 32px; }
.header h1 { margin: 0; font-size: 1.4em; }
.header p { margin: 4px 0 0; font-size: 0.85em; color: #aab; }
.content { max-width: 1200px; margin: 28px auto; padding: 0 24px; }
.section { background: #fff; border-radius: 8px; padding: 20px 24px;
           margin-bottom: 20px; box-shadow: 0 1px 4px rgba(0,0,0,.07); }
h2 { margin: 0 0 14px; font-size: 1.1em; color: #1a2233;
     border-bottom: 2px solid #eee; padding-bottom: 8px; }
h3 { font-size: 0.98em; color: #444; margin: 14px 0 8px; }
table { border-collapse: collapse; width: 100%; font-size: 0.84em; }
th { background: #1a2233; color: #fff; padding: 7px 10px; text-align: left; }
td { padding: 6px 10px; border-bottom: 1px solid #eee; }
tr:hover td { background: #f0f4ff; }
.pass { background: #d5f5e3; color: #196f3d; font-weight: 600; }
.warn { background: #fef9e7; color: #7d6608; }
.fail { background: #fadbd8; color: #922b21; font-weight: 600; }
.best { background: #1a2233 !important; color: #fff !important;
        font-weight: 700; border-radius: 3px; }
.metric-grid { display: grid;
    grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
    gap: 12px; margin-bottom: 18px; }
.metric-card { background: #f8fafc; border: 1px solid #e0e0e0;
    border-radius: 6px; padding: 12px 14px; text-align: center; }
.metric-card .val { font-size: 1.6em; font-weight: 700; color: #1a2233; }
.metric-card .lbl { font-size: 0.76em; color: #777; margin-top: 4px; }
.chart-row { display: flex; gap: 14px; flex-wrap: wrap;
             margin-bottom: 14px; align-items: flex-start; }
.chart-row img { max-width: 100%; border-radius: 6px;
    box-shadow: 0 1px 3px rgba(0,0,0,.1); }
.nav { background: #fff; padding: 10px 24px; display: flex; gap: 14px;
       flex-wrap: wrap; border-bottom: 1px solid #ddd; font-size: 0.85em; }
.nav a { color: #2563eb; text-decoration: none; }
.nav a:hover { text-decoration: underline; }
.badge { display: inline-block; padding: 2px 8px; border-radius: 10px;
         font-size: 0.72em; font-weight: 600; }
.ev { background: #dbeafe; color: #1e40af; }
.ln { background: #ede9fe; color: #5b21b6; }
.info-box { background: #f0f4ff; border-left: 4px solid #2563eb;
            padding: 12px 14px; margin-bottom: 16px;
            font-size: 0.84em; border-radius: 4px; }
.info-box > div { margin: 4px 0; }
.lead-grid { display: grid;
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: 18px; margin-bottom: 20px; }
.lead-card { background: #fff; border-radius: 8px; padding: 16px 20px;
    box-shadow: 0 1px 4px rgba(0,0,0,.07); border-left: 4px solid #2563eb; }
.lead-card h3 { margin: 0 0 10px; font-size: 0.95em; color: #1a2233; }
.months-grid { display: grid;
    grid-template-columns: repeat(auto-fill, minmax(175px, 1fr));
    gap: 11px; margin-bottom: 8px; }
.month-card { background: #f8fafc; border: 1px solid #dde3ee; border-radius: 7px;
    padding: 13px 16px; transition: box-shadow .15s; }
.month-card:hover { box-shadow: 0 2px 8px rgba(0,0,0,.10); }
.month-card a { font-size: 1.05em; font-weight: 700; color: #2563eb;
    text-decoration: none; }
.month-card a:hover { text-decoration: underline; }
.month-card .meta { font-size: 0.78em; color: #666; margin-top: 5px; line-height: 1.6; }
"""


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


def _threshold_table(m: dict) -> str:
    rows = ""
    for k in _THRESHOLDS:
        sk = str(k)
        mag = m["mag"][sk]
        prob = m["prob"].get(sk, {"recall": None, "precision": None, "n": 0})
        rows += (
            f"<tr><td>≥{k} ppb</td>"
            f"<td>{mag['n_pos']:,}</td>"
            f"<td class='{_color_cell(mag['recall'])}'>{_pct(mag['recall'])}</td>"
            f"<td class='{_color_cell(mag['precision'])}'>{_pct(mag['precision'])}</td>"
            f"<td class='{_color_cell(prob['recall'])}'>{_pct(prob['recall'])}</td>"
            f"<td class='{_color_cell(prob['precision'])}'>{_pct(prob['precision'])}</td>"
            f"</tr>"
        )
    return (
        "<table><thead><tr>"
        "<th>Threshold</th><th>Observed events</th>"
        "<th>Recall (magnitude)</th><th>Precision (magnitude)</th>"
        "<th>Recall (P&gt;0.5)</th><th>Precision (P&gt;0.5)</th>"
        f"</tr></thead><tbody>{rows}</tbody></table>"
    )


def _cards(m: dict) -> str:
    cards = [
        (_num(m["spearman"]), "Spearman"),
        (_num(m["mae"], "{:.1f}"), "MAE (ppb)"),
        (_pct(m["mag"]["5"]["recall"]), "recall@5"),
        (_pct(m["mag"]["8"]["recall"]), "recall@8 (complaint)"),
        (_pct(m["mag"]["10"]["recall"]), "recall@10"),
        (_pct(m["mag"]["30"]["recall"]), "recall@30"),
        (_pct(m["mag"]["100"]["recall"]), "recall@100"),
        (_pct(m["false_alarm_rate"]), "False alarm rate"),
    ]
    html = '<div class="metric-grid">'
    for val, lbl in cards:
        html += (f'<div class="metric-card">'
                 f'<div class="val">{val}</div>'
                 f'<div class="lbl">{lbl}</div></div>')
    return html + "</div>"


def _monthly_table(monthly: dict, months: list[str]) -> str:
    head = (
        "<tr><th>Month</th><th>n</th><th>Spearman</th><th>MAE</th>"
        "<th>recall@5</th><th>recall@8</th><th>recall@10</th>"
        "<th>recall@30</th><th>recall@100</th><th>FAR</th></tr>"
    )
    rows = ""
    for m in months:
        mm = monthly[m]
        rows += (
            f"<tr><td><b>{m}</b></td><td>{mm['n']:,}</td>"
            f"<td class='{_color_cell(mm['spearman'])}'>{_num(mm['spearman'])}</td>"
            f"<td>{_num(mm['mae'], '{:.1f}')}</td>"
            f"<td class='{_color_cell(mm['mag']['5']['recall'])}'>"
            f"{_pct(mm['mag']['5']['recall'])}</td>"
            f"<td class='{_color_cell(mm['mag']['8']['recall'])}'>"
            f"{_pct(mm['mag']['8']['recall'])}</td>"
            f"<td class='{_color_cell(mm['mag']['10']['recall'])}'>"
            f"{_pct(mm['mag']['10']['recall'])}</td>"
            f"<td class='{_color_cell(mm['mag']['30']['recall'])}'>"
            f"{_pct(mm['mag']['30']['recall'])}</td>"
            f"<td class='{_color_cell(mm['mag']['100']['recall'])}'>"
            f"{_pct(mm['mag']['100']['recall'])}</td>"
            f"<td class='{_color_cell(1 - mm['false_alarm_rate'], low=0.9, high=0.96)}'>"
            f"{_pct(mm['false_alarm_rate'])}</td></tr>"
        )
    return f"<table><thead>{head}</thead><tbody>{rows}</tbody></table>"


def _build_station_html(
    station_name: str,
    variant: str,
    oos_df: pd.DataFrame,
    monthly: dict,
    cutoff_date: str,
    month_key: str,
) -> str:
    overall = monthly["ALL"]
    months = [m for m in sorted(monthly) if m != "ALL"]
    vlabel = "Evidence (33 feat)" if variant == "evidence" else "Lean (19 feat)"
    rng = ""
    if len(oos_df):
        rng = (f"{oos_df['time'].min().strftime('%Y-%m-%d')} – "
               f"{oos_df['time'].max().strftime('%Y-%m-%d')}")

    charts_html = ""
    if len(oos_df) > 0:
        charts_html = f"""
    <div class="chart-row">
      <img src="data:image/png;base64,{_chart_confusion(overall, 'Confusion matrix')}"
           style="max-width:380px">
      <img src="data:image/png;base64,{_chart_scatter(oos_df, 'Predicted vs actual')}"
           style="max-width:430px">
    </div>"""
        if months:
            charts_html += f"""
    <div class="section"><h2>Monthly skill trends (OOS)</h2>
      <img src="data:image/png;base64,{_chart_spearman_recall(monthly)}" style="width:100%">
      <img src="data:image/png;base64,{_chart_mae(monthly)}" style="width:100%">
      <img src="data:image/png;base64,{_chart_category_recall(monthly)}" style="width:100%">
    </div>"""

    monthly_table_html = ""
    if months:
        monthly_table_html = (
            '<div class="section"><h2>Monthly OOS summary</h2>'
            + _monthly_table(monthly, months)
            + "</div>"
        )

    return f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Backfill backtest — {station_name} / {variant} / {month_key}</title>
<style>{_CSS}</style></head><body>
<div class="header"><h1>{station_name} — {vlabel} — backfill {month_key}</h1>
<p>Walk-forward: trained on data before {cutoff_date}, evaluated on {rng or 'no OOS data'} ({overall['n']:,} OOS hours)</p>
</div>
<div class="nav"><a href="index.html">↑ Back to {month_key} index</a></div>
<div class="content">
  <div class="section"><h2>Model info</h2>
    <div class="info-box">
      <div><b>Training cutoff:</b> {cutoff_date}</div>
      <div><b>OOS period:</b> {rng or '—'}</div>
      <div style="font-size:0.85em;color:#666;margin-top:6px;">Features use observed H2S lags (oracle one-step inputs) — not recursive forecasts</div>
    </div>
  </div>
  <div class="section"><h2>Overall OOS metrics</h2>
    {_cards(overall)}
    {charts_html}
    <h3>Threshold detection — magnitude cut vs classifier probability</h3>
    {_threshold_table(overall)}
  </div>
  {monthly_table_html}
</div></body></html>"""


def _build_month_index_html(
    month_key: str,
    cutoff_date: str,
    station_results: list[dict],
    s3_base_url: str,
) -> str:
    """Per-month index.html linking to station × variant detail pages."""
    rows = ""
    for r in station_results:
        m = r["overall"]
        badge = "ev" if r["variant"] == "evidence" else "ln"
        rows += (
            f"<tr>"
            f"<td><a href='{r['page']}'>{r['station_name']}</a></td>"
            f"<td><span class='badge {badge}'>{r['variant']}</span></td>"
            f"<td>{m['n']:,}</td>"
            f"<td class='{_color_cell(m['spearman'])}'>{_num(m['spearman'])}</td>"
            f"<td>{_num(m['mae'], '{:.1f}')}</td>"
            f"<td class='{_color_cell(m['mag']['5']['recall'])}'>"
            f"{_pct(m['mag']['5']['recall'])}</td>"
            f"<td class='{_color_cell(m['mag']['8']['recall'])}'>"
            f"{_pct(m['mag']['8']['recall'])}</td>"
            f"<td class='{_color_cell(m['mag']['10']['recall'])}'>"
            f"{_pct(m['mag']['10']['recall'])}</td>"
            f"<td class='{_color_cell(m['mag']['30']['recall'])}'>"
            f"{_pct(m['mag']['30']['recall'])}</td>"
            f"<td class='{_color_cell(m['balanced_accuracy'])}'>"
            f"{_pct(m['balanced_accuracy'])}</td>"
            f"<td class='{_color_cell(1-m['false_alarm_rate'], low=0.9, high=0.96)}'>"
            f"{_pct(m['false_alarm_rate'])}</td>"
            f"</tr>"
        )
    head = (
        "<tr><th>Station</th><th>Variant</th><th>n OOS</th>"
        "<th>Spearman</th><th>MAE</th>"
        "<th>recall@5</th><th>recall@8</th><th>recall@10</th><th>recall@30</th>"
        "<th>Balanced Acc</th><th>FAR</th></tr>"
    )
    return f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Backfill backtest — {month_key}</title>
<style>{_CSS}</style></head><body>
<div class="header"><h1>Backfill Backtest — {month_key}</h1>
<p>Walk-forward: trained on data before {cutoff_date} · OOS evaluation on post-cutoff period</p>
</div>
<div class="nav"><a href="../index.html">↑ All months</a></div>
<div class="content">
  <div class="section"><h2>All stations × variants — OOS metrics</h2>
    <table><thead>{head}</thead><tbody>{rows}</tbody></table>
  </div>
</div></body></html>"""


# ============================================================================
# Asset 3: OOS backtest + HTML generation (monthly partitioned)
# ============================================================================

@dg.asset(
    key_prefix="h2s",
    group_name="h2s_backfill",
    required_resource_keys={"s3"},
    kinds={"python", "s3"},
    partitions_def=BACKFILL_MONTHLY_PARTITIONS,
    ins={
        "backfill_training_data": dg.AssetIn(key=_KEY("backfill_training_data")),
        "backfill_station_models": dg.AssetIn(key=_KEY("backfill_station_models")),
    },
    description=(
        "Load backfill models from S3, evaluate OOS on post-cutoff data, "
        "generate per-(station, variant) HTML detail pages, and upload to S3. "
        "Returns metrics JSON for downstream comparison index."
    ),
)
def station_backtest_results(
    context: AssetExecutionContext,
    backfill_training_data: pd.DataFrame,
    backfill_station_models: dict,
) -> dict:
    """Score post-cutoff OOS data, generate HTML pages, upload to S3."""
    s3 = context.resources.s3
    cutoff = pd.Timestamp(context.partition_key, tz="UTC")
    month_key = cutoff.strftime("%Y-%m")
    cutoff_date_str = str(cutoff.date())

    oos_df_full = backfill_training_data[~backfill_training_data["is_pretrain"]].copy()
    stations_meta = backfill_station_models["stations"]

    results_per_station: dict[str, dict] = {}
    station_results_for_index: list[dict] = []
    s3_base_url = s3.publicUrl(f"{BACKTEST_RESULTS_BASE}/{month_key}/")

    context.log.info(
        f"OOS evaluation for {month_key}: {len(oos_df_full):,} post-cutoff rows"
    )

    for station_name, info in STATIONS.items():
        station_key = info["key"]
        if station_key not in stations_meta:
            context.log.warning(f"  {station_name}: no trained model found — skipping")
            continue

        meta = stations_meta[station_key]
        s3_model_base = meta["s3_base"]
        sdf_oos = (oos_df_full[oos_df_full["site_name"] == station_name]
                   .sort_values("time")
                   .reset_index(drop=True))

        context.log.info(f"  {station_name}: {len(sdf_oos):,} OOS rows")

        variant_results: dict[str, dict] = {}
        for variant, features in _VARIANTS.items():
            # Load models from S3
            models: dict = {}
            for task in _TASKS:
                pkl_path = f"{s3_model_base}/{task}_{variant}.pkl"
                try:
                    raw = s3.getFile(path=pkl_path, bucket=s3.S3_BUCKET)
                    models[task] = pickle.loads(raw)
                except Exception as e:
                    context.log.warning(f"    Could not load {pkl_path}: {e}")
                    models[task] = None

            reg = models.get("regression")
            if reg is None:
                context.log.warning(f"    {station_name}/{variant}: no regression model — skipping")
                continue

            # Score OOS rows
            if len(sdf_oos) > 0:
                X_oos = np.zeros((len(sdf_oos), len(features)))
                for i, col in enumerate(features):
                    if col in sdf_oos.columns:
                        X_oos[:, i] = sdf_oos[col].astype(float).fillna(0.0).values

                h2s_pred = np.clip(reg.predict(X_oos), 0.0, None)

                def _proba(clf):
                    if clf is None:
                        return np.full(len(X_oos), np.nan)
                    return clf.predict_proba(X_oos)[:, 1]

                scored = pd.DataFrame({
                    "time": sdf_oos["time"].values,
                    "actual_h2s": sdf_oos["H2S"].values,
                    "h2s_pred": h2s_pred,
                    "p5": _proba(models.get("clf_5ppb")),
                    "p10": _proba(models.get("clf_10ppb")),
                    "p30": _proba(models.get("clf_30ppb")),
                })
                scored["month"] = pd.to_datetime(scored["time"]).dt.to_period("M").astype(str)
                scored["actual_category"] = scored["actual_h2s"].apply(_categorize)
                scored["pred_category"] = scored["h2s_pred"].apply(_categorize)

                monthly = _monthly_metrics(scored)
                oos_metrics = monthly["ALL"]
            else:
                scored = pd.DataFrame()
                monthly = {}
                oos_metrics = {
                    "n": 0, "spearman": float("nan"), "mae": float("nan"),
                    "mag": {str(k): {"recall": None, "precision": None, "n_pos": 0}
                             for k in _THRESHOLDS},
                    "prob": {str(k): {"recall": None, "precision": None, "n": 0}
                              for k in _PROB_THRESHOLDS},
                    "per_class": {c: {"precision": 0.0, "recall": 0.0, "f1": 0.0, "n": 0}
                                   for c in ["green", "yellow", "orange"]},
                    "balanced_accuracy": float("nan"),
                    "false_alarm_rate": float("nan"),
                    "confusion_matrix": [[0]*3]*3,
                }

            variant_results[variant] = {
                "oos_metrics": oos_metrics,
                "monthly": {k: v for k, v in monthly.items() if k != "ALL"},
            }

            # Upload station detail HTML
            station_clean = station_name.replace(" ", "_").replace("-", "")
            page_name = f"{station_clean}_{variant}.html"
            html = _build_station_html(
                station_name, variant, scored, monthly,
                cutoff_date_str, month_key
            )
            html_path = f"{BACKTEST_RESULTS_BASE}/{month_key}/{page_name}"
            s3.putFile(html.encode("utf-8"), html_path,
                       bucket=s3.S3_BUCKET, content_type="text/html")
            context.log.info(f"    ✓ Uploaded {page_name}")

            station_results_for_index.append({
                "station_name": station_name,
                "station_key": station_key,
                "variant": variant,
                "page": page_name,
                "overall": oos_metrics,
            })

        results_per_station[station_key] = {
            "station_name": station_name,
            "n_train": meta["n_train"],
            "n_val": meta["n_val"],
            "n_oos": int(len(sdf_oos)),
            "variants": variant_results,
        }

    # Upload per-month index.html
    month_index_html = _build_month_index_html(
        month_key, cutoff_date_str, station_results_for_index, s3_base_url
    )
    month_index_path = f"{BACKTEST_RESULTS_BASE}/{month_key}/index.html"
    s3.putFile(month_index_html.encode("utf-8"), month_index_path,
               bucket=s3.S3_BUCKET, content_type="text/html")
    context.log.info(f"✓ Uploaded month index → {month_index_path}")

    # Build and upload backtest_results.json (machine-readable for comparison index)
    results_json = {
        "month_key": month_key,
        "cutoff_date": cutoff.isoformat(),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "stations": results_per_station,
    }
    results_json_path = f"{BACKTEST_RESULTS_BASE}/{month_key}/backtest_results.json"
    s3.putFile(
        json.dumps(results_json, indent=2, default=str).encode("utf-8"),
        results_json_path,
        bucket=s3.S3_BUCKET,
        content_type="application/json",
    )
    context.log.info(f"✓ Uploaded backtest_results.json → {results_json_path}")

    # Dagster metadata
    station_summaries = {}
    for sr in station_results_for_index:
        if sr["variant"] == "evidence":
            m = sr["overall"]
            station_summaries[sr["station_key"]] = {
                "spearman": m.get("spearman"),
                "balanced_accuracy": m.get("balanced_accuracy"),
                "n_oos": m.get("n"),
            }
    context.add_output_metadata({
        "month_key": month_key,
        "cutoff_date": cutoff_date_str,
        "stations_evaluated": list(results_per_station.keys()),
        "index_url": s3.publicUrl(month_index_path),
        **{f"{sk}_spearman": v.get("spearman") for sk, v in station_summaries.items()},
    })
    return results_json


# ============================================================================
# Asset 4: Cross-month comparison index (unpartitioned)
# ============================================================================

def _build_comparison_index_html(all_results: list[dict]) -> str:
    """Comparison table with best-value highlighting + lead section."""

    # Sort by month_key descending (most recent first)
    all_results = sorted(all_results, key=lambda r: r["month_key"], reverse=True)

    # Flatten into rows for the table
    TableRow = dict  # {month_key, station_name, station_key, variant, n_oos, spearman, mae, r5, r8, r10, r30, bal_acc, far}
    rows: list[TableRow] = []
    for res in all_results:
        month_key = res["month_key"]
        cutoff_date = res.get("cutoff_date", "")[:10]
        for station_key, sdata in res.get("stations", {}).items():
            for variant, vdata in sdata.get("variants", {}).items():
                m = vdata.get("oos_metrics", {})
                if not m or m.get("n", 0) == 0:
                    continue
                rows.append({
                    "month_key": month_key,
                    "cutoff_date": cutoff_date,
                    "station_name": sdata.get("station_name", station_key),
                    "station_key": station_key,
                    "variant": variant,
                    "n_oos": m.get("n", 0),
                    "spearman": m.get("spearman"),
                    "mae": m.get("mae"),
                    "r5": m.get("mag", {}).get("5", {}).get("recall"),
                    "r8": m.get("mag", {}).get("8", {}).get("recall"),
                    "r10": m.get("mag", {}).get("10", {}).get("recall"),
                    "r30": m.get("mag", {}).get("30", {}).get("recall"),
                    "bal_acc": m.get("balanced_accuracy"),
                    "far": m.get("false_alarm_rate"),
                })

    # Determine best per column (max for most, min for mae/far)
    def _best(col, lower_is_better=False):
        vals = [r[col] for r in rows
                if r[col] is not None and not (isinstance(r[col], float) and np.isnan(r[col]))]
        if not vals:
            return None
        return min(vals) if lower_is_better else max(vals)

    best = {
        "spearman": _best("spearman"),
        "mae": _best("mae", lower_is_better=True),
        "r5": _best("r5"),
        "r8": _best("r8"),
        "r10": _best("r10"),
        "r30": _best("r30"),
        "bal_acc": _best("bal_acc"),
        "far": _best("far", lower_is_better=True),
    }

    def _cls(col, val, lower_is_better=False):
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return ""
        bv = best.get(col)
        if bv is None:
            return ""
        tol = 0.001
        if lower_is_better:
            return "best" if abs(val - bv) < tol else ""
        return "best" if abs(val - bv) < tol else ""

    # Build table rows
    table_rows_html = ""
    for r in rows:
        badge = "ev" if r["variant"] == "evidence" else "ln"
        station_clean = r["station_name"].replace(" ", "_").replace("-", "")
        detail_href = f"{r['month_key']}/{station_clean}_{r['variant']}.html"
        month_href = f"{r['month_key']}/index.html"
        table_rows_html += (
            f"<tr>"
            f"<td><a href='{month_href}'>{r['month_key']}</a></td>"
            f"<td><small>{r['cutoff_date']}</small></td>"
            f"<td><a href='{detail_href}'>{r['station_name']}</a></td>"
            f"<td><span class='badge {badge}'>{r['variant']}</span></td>"
            f"<td>{r['n_oos']:,}</td>"
            f"<td class='{_cls('spearman', r['spearman'])}'>{_num(r['spearman'])}</td>"
            f"<td class='{_cls('mae', r['mae'], True)}'>{_num(r['mae'], '{:.2f}')}</td>"
            f"<td class='{_cls('r5', r['r5'])}'>{_pct(r['r5'])}</td>"
            f"<td class='{_cls('r8', r['r8'])}'>{_pct(r['r8'])}</td>"
            f"<td class='{_cls('r10', r['r10'])}'>{_pct(r['r10'])}</td>"
            f"<td class='{_cls('r30', r['r30'])}'>{_pct(r['r30'])}</td>"
            f"<td class='{_cls('bal_acc', r['bal_acc'])}'>{_pct(r['bal_acc'])}</td>"
            f"<td class='{_cls('far', r['far'], True)}'>{_pct(r['far'])}</td>"
            f"</tr>"
        )

    head = (
        "<tr>"
        "<th>Month</th><th>Cutoff</th><th>Station</th><th>Variant</th>"
        "<th>n OOS</th><th>Spearman ↑</th><th>MAE ↓</th>"
        "<th>recall@5 ↑</th><th>recall@8 ↑</th><th>recall@10 ↑</th>"
        "<th>recall@30 ↑</th><th>Bal.Acc ↑</th><th>FAR ↓</th>"
        "</tr>"
    )

    # Lead section: most recent month's evidence results per station
    lead_html = ""
    if all_results:
        latest = all_results[0]
        month_key_latest = latest["month_key"]
        cutoff_latest = latest.get("cutoff_date", "")[:10]
        lead_cards_html = ""
        for station_key, sdata in latest.get("stations", {}).items():
            ev = sdata.get("variants", {}).get("evidence", {})
            m = ev.get("oos_metrics", {})
            if not m or m.get("n", 0) == 0:
                continue
            station_name = sdata.get("station_name", station_key)
            station_clean = station_name.replace(" ", "_").replace("-", "")
            lead_cards_html += (
                f'<div class="lead-card">'
                f'<h3><a href="{month_key_latest}/{station_clean}_evidence.html">'
                f'{station_name}</a></h3>'
                f'<table style="font-size:0.82em;">'
                f'<tr><td>Spearman</td><td><b>{_num(m.get("spearman"))}</b></td></tr>'
                f'<tr><td>MAE</td><td><b>{_num(m.get("mae"), "{:.1f}")} ppb</b></td></tr>'
                f'<tr><td>recall@8</td><td><b>{_pct(m.get("mag", {}).get("8", {}).get("recall"))}</b></td></tr>'
                f'<tr><td>recall@30</td><td><b>{_pct(m.get("mag", {}).get("30", {}).get("recall"))}</b></td></tr>'
                f'<tr><td>Bal.Acc</td><td><b>{_pct(m.get("balanced_accuracy"))}</b></td></tr>'
                f'<tr><td>FAR</td><td><b>{_pct(m.get("false_alarm_rate"))}</b></td></tr>'
                f'<tr><td>n OOS</td><td>{m.get("n", 0):,}</td></tr>'
                f'</table></div>'
            )
        lead_html = (
            f'<div class="section"><h2>Latest backfill: {month_key_latest} '
            f'<small style="font-weight:400;font-size:0.85em;color:#555;">'
            f'(trained on data before {cutoff_latest}, Evidence variant)</small></h2>'
            f'<div class="lead-grid">{lead_cards_html}</div>'
            f'<p style="font-size:0.82em;color:#666;margin-top:0;">'
            f'<a href="{month_key_latest}/index.html">→ Full report for {month_key_latest}</a>'
            f'</p></div>'
        )

    # Monthly directory — one card per available month, most recent first
    months_seen = sorted({r["month_key"] for r in all_results}, reverse=True)
    month_cards_html = ""
    for mk in months_seen:
        res = next((r for r in all_results if r["month_key"] == mk), {})
        cutoff_d = res.get("cutoff_date", "")[:10]
        n_stations = len(res.get("stations", {}))
        # Gather evidence Spearman per station for the card subtitle
        ev_stats = []
        for sdata in res.get("stations", {}).values():
            ev = sdata.get("variants", {}).get("evidence", {})
            sp = ev.get("oos_metrics", {}).get("spearman")
            if sp is not None and not (isinstance(sp, float) and np.isnan(sp)):
                ev_stats.append(f"{_num(sp)}")
        sp_label = " / ".join(ev_stats) if ev_stats else "—"
        month_cards_html += (
            f'<div class="month-card">'
            f'<a href="{mk}/index.html">{mk}</a>'
            f'<div class="meta">'
            f'Cutoff: {cutoff_d}<br>'
            f'{n_stations} station{"s" if n_stations != 1 else ""}<br>'
            f'Spearman: {sp_label}'
            f'</div></div>'
        )
    months_section = (
        f'<div class="section"><h2>Monthly Reports</h2>'
        f'<div class="months-grid">{month_cards_html}</div>'
        f'</div>'
    ) if months_seen else ""

    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>H2S Backfill Backtest Comparison</title>
<style>{_CSS}</style></head><body>
<div class="header">
  <h1>H2S Walk-Forward Backfill Comparison</h1>
  <p>Monthly models trained on pre-cutoff data only · OOS evaluation on post-cutoff period · Generated {now_str}</p>
</div>
<div class="content">
  {lead_html}
  {months_section}
  <div class="section">
    <h2>All months × stations × variants</h2>
    <p style="font-size:0.84em;color:#555;">
      <b>Dark cells</b> highlight the best value in each column across all runs.
      ↑ = higher is better, ↓ = lower is better.
      recall@8 targets the resident-complaint band (~8 ppb).
      Features use observed H2S lags (oracle one-step), so these are upper-bound
      skill given good inputs — not the recursive multi-step forecast.
    </p>
    <div style="overflow-x:auto;">
      <table><thead>{head}</thead><tbody>{table_rows_html}</tbody></table>
    </div>
  </div>
</div></body></html>"""


@dg.asset(
    key_prefix="h2s",
    group_name="h2s_backfill",
    required_resource_keys={"s3"},
    kinds={"python", "s3"},
    # Declare a graph dependency on the partitioned asset so Dagster schedules
    # this after station_backtest_results completes within the same job run.
    # AllPartitionsMapping is the correct mapping for unpartitioned → partitioned.
    deps=[
        dg.AssetDep(
            asset=_KEY("station_backtest_results"),
            partition_mapping=dg.AllPartitionMapping(),
        )
    ],
    description=(
        "Scan all monthly backtest_results.json files in S3, generate "
        "cross-month comparison table with best-value highlighting, and "
        "upload to tijuana/forecast/backtest/index.html. "
        "Runs automatically at the end of station_backfill_training_job."
    ),
)
def backtest_comparison_index(context: AssetExecutionContext) -> str:
    """Rebuild the cross-month comparison index from all stored monthly results."""
    s3 = context.resources.s3

    # Scan S3 for all backtest_results.json files
    try:
        objects = list(s3.listPath(
            path=f"{BACKTEST_RESULTS_BASE}/",
            bucket=s3.S3_BUCKET
        ))
    except Exception as e:
        context.log.warning(f"Could not list S3 objects: {e}")
        objects = []

    result_paths = [
        obj.object_name for obj in objects
        if obj.object_name.endswith("backtest_results.json")
        and f"{BACKTEST_RESULTS_BASE}/" in obj.object_name
    ]
    context.log.info(f"Found {len(result_paths)} backtest_results.json files in S3")

    all_results: list[dict] = []
    for path in sorted(result_paths):
        try:
            raw = s3.getFile(path=path, bucket=s3.S3_BUCKET)
            data = json.loads(raw.decode("utf-8"))
            all_results.append(data)
            context.log.info(f"  ✓ Loaded {path}")
        except Exception as e:
            context.log.warning(f"  ! Could not load {path}: {e}")

    if not all_results:
        context.log.warning("No backtest results found — uploading empty index.")

    html = _build_comparison_index_html(all_results)
    index_path = f"{BACKTEST_RESULTS_BASE}/index.html"
    s3.putFile(html.encode("utf-8"), index_path,
               bucket=s3.S3_BUCKET, content_type="text/html")

    index_url = s3.publicUrl(index_path)
    context.log.info(f"✓ Uploaded comparison index → {index_path}")
    context.add_output_metadata({
        "n_monthly_results": len(all_results),
        "months": sorted(r["month_key"] for r in all_results),
        "index_url": index_url,
    })
    return index_url


# ============================================================================
# Job definitions
# ============================================================================

station_backfill_training_job = dg.define_asset_job(
    name="station_backfill_training_job",
    description=(
        "Monthly walk-forward: train per-station models on pre-cutoff data, "
        "evaluate OOS, and rebuild the cross-month comparison index. "
        "backtest_comparison_index runs last (unpartitioned) after the partitioned "
        "assets complete for the given month."
    ),
    selection=dg.AssetSelection.assets(
        backfill_training_data,
        backfill_station_models,
        station_backtest_results,
        backtest_comparison_index,
    ),
    partitions_def=BACKFILL_MONTHLY_PARTITIONS,
    tags={"environment": "production", "pipeline": "h2s_backfill"},
)

station_backtest_index_job = dg.define_asset_job(
    name="station_backtest_index_job",
    description=(
        "Manually rebuild the cross-month comparison index from all stored S3 "
        "backtest_results.json files. Useful for regenerating the index without "
        "re-running training (e.g. after a styling change)."
    ),
    selection=dg.AssetSelection.assets(backtest_comparison_index),
    tags={"environment": "production", "pipeline": "h2s_backfill"},
)
