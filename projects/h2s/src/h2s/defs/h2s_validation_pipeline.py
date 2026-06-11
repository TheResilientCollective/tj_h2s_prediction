"""Unified validation pipeline for all H2S forecast models.

Validates predictions from:
- Hourly forecast (nestor_xgboost) — refactored from h2s_pipeline.daily_validation_report
- Daily station forecasts (per-station regression + classification)
- Multi-horizon forecasts (0-6h, 6-24h bands)

Each validation asset writes a v2 metrics.json to:
    s3://{bucket}/tijuana/forecast/validation/{date}/{pipeline}/metrics.json

The accuracy_reporting_pipeline reads these to build scorecards.
"""

import json
import os
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import pandas as pd

import dagster as dg
from dagster import AssetExecutionContext

from h2s.constants import (
    DAILY_SUMMARY_PATH,
    H2S_CLASS_NAMES,
    H2S_CLASS_TO_INT,
    HOURLY_PREDICTIONS_PATH,
    OBS_DATA_PATH,
    PIPELINE_DAILY_STATION,
    PIPELINE_HOURLY,
    RISK_TO_3CLASS,
    STATIONS,
    VALIDATION_PATH,
    VALIDATION_SCHEMA_VERSION,
)
from h2s.training.validation import calculate_false_alarm_rate, calculate_metrics

# Reuse the daily partitions already defined in h2s_pipeline
forecast_daily_partitions = dg.DailyPartitionsDefinition(
    start_date="2026-01-01", timezone="UTC"
)

# Station name → canonical key mapping (observation data uses display names)
_STATION_NAME_TO_KEY = {info["key"]: info["key"] for info in STATIONS.values()}
for name, info in STATIONS.items():
    _STATION_NAME_TO_KEY[name] = info["key"]
# Common variants
_STATION_NAME_TO_KEY["NESTOR__BES"] = "NESTOR__BES"
_STATION_NAME_TO_KEY["NESTOR - BES"] = "NESTOR__BES"
_STATION_NAME_TO_KEY["IB CIVIC CTR"] = "IB_CIVIC_CTR"
_STATION_NAME_TO_KEY["SAN YSIDRO"] = "SAN_YSIDRO"


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _load_observations(
    s3, date_str: str, bucket: str | None = None,
) -> pd.DataFrame:
    """Load observation parquet from S3, filter to *date_str*, return UTC DataFrame.

    Columns guaranteed: time (UTC, rounded to hour), site_key, H2S.
    """
    if bucket is None:
        bucket = os.environ.get("PUBLIC_BUCKET", s3.S3_BUCKET)

    url = s3.publicUrl(path=OBS_DATA_PATH, bucket=bucket)
    df = pd.read_parquet(url)

    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df[(df["h2s_measured"] == True) & (df["H2S"] <= 500)].copy()  # noqa: E712
    df["H2S"] = df["H2S"].clip(lower=0)
    df["time"] = df["time"].dt.round("h")

    # Normalise station names to canonical keys
    df["site_key"] = df["site_name"].map(_STATION_NAME_TO_KEY)
    df = df.dropna(subset=["site_key"])

    # Filter to requested date
    df = df[df["time"].dt.strftime("%Y-%m-%d") == date_str].copy()
    return df[["time", "site_key", "H2S"]]


def _categorize_h2s(value: float) -> str | None:
    """Map raw H2S ppb to green/yellow/orange."""
    if pd.isna(value):
        return None
    if value < 5:
        return "green"
    if value < 30:
        return "yellow"
    return "orange"


def _build_site_metrics(
    preds: pd.DataFrame,
    actuals: pd.DataFrame,
    pred_category_col: str = "predicted_category",
) -> dict[str, Any]:
    """Build v2 per-site metrics dict from matched predictions and actuals.

    *preds* must have columns: time, predicted_category (green/yellow/orange).
    *actuals* must have columns: time, H2S.
    Merge is inner join on time (hourly).
    """
    merged = preds.merge(actuals[["time", "H2S"]], on="time", how="inner")
    merged["actual_category"] = merged["H2S"].apply(_categorize_h2s)
    merged = merged[merged["actual_category"].notna()].copy()

    if len(merged) == 0:
        return {
            "n_predictions": len(preds),
            "n_matched_observations": 0,
            "match_rate": 0.0,
            "balanced_accuracy": None,
            "false_alarm_rate": None,
            "confusion_matrix": None,
            "class_metrics": None,
            "regression_metrics": None,
        }

    y_true = merged["actual_category"].map(H2S_CLASS_TO_INT)
    y_pred = merged[pred_category_col].map(H2S_CLASS_TO_INT)

    metrics = calculate_metrics(y_true=y_true, y_pred=y_pred, class_names=H2S_CLASS_NAMES)
    far = calculate_false_alarm_rate(
        y_true=(merged["actual_category"] == "orange").astype(int),
        y_pred=(merged[pred_category_col] == "orange").astype(int),
        positive_class=1,
    )

    class_metrics = {}
    for cls in ["green", "yellow", "orange"]:
        class_metrics[cls] = {
            "precision": float(metrics.get(f"precision_{cls}", 0.0)),
            "recall": float(metrics.get(f"recall_{cls}", 0.0)),
            "f1_score": float(metrics.get(f"f1_{cls}", 0.0)),
        }

    result: dict[str, Any] = {
        "n_predictions": len(preds),
        "n_matched_observations": len(merged),
        "match_rate": float(len(merged) / len(preds)) if len(preds) else 0.0,
        "balanced_accuracy": float(metrics["balanced_accuracy"]),
        "false_alarm_rate": float(far),
        "confusion_matrix": metrics["confusion_matrix"],
        "class_metrics": class_metrics,
        "regression_metrics": None,
    }

    # Regression metrics if h2s_pred column available
    if "h2s_pred" in merged.columns:
        valid = merged[["h2s_pred", "H2S"]].dropna()
        if len(valid) > 0:
            residuals = valid["h2s_pred"].values - valid["H2S"].values
            ss_res = np.sum(residuals ** 2)
            ss_tot = np.sum((valid["H2S"].values - valid["H2S"].mean()) ** 2)
            result["regression_metrics"] = {
                "rmse": float(np.sqrt(np.mean(residuals ** 2))),
                "mae": float(np.mean(np.abs(residuals))),
                "bias": float(np.mean(residuals)),
                "r2": float(1 - ss_res / ss_tot) if ss_tot > 0 else None,
            }

    return result


def _write_validation_metrics(
    s3, date_str: str, pipeline: str, payload: dict[str, Any],
) -> None:
    """Write metrics.json to ``{VALIDATION_PATH}/{date}/{pipeline}/metrics.json``."""
    body = json.dumps(payload, default=str, indent=2)
    path = f"{VALIDATION_PATH}/{date_str}/{pipeline}/metrics.json"
    s3.putFile_text(data=body, path=path, content_type="application/json")


def _build_full_payload(
    date_str: str,
    pipeline: str,
    sites: dict[str, dict[str, Any]],
    horizon_metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble a complete v2 metrics.json payload."""
    return {
        "schema_version": VALIDATION_SCHEMA_VERSION,
        "date": date_str,
        "pipeline": pipeline,
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "sites": sites,
        "horizon_metrics": horizon_metrics,
    }


# ---------------------------------------------------------------------------
# Asset 1: Hourly forecast validation (refactored schema)
# ---------------------------------------------------------------------------
# The existing daily_validation_report in h2s_pipeline.py is kept as-is for
# now but its metrics_output dict will be updated to v2 schema in a separate
# edit to that file.  This module provides the *new* pipeline validations.


# ---------------------------------------------------------------------------
# Asset 2: Daily station forecast validation
# ---------------------------------------------------------------------------

@dg.asset(
    key_prefix="h2s",
    partitions_def=forecast_daily_partitions,
    group_name="h2s_validation",
    required_resource_keys={"s3"},
    kinds={"python", "s3"},
    description="Validate daily per-station forecasts against observations",
)
def daily_station_validation_report(context: AssetExecutionContext) -> dict[str, Any]:
    s3 = context.resources.s3
    date_str = context.partition_key  # e.g. "2026-04-20"

    # --- Load predictions ---
    # Try timestamped path first, fall back to latest
    csv_url = None
    for path_pattern in [
        f"{DAILY_SUMMARY_PATH}/{date_str}*/daily_station_forecasts.csv",
    ]:
        # List objects to find the timestamped folder
        objs = list(s3.listPath(path=f"{DAILY_SUMMARY_PATH}/{date_str}"))
        csv_objs = [o for o in objs if o.object_name.endswith("daily_station_forecasts.csv")]
        if csv_objs:
            csv_url = s3.publicUrl(path=csv_objs[0].object_name)
            break

    if csv_url is None:
        # Fall back to latest
        try:
            csv_url = s3.publicUrl(
                path="latest/tijuana/forecast_data/daily_station_forecasts.csv"
            )
        except Exception:
            raise dg.Failure(
                f"No daily station forecast CSV found for {date_str}. "
                f"Has daily_analysis_job run for this date?"
            )

    preds_df = pd.read_csv(csv_url)
    preds_df["time"] = pd.to_datetime(preds_df["time"], utc=True).dt.round("h")
    context.log.info(f"Loaded {len(preds_df)} daily station predictions")

    if len(preds_df) == 0:
        raise dg.Failure(f"Daily station forecast CSV is empty for {date_str}")

    # Map 4-tier risk → 3-class
    preds_df["predicted_category"] = preds_df["risk"].map(RISK_TO_3CLASS)

    # Normalise station names
    preds_df["site_key"] = preds_df["station"].map(_STATION_NAME_TO_KEY)

    # --- Load observations ---
    obs_df = _load_observations(s3, date_str)
    context.log.info(f"Loaded {len(obs_df)} observations for {date_str}")

    if len(obs_df) == 0:
        context.log.warning(
            f"No observation data for {date_str}. Sensors may be offline — skipping validation."
        )
        payload = _build_full_payload(date_str, PIPELINE_DAILY_STATION, {})
        payload["skipped"] = True
        payload["skip_reason"] = "no_observations"
        return payload

    # --- Per-station validation ---
    sites: dict[str, dict[str, Any]] = {}
    for site_name, info in STATIONS.items():
        key = info["key"]
        site_preds = preds_df[preds_df["site_key"] == key].copy()
        site_obs = obs_df[obs_df["site_key"] == key].copy()

        if len(site_preds) == 0:
            context.log.warning(f"No predictions for {key} — skipping")
            continue

        site_metrics = _build_site_metrics(site_preds, site_obs)
        sites[key] = site_metrics
        context.log.info(
            f"  {key}: matched={site_metrics['n_matched_observations']}, "
            f"ba={site_metrics.get('balanced_accuracy')}"
        )

    if not sites:
        context.log.warning(f"No stations had predictions for {date_str} — skipping validation.")
        payload = _build_full_payload(date_str, PIPELINE_DAILY_STATION, {})
        payload["skipped"] = True
        payload["skip_reason"] = "no_station_predictions"
        return payload

    payload = _build_full_payload(date_str, PIPELINE_DAILY_STATION, sites)
    _write_validation_metrics(s3, date_str, PIPELINE_DAILY_STATION, payload)

    context.log.info(f"Wrote daily_station validation for {date_str}: {len(sites)} stations")
    return payload
