"""Forecast validation store + accuracy reporting (Phase 5).

Joins the stored nowcast/nearcast/forecast product rows against the H2S
actually measured at each target hour, writes one consolidated — and fully
**rebuildable** — parquet, and computes per-lead-hour skill curves
(Spearman + recall@{5,10,30,100}, Evidence vs Lean).

Rebuildable by construction: every build recomputes from the immutable product
runs under ``PRODUCTS_PATH`` (idempotent). The daily job uses a recent window;
``station_forecast_validation_rebuild_job`` passes ``max_age_days=None`` to recompute
the whole history from scratch.

  forecast_validation_store → forecast_skill_report
"""

import io
import json
from datetime import datetime, timezone

import dagster as dg
import numpy as np
import pandas as pd

from h2s.constants import (
    FORECAST_SKILL_CURVES_PATH,
    FORECAST_SKILL_REPORT_LATEST_PATH,
    FORECAST_SKILL_REPORT_PATH,
    FORECAST_VALIDATION_MAX_AGE_DAYS,
    FORECAST_VALIDATION_SNAPSHOT_PATH,
    FORECAST_VALIDATION_STORE_PATH,
    PRODUCTS_PATH,
)
from h2s.forecasting.product_validation import build_validation_rows, skill_curves

_KEY = lambda name: dg.AssetKey(["h2s", name])

_OBS_RELPATH = "latest/tijuana/forecast_data/modeldata_h2s_nofill.parquet"


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _load_product_runs(s3, max_age_days, log) -> pd.DataFrame:
    """Concatenate stored product runs whose run_ts is within max_age_days
    (None = all runs). Each parquet is read via a presigned URL."""
    bucket = s3.S3_BUCKET
    try:
        objs = s3.getClient().list_objects(bucket, prefix=f"{PRODUCTS_PATH}/", recursive=True)
    except Exception as e:
        log.warning(f"Could not list product runs: {e}")
        return pd.DataFrame()

    cutoff = None
    if max_age_days is not None:
        cutoff = pd.Timestamp.now("UTC") - pd.Timedelta(days=max_age_days)

    chosen: list[tuple[pd.Timestamp, str]] = []
    for o in objs:
        name = o.object_name
        if not name.endswith("products.parquet"):
            continue
        seg = next((p for p in name.split("/") if p.startswith("run_ts=")), None)
        if seg is None:
            continue
        try:
            rt = pd.to_datetime(seg.split("=", 1)[1], format="%Y-%m-%dT%H%MZ", utc=True)
        except Exception:
            continue
        if cutoff is None or rt >= cutoff:
            chosen.append((rt, name))

    chosen.sort()
    frames = []
    for _, name in chosen:
        try:
            frames.append(pd.read_parquet(s3.get_presigned_url(name, bucket=bucket)))
        except Exception as e:
            log.warning(f"Could not read {name}: {e}")

    log.info(f"Loaded {len(frames)} product run(s) within "
             f"{'all history' if max_age_days is None else f'{max_age_days}d'}")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _load_observations(s3, bucket: str, log) -> pd.DataFrame:
    url = s3.publicUrl(path=_OBS_RELPATH, bucket=bucket)
    df = pd.read_parquet(url)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df[(df["h2s_measured"] == True) & (df["H2S"] <= 500)].copy()  # noqa: E712
    df["H2S"] = df["H2S"].clip(lower=0)
    log.info(f"Loaded {len(df)} measured observations")
    return df


def _put_parquet(s3, df: pd.DataFrame, path: str) -> bytes:
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    data = buf.getvalue()
    s3.putFile(data, path, bucket=s3.S3_BUCKET, content_type="application/octet-stream")
    return data


# ---------------------------------------------------------------------------
# Asset 1: rebuildable validation store
# ---------------------------------------------------------------------------

@dg.asset(
    key_prefix="h2s",
    group_name="forecast_validation",
    required_resource_keys={"s3"},
    kinds={"python", "s3"},
    description="Rebuildable join of stored product rows vs measured H2S (Phase-5 validation substrate)",
    config_schema={
        "max_age_days": dg.Field(
            dg.Noneable(int), default_value=FORECAST_VALIDATION_MAX_AGE_DAYS,
            description="Only product runs newer than this are joined; None = all runs (full rebuild)",
        ),
        "obs_bucket": dg.Field(str, default_value="resilentpublic",
                               description="Bucket holding the measured-observation parquet"),
    },
)
def forecast_validation_store(context: dg.AssetExecutionContext) -> pd.DataFrame:
    s3 = context.resources.s3
    max_age = context.op_config["max_age_days"]
    obs_bucket = context.op_config["obs_bucket"]

    products = _load_product_runs(s3, max_age, context.log)
    if products.empty:
        context.log.warning("No product runs found — writing an empty validation store.")
        empty = build_validation_rows(products, pd.DataFrame(columns=["site_name", "time", "H2S"]))
        _put_parquet(s3, empty, FORECAST_VALIDATION_STORE_PATH)
        context.add_output_metadata({"validation_rows": 0, "product_runs": 0, "rebuild": max_age is None})
        return empty

    obs = _load_observations(s3, obs_bucket, context.log)
    v = build_validation_rows(products, obs)

    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    data = _put_parquet(s3, v, FORECAST_VALIDATION_STORE_PATH)
    s3.putFile(data, FORECAST_VALIDATION_SNAPSHOT_PATH.format(date_str=date_str),
               bucket=s3.S3_BUCKET, content_type="application/octet-stream")

    n_runs = int(products["run_ts"].nunique()) if "run_ts" in products.columns else 0
    context.log.info(f"✓ Validation store: {len(v)} rows from {n_runs} product run(s)")
    context.add_output_metadata({
        "validation_rows": int(len(v)),
        "product_runs": n_runs,
        "products": sorted(v["product"].unique().tolist()) if not v.empty else [],
        "variants": sorted(v["variant"].unique().tolist()) if not v.empty else [],
        "time_min": str(v["time"].min()) if not v.empty else "n/a",
        "time_max": str(v["time"].max()) if not v.empty else "n/a",
        "rebuild": max_age is None,
        "s3_path": FORECAST_VALIDATION_STORE_PATH,
    })
    return v


# ---------------------------------------------------------------------------
# Asset 2: per-lead-hour skill report
# ---------------------------------------------------------------------------

def _summarize(curves: pd.DataFrame, v: pd.DataFrame) -> dict:
    meta = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_validation_rows": int(len(v)),
        "time_min": str(v["time"].min()) if not v.empty else None,
        "time_max": str(v["time"].max()) if not v.empty else None,
    }
    if curves.empty:
        return {**meta, "headline": [], "note": "no overlapping forecast/actual rows yet"}

    headline = []
    for (product, variant), g in curves.groupby(["product", "variant"]):
        scored = g.dropna(subset=["spearman"])
        wsp = (float(np.average(scored["spearman"], weights=scored["n"]))
               if len(scored) else None)
        headline.append({
            "product": product,
            "variant": variant,
            "n": int(g["n"].sum()),
            "spearman_weighted": round(wsp, 3) if wsp is not None else None,
            "spearman_by_lead": {
                int(r.lead_hour): (None if pd.isna(r.spearman) else round(r.spearman, 3))
                for r in g.itertuples()
            },
            "mae_by_lead": {int(r.lead_hour): round(r.mae, 2) for r in g.itertuples()},
            "prob_recall_5_by_lead": {
                int(r.lead_hour): (None if getattr(r, "prob_recall_5", None) is None
                                   else round(r.prob_recall_5, 3))
                for r in g.itertuples()
            },
            "prob_recall_10_by_lead": {
                int(r.lead_hour): (None if getattr(r, "prob_recall_10", None) is None
                                   else round(r.prob_recall_10, 3))
                for r in g.itertuples()
            },
            "prob_recall_30_by_lead": {
                int(r.lead_hour): (None if getattr(r, "prob_recall_30", None) is None
                                   else round(r.prob_recall_30, 3))
                for r in g.itertuples()
            },
        })
    return {**meta, "headline": headline, "curves": curves.to_dict(orient="records")}


@dg.asset(
    key_prefix="h2s",
    group_name="forecast_validation",
    required_resource_keys={"s3"},
    kinds={"python", "s3"},
    description="Per-(product, variant, lead-hour) skill curves: Spearman + recall@{5,10,30,100}, Evidence vs Lean",
    ins={"forecast_validation_store": dg.AssetIn(key=_KEY("forecast_validation_store"))},
)
def forecast_skill_report(
    context: dg.AssetExecutionContext,
    forecast_validation_store: pd.DataFrame,
) -> pd.DataFrame:
    s3 = context.resources.s3
    v = forecast_validation_store
    curves = skill_curves(v)

    if not curves.empty:
        _put_parquet(s3, curves, FORECAST_SKILL_CURVES_PATH)

    report = _summarize(curves, v)
    payload = json.dumps(report, indent=2, default=str)
    s3.putFile(payload, FORECAST_SKILL_REPORT_PATH, bucket=s3.S3_BUCKET, content_type="application/json")
    s3.putFile(payload, FORECAST_SKILL_REPORT_LATEST_PATH, bucket=s3.S3_BUCKET, content_type="application/json")

    context.log.info(f"✓ Skill report: {len(curves)} (product, variant, lead) cells")
    context.add_output_metadata({
        "skill_cells": int(len(curves)),
        "products": sorted(curves["product"].unique().tolist()) if not curves.empty else [],
        "report_path": FORECAST_SKILL_REPORT_PATH,
        "headline": json.dumps(report.get("headline", []))[:2000],
    })
    return curves


# ---------------------------------------------------------------------------
# Jobs + schedule
# ---------------------------------------------------------------------------

forecast_validation_job = dg.define_asset_job(
    name="forecast_validation_job",
    selection=dg.AssetSelection.assets(forecast_validation_store, forecast_skill_report),
    description="Recompute the forecast validation store (recent window) + skill curves",
    tags={"environment": "production", "pipeline": "forecast_validation"},
)

station_forecast_validation_rebuild_job = dg.define_asset_job(
    name="station_forecast_validation_rebuild_job",
    selection=dg.AssetSelection.assets(forecast_validation_store, forecast_skill_report),
    config={"ops": {"h2s__forecast_validation_store": {"config": {"max_age_days": None}}}},
    description="Full rebuild of the validation store + skill curves from ALL stored product runs",
    tags={"environment": "production", "pipeline": "forecast_validation"},
)

forecast_validation_schedule = dg.ScheduleDefinition(
    job=forecast_validation_job,
    cron_schedule="0 9 * * *",
    default_status=dg.DefaultScheduleStatus.STOPPED,
    description="Daily 09:00 UTC: refresh the forecast validation store + skill curves",
)
