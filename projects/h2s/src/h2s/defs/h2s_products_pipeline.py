"""Nowcast / Nearcast / Forecast products pipeline (Phase 3).

Runs the recursive inference engine (h2s.forecasting.recursive) for every
station × variant and stores one parquet of product rows per run:

  tijuana/forecast/products/run_ts={ISO}/products.parquet
  latest/tijuana/forecast_data/products_latest.parquet   (mirror)

Row schema (the Phase-5 validation substrate):
  run_ts, product, station, lead_hour, time, variant, model_version,
  h2s_pred, p5, p10, p30

Both variants run every cycle — Evidence drives downstream alert triggers
(Phase 4); Lean is reported alongside. Each row carries the model_version
stamped at deployment so any analysis can be replayed against the exact
archived models (Phase 2).

This pipeline computes ALL THREE products unconditionally so the validation
store accumulates skill-vs-lead-hour data; the compute-on-trigger cascade
(Phase 4) gates only the alerting path, not storage.
"""

import io
import json
import pickle
from datetime import datetime, timezone

import dagster as dg
import pandas as pd

from h2s.constants import (
    FLOW_COL,
    MODEL_FEATURES,
    MODEL_FEATURES_LEAN,
    PRODUCTS_LATEST_PATH,
    PRODUCTS_PATH,
    STATION_MODELS_S3_BASE,
    STATIONS,
)
from h2s.defs.h2s_daily_pipeline import _engineer_forecast_features
from h2s.forecasting.recursive import VariantModels, run_products

_KEY = lambda name: dg.AssetKey(["h2s", name])

_PRODUCT_TASKS = ("regression", "clf_5ppb", "clf_10ppb", "clf_30ppb")
_PRODUCT_VARIANTS: dict[str, list[str]] = {
    "evidence": MODEL_FEATURES,
    "lean": MODEL_FEATURES_LEAN,
}


# ==============================================================================
# Asset 1: Load all product models (3 stations × 2 variants × 4 tasks)
# ==============================================================================

@dg.asset(
    key_prefix="h2s",
    group_name="h2s_products",
    required_resource_keys={"s3"},
    kinds={"python", "s3"},
    description="Load per-station Evidence + Lean model sets for the product pipeline",
)
def products_model_artifacts(context: dg.AssetExecutionContext) -> dict:
    """Load both variants' models per station from the production prefix.

    Returns {station_name: {variant: VariantModels-ready dict,
    '_features': {variant: cols}, '_model_version': str}}.
    A station is skipped only if a variant's REGRESSION model is missing;
    missing classifiers degrade to NaN probabilities downstream.
    """
    s3 = context.resources.s3
    artifacts: dict = {}

    for site_name, info in STATIONS.items():
        station_key = info["key"]
        base_path = f"{STATION_MODELS_S3_BASE}/{station_key}"
        station: dict = {"_features": {}}

        for variant in _PRODUCT_VARIANTS:
            models: dict = {}
            for task in _PRODUCT_TASKS:
                s3_path = f"{base_path}/{task}_{variant}.pkl"
                try:
                    models[task] = pickle.loads(s3.getFile(path=s3_path, bucket=s3.S3_BUCKET))
                except Exception:
                    models[task] = None
                    context.log.warning(f"  ✗ {site_name} / {task}_{variant} missing")
            if models.get("regression") is None:
                context.log.warning(f"  ✗ {site_name} / {variant}: no regression model — variant skipped")
                continue

            try:
                feat_bytes = s3.getFile(path=f"{base_path}/features_{variant}.json", bucket=s3.S3_BUCKET)
                station["_features"][variant] = json.loads(feat_bytes.decode("utf-8"))
            except Exception:
                station["_features"][variant] = list(_PRODUCT_VARIANTS[variant])
                context.log.info(f"  ⚠ {site_name} / features_{variant}.json missing — using constants")

            station[variant] = models
            context.log.info(f"  ✓ {site_name} / {variant} ({len(station['_features'][variant])} features)")

        try:
            meta_bytes = s3.getFile(path=f"{base_path}/deployment_metadata.json", bucket=s3.S3_BUCKET)
            station["_model_version"] = json.loads(meta_bytes.decode("utf-8")).get(
                "model_version", "unversioned")
        except Exception:
            station["_model_version"] = "unversioned"

        if any(v in station for v in _PRODUCT_VARIANTS):
            artifacts[site_name] = station

    context.add_output_metadata({
        "stations_loaded": list(artifacts.keys()),
        "model_versions": {s: a.get("_model_version") for s, a in artifacts.items()},
    })
    return artifacts


# ==============================================================================
# Asset 2: Run all three products for every station × variant
# ==============================================================================

@dg.asset(
    key_prefix="h2s",
    group_name="h2s_products",
    required_resource_keys={"s3"},
    kinds={"python", "ml", "s3"},
    description="Nowcast (0-3h) + nearcast (3-6h) + forecast (6-24h) rows per station × variant",
    ins={"products_model_artifacts": dg.AssetIn(key=_KEY("products_model_artifacts"))},
    config_schema={
        "obs_bucket": dg.Field(str, default_value="resilentpublic",
                               description="Bucket holding observations + forecast met data"),
        "forecast_hours": dg.Field(int, default_value=24),
    },
)
def h2s_products(
    context: dg.AssetExecutionContext,
    products_model_artifacts: dict,
) -> pd.DataFrame:
    """Compute the three products and store the rows to S3.

    Honest scope: the forecast tier (leads 7-24) recursion feeds the model's
    own predictions back through the autoregressive features, so magnitude
    skill decays toward the exogenous ceiling. Treat forecast-tier output as
    a risk ranking; the validation store (Phase 5) quantifies the decay.
    """
    s3 = context.resources.s3
    bucket = context.op_config["obs_bucket"]
    forecast_hours = context.op_config["forecast_hours"]
    run_ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%MZ")

    # --- Observations (H2S seed history per station) -----------------------
    obs_url = s3.publicUrl(path="latest/tijuana/forecast_data/modeldata_h2s_nofill.parquet",
                           bucket=bucket)
    obs_df = pd.read_parquet(obs_url)
    obs_df["time"] = pd.to_datetime(obs_df["time"], utc=True)
    obs_df = obs_df[(obs_df["h2s_measured"] == True) & (obs_df["H2S"] <= 500)].copy()  # noqa: E712
    obs_df["H2S"] = obs_df["H2S"].clip(lower=0)
    context.log.info(f"✓ Observations: {len(obs_df)} rows")

    # --- Forecast met data (same loading as the daily pipeline) ------------
    try:
        fc_df = pd.read_parquet(s3.publicUrl(path="latest/tijuana/forecast_data/model_forecast.parquet"))
        context.log.info(f"✓ Model forecast (parquet): {len(fc_df)} rows")
    except Exception:
        fc_df = pd.read_csv(s3.publicUrl(path="latest/tijuana/forecast_data/model_forecast.csv"))
        context.log.info(f"✓ Model forecast (csv): {len(fc_df)} rows")
    if "time" not in fc_df.columns and "date" in fc_df.columns:
        fc_df = fc_df.rename(columns={"date": "time"})
    fc_df["time"] = pd.to_datetime(fc_df["time"], utc=True)

    try:
        tidal_df = pd.read_csv(s3.publicUrl(path="latest/tijuana/tidal_forecast/latest.csv"))
        tidal_df["time"] = pd.to_datetime(tidal_df["time"], utc=True)
        tidal_df["_mtime"] = tidal_df["time"].dt.floor("h")
        fc_df["_mtime"] = pd.to_datetime(fc_df["time"]).dt.floor("h")
        fc_df = fc_df.merge(
            tidal_df[["_mtime", "tide_height", "tidal_state"]].drop_duplicates("_mtime"),
            on="_mtime", how="left",
        ).drop(columns=["_mtime"])
    except Exception as e:
        context.log.warning(f"Tidal forecast unavailable: {e}")
        fc_df["tide_height"] = 0.5
        fc_df["tidal_state"] = "ebb"

    if FLOW_COL not in fc_df.columns:
        fc_df[FLOW_COL] = 2.0
    else:
        fc_df[FLOW_COL] = fc_df[FLOW_COL].fillna(2.0)

    # --- Run products per station × variant ---------------------------------
    all_rows: list[pd.DataFrame] = []
    for site_name, station in products_model_artifacts.items():
        ss = obs_df[obs_df["site_name"] == site_name].sort_values("time")
        if len(ss) == 0:
            context.log.warning(f"No observations for {site_name} — skipping")
            continue
        h2s_history = ss["H2S"].tail(24).tolist()
        last_state = {
            "h2s": float(ss.iloc[-1]["H2S"]),
            "h2s_6h": float(ss.tail(6)["H2S"].mean()),
            "h2s_24h": float(ss.tail(24)["H2S"].mean()),
            "flow": float(ss.iloc[-1].get(FLOW_COL, 2.0) or 2.0),
            "flow_24h": float(ss.tail(24).get(FLOW_COL, pd.Series([2.0])).mean()),
        }

        sfc = fc_df.head(forecast_hours).copy().reset_index(drop=True)
        sfc["site_name"] = site_name
        sfc = _engineer_forecast_features(sfc, last_state)

        for variant in _PRODUCT_VARIANTS:
            if variant not in station:
                continue
            feature_cols = station["_features"][variant]
            for col in feature_cols:
                if col not in sfc.columns:
                    sfc[col] = 0.0

            tasks = station[variant]
            models = VariantModels(
                regression=tasks["regression"],
                clf_5ppb=tasks.get("clf_5ppb"),
                clf_10ppb=tasks.get("clf_10ppb"),
                clf_30ppb=tasks.get("clf_30ppb"),
            )
            product_rows = run_products(sfc, h2s_history, models, feature_cols)
            product_rows["station"] = site_name
            product_rows["variant"] = variant
            product_rows["model_version"] = station.get("_model_version", "unversioned")
            all_rows.append(product_rows)
            context.log.info(
                f"  ✓ {site_name} / {variant}: {len(product_rows)} rows "
                f"(max h2s_pred={product_rows['h2s_pred'].max():.1f} ppb)"
            )

    if not all_rows:
        raise dg.Failure("No product rows generated — no stations had models + observations")

    out = pd.concat(all_rows, ignore_index=True)
    out.insert(0, "run_ts", run_ts)

    # --- Store: per-run parquet + latest mirror -----------------------------
    buf = io.BytesIO()
    out.to_parquet(buf, index=False)
    data = buf.getvalue()
    run_path = f"{PRODUCTS_PATH}/run_ts={run_ts}/products.parquet"
    s3.putFile(data, run_path, bucket=s3.S3_BUCKET, content_type="application/octet-stream")
    s3.putFile(data, PRODUCTS_LATEST_PATH, bucket=s3.S3_BUCKET, content_type="application/octet-stream")
    context.log.info(f"✓ Stored {len(out)} product rows → {run_path} (+ latest mirror)")

    context.add_output_metadata({
        "run_ts": run_ts,
        "rows": len(out),
        "stations": list(out["station"].unique()),
        "variants": list(out["variant"].unique()),
        "products": list(out["product"].unique()),
        "s3_path": run_path,
        "max_h2s_pred": float(out["h2s_pred"].max()),
    })
    return out


products_forecast_job = dg.define_asset_job(
    name="products_forecast_job",
    description=(
        "Run nowcast + nearcast + forecast for all stations × variants and "
        "store the product rows to S3. Cascade-triggered alerting consumes "
        "these in Phase 4; the validation store in Phase 5."
    ),
    selection=dg.AssetSelection.assets(products_model_artifacts, h2s_products),
    tags={"environment": "production", "pipeline": "h2s_products"},
)
