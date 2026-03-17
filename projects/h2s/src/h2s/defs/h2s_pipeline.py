"""H2S Prediction Pipeline - S3-integrated production classification model.

This pipeline loads a pre-trained XGBoost classification model from S3,
processes environmental data, generates H2S predictions (green/yellow/orange),
and exports results to S3 with visualizations.
"""

import json
from datetime import datetime, timedelta
from io import BytesIO

import dagster as dg
import pandas as pd

from h2s.utils import store_assets
from h2s.constants import (
    MODEL_PATH,
    PREDICTIONS_PATH,
    VISUALIZATIONS_PATH,
    LATEST_FORECAST_DATA,
    VALIDATION_PATH,
)

STORE_ASSETS_AVAILABLE = True

_KEY = lambda name: dg.AssetKey(["h2s", name])


def _derive_tidal_state(heights: pd.Series) -> pd.Series:
    """Classify each hourly tide height as flood, ebb, slack high, or slack low."""
    states = ['ebb'] * len(heights)
    for i in range(len(heights)):
        h = heights.iloc[i]
        h_prev = heights.iloc[i - 1] if i > 0 else h
        h_next = heights.iloc[i + 1] if i < len(heights) - 1 else h
        if i > 0 and i < len(heights) - 1:
            if h >= h_prev and h >= h_next:
                states[i] = 'slack high'
            elif h <= h_prev and h <= h_next:
                states[i] = 'slack low'
            elif h > h_prev:
                states[i] = 'flood'
            # else: ebb (default)
        elif h > h_prev:
            states[i] = 'flood'
    return pd.Series(states, index=heights.index)

MODEL_VARIANTS = ["xgboost_base", "xgboost_smote", "random_forest"]


# ==============================================================================
# Asset Group: Model Management
# ==============================================================================

@dg.asset(
    key_prefix="h2s",
    group_name="h2s_model",
    required_resource_keys={"s3"},
    kinds={"xgboost", "s3"},
    description="Pre-trained H2S classification model loaded from S3",
)
def h2s_model_artifacts(context: dg.AssetExecutionContext):
    """Load pre-trained classification model from S3.

    Returns H2SPredictor instance with model and preprocessing info.
    """
    from h2s.predictor.h2s_predictor import H2SPredictor

    s3_resource = context.resources.s3

    context.log.info(f"Loading model from S3: {MODEL_PATH}")

    # Resolve display name from deployment metadata (written by production_model_deployment)
    model_display_name = "XGBoost Weighted"
    try:
        meta_bytes = s3_resource.getFile(f"{MODEL_PATH}/deployment_metadata.json", bucket=s3_resource.S3_BUCKET)
        meta = json.loads(meta_bytes.decode("utf-8"))
        variant = meta.get("approval_metadata", {}).get("variant", "")
        if variant:
            model_display_name = variant.replace("_", " ").title()
    except Exception:
        pass  # No deployment metadata yet — use default name

    predictor = H2SPredictor.from_s3(
        s3_resource,
        f"{MODEL_PATH}/nestor_xgboost_weighted_model.json",
        f"{MODEL_PATH}/nestor_preprocessing_info.json",
        model_name=model_display_name,
    )

    context.log.info(f"Model loaded successfully")
    context.log.info(f"  Features: {len(predictor.feature_cols)}")
    context.log.info(f"  Classes: {predictor.class_names}")
    context.log.info(f"  Site: {predictor.site_name}")

    return predictor


# ==============================================================================
# Asset Group: Data Ingestion
# ==============================================================================

@dg.asset(
    key_prefix="h2s",
    group_name="h2s_prediction",
    required_resource_keys={"s3"},
    kinds={"python", "s3"},
    description="Streamflow forecast using diurnal (month, hour) median profile from historical data",
    config_schema={
        "local_fallback_path": dg.Field(
            str,
            default_value="/Users/valentin/development/dev_resilient/tj_h2s_prediction/data/modeldata_h2s_nofill.parquet",
            description="Local fallback path if S3 historical data is unavailable",
        ),
        "forecast_days": dg.Field(
            int,
            default_value=10,
            description="Number of days forward to generate streamflow forecast",
        ),
    },
)
def streamflow_forecast(context: dg.AssetExecutionContext) -> pd.DataFrame:
    """Generate flow rate forecast using historical diurnal (month, hour) median profile.

    Loads historical training data, computes median Flow (m^3/s)--Border per (month, hour)
    bucket, and projects that profile over the upcoming forecast window.
    Returns columns: time, Flow (m^3/s)--Border
    """
    s3_resource = context.resources.s3
    local_fallback = context.op_config["local_fallback_path"]
    forecast_days = context.op_config["forecast_days"]
    flow_col = "Flow (m^3/s)--Border"

    # --- Load historical data (S3 with local fallback) ---
    hist_df = None
    s3_path = "latest/tijuana/forecast_data/modeldata_h2s_nofill.parquet"
    try:
        stream = s3_resource.get_stream(path=s3_path)
        hist_df = pd.read_parquet(stream)
        context.log.info(f"✓ Loaded historical data from S3: {s3_path} ({len(hist_df)} rows)")
    except Exception as e:
        context.log.warning(f"S3 load failed ({e}), falling back to local: {local_fallback}")
        hist_df = pd.read_parquet(local_fallback)
        context.log.info(f"✓ Loaded historical data from local file ({len(hist_df)} rows)")

    # --- Parse time column ---
    time_col = next((c for c in ["time", "date", "Time", "Date"] if c in hist_df.columns), None)
    if time_col is None:
        raise ValueError(f"No time column found in historical data. Columns: {list(hist_df.columns)}")
    if flow_col not in hist_df.columns:
        raise ValueError(f"Column '{flow_col}' not found. Available: {list(hist_df.columns)}")

    hist_df["_time"] = pd.to_datetime(hist_df[time_col])
    hist_valid = hist_df[hist_df[flow_col].notna()].copy()
    hist_valid["_month"] = hist_valid["_time"].dt.month
    hist_valid["_hour"] = hist_valid["_time"].dt.hour

    # --- Compute (month, hour) median profile ---
    profile = hist_valid.groupby(["_month", "_hour"])[flow_col].median()
    global_median = hist_valid[flow_col].median()

    context.log.info(f"✓ Computed flow profile from {len(hist_valid)} rows")
    context.log.info(f"  Date range: {hist_valid['_time'].min()} → {hist_valid['_time'].max()}")
    context.log.info(f"  Flow range: {hist_valid[flow_col].min():.2f} – {hist_valid[flow_col].max():.2f} m³/s")
    context.log.info(f"  Global median: {global_median:.3f} m³/s")

    # --- Generate forecast timestamps (hourly from now) ---
    now_utc = pd.Timestamp.utcnow().floor("h").tz_localize(None)
    forecast_times = pd.date_range(start=now_utc, periods=forecast_days * 24, freq="h")

    rows = [
        {"time": ts, flow_col: profile.get((ts.month, ts.hour), global_median)}
        for ts in forecast_times
    ]
    forecast_df = pd.DataFrame(rows)

    context.log.info(f"✓ Generated {len(forecast_df)} hourly streamflow forecast rows")
    context.log.info(f"  Flow range: {forecast_df[flow_col].min():.3f} – {forecast_df[flow_col].max():.3f} m³/s")

    # --- Upload to S3 ---
    csv_path = "latest/tijuana/streamflow_forecast/latest.csv"
    try:
        csv_bytes = forecast_df.to_csv(index=False).encode("utf-8")
        s3_resource.putFile(csv_bytes, csv_path, bucket=s3_resource.S3_BUCKET, content_type="text/csv")
        context.log.info(f"✓ Uploaded streamflow forecast to S3: {csv_path}")
    except Exception as e:
        context.log.warning(f"Could not upload streamflow forecast to S3: {e}")

    context.add_output_metadata({
        "row_count": len(forecast_df),
        "flow_min": float(forecast_df[flow_col].min()),
        "flow_max": float(forecast_df[flow_col].max()),
        "flow_median": float(forecast_df[flow_col].median()),
        "profile_buckets": int(len(profile)),
        "historical_rows": int(len(hist_valid)),
    })
    return forecast_df


@dg.asset(
    key_prefix="h2s",
    group_name="h2s_prediction",
    required_resource_keys={"s3"},
    kinds={"python", "s3"},
    description="Tidal predictions from NOAA CO-OPS API (Station 9410170, San Diego)",
    config_schema={
        "forecast_days": dg.Field(
            int,
            default_value=10,
            description="Number of days forward to fetch tidal predictions",
        ),
        "noaa_station": dg.Field(
            str,
            default_value="9410170",
            description="NOAA CO-OPS station ID (default: San Diego, CA — closest to Tijuana River mouth)",
        ),
    },
)
def tidal_forecast(context: dg.AssetExecutionContext) -> pd.DataFrame:
    """Fetch deterministic hourly tidal predictions from NOAA CO-OPS API.

    Station 9410170 (San Diego) is the closest official NOAA gauge to the
    Tijuana River mouth. Derives tidal_state (flood/ebb/slack high/slack low)
    from the slope of the predicted tide height series.
    Returns columns: time, tide_height, tidal_state
    """
    import json as _json
    import urllib.request

    s3_resource = context.resources.s3
    forecast_days = context.op_config["forecast_days"]
    station = context.op_config["noaa_station"]

    now_utc = pd.Timestamp.utcnow()
    begin_date = now_utc.strftime("%Y%m%d")
    end_date = (now_utc + pd.Timedelta(days=forecast_days)).strftime("%Y%m%d")

    api_url = (
        f"https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"
        f"?product=predictions&datum=MLLW&interval=h&units=metric&time_zone=gmt"
        f"&format=json&station={station}&begin_date={begin_date}&end_date={end_date}"
    )

    context.log.info(f"Fetching tidal predictions from NOAA CO-OPS (station {station})...")
    context.log.info(f"  Date range: {begin_date} → {end_date}")

    try:
        with urllib.request.urlopen(api_url, timeout=30) as response:
            data = _json.loads(response.read().decode("utf-8"))
    except Exception as e:
        raise RuntimeError(f"Failed to fetch NOAA tidal predictions: {e}")

    if "error" in data:
        raise RuntimeError(f"NOAA API error: {data['error'].get('message', data['error'])}")

    predictions = data.get("predictions", [])
    if not predictions:
        raise RuntimeError("NOAA API returned no predictions")

    context.log.info(f"✓ Received {len(predictions)} tidal predictions from NOAA")

    tide_df = pd.DataFrame([
        {"time": pd.to_datetime(p["t"]), "tide_height": float(p["v"])}
        for p in predictions
    ]).sort_values("time").reset_index(drop=True)

    tide_df["tidal_state"] = _derive_tidal_state(tide_df["tide_height"])

    state_counts = tide_df["tidal_state"].value_counts().to_dict()
    context.log.info(f"✓ Tide height range: {tide_df['tide_height'].min():.3f} – {tide_df['tide_height'].max():.3f} m")
    context.log.info(f"  Tidal states: {state_counts}")

    # --- Upload to S3 ---
    csv_path = "latest/tijuana/tidal_forecast/latest.csv"
    try:
        csv_bytes = tide_df.to_csv(index=False).encode("utf-8")
        s3_resource.putFile(csv_bytes, csv_path, bucket=s3_resource.S3_BUCKET, content_type="text/csv")
        context.log.info(f"✓ Uploaded tidal forecast to S3: {csv_path}")
    except Exception as e:
        context.log.warning(f"Could not upload tidal forecast to S3: {e}")

    context.add_output_metadata({
        "row_count": len(tide_df),
        "tide_min": float(tide_df["tide_height"].min()),
        "tide_max": float(tide_df["tide_height"].max()),
        "tidal_states": state_counts,
        "station": station,
    })
    return tide_df


@dg.asset(
    key_prefix="h2s",
    group_name="h2s_prediction",
    required_resource_keys={"s3"},
    kinds={"csv", "s3"},
    description="Environmental data loaded from S3 or local test data",
    config_schema={
        "use_local_data": dg.Field(
            bool,
            default_value=False,
            description="Use local test data from data/ directory instead of S3 (for testing only)"
        ),
        "local_data_path": dg.Field(
            str,
            default_value="/Users/valentin/development/dev_resilient/tj_h2s_prediction/data/modeldata_h2s_nofill.parquet",
            description="Path to local test data file (used when use_local_data=True)"
        ),
    },
    ins={
        "streamflow_forecast": dg.AssetIn(key=_KEY("streamflow_forecast")),
        "tidal_forecast": dg.AssetIn(key=_KEY("tidal_forecast")),
    },
)
def raw_environmental_data(
    context: dg.AssetExecutionContext,
    streamflow_forecast: pd.DataFrame,
    tidal_forecast: pd.DataFrame,
) -> pd.DataFrame:
    """Load environmental data from S3 or local test data.

    Production Mode (use_local_data=False):
    - Loads from S3: latest/tijuana/weather_forecast/latest.csv
    - FAILS if S3 data is not available (no fallback)
    - If S3 data lacks H2S measurements, merges from local latest.csv

    Test Mode (use_local_data=True):
    - Loads from local data directory
    - Use for testing when S3 is not available
    """
    s3_resource = context.resources.s3
    use_local = context.op_config["use_local_data"]
    local_path = context.op_config["local_data_path"]

    if use_local:
        # TEST MODE: Load from local data directory
        context.log.info(f"TEST MODE: Loading from local file: {local_path}")
        try:
            if local_path.endswith(".parquet"):
                df = pd.read_parquet(local_path)
            else:
                df = pd.read_csv(local_path)
            context.log.info(f"✓ Loaded {len(df)} rows from local test data")
            source = "local"
        except Exception as e:
            raise RuntimeError(f"Failed to load local test data from {local_path}: {e}")
    else:
        # PRODUCTION MODE: Load from S3 (no fallback)
        context.log.info("PRODUCTION MODE: Loading from S3...")
        s3_path = "latest/tijuana/weather_forecast/latest.csv"
        try:
            stream = s3_resource.get_stream(path=s3_path)
            df = pd.read_csv(stream)
            context.log.info(f"✓ Loaded from S3: {s3_path}")
            source = "s3"
        except Exception as e:
            raise RuntimeError(
                f"Failed to load environmental data from S3 path '{s3_path}'. "
                f"Error: {e}\n"
                f"For testing, set use_local_data=True in asset config."
            )

    # Standardize time column
    if 'time' in df.columns:
        df['date'] = pd.to_datetime(df['time'])
    elif 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])

    # Check if H2S measurements are present
    h2s_cols = [col for col in df.columns if col.upper() == 'H2S' or ('h2s' in col.lower() and col.lower() != 'h2s_measured')]

    # If no H2S measurements and we loaded from S3, try to merge from local modeldata_h2s.csv
    if not h2s_cols and source == "s3":
        try:
            context.log.info("No H2S measurements in S3 data - attempting to merge from local modeldata_h2s.csv")
            local_path = "/Users/valentin/development/dev_resilient/tj_h2s_prediction/data/latest.csv"
            h2s_df = pd.read_parquet(local_path)

            # Standardize time column in H2S data
            if 'time' in h2s_df.columns:
                h2s_df['merge_time'] = pd.to_datetime(h2s_df['time'])
            elif 'date' in h2s_df.columns:
                h2s_df['merge_time'] = pd.to_datetime(h2s_df['date'])

            df['merge_time'] = df['date']

            # Select only H2S columns from local file
            h2s_measurement_cols = [col for col in h2s_df.columns if col.upper() == 'H2S' or col.lower() == 'h2s_measured']
            if h2s_measurement_cols:
                merge_cols = ['merge_time'] + h2s_measurement_cols
                h2s_subset = h2s_df[merge_cols].copy()

                # Merge on time
                df = df.merge(h2s_subset, on='merge_time', how='left')
                df = df.drop(columns=['merge_time'])

                context.log.info(f"✓ Merged H2S measurements from local file")
                h2s_cols = h2s_measurement_cols

        except Exception as e:
            context.log.warning(f"Could not merge H2S measurements from local file: {e}")

    # Log H2S column availability
    if h2s_cols:
        context.log.info(f"✓ H2S measurements available in column: {h2s_cols[0]}")
        h2s_col = h2s_cols[0]
        non_null_count = df[h2s_col].notna().sum()
        if non_null_count > 0:
            context.log.info(f"  H2S values: {non_null_count}/{len(df)} rows")
            context.log.info(f"  H2S range: {df[h2s_col].min():.2f} - {df[h2s_col].max():.2f} ppb")
        else:
            context.log.warning(f"  ⚠ H2S column exists but has no values")

    # --- Merge streamflow forecast ---
    flow_col = "Flow (m^3/s)--Border"
    merge_times = df["date"].dt.floor("h")
    if merge_times.dt.tz is not None:
        merge_times = merge_times.dt.tz_convert("UTC").dt.tz_localize(None)
    df["_merge_time"] = merge_times

    sf = streamflow_forecast[[c for c in ["time", flow_col] if c in streamflow_forecast.columns]].copy()
    sf_times = pd.to_datetime(sf["time"])
    if sf_times.dt.tz is not None:
        sf_times = sf_times.dt.tz_convert("UTC").dt.tz_localize(None)
    sf["_merge_time"] = sf_times.dt.floor("h")
    flow_merge = sf[["_merge_time", flow_col]].drop_duplicates("_merge_time")
    df = df.merge(flow_merge, on="_merge_time", how="left")
    flow_matched = int(df[flow_col].notna().sum())
    context.log.info(f"✓ Streamflow merged: {flow_matched}/{len(df)} rows matched")
    if flow_matched < len(df):
        context.log.warning(f"  {len(df) - flow_matched} rows have no streamflow match — will default to 0.0 in preprocessor")

    # --- Merge tidal forecast ---
    tf = tidal_forecast[["time", "tide_height", "tidal_state"]].copy()
    tf_times = pd.to_datetime(tf["time"])
    if tf_times.dt.tz is not None:
        tf_times = tf_times.dt.tz_convert("UTC").dt.tz_localize(None)
    tf["_merge_time"] = tf_times.dt.floor("h")
    tide_merge = tf[["_merge_time", "tide_height", "tidal_state"]].drop_duplicates("_merge_time")
    df = df.merge(tide_merge, on="_merge_time", how="left")
    tide_matched = int(df["tide_height"].notna().sum())
    context.log.info(f"✓ Tidal merged: {tide_matched}/{len(df)} rows matched")
    if tide_matched < len(df):
        context.log.warning(f"  {len(df) - tide_matched} rows have no tidal match — will default to 0.0 in preprocessor")

    df = df.drop(columns=["_merge_time"])

    context.log.info(f"Loaded {len(df)} rows")
    context.log.info(f"Date range: {df['date'].min()} to {df['date'].max()}")
    context.log.info(f"Columns: {len(df.columns)}")

    return df


# ==============================================================================
# Asset Group: Prediction Pipeline
# ==============================================================================

@dg.asset(
    key_prefix="h2s",
    group_name="h2s_prediction",
    kinds={"python"},
    description="Preprocessed features ready for model prediction",
    auto_materialize_policy=dg.AutoMaterializePolicy.eager(),
    ins={
        "h2s_model_artifacts": dg.AssetIn(key=_KEY("h2s_model_artifacts")),
        "raw_environmental_data": dg.AssetIn(key=_KEY("raw_environmental_data")),
    },
)
def preprocessed_features(
    context: dg.AssetExecutionContext,
    h2s_model_artifacts,
    raw_environmental_data: pd.DataFrame
) -> pd.DataFrame:
    """Apply production preprocessing to raw data.

    Creates cyclical encodings, interaction features, and categorical mappings.
    """
    context.log.info("Preprocessing data...")

    df_processed = h2s_model_artifacts.preprocess_data(raw_environmental_data)

    context.log.info(f"✓ Preprocessed {len(df_processed)} samples")
    context.log.info(f"  Features: {len(df_processed.columns)}")

    return df_processed


@dg.asset(
    key_prefix="h2s",
    group_name="h2s_prediction",
    kinds={"xgboost", "ml"},
    description="H2S category predictions with probabilities",
    auto_materialize_policy=dg.AutoMaterializePolicy.eager(),
    ins={
        "h2s_model_artifacts": dg.AssetIn(key=_KEY("h2s_model_artifacts")),
        "preprocessed_features": dg.AssetIn(key=_KEY("preprocessed_features")),
    },
)
def h2s_predictions(
    context: dg.AssetExecutionContext,
    h2s_model_artifacts,
    preprocessed_features: pd.DataFrame
) -> pd.DataFrame:
    """Generate H2S predictions with probabilities.

    Classifies each sample as green (<5 ppb), yellow (5-30 ppb), or orange (>=30 ppb).
    """
    context.log.info("Generating predictions...")

    results = h2s_model_artifacts.predict(preprocessed_features)

    # Log summary statistics
    orange_count = int((results['predicted_category'] == 'orange').sum())
    yellow_count = int((results['predicted_category'] == 'yellow').sum())
    green_count = int((results['predicted_category'] == 'green').sum())

    context.log.info(f"✓ Generated {len(results)} predictions")
    context.log.info(f"  Green: {green_count} ({green_count/len(results)*100:.1f}%)")
    context.log.info(f"  Yellow: {yellow_count} ({yellow_count/len(results)*100:.1f}%)")
    context.log.info(f"  Orange: {orange_count} ({orange_count/len(results)*100:.1f}%)")

    context.add_output_metadata({
        "total_predictions": len(results),
        "orange_count": orange_count,
        "yellow_count": yellow_count,
        "green_count": green_count,
        "alert_percentage": float((orange_count + yellow_count) / len(results) * 100),
    })

    return results


@dg.asset(
    key_prefix="h2s",
    group_name="h2s_prediction",
    kinds={"python"},
    description="Filtered predictions showing only alerts (orange/yellow)",
    auto_materialize_policy=dg.AutoMaterializePolicy.eager(),
    ins={
        "h2s_predictions": dg.AssetIn(key=_KEY("h2s_predictions")),
    },
)
def h2s_alerts(
    context: dg.AssetExecutionContext,
    h2s_predictions: pd.DataFrame
) -> pd.DataFrame:
    """Filter to orange/yellow alerts only.

    Returns only rows where alert flag is True.
    """
    alerts = h2s_predictions[h2s_predictions['alert'] == True].copy()

    context.log.info(f"Found {len(alerts)} alerts out of {len(h2s_predictions)} total predictions")
    context.log.info(f"  Orange: {(alerts['predicted_category']=='orange').sum()}")
    context.log.info(f"  Yellow: {(alerts['predicted_category']=='yellow').sum()}")

    return alerts


@dg.asset(
    key_prefix="h2s",
    group_name="h2s_prediction",
    required_resource_keys={"s3"},
    kinds={"ml", "xgboost"},
    description="Run predictions for each deployed model variant; skips missing variants",
)
def h2s_variant_predictions(
    context: dg.AssetExecutionContext,
    preprocessed_features: pd.DataFrame,
) -> dict:
    """Load each variant model from S3 and generate predictions.

    Returns:
        Dict mapping variant name → predictions DataFrame.
        Variants whose models are not yet deployed are silently skipped.
    """
    from h2s.predictor.h2s_predictor import H2SPredictor

    s3_resource = context.resources.s3
    prep_path = f"{MODEL_PATH}/nestor_preprocessing_info.json"
    results = {}

    for variant in MODEL_VARIANTS:
        for model_filename in ("model.json", "model.joblib"):
            model_path = f"{MODEL_PATH}/{variant}/{model_filename}"
            try:
                predictor = H2SPredictor.from_s3(s3_resource, model_path, prep_path, model_name=variant.replace("_", " ").title())
                preds = predictor.predict(preprocessed_features)
                results[variant] = preds
                context.log.info(f"✓ {variant}: {len(preds)} predictions")
                break
            except Exception:
                continue
        else:
            context.log.info(f"⚠ {variant}: no deployed model found, skipping")

    context.add_output_metadata({"variants_loaded": list(results.keys()), "n_variants": len(results)})
    return results


@dg.asset(
    key_prefix="h2s",
    group_name="h2s_prediction",
    kinds={"ml"},
    description="Ensemble predictions by averaging probabilities across all available variants",
)
def h2s_ensemble_predictions(
    context: dg.AssetExecutionContext,
    h2s_variant_predictions: dict,
    h2s_predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Average class probabilities across all variant models.

    Falls back to primary model (h2s_predictions) if no variants are loaded.

    Returns:
        DataFrame with ensemble predicted_category, probability_green/yellow/orange, confidence, alert
    """
    variant_dfs = list(h2s_variant_predictions.values())

    if not variant_dfs:
        context.log.warning("No variant models loaded — returning primary model predictions as ensemble")
        return h2s_predictions.copy()

    # Include the primary model in the ensemble
    all_dfs = [h2s_predictions] + variant_dfs

    # Average probabilities
    prob_cols = ["probability_green", "probability_yellow", "probability_orange"]

    avg_probs = pd.DataFrame(0.0, index=all_dfs[0].index, columns=prob_cols)
    for df in all_dfs:
        for col in prob_cols:
            if col in df.columns:
                avg_probs[col] += df[col]
    avg_probs = avg_probs / len(all_dfs)

    # Classify by argmax
    class_map = {0: "green", 1: "yellow", 2: "orange"}
    avg_probs_array = avg_probs[["probability_green", "probability_yellow", "probability_orange"]].values
    predicted_idx = avg_probs_array.argmax(axis=1)
    predicted_category = pd.Series([class_map[i] for i in predicted_idx], index=all_dfs[0].index)

    ensemble_df = all_dfs[0][["time"]].copy() if "time" in all_dfs[0].columns else pd.DataFrame(index=all_dfs[0].index)
    ensemble_df["predicted_category"] = predicted_category
    ensemble_df["probability_green"] = avg_probs["probability_green"]
    ensemble_df["probability_yellow"] = avg_probs["probability_yellow"]
    ensemble_df["probability_orange"] = avg_probs["probability_orange"]
    ensemble_df["confidence"] = avg_probs_array.max(axis=1)
    ensemble_df["alert"] = predicted_category.isin(["yellow", "orange"])
    ensemble_df["n_models"] = len(all_dfs)

    context.log.info(f"✓ Ensemble from {len(all_dfs)} models: {predicted_category.value_counts().to_dict()}")
    context.add_output_metadata({
        "n_models": len(all_dfs),
        "variants": list(h2s_variant_predictions.keys()),
        "alert_count": int(ensemble_df["alert"].sum()),
    })
    return ensemble_df


# ==============================================================================
# Asset Group: Actual Data (Optional)
# ==============================================================================

@dg.asset(
    key_prefix="h2s",
    group_name="h2s_data",
    required_resource_keys={"s3"},
    kinds={"csv", "s3"},
    description="Actual H2S measurements for validation (optional)",
)
def actual_h2s_data(context: dg.AssetExecutionContext) -> pd.DataFrame:
    """Load actual H2S measurements from S3 for model validation.

    This is optional - visualizations will gracefully handle if not available.
    Tries to load from S3: tijuana/forecast/actuals/latest.csv
    """
    s3_resource = context.resources.s3

    try:
        stream = s3_resource.get_stream(path="tijuana/forecast/actuals/latest.csv")
        df = pd.read_csv(stream)
        context.log.info(f"✓ Loaded actual H2S data from S3: {len(df)} rows")

        # Ensure time column is datetime
        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'])

        return df
    except Exception as e:
        context.log.warning(f"Could not load actual H2S data: {e}")
        context.log.info("Returning empty DataFrame - visualizations requiring actuals will be skipped")
        return pd.DataFrame()


# ==============================================================================
# Asset Group: Visualization & Export
# ==============================================================================

@dg.asset(
    key_prefix="h2s",
    group_name="h2s_visualization",
    required_resource_keys={"s3"},
    description="Feature importance visualization stored to S3",
    auto_materialize_policy=dg.AutoMaterializePolicy.eager(),
    ins={
        "h2s_model_artifacts": dg.AssetIn(key=_KEY("h2s_model_artifacts")),
    },
)
def feature_importance_viz(
    context: dg.AssetExecutionContext,
    h2s_model_artifacts
) -> None:
    """Generate and upload feature importance plot to S3."""
    from h2s.predictor.visualizations import generate_feature_importance

    context.log.info("Generating feature importance visualization...")

    s3_resource = context.resources.s3
    model_name = h2s_model_artifacts.model_name
    plot_bytes = generate_feature_importance(
        h2s_model_artifacts.model,
        h2s_model_artifacts.prep_info,
        model_name=model_name,
    )

    # Upload to timestamped path
    timestamp = datetime.now().strftime("%Y-%m-%d")
    timestamped_path = f"{VISUALIZATIONS_PATH}/{timestamp}/feature_importance.png"
    s3_resource.putFile(plot_bytes.read(), timestamped_path, bucket=s3_resource.S3_BUCKET, content_type='image/png')

    # Upload to latest path
    plot_bytes.seek(0)
    latest_path = f"latest/{LATEST_FORECAST_DATA}/visualizations/feature_importance.png"
    s3_resource.putFile(plot_bytes.read(), latest_path, bucket=s3_resource.S3_BUCKET, content_type='image/png')

    context.log.info(f"✓ Uploaded to S3: {timestamped_path}")
    context.log.info(f"✓ Uploaded to S3: {latest_path}")


@dg.asset(
    key_prefix="h2s",
    group_name="h2s_visualization",
    required_resource_keys={"s3"},
    description="Confusion matrix comparing predictions vs actuals (requires actual H2S data in raw_environmental_data)",
    auto_materialize_policy=dg.AutoMaterializePolicy.eager(),
    ins={
        "h2s_predictions": dg.AssetIn(key=_KEY("h2s_predictions")),
        "raw_environmental_data": dg.AssetIn(key=_KEY("raw_environmental_data")),
        "h2s_model_artifacts": dg.AssetIn(key=_KEY("h2s_model_artifacts")),
    },
)
def confusion_matrix_viz(
    context: dg.AssetExecutionContext,
    h2s_predictions: pd.DataFrame,
    raw_environmental_data: pd.DataFrame,
    h2s_model_artifacts,
) -> None:
    """Generate and upload confusion matrix plot to S3.

    Requires actual H2S measurements in raw_environmental_data.
    Skips if H2S measurements are not available in the data.
    """
    from h2s.predictor.visualizations import generate_confusion_matrix_with_metrics

    s3_resource = context.resources.s3

    # Check if raw data has H2S measurements
    h2s_cols = [col for col in raw_environmental_data.columns if col.upper() == 'H2S' or 'h2s' in col.lower()]

    if not h2s_cols:
        context.log.warning("⚠ No H2S column found in raw_environmental_data - skipping confusion matrix")
        context.log.info(f"Available columns: {list(raw_environmental_data.columns)}")
        context.log.info("To generate confusion matrix, ensure raw environmental data includes an 'H2S' or 'h2s' column with actual measurements")
        context.add_output_metadata({
            "status": "skipped",
            "reason": "No H2S measurements in raw_environmental_data",
            "available_columns": list(raw_environmental_data.columns)
        })
        return

    # Prepare actuals data
    actuals_df = raw_environmental_data.copy()

    # Ensure actuals has 'time' column (case-insensitive check)
    if 'time' not in actuals_df.columns:
        # Try to find time-like column (case-insensitive)
        time_cols = [col for col in actuals_df.columns if col.lower() == 'time']
        if time_cols:
            actuals_df['time'] = actuals_df[time_cols[0]]
        elif 'date' in actuals_df.columns:
            actuals_df['time'] = actuals_df['date']
        else:
            context.log.error(f"❌ Raw data missing time column. Available columns: {list(actuals_df.columns)}")
            return

    # Rename H2S column if needed
    if 'H2S' not in actuals_df.columns:
        context.log.info(f"Renaming column '{h2s_cols[0]}' to 'H2S'")
        actuals_df['H2S'] = actuals_df[h2s_cols[0]]

    # Prepare predictions DataFrame with time column
    predictions_df = h2s_predictions.copy()

    # Ensure predictions has 'time' column
    if 'time' not in predictions_df.columns:
        if 'date' in predictions_df.columns:
            predictions_df['time'] = predictions_df['date']
        else:
            context.log.error("❌ Predictions DataFrame missing both 'time' and 'date' columns")
            return

    context.log.info("Generating confusion matrix visualization...")
    context.log.info(f"  Found H2S measurements: {len(actuals_df)} rows")
    context.log.info(f"  Predictions: {len(predictions_df)} rows")
    context.log.info(f"  H2S value range: {actuals_df['H2S'].min():.2f} - {actuals_df['H2S'].max():.2f} ppb")

    model_name = h2s_model_artifacts.model_name
    plot_bytes = generate_confusion_matrix_with_metrics(
        predictions_df,
        actuals_df,
        time_col='time',
        model_name=model_name,
    )

    # Upload to timestamped path
    timestamp = datetime.now().strftime("%Y-%m-%d")
    timestamped_path = f"{VISUALIZATIONS_PATH}/{timestamp}/confusion_matrix.png"
    s3_resource.putFile(plot_bytes.read(), timestamped_path, bucket=s3_resource.S3_BUCKET, content_type='image/png')

    # Upload to latest path
    plot_bytes.seek(0)
    latest_path = f"latest/{LATEST_FORECAST_DATA}/visualizations/confusion_matrix.png"
    s3_resource.putFile(plot_bytes.read(), latest_path, bucket=s3_resource.S3_BUCKET, content_type='image/png')

    context.log.info(f"✓ Uploaded to S3: {timestamped_path}")
    context.log.info(f"✓ Uploaded to S3: {latest_path}")


@dg.asset(
    key_prefix="h2s",
    group_name="h2s_visualization",
    required_resource_keys={"s3"},
    description="Model performance comparison plot (requires actual H2S data)",
    auto_materialize_policy=dg.AutoMaterializePolicy.eager(),
    ins={
        "h2s_predictions": dg.AssetIn(key=_KEY("h2s_predictions")),
        "raw_environmental_data": dg.AssetIn(key=_KEY("raw_environmental_data")),
        "h2s_model_artifacts": dg.AssetIn(key=_KEY("h2s_model_artifacts")),
    },
)
def model_comparison_viz(
    context: dg.AssetExecutionContext,
    h2s_predictions: pd.DataFrame,
    raw_environmental_data: pd.DataFrame,
    h2s_model_artifacts,
) -> None:
    """Generate and upload model comparison plot to S3.

    Shows balanced accuracy, recall, precision, and confusion matrix.
    Requires actual H2S measurements to compare against predictions.
    """
    from h2s.predictor.visualizations import generate_model_comparison

    s3_resource = context.resources.s3

    # Prepare predictions DataFrame with time column
    predictions_df = h2s_predictions.copy()
    if 'time' not in predictions_df.columns:
        if 'date' in predictions_df.columns:
            predictions_df['time'] = predictions_df['date']
        else:
            context.log.error("❌ Predictions DataFrame missing both 'time' and 'date' columns")
            return

    # Ensure actuals has required columns
    actuals_df = raw_environmental_data.copy()
    if 'time' not in actuals_df.columns:
        time_cols = [col for col in actuals_df.columns if col.lower() == 'time']
        if time_cols:
            actuals_df['time'] = actuals_df[time_cols[0]]
        elif 'date' in actuals_df.columns:
            actuals_df['time'] = actuals_df['date']
        else:
            context.log.error(f"❌ Actuals DataFrame missing time column. Available columns: {list(actuals_df.columns)}")
            return

    # Check for H2S measurements
    h2s_cols = [col for col in actuals_df.columns if col.upper() == 'H2S' or 'h2s' in col.lower()]
    if not h2s_cols:
        context.log.warning("⚠ No H2S measurements available - skipping model comparison")
        return

    if 'H2S' not in actuals_df.columns:
        actuals_df['H2S'] = actuals_df[h2s_cols[0]]

    # Filter to only rows with non-null H2S measurements
    actuals_df = actuals_df[actuals_df['H2S'].notna()].copy()

    if len(actuals_df) == 0:
        context.log.warning("⚠ No non-null H2S measurements available - skipping model comparison")
        return

    context.log.info(f"Generating model comparison visualization with {len(actuals_df)} H2S measurements...")

    model_name = h2s_model_artifacts.model_name
    plot_bytes = generate_model_comparison(
        predictions_df,
        actuals_df,
        model_name=model_name,
        time_col='time'
    )

    # Upload to timestamped path
    timestamp = datetime.now().strftime("%Y-%m-%d")
    timestamped_path = f"{VISUALIZATIONS_PATH}/{timestamp}/model_comparison.png"
    s3_resource.putFile(plot_bytes.read(), timestamped_path, bucket=s3_resource.S3_BUCKET, content_type='image/png')

    # Upload to latest path
    plot_bytes.seek(0)
    latest_path = f"latest/{LATEST_FORECAST_DATA}/visualizations/model_comparison.png"
    s3_resource.putFile(plot_bytes.read(), latest_path, bucket=s3_resource.S3_BUCKET, content_type='image/png')

    context.log.info(f"✓ Uploaded to S3: {timestamped_path}")
    context.log.info(f"✓ Uploaded to S3: {latest_path}")


@dg.asset(
    key_prefix="h2s",
    group_name="h2s_visualization",
    required_resource_keys={"s3"},
    description="Prediction timeline plot showing H2S predictions with environmental variables",
    auto_materialize_policy=dg.AutoMaterializePolicy.eager(),
    ins={
        "h2s_predictions": dg.AssetIn(key=_KEY("h2s_predictions")),
        "raw_environmental_data": dg.AssetIn(key=_KEY("raw_environmental_data")),
    },
)
def prediction_timeline_viz(
    context: dg.AssetExecutionContext,
    h2s_predictions: pd.DataFrame,
    raw_environmental_data: pd.DataFrame
) -> None:
    """Generate and upload prediction timeline plot to S3.

    Shows predictions over time with environmental variables.
    Includes actual H2S values if present in raw_environmental_data.
    """
    from h2s.predictor.visualizations import generate_prediction_timeline

    s3_resource = context.resources.s3

    # Prepare predictions DataFrame with time column
    predictions_df = h2s_predictions.copy()
    if 'date' in predictions_df.columns and 'time' not in predictions_df.columns:
        predictions_df['time'] = predictions_df['date']

    # Check if raw data has H2S measurements
    h2s_cols = [col for col in raw_environmental_data.columns if col.upper() == 'H2S' or 'h2s' in col.lower()]

    if h2s_cols:
        context.log.info("Generating prediction timeline (with H2S actuals)...")
    else:
        context.log.info("Generating prediction timeline (predictions + environmental variables)...")

    plot_bytes = generate_prediction_timeline(predictions_df, raw_environmental_data)

    # Upload to timestamped path
    timestamp = datetime.now().strftime("%Y-%m-%d")
    timestamped_path = f"{VISUALIZATIONS_PATH}/{timestamp}/prediction_timeline.png"
    s3_resource.putFile(plot_bytes.read(), timestamped_path, bucket=s3_resource.S3_BUCKET, content_type='image/png')

    # Upload to latest path
    plot_bytes.seek(0)
    latest_path = f"latest/{LATEST_FORECAST_DATA}/visualizations/prediction_timeline.png"
    s3_resource.putFile(plot_bytes.read(), latest_path, bucket=s3_resource.S3_BUCKET, content_type='image/png')

    context.log.info(f"✓ Uploaded to S3: {timestamped_path}")
    context.log.info(f"✓ Uploaded to S3: {latest_path}")


@dg.asset(
    key_prefix="h2s",
    group_name="h2s_export",
    required_resource_keys={"s3"},
    kinds={"s3", "export"},
    description="Predictions exported to S3 as CSV and JSON",
    auto_materialize_policy=dg.AutoMaterializePolicy.eager(),
    ins={
        "h2s_predictions": dg.AssetIn(key=_KEY("h2s_predictions")),
        "h2s_variant_predictions": dg.AssetIn(key=_KEY("h2s_variant_predictions")),
        "h2s_ensemble_predictions": dg.AssetIn(key=_KEY("h2s_ensemble_predictions")),
    },
)
def predictions_export(
    context: dg.AssetExecutionContext,
    h2s_predictions: pd.DataFrame,
    h2s_variant_predictions: dict,
    h2s_ensemble_predictions: pd.DataFrame,
) -> None:
    """Export predictions to S3 with versioning.

    Stores predictions in both timestamped and latest paths.
    S3 Path: tijuana/forecast/predictions/
    """
    s3_resource = context.resources.s3

    context.log.info("Using store_assets utility for export...")

    run_timestamp = datetime.now().strftime("%Y-%m-%d_%H")
    timestamped_path = f"{PREDICTIONS_PATH}/{run_timestamp}"

    metadata = store_assets.objectMetadata(
        name="H2S Predictions - NESTOR BES",
        description="H2S category predictions with probabilities for NESTOR - BES site",
        variableMeasured=["H2S Category", "Probability Scores", "Alert Status"],
    )

    store_assets.store_dataframe_to_s3(
        df=h2s_predictions,
        path=timestamped_path,
        dataset_identifier="h2s_predictions",
        s3_resource=s3_resource,
        metadata=metadata,
        latestdatasetpath=LATEST_FORECAST_DATA,
        enable_latest_path=True,
        formats=['csv', 'json']
    )

    context.log.info(f"✓ Exported predictions to {timestamped_path}")
    context.log.info(f"✓ Latest path: latest/{LATEST_FORECAST_DATA}")

    # === PER-VARIANT PREDICTIONS ===
    for variant, variant_df in h2s_variant_predictions.items():
        variant_csv = variant_df.to_csv(index=False)
        s3_resource.putFile_text(
            variant_csv,
            path=f"latest/{LATEST_FORECAST_DATA}/h2s_predictions_{variant}.csv",
            bucket=s3_resource.S3_BUCKET,
            content_type='text/csv',
        )
        context.log.info(f"✓ Exported variant predictions: h2s_predictions_{variant}.csv")

    # === ENSEMBLE PREDICTIONS ===
    ensemble_csv = h2s_ensemble_predictions.to_csv(index=False)
    s3_resource.putFile_text(
        ensemble_csv,
        path=f"latest/{LATEST_FORECAST_DATA}/h2s_predictions_ensemble.csv",
        bucket=s3_resource.S3_BUCKET,
        content_type='text/csv',
    )
    context.log.info(f"✓ Exported ensemble predictions: h2s_predictions_ensemble.csv")

    context.add_output_metadata({
        "row_count": len(h2s_predictions),
        "export_timestamp": datetime.now().isoformat(),
        "s3_path": timestamped_path,
    })


# ==============================================================================
# Asset Group: Validation
# ==============================================================================

@dg.asset(
    key_prefix="h2s",
    group_name="h2s_validation",
    required_resource_keys={"s3"},
    kinds={"python", "s3"},
    description="Daily report comparing previous day's 6-hourly predictions to actual H2S measurements",
)
def daily_validation_report(context: dg.AssetExecutionContext) -> None:
    """Compare yesterday's 6-hourly predictions against actual H2S measurements.

    Loads predictions from each of the 4 hourly runs (_00, _06, _12, _18),
    combines them, and generates validation plots uploaded to S3.
    Skips gracefully if predictions or actuals are unavailable.
    """
    from h2s.predictor.visualizations import (
        generate_confusion_matrix_with_metrics,
        generate_model_comparison,
        generate_prediction_timeline,
    )

    s3_resource = context.resources.s3
    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

    # Load yesterday's 6-hourly predictions
    prediction_dfs = []
    for hour in ["00", "06", "12", "18"]:
        s3_path = f"{PREDICTIONS_PATH}/{yesterday}_{hour}/h2s_predictions.csv"
        try:
            stream = s3_resource.get_stream(path=s3_path)
            df = pd.read_csv(stream)
            df["run_hour"] = hour
            prediction_dfs.append(df)
            context.log.info(f"✓ Loaded predictions for {yesterday}_{hour}: {len(df)} rows")
        except Exception as e:
            context.log.warning(f"No predictions found for {yesterday}_{hour}: {e}")

    if not prediction_dfs:
        context.log.warning(f"No predictions found for {yesterday} — skipping validation report")
        return

    predictions_df = pd.concat(prediction_dfs, ignore_index=True)
    context.log.info(f"Combined {len(predictions_df)} predictions across {len(prediction_dfs)} runs")

    # Ensure time column exists
    if 'time' not in predictions_df.columns and 'date' in predictions_df.columns:
        predictions_df['time'] = pd.to_datetime(predictions_df['date'])

    # Load actual H2S measurements
    actuals_df = pd.DataFrame()
    try:
        stream = s3_resource.get_stream(path="tijuana/forecast/actuals/latest.csv")
        actuals_df = pd.read_csv(stream)
        if 'time' in actuals_df.columns:
            actuals_df['time'] = pd.to_datetime(actuals_df['time'])
            # Filter to yesterday's date range
            actuals_df = actuals_df[actuals_df['time'].dt.strftime("%Y-%m-%d") == yesterday].copy()
        context.log.info(f"✓ Loaded {len(actuals_df)} actual H2S measurements for {yesterday}")
    except Exception as e:
        context.log.warning(f"Could not load actual H2S data: {e} — plots will be skipped")

    validation_base = f"{VALIDATION_PATH}/{yesterday}"

    # Generate and upload validation plots (only when actuals are available)
    if not actuals_df.empty:
        h2s_cols = [col for col in actuals_df.columns if col.upper() == 'H2S' or 'h2s' in col.lower()]
        if h2s_cols:
            if 'H2S' not in actuals_df.columns:
                actuals_df['H2S'] = actuals_df[h2s_cols[0]]

            for plot_name, plot_fn, kwargs in [
                ("confusion_matrix", generate_confusion_matrix_with_metrics, {"time_col": "time"}),
                ("model_comparison", generate_model_comparison, {"model_name": "XGBoost Weighted", "time_col": "time"}),
                ("prediction_timeline", generate_prediction_timeline, {}),
            ]:
                try:
                    plot_bytes = plot_fn(predictions_df, actuals_df, **kwargs)
                    s3_path = f"{validation_base}/{plot_name}.png"
                    s3_resource.putFile(plot_bytes.read(), s3_path, bucket=s3_resource.S3_BUCKET, content_type='image/png')
                    context.log.info(f"✓ Uploaded {plot_name} to {s3_path}")
                except Exception as e:
                    context.log.warning(f"Could not generate {plot_name}: {e}")
        else:
            context.log.warning("No H2S column in actuals data — skipping validation plots")

    # Export combined predictions CSV (historical record, no latest/ overwrite)
    metadata = store_assets.objectMetadata(
        name=f"H2S Daily Predictions Combined - {yesterday}",
        description=f"Combined 6-hourly H2S predictions for {yesterday}",
        variableMeasured=["H2S Category", "Probability Scores", "Alert Status"],
    )
    store_assets.store_dataframe_to_s3(
        df=predictions_df,
        path=validation_base,
        dataset_identifier="daily_predictions_combined",
        s3_resource=s3_resource,
        metadata=metadata,
        enable_latest_path=False,
        formats=['csv', 'json'],
    )

    context.log.info(f"✓ Validation report saved to {validation_base}")
    context.add_output_metadata({
        "date": yesterday,
        "prediction_runs": len(prediction_dfs),
        "total_predictions": len(predictions_df),
        "actuals_available": not actuals_df.empty,
        "s3_path": validation_base,
    })
