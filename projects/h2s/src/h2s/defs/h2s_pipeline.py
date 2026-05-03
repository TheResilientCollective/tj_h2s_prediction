"""H2S Prediction Pipeline - S3-integrated production classification model.

This pipeline loads a pre-trained XGBoost classification model from S3,
processes environmental data, generates H2S predictions (green/yellow/orange),
and exports results to S3 with visualizations.
"""

import json
import os
from datetime import datetime, timedelta, timezone
from io import BytesIO
from zoneinfo import ZoneInfo

import dagster as dg
import pandas as pd

from h2s.utils import store_assets
from h2s.constants import (
    FORECAST_DATA_PATH,
    H2S_CLASS_NAMES,
    H2S_CLASS_TO_INT,
    HOURLY_PREDICTIONS_PATH,
    LATEST_FORECAST,
    MODEL_PATH,
    OBS_DATA_PATH,
    PIPELINE_DAILY_STATION,
    VALIDATION_PATH,
    VISUALIZATIONS_PATH,
)

STORE_ASSETS_AVAILABLE = True

_KEY = lambda name: dg.AssetKey(["h2s", name])

MODEL_VARIANTS = ["xgboost_base", "xgboost_smote", "random_forest"]


# ==============================================================================
# Partition Definition: Daily Forecast/Validation Runs
# ==============================================================================

forecast_daily_partitions = dg.DailyPartitionsDefinition(
    start_date="2026-01-01",
    end_offset=1,  # Include today's partition (0 would exclude it until day completes)
    timezone="UTC",
)


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
# Asset Group: Prediction Pipeline
# ==============================================================================

@dg.asset(
    key_prefix="h2s",
    group_name="h2s_prediction",
    required_resource_keys={"s3"},
    kinds={"python", "s3"},
    description="Preprocessed features ready for model prediction (loaded from pre-featurized S3 data)",
    partitions_def=forecast_daily_partitions,
    auto_materialize_policy=dg.AutoMaterializePolicy.eager(),
    ins={
        "h2s_model_artifacts": dg.AssetIn(
            key=_KEY("h2s_model_artifacts"),
            partition_mapping=dg.AllPartitionMapping()
        ),
    },
)
def preprocessed_features(
    context: dg.AssetExecutionContext,
    h2s_model_artifacts,
) -> pd.DataFrame:
    """Load forecast/observation data and apply model preprocessing.

    Data source selection:
    - Current partition (today): Load from FORECAST_DATA_PATH (forecast data)
    - Historical partition (< today): Load from OBS_DATA_PATH (observation data for backfills)

    Reads input data from PUBLIC_BUCKET (production sensor data).
    Writes outputs to S3_BUCKET (predictions, visualizations).
    """
    s3 = context.resources.s3
    partition_key = context.partition_key  # e.g., "2026-04-02"
    today = datetime.now(ZoneInfo("UTC")).date().strftime("%Y-%m-%d")

    # Read input data from PUBLIC_BUCKET (production data from sensors)
    public_bucket = os.environ.get('PUBLIC_BUCKET', s3.S3_BUCKET)

    # Determine data source based on partition
    is_backfill = partition_key < today

    if is_backfill:
        # BACKFILL MODE: Use observation data (actual conditions that occurred)
        data_path = OBS_DATA_PATH
        context.log.info(f"🔄 Backfill mode: Loading observation data from {data_path}")
        context.log.info(f"   Partition: {partition_key}, Today: {today}")
        context.log.info(f"   Reading from PUBLIC_BUCKET: {public_bucket}")

        try:
            data_url = s3.publicUrl(path=data_path, bucket=public_bucket)
            df = pd.read_parquet(data_url)
            context.log.info(f"✓ Loaded {len(df)} rows from observation data")
        except Exception as e:
            raise RuntimeError(
                f"Failed to load observation data from S3 path '{data_path}': {e}\n"
                f"Observation data is required for historical backfills (partition < today)."
            )

        # Filter to partition date (observation data spans full history)
        df['time'] = pd.to_datetime(df['time'], utc=True)
        partition_dt = datetime.strptime(partition_key, "%Y-%m-%d").replace(tzinfo=ZoneInfo("UTC"))
        next_day = partition_dt + timedelta(days=1)

        df_filtered = df[(df['time'] >= partition_dt) & (df['time'] < next_day)].copy()
        context.log.info(f"✓ Filtered to partition date {partition_key}: {len(df_filtered)} rows")

        if len(df_filtered) == 0:
            # Find recent dates with data to help user
            df['date_str'] = df['time'].dt.strftime('%Y-%m-%d')
            available_dates = sorted(df['date_str'].unique(), reverse=True)[:10]

            raise ValueError(
                f"No observation data found for partition {partition_key} — sensors likely offline that day.\n"
                f"Cannot backfill without observation data.\n"
                f"Recent dates WITH available data: {', '.join(available_dates[:5])}\n"
                f"Try backfilling one of these dates instead."
            )

        # Filter to valid measurements only
        if 'h2s_measured' in df_filtered.columns:
            df_filtered = df_filtered[df_filtered['h2s_measured'] == True].copy()
            context.log.info(f"✓ Filtered to h2s_measured==True: {len(df_filtered)} rows")

        # Drop H2S column if present (not used as input feature)
        if 'H2S' in df_filtered.columns:
            df_filtered = df_filtered.drop(columns=['H2S'])
            context.log.info("✓ Dropped H2S column (target variable, not input)")

        df = df_filtered

    else:
        # LIVE FORECAST MODE: Use forecast data (current/future predictions)
        data_path = FORECAST_DATA_PATH
        context.log.info(f"📡 Live forecast mode: Loading forecast data from {data_path}")
        context.log.info(f"   Reading from PUBLIC_BUCKET: {public_bucket}")

        try:
            data_url = s3.publicUrl(path=data_path, bucket=public_bucket)
            df = pd.read_parquet(data_url)
            context.log.info(f"✓ Loaded {len(df)} rows from forecast data")
        except Exception as e:
            raise RuntimeError(
                f"Failed to load forecast data from S3 path '{data_path}': {e}\n"
                f"Forecast data is required for current/future predictions (partition >= today)."
            )

    # Apply model preprocessing (idempotent — fills only missing features)
    context.log.info("Applying model preprocessing...")
    df_processed = h2s_model_artifacts.preprocess_data(df)

    context.log.info(f"✓ Preprocessed {len(df_processed)} samples")
    context.log.info(f"  Features: {len(df_processed.columns)}")

    return df_processed


@dg.asset(
    key_prefix="h2s",
    group_name="h2s_prediction",
    kinds={"xgboost", "ml"},
    description="H2S category predictions with probabilities",
    partitions_def=forecast_daily_partitions,
    auto_materialize_policy=dg.AutoMaterializePolicy.eager(),
    ins={
        "h2s_model_artifacts": dg.AssetIn(
            key=_KEY("h2s_model_artifacts"),
            partition_mapping=dg.AllPartitionMapping()
        ),
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
    partitions_def=forecast_daily_partitions,
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
    group_name="h2s_alerts",
    kinds={"slack"},
    required_resource_keys={"slack", "s3"},
    description="Send H2S forecast summary to Slack",
    partitions_def=forecast_daily_partitions,
    auto_materialize_policy=dg.AutoMaterializePolicy.eager(),
    ins={
        "h2s_alerts": dg.AssetIn(key=_KEY("h2s_alerts")),
        "h2s_predictions": dg.AssetIn(key=_KEY("h2s_predictions")),
    },
)
def slack_alerts(
    context: dg.AssetExecutionContext,
    h2s_alerts: pd.DataFrame,
    h2s_predictions: pd.DataFrame,
) -> None:
    """Post H2S forecast summary to Slack with 48h chart."""
    # Skip Slack notifications for backfills (only send for today's partition)
    today = datetime.now(ZoneInfo("UTC")).date().strftime("%Y-%m-%d")
    if context.partition_key != today:
        context.log.info(
            f"Skipping Slack notification for partition {context.partition_key} "
            f"(only sending for today's partition: {today})"
        )
        return

    orange_count = int((h2s_alerts['predicted_category'] == 'orange').sum()) if not h2s_alerts.empty else 0
    yellow_count = int((h2s_alerts['predicted_category'] == 'yellow').sum()) if not h2s_alerts.empty else 0
    total = len(h2s_alerts)

    # Build time range from full predictions in Pacific time
    pacific = ZoneInfo("America/Los_Angeles")
    time_col = 'time' if 'time' in h2s_predictions.columns else None
    if time_col and not h2s_predictions.empty:
        t_min = h2s_predictions[time_col].min().astimezone(pacific).strftime("%-I %p %-m/%-d")
        t_max = h2s_predictions[time_col].max().astimezone(pacific).strftime("%-I %p %-m/%-d")
        time_range = f"{t_min} → {t_max} PT"
    else:
        time_range = "unknown"

    # Peak risk info
    max_confidence = float(h2s_alerts['confidence'].max()) if not h2s_alerts.empty and 'confidence' in h2s_alerts.columns else 0
    max_risk = float(h2s_alerts['h2s_risk'].max()) if not h2s_alerts.empty and 'h2s_risk' in h2s_alerts.columns else 0

    all_green = h2s_alerts.empty
    header_text = "H2S Forecast — All Green ✓" if all_green else "H2S Alert — Elevated Levels Forecast"

    # Compose Slack message using Block Kit
    blocks = [
        {
            "type": "header",
            "text": {"type": "plain_text", "text": header_text},
        },
        {
            "type": "section",
            "fields": [
                {"type": "mrkdwn", "text": f"*Orange (>30 ppb):*\n{orange_count} hours"},
                {"type": "mrkdwn", "text": f"*Yellow (5-30 ppb):*\n{yellow_count} hours"},
                {"type": "mrkdwn", "text": f"*Total alerts:*\n{total}"},
                {"type": "mrkdwn", "text": f"*Peak risk score:*\n{max_risk:.2f}"},
            ],
        },
        {
            "type": "context",
            "elements": [
                {"type": "mrkdwn", "text": f"Forecast window: {time_range} | Max confidence: {max_confidence:.0%}"},
                {"type": "mrkdwn", "text": f"<{context.resources.s3.publicUrl(f'{VISUALIZATIONS_PATH}/{context.partition_key}/prediction_timeline.png', bucket=context.resources.s3.S3_BUCKET)}|View Dashboard>"},
            ],
        },
    ]

    slack = context.resources.slack
    slack.get_client().chat_postMessage(
        channel=slack.channel,
        text=f"H2S Alert: {orange_count} orange, {yellow_count} yellow hours forecast",
        blocks=blocks,
    )

    context.log.info(f"Slack alert sent: {orange_count} orange, {yellow_count} yellow")
    context.add_output_metadata({
        "orange_count": orange_count,
        "yellow_count": yellow_count,
        "total_alerts": total,
    })


@dg.asset(
    key_prefix="h2s",
    group_name="h2s_prediction",
    required_resource_keys={"s3"},
    kinds={"ml", "xgboost"},
    description="Run predictions for each deployed model variant; skips missing variants",
    partitions_def=forecast_daily_partitions,
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
    partitions_def=forecast_daily_partitions,
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
# ==============================================================================
# Asset Group: Visualization & Export
# ==============================================================================

@dg.asset(
    key_prefix="h2s",
    group_name="h2s_visualization",
    required_resource_keys={"s3"},
    description="Feature importance visualization stored to S3",
    partitions_def=forecast_daily_partitions,
    auto_materialize_policy=dg.AutoMaterializePolicy.eager(),
    ins={
        "h2s_model_artifacts": dg.AssetIn(
            key=_KEY("h2s_model_artifacts"),
            partition_mapping=dg.AllPartitionMapping()
        ),
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
    timestamp = context.partition_key
    timestamped_path = f"{VISUALIZATIONS_PATH}/{timestamp}/feature_importance.png"
    s3_resource.putFile(plot_bytes.read(), timestamped_path, bucket=s3_resource.S3_BUCKET, content_type='image/png')

    # Upload to latest path
    plot_bytes.seek(0)
    latest_path = f"latest/{LATEST_FORECAST}/visualizations/models/feature_importance.png"
    s3_resource.putFile(plot_bytes.read(), latest_path, bucket=s3_resource.S3_BUCKET, content_type='image/png')

    context.log.info(f"✓ Uploaded to S3: {timestamped_path}")
    context.log.info(f"✓ Uploaded to S3: {latest_path}")


@dg.asset(
    key_prefix="h2s",
    group_name="h2s_visualization",
    required_resource_keys={"s3"},
    description="Confusion matrix comparing predictions vs actuals (requires actual H2S data from S3)",
    partitions_def=forecast_daily_partitions,
    auto_materialize_policy=dg.AutoMaterializePolicy.eager(),
    ins={
        "h2s_predictions": dg.AssetIn(key=_KEY("h2s_predictions")),
        "h2s_model_artifacts": dg.AssetIn(
            key=_KEY("h2s_model_artifacts"),
            partition_mapping=dg.AllPartitionMapping()
        ),
    },
)
def confusion_matrix_viz(
    context: dg.AssetExecutionContext,
    h2s_predictions: pd.DataFrame,
    h2s_model_artifacts,
) -> None:
    """Generate and upload confusion matrix plot to S3.

    Loads actual H2S measurements from S3 observation data.
    Skips if H2S measurements are not available in the data or partition is today/future.
    """
    from datetime import date
    from h2s.predictor.visualizations import generate_confusion_matrix_with_metrics

    partition_date = datetime.strptime(context.partition_key, "%Y-%m-%d").date()
    if partition_date >= date.today():
        context.log.info(
            f"Skipping confusion matrix for partition {context.partition_key} — "
            f"no actuals available yet for current/future forecasts"
        )
        context.add_output_metadata({
            "status": "skipped",
            "reason": "Partition is today or future — no actuals available yet",
        })
        return

    s3_resource = context.resources.s3
    public_bucket = os.environ.get('PUBLIC_BUCKET', s3_resource.S3_BUCKET)

    # Load observation data from S3
    context.log.info(f"Loading observation data from S3: {OBS_DATA_PATH}")
    try:
        obs_url = s3_resource.publicUrl(path=OBS_DATA_PATH, bucket=public_bucket)
        actuals_df = pd.read_parquet(obs_url)
        context.log.info(f"✓ Loaded {len(actuals_df)} rows from S3 (bucket: {public_bucket})")
    except Exception as e:
        context.log.warning(f"⚠ Could not load observation data from S3: {e}")
        context.add_output_metadata({"status": "skipped", "reason": f"S3 load failed: {e}"})
        return

    # Check if H2S measurements are present
    h2s_cols = [col for col in actuals_df.columns if col.upper() == 'H2S' or 'h2s' in col.lower()]
    if not h2s_cols:
        context.log.warning("⚠ No H2S column found in observation data - skipping confusion matrix")
        context.add_output_metadata({
            "status": "skipped",
            "reason": "No H2S measurements in observation data",
            "available_columns": list(actuals_df.columns)
        })
        return

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
    timestamp = context.partition_key
    timestamped_path = f"{VISUALIZATIONS_PATH}/{timestamp}/confusion_matrix.png"
    s3_resource.putFile(plot_bytes.read(), timestamped_path, bucket=s3_resource.S3_BUCKET, content_type='image/png')

    # Upload to latest path
    plot_bytes.seek(0)
    latest_path = f"latest/{LATEST_FORECAST}/visualizations/models/confusion_matrix.png"
    s3_resource.putFile(plot_bytes.read(), latest_path, bucket=s3_resource.S3_BUCKET, content_type='image/png')

    context.log.info(f"✓ Uploaded to S3: {timestamped_path}")
    context.log.info(f"✓ Uploaded to S3: {latest_path}")


@dg.asset(
    key_prefix="h2s",
    group_name="h2s_visualization",
    required_resource_keys={"s3"},
    description="Model performance comparison plot (requires actual H2S data from S3)",
    partitions_def=forecast_daily_partitions,
    auto_materialize_policy=dg.AutoMaterializePolicy.eager(),
    ins={
        "h2s_predictions": dg.AssetIn(key=_KEY("h2s_predictions")),
        "h2s_model_artifacts": dg.AssetIn(
            key=_KEY("h2s_model_artifacts"),
            partition_mapping=dg.AllPartitionMapping()
        ),
    },
)
def model_comparison_viz(
    context: dg.AssetExecutionContext,
    h2s_predictions: pd.DataFrame,
    h2s_model_artifacts,
) -> None:
    """Generate and upload model comparison plot to S3.

    Shows balanced accuracy, recall, precision, and confusion matrix.
    Requires actual H2S measurements to compare against predictions.
    Skips if partition is today/future (no actuals available).
    """
    from datetime import date
    from h2s.predictor.visualizations import generate_model_comparison

    partition_date = datetime.strptime(context.partition_key, "%Y-%m-%d").date()
    if partition_date >= date.today():
        context.log.info(
            f"Skipping model comparison for partition {context.partition_key} — "
            f"no actuals available yet for current/future forecasts"
        )
        context.add_output_metadata({
            "status": "skipped",
            "reason": "Partition is today or future — no actuals available yet",
        })
        return

    s3_resource = context.resources.s3
    public_bucket = os.environ.get('PUBLIC_BUCKET', s3_resource.S3_BUCKET)

    # Load observation data from S3
    context.log.info(f"Loading observation data from S3: {OBS_DATA_PATH}")
    try:
        obs_url = s3_resource.publicUrl(path=OBS_DATA_PATH, bucket=public_bucket)
        actuals_df = pd.read_parquet(obs_url)
        context.log.info(f"✓ Loaded {len(actuals_df)} rows from S3 (bucket: {public_bucket})")
    except Exception as e:
        context.log.warning(f"⚠ Could not load observation data from S3: {e}")
        context.add_output_metadata({"status": "skipped", "reason": f"S3 load failed: {e}"})
        return

    # Prepare predictions DataFrame with time column
    predictions_df = h2s_predictions.copy()
    if 'time' not in predictions_df.columns:
        if 'date' in predictions_df.columns:
            predictions_df['time'] = predictions_df['date']
        else:
            context.log.error("❌ Predictions DataFrame missing both 'time' and 'date' columns")
            return
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
    timestamp = context.partition_key
    timestamped_path = f"{VISUALIZATIONS_PATH}/{timestamp}/model_comparison.png"
    s3_resource.putFile(plot_bytes.read(), timestamped_path, bucket=s3_resource.S3_BUCKET, content_type='image/png')

    # Upload to latest path
    plot_bytes.seek(0)
    latest_path = f"latest/{LATEST_FORECAST}/visualizations/models/model_comparison.png"
    s3_resource.putFile(plot_bytes.read(), latest_path, bucket=s3_resource.S3_BUCKET, content_type='image/png')

    context.log.info(f"✓ Uploaded to S3: {timestamped_path}")
    context.log.info(f"✓ Uploaded to S3: {latest_path}")


@dg.asset(
    key_prefix="h2s",
    group_name="h2s_visualization",
    required_resource_keys={"s3"},
    description="Prediction timeline plot showing H2S predictions with environmental variables",
    partitions_def=forecast_daily_partitions,
    auto_materialize_policy=dg.AutoMaterializePolicy.eager(),
    ins={
        "h2s_predictions": dg.AssetIn(key=_KEY("h2s_predictions")),
    },
)
def prediction_timeline_viz(
    context: dg.AssetExecutionContext,
    h2s_predictions: pd.DataFrame,
) -> None:
    """Generate and upload prediction timeline plot to S3.

    Shows predictions over time with environmental variables.
    Includes actual H2S values if present in observation data from S3.
    """
    from h2s.predictor.visualizations import generate_prediction_timeline

    s3_resource = context.resources.s3
    public_bucket = os.environ.get('PUBLIC_BUCKET', s3_resource.S3_BUCKET)

    # Load observation data from S3 (optional — viz works without it)
    obs_data = None
    try:
        obs_url = s3_resource.publicUrl(path=OBS_DATA_PATH, bucket=public_bucket)
        obs_data = pd.read_parquet(obs_url)
        context.log.info(f"✓ Loaded {len(obs_data)} rows from observation data (bucket: {public_bucket})")
        h2s_cols = [col for col in obs_data.columns if col.upper() == 'H2S' or 'h2s' in col.lower()]
        if h2s_cols:
            context.log.info("Generating prediction timeline (with H2S actuals)...")
        else:
            context.log.info("Generating prediction timeline (predictions + environmental variables)...")
    except Exception as e:
        context.log.info(f"No observation data available ({e}) — timeline will show predictions only")

    # Prepare predictions DataFrame with time column
    predictions_df = h2s_predictions.copy()
    if 'date' in predictions_df.columns and 'time' not in predictions_df.columns:
        predictions_df['time'] = predictions_df['date']

    plot_bytes = generate_prediction_timeline(predictions_df, obs_data)

    # Upload to timestamped path
    timestamp = context.partition_key
    timestamped_path = f"{VISUALIZATIONS_PATH}/{timestamp}/prediction_timeline.png"
    s3_resource.putFile(plot_bytes.read(), timestamped_path, bucket=s3_resource.S3_BUCKET, content_type='image/png')

    # Upload to latest path
    plot_bytes.seek(0)
    latest_path = f"latest/{LATEST_FORECAST}/visualizations/models/prediction_timeline.png"
    s3_resource.putFile(plot_bytes.read(), latest_path, bucket=s3_resource.S3_BUCKET, content_type='image/png')

    context.log.info(f"✓ Uploaded to S3: {timestamped_path}")
    context.log.info(f"✓ Uploaded to S3: {latest_path}")


@dg.asset(
    key_prefix="h2s",
    group_name="h2s_visualization",
    name="cross_correlation_viz",
    required_resource_keys={"s3"},
    description="Time-lagged cross-correlation between actual H2S and environmental drivers",
    partitions_def=forecast_daily_partitions,
    auto_materialize_policy=dg.AutoMaterializePolicy.eager(),
)
def cross_correlation_viz(
    context: dg.AssetExecutionContext,
) -> None:
    """Generate and upload cross-correlation plot to S3.

    Loads actual H2S measurements from S3 observation data.
    Computes corr(H2S(t), feature(t - lag)) for each environmental driver
    at lags -24 h … +24 h, revealing which features *precede* H2S events.
    Skipped gracefully if no H2S measurements are present.
    """
    from h2s.predictor.visualizations import generate_cross_correlation_viz

    s3_resource = context.resources.s3
    public_bucket = os.environ.get('PUBLIC_BUCKET', s3_resource.S3_BUCKET)

    # Load observation data from S3
    context.log.info(f"Loading observation data from S3: {OBS_DATA_PATH}")
    try:
        obs_url = s3_resource.publicUrl(path=OBS_DATA_PATH, bucket=public_bucket)
        obs_data = pd.read_parquet(obs_url)
        context.log.info(f"✓ Loaded {len(obs_data)} rows from S3 (bucket: {public_bucket})")
    except Exception as e:
        context.log.warning(f"⚠ Could not load observation data from S3: {e}")
        context.add_output_metadata({"status": "skipped", "reason": f"S3 load failed: {e}"})
        return

    h2s_cols = [col for col in obs_data.columns
                if col.upper() == "H2S" or "h2s" in col.lower()]

    if not h2s_cols:
        context.log.warning("⚠ No H2S measurements in observation data — skipping cross-correlation")
        context.add_output_metadata({"status": "skipped", "reason": "No H2S column"})
        return

    h2s_col = "H2S" if "H2S" in obs_data.columns else h2s_cols[0]
    n_valid = int(obs_data[h2s_col].notna().sum())

    if n_valid < 48:
        context.log.warning(f"⚠ Only {n_valid} H2S measurements — need ≥48 for meaningful cross-correlation, skipping")
        context.add_output_metadata({"status": "skipped", "reason": f"Insufficient H2S rows: {n_valid}"})
        return

    # Filter to one station to avoid duplicate timestamps across stations
    if 'site_name' in obs_data.columns:
        best_station = obs_data.groupby('site_name')[h2s_col].count().idxmax()
        obs_data = obs_data[obs_data['site_name'] == best_station].copy().reset_index(drop=True)
        context.log.info(f"Using station '{best_station}' for cross-correlation ({int(obs_data[h2s_col].notna().sum())} H2S rows)")
    else:
        obs_data = obs_data.reset_index(drop=True)

    context.log.info(f"Computing cross-correlation over {n_valid} H2S measurements...")

    plot_bytes = generate_cross_correlation_viz(obs_data, h2s_col=h2s_col)

    timestamp = context.partition_key
    timestamped_path = f"{VISUALIZATIONS_PATH}/{timestamp}/cross_correlation.png"
    s3_resource.putFile(plot_bytes.read(), timestamped_path, bucket=s3_resource.S3_BUCKET, content_type="image/png")

    plot_bytes.seek(0)
    latest_path = f"latest/{LATEST_FORECAST}/visualizations/models/cross_correlation.png"
    s3_resource.putFile(plot_bytes.read(), latest_path, bucket=s3_resource.S3_BUCKET, content_type="image/png")

    context.log.info(f"✓ Uploaded to S3: {timestamped_path}")
    context.log.info(f"✓ Uploaded to S3: {latest_path}")
    context.add_output_metadata({"status": "ok", "h2s_rows": n_valid, "s3_path": timestamped_path})


@dg.asset(
    key_prefix="h2s",
    group_name="h2s_export",
    required_resource_keys={"s3"},
    kinds={"s3", "export"},
    description="Predictions exported to S3 as CSV and JSON",
    partitions_def=forecast_daily_partitions,
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
    S3 Path: tijuana/forecast/hourly/model=.../year=.../month=.../day=.../hour=.../
    """
    s3_resource = context.resources.s3

    context.log.info("Using store_assets utility for export...")

    # Parse partition_key for Hive paths
    partition_dt = datetime.strptime(context.partition_key, "%Y-%m-%d")

    # Extract hour from schedule tag (default to 0 for backfills)
    scheduled_time = context.run.tags.get("dagster/schedule_execution_time")
    if scheduled_time:
        hour = datetime.fromisoformat(scheduled_time.replace('Z', '+00:00')).hour
    else:
        hour = 0

    hive_path = (
        f"{HOURLY_PREDICTIONS_PATH}"
        f"/model=nestor_xgboost"
        f"/year={partition_dt.strftime('%Y')}"
        f"/month={partition_dt.strftime('%m')}"
        f"/day={partition_dt.strftime('%d')}"
        f"/hour={hour:02d}"
    )
    timestamped_path = hive_path

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
        latestdatasetpath=f"{LATEST_FORECAST}/predictions",
        enable_latest_path=True,
        formats=['csv', 'json', 'parquet']
    )

    context.log.info(f"✓ Exported predictions to {timestamped_path}")
    context.log.info(f"✓ Latest path: latest/{LATEST_FORECAST}/predictions")

    # === PER-VARIANT PREDICTIONS ===
    for variant, variant_df in h2s_variant_predictions.items():
        variant_csv = variant_df.to_csv(index=False)
        s3_resource.putFile_text(
            variant_csv,
            path=f"latest/{LATEST_FORECAST}/predictions/h2s_predictions_{variant}.csv",
            bucket=s3_resource.S3_BUCKET,
            content_type='text/csv',
        )
        context.log.info(f"✓ Exported variant predictions: h2s_predictions_{variant}.csv")

    # === ENSEMBLE PREDICTIONS ===
    ensemble_csv = h2s_ensemble_predictions.to_csv(index=False)
    s3_resource.putFile_text(
        ensemble_csv,
        path=f"latest/{LATEST_FORECAST}/predictions/h2s_predictions_ensemble.csv",
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
    description="Daily report comparing previous day's 6-hourly predictions to actual H2S measurements with metrics",
    partitions_def=forecast_daily_partitions,
)
def daily_validation_report(context: dg.AssetExecutionContext) -> None:
    """Compare yesterday's 6-hourly predictions against actual H2S measurements.

    Loads predictions from each of the 4 hourly runs (_00, _06, _12, _18),
    combines them, merges with actual observations, calculates performance metrics,
    and generates validation plots and metrics.json uploaded to S3.
    Skips gracefully if predictions or actuals are unavailable.
    """
    import json
    from h2s.predictor.visualizations import (
        generate_cell_comparison_html,
        generate_cell_comparison_png,
        generate_confusion_matrix_with_metrics,
        generate_h2s_line_chart,
        generate_model_comparison,
        generate_prediction_timeline,
    )
    from h2s.training.validation import calculate_metrics, calculate_false_alarm_rate

    s3_resource = context.resources.s3
    public_bucket = os.environ.get('PUBLIC_BUCKET', s3_resource.S3_BUCKET)
    validation_date = context.partition_key  # e.g., "2026-04-01"
    yesterday_dt = datetime.strptime(validation_date, "%Y-%m-%d")
    yesterday = validation_date
    y, m, d = yesterday_dt.strftime("%Y"), yesterday_dt.strftime("%m"), yesterday_dt.strftime("%d")

    # Load yesterday's 6-hourly predictions
    prediction_dfs = []
    for hour in ["00", "06", "12", "18"]:
        s3_path = f"{HOURLY_PREDICTIONS_PATH}/model=nestor_xgboost/year={y}/month={m}/day={d}/hour={hour}/h2s_predictions.csv"
        try:
            csv_url = s3_resource.publicUrl(path=s3_path)
            df = pd.read_csv(csv_url)
            df["run_hour"] = hour
            prediction_dfs.append(df)
            context.log.info(f"✓ Loaded predictions for {yesterday}_{hour}: {len(df)} rows")
        except Exception as e:
            context.log.warning(f"No predictions found for {yesterday}_{hour}: {e}")

    if not prediction_dfs:
        raise dg.Failure(
            f"No predictions found for {yesterday} in "
            f"{HOURLY_PREDICTIONS_PATH}/model=nestor_xgboost/year={y}/month={m}/day={d}/hour={{00,06,12,18}}/. "
            f"Is forecast_prediction_schedule running?"
        )

    predictions_df = pd.concat(prediction_dfs, ignore_index=True)
    context.log.info(f"Combined {len(predictions_df)} predictions across {len(prediction_dfs)} runs")

    # Ensure time column exists and is timezone-aware UTC
    if 'time' not in predictions_df.columns and 'date' in predictions_df.columns:
        predictions_df['time'] = pd.to_datetime(predictions_df['date'])

    predictions_df['time'] = pd.to_datetime(predictions_df['time'])
    # Convert to UTC (handles both naive and timezone-aware datetimes)
    if predictions_df['time'].dt.tz is None:
        predictions_df['time'] = predictions_df['time'].dt.tz_localize('UTC')
    else:
        predictions_df['time'] = predictions_df['time'].dt.tz_convert('UTC')
    # Round to nearest hour for consistent merging with actuals
    predictions_df['time'] = predictions_df['time'].dt.round('h')

    # Load actual H2S measurements from OBS_DATA_PATH (FAIL if missing)
    context.log.info(f"Loading observation data from {OBS_DATA_PATH} (bucket: {public_bucket})")
    parquet_url = s3_resource.publicUrl(path=OBS_DATA_PATH, bucket=public_bucket)
    actuals_df = pd.read_parquet(parquet_url)
    context.log.info(f"✓ Loaded {len(actuals_df)} total observation rows from S3")

    # Filter to NESTOR-BES (handle both name variants)
    actuals_df = actuals_df[
        actuals_df['site_name'].isin(['NESTOR__BES', 'NESTOR - BES'])
    ].copy()

    if 'time' not in actuals_df.columns:
        raise ValueError(f"Missing 'time' column in observation data from {OBS_DATA_PATH}")

    # Convert to UTC (handles both naive and timezone-aware datetimes)
    actuals_df['time'] = pd.to_datetime(actuals_df['time'])
    if actuals_df['time'].dt.tz is None:
        # Assume Pacific time if naive, convert to UTC
        actuals_df['time'] = actuals_df['time'].dt.tz_localize('America/Los_Angeles').dt.tz_convert('UTC')
    else:
        # Already timezone-aware, just convert to UTC
        actuals_df['time'] = actuals_df['time'].dt.tz_convert('UTC')
    # Round to nearest hour for consistent merging with predictions
    actuals_df['time'] = actuals_df['time'].dt.round('h')
    # Filter to yesterday's date range
    actuals_df = actuals_df[actuals_df['time'].dt.strftime("%Y-%m-%d") == yesterday].copy()

    context.log.info(f"✓ Loaded {len(actuals_df)} actual H2S measurements for NESTOR-BES on {yesterday}")

    if len(actuals_df) == 0:
        raise dg.Failure(
            f"No observation data found for NESTOR-BES on {yesterday}. "
            f"Sensors may be offline or {OBS_DATA_PATH} may not cover this date."
        )

    validation_base = f"{VALIDATION_PATH}/{yesterday}"

    # Calculate metrics and generate plots (FAIL if H2S data missing)
    h2s_cols = [col for col in actuals_df.columns if col.upper() == 'H2S' or 'h2s' in col.lower()]
    if not h2s_cols:
        raise ValueError(f"No H2S column found in observation data (columns: {actuals_df.columns.tolist()})")

    if 'H2S' not in actuals_df.columns:
        actuals_df['H2S'] = actuals_df[h2s_cols[0]]

    # Deduplicate predictions per hour (keep latest run's prediction for overlapping forecast hours)
    predictions_df = predictions_df.sort_values('run_hour', ascending=False).drop_duplicates(subset=['time'], keep='first')

    # Diagnostic logging for time alignment
    context.log.info(f"  Predictions time range: {predictions_df['time'].min()} to {predictions_df['time'].max()}")
    context.log.info(f"  Actuals time range: {actuals_df['time'].min()} to {actuals_df['time'].max()}")

    # Merge predictions with actuals on time
    merged = predictions_df.merge(actuals_df[['time', 'H2S']], on='time', how='inner')
    context.log.info(f"Merged {len(merged)} predictions with actuals (match rate: {len(merged)/len(predictions_df):.1%})")

    if len(merged) == 0:
        raise dg.Failure(
            f"No predictions matched with actuals for {yesterday}. "
            f"Predictions: {predictions_df['time'].min()} to {predictions_df['time'].max()}, "
            f"Actuals: {actuals_df['time'].min()} to {actuals_df['time'].max()}"
        )

    # Categorize actual H2S values
    def categorize_h2s(value):
        if pd.isna(value):
            return None
        if value < 5:
            return 'green'
        if value < 30:
            return 'yellow'
        return 'orange'

    merged['actual_category'] = merged['H2S'].apply(categorize_h2s)
    merged = merged[merged['actual_category'].notna()].copy()

    if len(merged) == 0:
        raise dg.Failure(
            f"No valid H2S measurements for {yesterday} after filtering NaNs. "
            f"Sensors may be offline for maintenance."
        )

    # Convert string categories to integers (for calculate_metrics compatibility)
    y_true_int = merged['actual_category'].map(H2S_CLASS_TO_INT)
    y_pred_int = merged['predicted_category'].map(H2S_CLASS_TO_INT)

    # Calculate performance metrics (FAIL on error)
    metrics_dict = calculate_metrics(
        y_true=y_true_int,
        y_pred=y_pred_int,
        class_names=H2S_CLASS_NAMES,
    )

    # Calculate false alarm rate (orange predicted when actually green)
    far = calculate_false_alarm_rate(
        y_true=(merged['actual_category'] == 'orange').astype(int),
        y_pred=(merged['predicted_category'] == 'orange').astype(int),
        positive_class=1
    )

    # Build v2 metrics output structure (sites dict for accuracy_reporting compat)
    site_metrics = {
        "n_predictions": len(predictions_df),
        "n_matched_observations": len(merged),
        "match_rate": float(len(merged) / len(predictions_df)),
        "balanced_accuracy": float(metrics_dict['balanced_accuracy']),
        "false_alarm_rate": float(far),
        "confusion_matrix": metrics_dict['confusion_matrix'],
        "class_metrics": {
            cls: {
                "precision": float(metrics_dict.get(f'precision_{cls}', 0.0)),
                "recall": float(metrics_dict.get(f'recall_{cls}', 0.0)),
                "f1_score": float(metrics_dict.get(f'f1_{cls}', 0.0))
            }
            for cls in ['green', 'yellow', 'orange']
        },
        "regression_metrics": None,
    }
    metrics_output = {
        "schema_version": 2,
        "date": yesterday,
        "pipeline": "hourly",
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "sites": {
            "NESTOR__BES": site_metrics,
        },
        "horizon_metrics": None,
    }

    # Upload metrics.json to S3 (both legacy root and hourly subdir)
    metrics_json = json.dumps(metrics_output, indent=2)
    for metrics_path in [
        f"{validation_base}/metrics.json",
        f"{validation_base}/hourly/metrics.json",
    ]:
        s3_resource.putFile(
            metrics_json.encode('utf-8'),
            metrics_path,
            bucket=s3_resource.S3_BUCKET,
            content_type='application/json'
        )
    context.log.info(f"✓ Uploaded metrics.json (balanced_accuracy={site_metrics['balanced_accuracy']:.3f}, FAR={far:.3f})")

    # Generate and upload validation plots (FAIL on error)
    all_stations = [
        ("NESTOR - BES", "NESTOR__BES"),
        ("IB CIVIC CTR", "IB_CIVIC_CTR"),
        ("SAN YSIDRO", "SAN_YSIDRO"),
    ]

    for plot_name, plot_fn, kwargs in [
        ("confusion_matrix", generate_confusion_matrix_with_metrics, {"time_col": "time"}),
        ("model_comparison", generate_model_comparison, {"model_name": "XGBoost Weighted", "time_col": "time"}),
        ("prediction_timeline", generate_prediction_timeline, {}),
    ]:
        plot_bytes = plot_fn(predictions_df, actuals_df, **kwargs)
        s3_path = f"{validation_base}/{plot_name}.png"
        s3_resource.putFile(plot_bytes.read(), s3_path, bucket=s3_resource.S3_BUCKET, content_type='image/png')
        context.log.info(f"✓ Uploaded {plot_name} to {s3_path}")

    # Cell comparison PNG (all 3 stations)
    cell_png = generate_cell_comparison_png(predictions_df, actuals_df, stations=all_stations, time_col='time')
    s3_resource.putFile(cell_png.read(), f"{validation_base}/cell_comparison.png",
                        bucket=s3_resource.S3_BUCKET, content_type='image/png')
    context.log.info(f"✓ Uploaded cell_comparison.png to {validation_base}")

    # Cell comparison HTML (scrollable, all 3 stations)
    cell_html = generate_cell_comparison_html(predictions_df, actuals_df, stations=all_stations, time_col='time')
    s3_resource.putFile(cell_html.read(), f"{validation_base}/cell_comparison.html",
                        bucket=s3_resource.S3_BUCKET, content_type='text/html')
    context.log.info(f"✓ Uploaded cell_comparison.html to {validation_base}")

    # H2S line chart (all 3 stations)
    line_chart = generate_h2s_line_chart(actuals_df, stations=all_stations, time_col='time')
    s3_resource.putFile(line_chart.read(), f"{validation_base}/h2s_line_chart.png",
                        bucket=s3_resource.S3_BUCKET, content_type='image/png')
    context.log.info(f"✓ Uploaded h2s_line_chart.png to {validation_base}")

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

    # Add output metadata
    metadata = {
        "date": yesterday,
        "prediction_runs": len(prediction_dfs),
        "total_predictions": len(predictions_df),
        "actuals_available": not actuals_df.empty,
        "s3_path": validation_base,
    }

    if metrics_output:
        metadata.update({
            "balanced_accuracy": metrics_output['balanced_accuracy'],
            "false_alarm_rate": metrics_output['false_alarm_rate'],
            "n_matched": metrics_output['n_matched'],
            "match_rate": metrics_output['match_rate'],
        })

    context.add_output_metadata(metadata)


@dg.asset(
    key_prefix="h2s",
    group_name="h2s_validation",
    required_resource_keys={"s3"},
    kinds={"python", "s3"},
    description="Generate 30-day performance metrics dashboard from daily metrics.json files",
    partitions_def=forecast_daily_partitions,
)
def monthly_performance_viz(context: dg.AssetExecutionContext) -> None:
    """Generate monthly performance dashboard from daily metrics.

    Loads last 30 days of metrics.json files and creates:
    - Aggregate confusion matrix (30-day sum, normalized %)
    - Balanced accuracy trend
    - Per-class recall trends
    - False alarm rate trend
    """
    import json
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import numpy as np
    from io import BytesIO

    s3_resource = context.resources.s3

    # Load last 30 days of metrics (FAIL if insufficient data)
    partition_dt = datetime.strptime(context.partition_key, "%Y-%m-%d").date()
    today = partition_dt
    metrics_list = []
    missing_dates = []

    for i in range(30):
        date = today - timedelta(days=i+1)
        date_str = date.strftime("%Y-%m-%d")
        metrics_path = f"{VALIDATION_PATH}/{date_str}/{PIPELINE_DAILY_STATION}/metrics.json"

        try:
            metrics_bytes = s3_resource.getFile(metrics_path, bucket=s3_resource.S3_BUCKET)
            metrics_data = json.loads(metrics_bytes.decode('utf-8'))
            metrics_data['date_obj'] = date
            metrics_list.append(metrics_data)
            context.log.debug(f"✓ Loaded metrics for {date_str}")
        except Exception as e:
            missing_dates.append(date_str)
            context.log.debug(f"Missing metrics for {date_str}: {e}")

    if not metrics_list:
        raise ValueError(f"No metrics found for last 30 days — cannot generate performance dashboard. Missing all dates.")

    context.log.info(f"Loaded metrics for {len(metrics_list)} days (missing {len(missing_dates)} days)")

    # Sort by date (oldest first for trends)
    metrics_list.sort(key=lambda x: x['date_obj'])

    # Extract data for plots
    dates = [m['date_obj'] for m in metrics_list]
    balanced_acc = [m['balanced_accuracy'] for m in metrics_list]

    # Per-class recall
    orange_recall = [m['class_metrics']['orange']['recall'] for m in metrics_list]
    yellow_recall = [m['class_metrics']['yellow']['recall'] for m in metrics_list]
    green_recall = [m['class_metrics']['green']['recall'] for m in metrics_list]

    false_alarm_rate = [m.get('false_alarm_rate', 0) for m in metrics_list]

    # Aggregate confusion matrix (sum counts)
    # Handle days where not all 3 classes were present (e.g., 2x2 matrix)
    cm_sum = np.zeros((3, 3))
    for m in metrics_list:
        cm = np.array(m['confusion_matrix'])
        if cm.shape == (3, 3):
            cm_sum += cm
        else:
            context.log.warning(
                f"Skipping {m.get('date', '?')} confusion matrix "
                f"(shape {cm.shape}, expected 3x3)"
            )

    # Normalize to percentages
    cm_norm = cm_sum / cm_sum.sum(axis=1, keepdims=True) * 100

    # Create 2×2 grid figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'{len(metrics_list)}-Day Performance Dashboard (NESTOR-BES)',
                 fontsize=16, fontweight='bold')

    # Panel 1: Aggregate Confusion Matrix
    ax1 = axes[0, 0]
    im = ax1.imshow(cm_norm, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=100)
    ax1.set_xticks(range(3))
    ax1.set_yticks(range(3))
    ax1.set_xticklabels(['Green', 'Yellow', 'Orange'])
    ax1.set_yticklabels(['Green', 'Yellow', 'Orange'])
    ax1.set_xlabel('Predicted Category')
    ax1.set_ylabel('Actual Category')
    ax1.set_title(f'Confusion Matrix ({len(metrics_list)}-day aggregate, normalized)')

    # Annotate cells
    for i in range(3):
        for j in range(3):
            text = ax1.text(j, i, f'{cm_norm[i, j]:.1f}%\n({int(cm_sum[i, j])})',
                           ha='center', va='center', fontsize=10,
                           color='white' if cm_norm[i, j] > 50 else 'black')

    plt.colorbar(im, ax=ax1, label='Percentage (%)')

    # Panel 2: Balanced Accuracy Trend
    ax2 = axes[0, 1]
    ax2.plot(dates, balanced_acc, marker='o', linewidth=2, markersize=4)
    ax2.axhline(0.61, color='red', linestyle='--', linewidth=1, label='Target (61%)')
    ax2.set_ylabel('Balanced Accuracy')
    ax2.set_title('Daily Balanced Accuracy')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    ax2.tick_params(axis='x', rotation=45)

    # Panel 3: Per-Class Recall
    ax3 = axes[1, 0]
    ax3.plot(dates, orange_recall, marker='o', label='Orange', color='#e74c3c', linewidth=2, markersize=3)
    ax3.plot(dates, yellow_recall, marker='s', label='Yellow', color='#f39c12', linewidth=2, markersize=3)
    ax3.plot(dates, green_recall, marker='^', label='Green', color='#27ae60', linewidth=2, markersize=3)
    ax3.set_ylabel('Recall (Sensitivity)')
    ax3.set_title('Daily Recall by Category')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    ax3.tick_params(axis='x', rotation=45)

    # Panel 4: False Alarm Rate
    ax4 = axes[1, 1]
    ax4.plot(dates, false_alarm_rate, marker='o', color='#e67e22', linewidth=2, markersize=4)
    ax4.axhline(0.054, color='red', linestyle='--', linewidth=1, label='Target (5.4%)')
    ax4.set_ylabel('False Alarm Rate')
    ax4.set_title('Daily False Alarm Rate (Orange predicted | Actually Green)')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    ax4.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    ax4.tick_params(axis='x', rotation=45)

    plt.tight_layout()

    # Save to BytesIO
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    plt.close()

    # Upload to S3 (timestamped + latest)
    month_str = today.strftime("%Y-%m")
    timestamped_path = f"{VALIDATION_PATH}/monthly/{month_str}/performance_dashboard.png"
    latest_path = f"latest/{LATEST_FORECAST}/visualizations/performance_dashboard.png"

    s3_resource.putFile(buf.read(), timestamped_path, bucket=s3_resource.S3_BUCKET,
                       content_type='image/png')

    buf.seek(0)
    s3_resource.putFile(buf.read(), latest_path, bucket=s3_resource.S3_BUCKET,
                       content_type='image/png')

    context.log.info(f"✓ Performance dashboard uploaded to {timestamped_path} and {latest_path}")

    # Add metadata
    context.add_output_metadata({
        "days_included": len(metrics_list),
        "date_range": f"{dates[0]} to {dates[-1]}",
        "avg_balanced_accuracy": float(np.mean(balanced_acc)),
        "avg_false_alarm_rate": float(np.mean(false_alarm_rate)),
        "timestamped_path": timestamped_path,
        "latest_path": latest_path,
    })
