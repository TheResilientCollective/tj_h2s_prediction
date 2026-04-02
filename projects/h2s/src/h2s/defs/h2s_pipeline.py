"""H2S Prediction Pipeline - S3-integrated production classification model.

This pipeline loads a pre-trained XGBoost classification model from S3,
processes environmental data, generates H2S predictions (green/yellow/orange),
and exports results to S3 with visualizations.
"""

import json
from datetime import datetime, timedelta
from io import BytesIO
from zoneinfo import ZoneInfo

import dagster as dg
import pandas as pd

from h2s.utils import store_assets
from h2s.constants import (
    FORECAST_DATA_PATH,
    HOURLY_PREDICTIONS_PATH,
    LATEST_FORECAST,
    MODEL_PATH,
    OBS_DATA_PATH,
    VALIDATION_PATH,
    VISUALIZATIONS_PATH,
)

STORE_ASSETS_AVAILABLE = True

_KEY = lambda name: dg.AssetKey(["h2s", name])

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
# Asset Group: Prediction Pipeline
# ==============================================================================

@dg.asset(
    key_prefix="h2s",
    group_name="h2s_prediction",
    required_resource_keys={"s3"},
    kinds={"python", "s3"},
    description="Preprocessed features ready for model prediction (loaded from pre-featurized S3 data)",
    auto_materialize_policy=dg.AutoMaterializePolicy.eager(),
    ins={
        "h2s_model_artifacts": dg.AssetIn(key=_KEY("h2s_model_artifacts")),
    },
)
def preprocessed_features(
    context: dg.AssetExecutionContext,
    h2s_model_artifacts,
) -> pd.DataFrame:
    """Load pre-featurized forecast data from S3 and apply model preprocessing.

    Loads model_forecast.parquet from S3 (already contains 43 MODEL_FEATURES).
    Applies H2SPredictor.preprocess_data() to add any missing features and
    handle H2S lags (set to 0 in forecast mode).
    """
    s3 = context.resources.s3

    context.log.info(f"Loading forecast data from S3: {FORECAST_DATA_PATH}")

    try:
        forecast_url = s3.get_presigned_url(path=FORECAST_DATA_PATH, bucket=s3.S3_BUCKET)
        df = pd.read_parquet(forecast_url)
        context.log.info(f"✓ Loaded {len(df)} rows from S3")
        context.log.info(f"  Columns: {len(df.columns)}")
    except Exception as e:
        raise RuntimeError(f"Failed to load forecast data from S3 path '{FORECAST_DATA_PATH}': {e}")

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
    group_name="h2s_alerts",
    kinds={"slack"},
    required_resource_keys={"slack"},
    description="Send YELLOW_HIGH and ORANGE alerts to Slack",
    auto_materialize_policy=dg.AutoMaterializePolicy.eager(),
    ins={
        "h2s_alerts": dg.AssetIn(key=_KEY("h2s_alerts")),
    },
)
def slack_alerts(
    context: dg.AssetExecutionContext,
    h2s_alerts: pd.DataFrame,
) -> None:
    """Post H2S alert summary to Slack when YELLOW_HIGH or ORANGE predictions exist."""
    if h2s_alerts.empty:
        context.log.info("No alerts to send")
        return

    orange_count = int((h2s_alerts['predicted_category'] == 'orange').sum())
    yellow_count = int((h2s_alerts['predicted_category'] == 'yellow').sum())
    total = len(h2s_alerts)

    # Build time range in Pacific time
    time_col = 'time' if 'time' in h2s_alerts.columns else None
    if time_col:
        pacific = ZoneInfo("America/Los_Angeles")
        t_min = h2s_alerts[time_col].min().astimezone(pacific).strftime("%-I %p %-m/%-d")
        t_max = h2s_alerts[time_col].max().astimezone(pacific).strftime("%-I %p %-m/%-d")
        time_range = f"{t_min} → {t_max} PT"
    else:
        time_range = "unknown"

    # Peak risk info
    max_confidence = float(h2s_alerts['confidence'].max()) if 'confidence' in h2s_alerts.columns else 0
    max_risk = float(h2s_alerts['h2s_risk'].max()) if 'h2s_risk' in h2s_alerts.columns else 0

    # Compose Slack message using Block Kit
    blocks = [
        {
            "type": "header",
            "text": {"type": "plain_text", "text": "H2S Alert — Elevated Levels Forecast"},
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
    latest_path = f"latest/{LATEST_FORECAST}/visualizations/models/feature_importance.png"
    s3_resource.putFile(plot_bytes.read(), latest_path, bucket=s3_resource.S3_BUCKET, content_type='image/png')

    context.log.info(f"✓ Uploaded to S3: {timestamped_path}")
    context.log.info(f"✓ Uploaded to S3: {latest_path}")


@dg.asset(
    key_prefix="h2s",
    group_name="h2s_visualization",
    required_resource_keys={"s3"},
    description="Confusion matrix comparing predictions vs actuals (requires actual H2S data from S3)",
    auto_materialize_policy=dg.AutoMaterializePolicy.eager(),
    ins={
        "h2s_predictions": dg.AssetIn(key=_KEY("h2s_predictions")),
        "h2s_model_artifacts": dg.AssetIn(key=_KEY("h2s_model_artifacts")),
    },
)
def confusion_matrix_viz(
    context: dg.AssetExecutionContext,
    h2s_predictions: pd.DataFrame,
    h2s_model_artifacts,
) -> None:
    """Generate and upload confusion matrix plot to S3.

    Loads actual H2S measurements from S3 observation data.
    Skips if H2S measurements are not available in the data.
    """
    from h2s.predictor.visualizations import generate_confusion_matrix_with_metrics

    s3_resource = context.resources.s3

    # Load observation data from S3
    context.log.info(f"Loading observation data from S3: {OBS_DATA_PATH}")
    try:
        obs_url = s3_resource.get_presigned_url(path=OBS_DATA_PATH, bucket=s3_resource.S3_BUCKET)
        actuals_df = pd.read_parquet(obs_url)
        context.log.info(f"✓ Loaded {len(actuals_df)} rows from S3")
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
    timestamp = datetime.now().strftime("%Y-%m-%d")
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
    auto_materialize_policy=dg.AutoMaterializePolicy.eager(),
    ins={
        "h2s_predictions": dg.AssetIn(key=_KEY("h2s_predictions")),
        "h2s_model_artifacts": dg.AssetIn(key=_KEY("h2s_model_artifacts")),
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
    """
    from h2s.predictor.visualizations import generate_model_comparison

    s3_resource = context.resources.s3

    # Load observation data from S3
    context.log.info(f"Loading observation data from S3: {OBS_DATA_PATH}")
    try:
        obs_url = s3_resource.get_presigned_url(path=OBS_DATA_PATH, bucket=s3_resource.S3_BUCKET)
        actuals_df = pd.read_parquet(obs_url)
        context.log.info(f"✓ Loaded {len(actuals_df)} rows from S3")
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
    timestamp = datetime.now().strftime("%Y-%m-%d")
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

    # Load observation data from S3 (optional — viz works without it)
    obs_data = None
    try:
        obs_url = s3_resource.get_presigned_url(path=OBS_DATA_PATH, bucket=s3_resource.S3_BUCKET)
        obs_data = pd.read_parquet(obs_url)
        context.log.info(f"✓ Loaded {len(obs_data)} rows from observation data")
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
    timestamp = datetime.now().strftime("%Y-%m-%d")
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

    # Load observation data from S3
    context.log.info(f"Loading observation data from S3: {OBS_DATA_PATH}")
    try:
        obs_url = s3_resource.get_presigned_url(path=OBS_DATA_PATH, bucket=s3_resource.S3_BUCKET)
        obs_data = pd.read_parquet(obs_url)
        context.log.info(f"✓ Loaded {len(obs_data)} rows from S3")
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

    timestamp = datetime.now().strftime("%Y-%m-%d")
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

    now = datetime.now()
    hive_path = (
        f"{HOURLY_PREDICTIONS_PATH}"
        f"/model=nestor_xgboost"
        f"/year={now.strftime('%Y')}"
        f"/month={now.strftime('%m')}"
        f"/day={now.strftime('%d')}"
        f"/hour={now.strftime('%H')}"
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
    yesterday_dt = datetime.now() - timedelta(days=1)
    yesterday = yesterday_dt.strftime("%Y-%m-%d")
    y, m, d = yesterday_dt.strftime("%Y"), yesterday_dt.strftime("%m"), yesterday_dt.strftime("%d")

    # Load yesterday's 6-hourly predictions
    prediction_dfs = []
    for hour in ["00", "06", "12", "18"]:
        s3_path = f"{HOURLY_PREDICTIONS_PATH}/model=nestor_xgboost/year={y}/month={m}/day={d}/hour={hour}/h2s_predictions.csv"
        try:
            csv_url = s3_resource.get_presigned_url(path=s3_path)
            df = pd.read_csv(csv_url)
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
        csv_url = s3_resource.get_presigned_url(path="tijuana/forecast/actuals/latest.csv")
        actuals_df = pd.read_csv(csv_url)
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
