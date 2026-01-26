"""H2S Prediction Pipeline - S3-integrated production classification model.

This pipeline loads a pre-trained XGBoost classification model from S3,
processes environmental data, generates H2S predictions (green/yellow/orange),
and exports results to S3 with visualizations.
"""

from datetime import datetime
from io import BytesIO

import dagster as dg
import pandas as pd

from h2s.utils import store_assets

STORE_ASSETS_AVAILABLE = True

# Define S3 paths following tijuana/forecast conventions
MODEL_PATH = 'tijuana/forecast/models'
OUTPUT_PATH = 'tijuana/forecast/output'
LATEST = 'tijuana/forecast_data'


# ==============================================================================
# Asset Group: Model Management
# ==============================================================================

@dg.asset(
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

    predictor = H2SPredictor.from_s3(
        s3_resource,
        f"{MODEL_PATH}/nestor_xgboost_weighted_model.json",
        f"{MODEL_PATH}/nestor_preprocessing_info.json"
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
    group_name="h2s_prediction",
    required_resource_keys={"s3"},
    kinds={"csv", "s3"},
    description="Environmental data loaded from S3 or local",
)
def raw_environmental_data(context: dg.AssetExecutionContext) -> pd.DataFrame:
    """Load environmental data from S3 (latest/) or fallback to local.

    Tries to load from:
    1. S3: latest/tijuana/weather_forecast/latest.csv (from openmeteo.py)
    2. Fallback: Local data/latest.csv for testing
    """
    s3_resource = context.resources.s3

    # Try S3 first
    try:
        stream = s3_resource.get_stream(path="latest/tijuana/weather_forecast/latest.csv")
        df = pd.read_csv(stream)
        context.log.info("✓ Loaded from S3: latest/tijuana/weather_forecast/latest.csv")
    except Exception as e:
        # Fallback to local data for testing
        context.log.warning(f"Could not load from S3: {e}")
        context.log.info("Falling back to local data/latest.csv")

        local_path = "/Users/valentin/development/dev_resilient/tj_h2s_prediction/data/latest.csv"
        df = pd.read_csv(local_path)
        context.log.info(f"✓ Loaded {len(df)} rows from local file")

    # Convert time to datetime
    df['date'] = pd.to_datetime(df['date'])

    context.log.info(f"Loaded {len(df)} rows")
    context.log.info(f"Date range: {df['date'].min()} to {df['date'].max()}")
    context.log.info(f"Columns: {len(df.columns)}")

    return df


# ==============================================================================
# Asset Group: Prediction Pipeline
# ==============================================================================

@dg.asset(
    group_name="h2s_prediction",
    kinds={"python"},
    description="Preprocessed features ready for model prediction",
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
    group_name="h2s_prediction",
    kinds={"xgboost", "ml"},
    description="H2S category predictions with probabilities",
)
def h2s_predictions(
    context: dg.AssetExecutionContext,
    h2s_model_artifacts,
    preprocessed_features: pd.DataFrame
) -> pd.DataFrame:
    """Generate H2S predictions with probabilities.

    Classifies each sample as green (<5 ppb), yellow (5-15 ppb), or orange (>=15 ppb).
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
    group_name="h2s_prediction",
    kinds={"python"},
    description="Filtered predictions showing only alerts (orange/yellow)",
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


# ==============================================================================
# Asset Group: Visualization & Export
# ==============================================================================

@dg.asset(
    group_name="h2s_visualization",
    required_resource_keys={"s3"},
    kinds={"matplotlib", "s3"},
    description="Feature importance visualization stored to S3",
)
def feature_importance_viz(
    context: dg.AssetExecutionContext,
    h2s_model_artifacts
) -> None:
    """Generate and upload feature importance plot to S3."""
    from h2s.predictor.visualizations import generate_feature_importance

    context.log.info("Generating feature importance visualization...")

    s3_resource = context.resources.s3
    plot_bytes = generate_feature_importance(
        h2s_model_artifacts.model,
        h2s_model_artifacts.prep_info
    )

    # Upload to timestamped path
    timestamp = datetime.now().strftime("%Y-%m-%d")
    timestamped_path = f"{OUTPUT_PATH}/visualizations/{timestamp}/feature_importance.png"
    s3_resource.putFile(plot_bytes, timestamped_path, bucket=s3_resource.S3_BUCKET)

    # Upload to latest path
    plot_bytes.seek(0)  # Reset BytesIO position
    latest_path = f"latest/{LATEST}/visualizations/feature_importance.png"
    s3_resource.putFile(plot_bytes, latest_path, bucket=s3_resource.S3_BUCKET)

    context.log.info(f"✓ Uploaded to S3: {timestamped_path}")
    context.log.info(f"✓ Uploaded to S3: {latest_path}")


@dg.asset(
    group_name="h2s_export",
    required_resource_keys={"s3"},
    kinds={"s3", "export"},
    description="Predictions exported to S3 as CSV and JSON",
)
def predictions_export(
    context: dg.AssetExecutionContext,
    h2s_predictions: pd.DataFrame
) -> None:
    """Export predictions to S3 with versioning.

    Stores predictions in both timestamped and latest paths.
    Uses store_assets utility if available, otherwise manual upload.
    """
    s3_resource = context.resources.s3

    if STORE_ASSETS_AVAILABLE:
        context.log.info("Using store_assets utility for export...")

        metadata = store_assets.objectMetadata(
            name="H2S Predictions - NESTOR BES",
            description="H2S category predictions with probabilities for NESTOR - BES site",
            variableMeasured=["H2S Category", "Probability Scores", "Alert Status"],
        )

        store_assets.store_dataframe_to_s3(
            df=h2s_predictions,
            path=OUTPUT_PATH,
            dataset_identifier="h2s_predictions",
            s3_resource=s3_resource,
            metadata=metadata,
            latestdatasetpath=f"latest/{LATEST}",
            enable_latest_path=True,
            formats=['csv', 'json']
        )

        context.log.info(f"✓ Exported using store_assets to {OUTPUT_PATH}")

    else:
        # Manual export if store_assets not available
        context.log.info("store_assets not available, using manual export...")

        timestamp = datetime.now().strftime("%Y-%m-%d_%H")

        # Export CSV
        csv_buffer = BytesIO()
        h2s_predictions.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)

        csv_path = f"{OUTPUT_PATH}/{timestamp}/h2s_predictions.csv"
        s3_resource.putFile(csv_buffer, csv_path, bucket=s3_resource.S3_BUCKET)

        # Export to latest
        csv_buffer.seek(0)
        latest_csv_path = f"latest/{LATEST}/h2s_predictions.csv"
        s3_resource.putFile(csv_buffer, latest_csv_path, bucket=s3_resource.S3_BUCKET)

        context.log.info(f"✓ Exported CSV to {csv_path}")
        context.log.info(f"✓ Exported CSV to {latest_csv_path}")

    context.add_output_metadata({
        "row_count": len(h2s_predictions),
        "export_timestamp": datetime.now().isoformat(),
        "s3_path": OUTPUT_PATH,
    })
