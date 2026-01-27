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
    description="Environmental data loaded from S3 or local test data",
    config_schema={
        "use_local_data": dg.Field(
            bool,
            default_value=False,
            description="Use local test data from data/ directory instead of S3 (for testing only)"
        ),
        "local_data_path": dg.Field(
            str,
            default_value="/Users/valentin/development/dev_resilient/tj_h2s_prediction/data/latest.csv",
            description="Path to local test data file (used when use_local_data=True)"
        ),
    }
)
def raw_environmental_data(context: dg.AssetExecutionContext) -> pd.DataFrame:
    """Load environmental data from S3 or local test data.

    Production Mode (use_local_data=False):
    - Loads from S3: latest/tijuana/weather_forecast/latest.csv
    - FAILS if S3 data is not available (no fallback)
    - If S3 data lacks H2S measurements, merges from local modeldata_h2s.csv

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
            local_path = "/Users/valentin/development/dev_resilient/tj_h2s_prediction/data/modeldata_h2s.csv"
            h2s_df = pd.read_csv(local_path)

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
# Asset Group: Actual Data (Optional)
# ==============================================================================

@dg.asset(
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
    group_name="h2s_visualization",
    required_resource_keys={"s3"},
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
    s3_resource.putFile(plot_bytes.read(), timestamped_path, bucket=s3_resource.S3_BUCKET, content_type='image/png')

    # Upload to latest path
    plot_bytes.seek(0)
    latest_path = f"{LATEST}/visualizations/feature_importance.png"
    s3_resource.putFile(plot_bytes.read(), latest_path, bucket=s3_resource.S3_BUCKET, content_type='image/png')

    context.log.info(f"✓ Uploaded to S3: {timestamped_path}")
    context.log.info(f"✓ Uploaded to S3: {latest_path}")


@dg.asset(
    group_name="h2s_visualization",
    required_resource_keys={"s3"},
    description="Confusion matrix comparing predictions vs actuals (requires actual H2S data in raw_environmental_data)",
)
def confusion_matrix_viz(
    context: dg.AssetExecutionContext,
    h2s_predictions: pd.DataFrame,
    raw_environmental_data: pd.DataFrame
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

    plot_bytes = generate_confusion_matrix_with_metrics(
        predictions_df,
        actuals_df,
        time_col='time'
    )

    # Upload to timestamped path
    timestamp = datetime.now().strftime("%Y-%m-%d")
    timestamped_path = f"{OUTPUT_PATH}/visualizations/{timestamp}/confusion_matrix.png"
    s3_resource.putFile(plot_bytes.read(), timestamped_path, bucket=s3_resource.S3_BUCKET, content_type='image/png')

    # Upload to latest path
    plot_bytes.seek(0)
    latest_path = f"{LATEST}/visualizations/confusion_matrix.png"
    s3_resource.putFile(plot_bytes.read(), latest_path, bucket=s3_resource.S3_BUCKET, content_type='image/png')

    context.log.info(f"✓ Uploaded to S3: {timestamped_path}")
    context.log.info(f"✓ Uploaded to S3: {latest_path}")


@dg.asset(
    group_name="h2s_visualization",
    required_resource_keys={"s3"},
    description="Model performance comparison plot (requires actual H2S data)",
)
def model_comparison_viz(
    context: dg.AssetExecutionContext,
    h2s_predictions: pd.DataFrame,
    raw_environmental_data: pd.DataFrame
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

    plot_bytes = generate_model_comparison(
        predictions_df,
        actuals_df,
        model_name="XGBoost Weighted",
        time_col='time'
    )

    # Upload to timestamped path
    timestamp = datetime.now().strftime("%Y-%m-%d")
    timestamped_path = f"{OUTPUT_PATH}/visualizations/{timestamp}/model_comparison.png"
    s3_resource.putFile(plot_bytes.read(), timestamped_path, bucket=s3_resource.S3_BUCKET, content_type='image/png')

    # Upload to latest path
    plot_bytes.seek(0)
    latest_path = f"{LATEST}/visualizations/model_comparison.png"
    s3_resource.putFile(plot_bytes.read(), latest_path, bucket=s3_resource.S3_BUCKET, content_type='image/png')

    context.log.info(f"✓ Uploaded to S3: {timestamped_path}")
    context.log.info(f"✓ Uploaded to S3: {latest_path}")


@dg.asset(
    group_name="h2s_visualization",
    required_resource_keys={"s3"},
    description="Prediction timeline plot showing H2S predictions with environmental variables",
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
    timestamped_path = f"{OUTPUT_PATH}/visualizations/{timestamp}/prediction_timeline.png"
    s3_resource.putFile(plot_bytes.read(), timestamped_path, bucket=s3_resource.S3_BUCKET, content_type='image/png')

    # Upload to latest path
    plot_bytes.seek(0)
    latest_path = f"{LATEST}/visualizations/prediction_timeline.png"
    s3_resource.putFile(plot_bytes.read(), latest_path, bucket=s3_resource.S3_BUCKET, content_type='image/png')

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
