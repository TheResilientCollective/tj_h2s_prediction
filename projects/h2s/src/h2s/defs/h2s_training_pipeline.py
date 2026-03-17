"""H2S Model Retraining Pipeline - Monthly Automated Training.

This pipeline implements automated monthly model retraining with the following phases:

Phase 1: Data Extraction & Preparation
    - monthly_training_data: Load historical data from S3/local
    - relabeled_training_data: Apply new H2S thresholds (5-30-30+)
    - data_quality_report: Validate data completeness and quality
    - training_validation_split: Time-based split (80/20)

Phase 2: Model Training
    - trained_model_cv: Train XGBoost with 5-fold time-series CV
    - model_training_metrics: Export CV metrics to S3
    - feature_importance_analysis: Generate feature importance visualization

Phase 3: Validation
    - validation_predictions: Test new model on held-out validation set
    - validation_report: Compare new vs current model performance
    - model_comparison_report: Generate visual comparison

Phase 4: Deployment
    - deployment_approval: Manual approval gate (BLOCKS auto-deployment)
    - production_model_deployment: Deploy new model to production S3 paths

Scheduling:
    - Monthly schedule (1st of month, 2 AM)
    - Defined in h2s_schedules.py
"""

import io
import json
import os
import tempfile
from datetime import datetime
from typing import Dict, Optional

import dagster as dg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from h2s.predictor.visualizations import generate_feature_importance
from h2s.training.relabeling import categorize_h2s, apply_categorization, get_threshold_info
from h2s.training.model_trainer import (
    train_model_with_cv, train_random_forest_with_cv,
    get_feature_importance, calculate_cv_summary
)
from h2s.training.validation import calculate_metrics, compare_models, format_metrics_report
from h2s.utils import store_assets
from h2s.constants import MODEL_PATH, TRAINING_PATH, LATEST_FORECAST_DATA


# ==============================================================================
# Partition Definition: Monthly Training Runs
# ==============================================================================

monthly_training_partitions = dg.MonthlyPartitionsDefinition(
    start_date="2025-09-01",
    end_offset=0,  # Run up to current month
)

# Model variants — add new entries here to support additional models / ensemble members
model_variant_partitions = dg.StaticPartitionsDefinition([
    "xgboost_base",    # Standard XGBoost with class weights
    "xgboost_smote",   # XGBoost with SMOTE oversampling on hazard classes
    "random_forest",   # Random Forest with balanced class weights
])

# Combined partition: every (month, variant) pair gets its own run and S3 path
model_run_partitions = dg.MultiPartitionsDefinition({
    "month": monthly_training_partitions,
    "variant": model_variant_partitions,
})


def _KEY(name: str) -> dg.AssetKey:
    """Helper: resolve an intra-file asset key under the 'h2s' prefix."""
    return dg.AssetKey(["h2s", name])


# ==============================================================================
# PHASE 1: Data Extraction & Preparation
# ==============================================================================

@dg.asset(
    key_prefix="h2s",
    group_name="h2s_training_data",
    kinds={"csv"},
    required_resource_keys={"s3"},
    description="Historical H2S training data from modeldata_h2s.csv - Partitioned by month",
    partitions_def=monthly_training_partitions,
    config_schema={
        "use_local_data": dg.Field(
            bool,
            default_value=False,
            description="Load from local file (True) or S3 (False, default)"
        ),
        "local_data_path": dg.Field(
            str,
            default_value="/Users/valentin/development/dev_resilient/tj_h2s_prediction/data/modeldata_h2s_nofill.parquet",
            description="Path to local training data file (.parquet or .csv) — only used when use_local_data=True"
        ),
        "s3_data_path": dg.Field(
            str,
            default_value="latest/tijuana/forecast_data/modeldata_h2s_nofill.parquet",
            description="S3 path to training data parquet file"
        ),
        "site_filter": dg.Field(
            str,
            default_value="NESTOR - BES",
            description="Site to filter training data (NESTOR - BES, IB CIVIC CTR, SAN YSIDRO)"
        )
    }
)
def monthly_training_data(context: dg.AssetExecutionContext) -> pd.DataFrame:
    """Extract historical training data for model retraining.

    Data Sources:
        - Local: /data/modeldata_h2s.csv (288 rows, 24 columns)
        - S3: latest/tijuana/forecast_data/modeldata_h2s.csv (future)

    Returns:
        DataFrame with columns:
            - time: Timestamp
            - site_name: Monitoring site
            - H2S: Measured H2S value (ppb)
            - h2s_measured: Boolean flag
            - 20 environmental features (temperature, wind, tide, etc.)
    """
    use_local = context.op_config["use_local_data"]
    site_filter = context.op_config["site_filter"]

    context.log.info(f"Loading training data (local={use_local}, site={site_filter})")

    # Load data
    if use_local:
        local_path = context.op_config["local_data_path"]
        context.log.info(f"Loading from local file: {local_path}")

        if not os.path.exists(local_path):
            raise FileNotFoundError(
                f"Local training data not found at {local_path}. "
                f"Ensure modeldata_h2s.csv exists in /data/ directory."
            )

        if local_path.endswith('.parquet'):
            df = pd.read_parquet(local_path)
        else:
            df = pd.read_csv(local_path)
            # Legacy CSV used 'D' as the time column
            if 'D' in df.columns and 'time' not in df.columns:
                df['time'] = pd.to_datetime(df['D'])
    else:
        s3_resource = context.resources.s3
        s3_path = context.op_config["s3_data_path"]
        context.log.info(f"Loading from S3: {s3_path}")
        try:
            from io import BytesIO
            data_bytes = s3_resource.getFile(s3_path)
            if s3_path.endswith('.parquet'):
                df = pd.read_parquet(BytesIO(data_bytes))
            else:
                import io
                df = pd.read_csv(io.StringIO(data_bytes.decode('utf-8')))
                if 'D' in df.columns and 'time' not in df.columns:
                    df['time'] = pd.to_datetime(df['D'])
        except Exception as e:
            raise RuntimeError(
                f"Failed to load training data from S3 path '{s3_path}': {e}\n"
                f"Upload training data to S3 first, or set use_local_data=True with a valid local_data_path."
            )

    context.log.info(f"Loaded {len(df)} rows")

    # Normalize time column — strip timezone so partition comparisons work uniformly
    df['time'] = pd.to_datetime(df['time']).dt.tz_localize(None)

    # Filter by site
    if site_filter:
        df = df[df['site_name'] == site_filter].copy()
        context.log.info(f"Filtered to {site_filter}: {len(df)} rows")

    # Filter by partition month - CUMULATIVE (all data up to end of partition month)
    # e.g., partition "2024-01-01" includes all data from beginning → 2024-01-31
    partition_key = context.partition_key
    if partition_key:
        # Parse partition key: "2024-01-01" -> first day of month
        partition_start = pd.to_datetime(partition_key)  # tz-naive to match df['time']

        # Create end date for the month (last day)
        if partition_start.month == 12:
            end_date = pd.Timestamp(partition_start.year + 1, 1, 1) - pd.Timedelta(days=1)
        else:
            end_date = pd.Timestamp(partition_start.year, partition_start.month + 1, 1) - pd.Timedelta(days=1)

        # Cumulative filter: all data from beginning up to end of partition month
        df = df[df['time'] <= end_date].copy()
        context.log.info(
            f"Filtered to partition {partition_key} (CUMULATIVE) "
            f"(all data up to {end_date.date()}): {len(df)} rows"
        )

    # Validate required columns
    required_cols = ['time', 'site_name', 'H2S', 'h2s_measured']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Filter to valid H2S measurements
    df = df[df['h2s_measured'] == True].copy()  # noqa: E712
    context.log.info(f"Filtered to valid H2S measurements: {len(df)} rows")

    # Log H2S value distribution
    h2s_stats = df['H2S'].describe()
    context.log.info(f"H2S distribution:\n{h2s_stats}")

    # Add metadata
    context.add_output_metadata({
        "total_rows": len(df),
        "site": site_filter,
        "date_range_start": str(df['time'].min()),
        "date_range_end": str(df['time'].max()),
        "h2s_min": float(df['H2S'].min()),
        "h2s_max": float(df['H2S'].max()),
        "h2s_mean": float(df['H2S'].mean()),
    })

    return df


@dg.asset(
    key_prefix="h2s",
    group_name="h2s_training_data",
    kinds={"python"},
    description="Apply new H2S thresholds (Yellow: 5-30 ppb, Orange: ≥30 ppb)",
    partitions_def=monthly_training_partitions,
    ins={
        "monthly_training_data": dg.AssetIn(key=_KEY("monthly_training_data")),
    },
)
def relabeled_training_data(
    context: dg.AssetExecutionContext,
    monthly_training_data: pd.DataFrame
) -> pd.DataFrame:
    """Apply new H2S category thresholds to training data.

    NEW THRESHOLDS (Client Specification, Jan 2026):
        - Green: H2S < 5 ppb
        - Yellow: 5 ≤ H2S < 30 ppb
        - Orange: H2S ≥ 30 ppb

    OLD THRESHOLDS (Historical):
        - Green: H2S < 5 ppb
        - Yellow: 5 ≤ H2S < 15 ppb
        - Orange: H2S ≥ 15 ppb

    Returns:
        DataFrame with added 'h2s_category' column
    """
    context.log.info("Applying new H2S categorization thresholds...")

    # Apply new categorization
    df = apply_categorization(monthly_training_data, h2s_column='H2S')

    # Log category distribution
    category_counts = df['h2s_category'].value_counts()
    context.log.info(f"Category distribution:\n{category_counts}")

    # Calculate what changed from old thresholds
    old_orange_threshold = 15
    new_orange_threshold = 30
    reclassified = df[(df['H2S'] >= old_orange_threshold) & (df['H2S'] < new_orange_threshold)]
    context.log.info(
        f"Reclassified {len(reclassified)} samples from orange→yellow "
        f"(H2S in range [{old_orange_threshold}, {new_orange_threshold}))"
    )

    # Log threshold info
    threshold_info = get_threshold_info()
    context.log.info(f"Threshold version: {threshold_info['version']}")

    # Add metadata
    context.add_output_metadata({
        "green_count": int(category_counts.get('green', 0)),
        "yellow_count": int(category_counts.get('yellow', 0)),
        "orange_count": int(category_counts.get('orange', 0)),
        "reclassified_orange_to_yellow": len(reclassified),
        "threshold_version": threshold_info['version'],
    })

    return df


@dg.asset(
    key_prefix="h2s",
    group_name="h2s_training_data",
    kinds={"validation"},
    description="Data quality validation report",
    partitions_def=monthly_training_partitions,
    ins={
        "relabeled_training_data": dg.AssetIn(key=_KEY("relabeled_training_data")),
    },
)
def data_quality_report(
    context: dg.AssetExecutionContext,
    relabeled_training_data: pd.DataFrame
) -> None:
    """Generate data quality validation report.

    Checks:
        - Missing values per feature (<10% threshold)
        - H2S value range (0-100 ppb reasonable)
        - Class imbalance (warn if any class <5%)
        - Time continuity (no large gaps)

    Raises:
        RuntimeError: If critical data quality issues found
    """
    context.log.info("Validating training data quality...")

    df = relabeled_training_data
    issues = []

    # Check 1: Missing values
    missing_pct = (df.isnull().sum() / len(df) * 100).round(2)
    high_missing = missing_pct[missing_pct > 10]
    if len(high_missing) > 0:
        issues.append(f"High missing values: {high_missing.to_dict()}")
        context.log.warning(f"Features with >10% missing: {list(high_missing.index)}")

    # Check 2: H2S range
    h2s_min = df['H2S'].min()
    h2s_max = df['H2S'].max()
    if h2s_min < 0:
        issues.append(f"H2S has negative values: min={h2s_min}")
    if h2s_max > 650:
        context.log.warning(f"H2S max ({h2s_max} ppb) exceeds expected upper bound of 650 ppb")

    # Check 3: Class imbalance
    category_pct = (df['h2s_category'].value_counts() / len(df) * 100).round(2)
    low_classes = category_pct[category_pct < 5]
    if len(low_classes) > 0:
        context.log.warning(
            f"Classes with <5% representation: {low_classes.to_dict()}"
        )
        context.log.warning("Consider collecting more data or adjusting class weights")

    # Check 4: Time continuity
    df_sorted = df.sort_values('time')
    time_diffs = df_sorted['time'].diff()
    large_gaps = time_diffs[time_diffs > pd.Timedelta(hours=24)]
    if len(large_gaps) > 0:
        context.log.warning(f"Found {len(large_gaps)} gaps >24 hours in time series")

    # Summary
    if len(issues) > 0:
        error_msg = "Data quality issues found:\n" + "\n".join(f"  - {issue}" for issue in issues)
        context.log.error(error_msg)
        raise RuntimeError(
            f"Training data quality validation failed with {len(issues)} critical issues. "
            "Review logs and fix data before proceeding."
        )

    context.log.info("✓ Data quality validation passed")

    # Metadata
    context.add_output_metadata({
        "validation_passed": True,
        "total_samples": len(df),
        "features_with_missing": int((df.isnull().sum() > 0).sum()),
        "max_missing_pct": float(missing_pct.max()),
        "h2s_range": [float(h2s_min), float(h2s_max)],
        "class_distribution": category_pct.to_dict(),
        "time_gaps_gt_24h": len(large_gaps),
    })


@dg.asset(
    key_prefix="h2s",
    group_name="h2s_training_data",
    kinds={"python"},
    description="Time-based training/validation split (80/20)",
    partitions_def=monthly_training_partitions,
    ins={
        "relabeled_training_data": dg.AssetIn(key=_KEY("relabeled_training_data")),
    },
    config_schema={
        "validation_split": dg.Field(
            float,
            default_value=0.2,
            description="Fraction of data for validation (default: 0.2 = 20%)"
        ),
    }
)
def training_data(
    context: dg.AssetExecutionContext,
    relabeled_training_data: pd.DataFrame
) -> pd.DataFrame:
    """Extract training portion of data using time-based split.

    Returns first (100-validation_split)% of time-sorted data.
    """
    validation_split = context.op_config["validation_split"]

    # Sort by time
    df = relabeled_training_data.sort_values('time').reset_index(drop=True)

    # Calculate split index
    split_idx = int(len(df) * (1 - validation_split))
    train_df = df.iloc[:split_idx].copy()

    context.log.info(f"Training set: {len(train_df)} rows ({train_df['time'].min()} to {train_df['time'].max()})")

    train_dist = train_df['h2s_category'].value_counts()
    context.log.info(f"Training set categories:\n{train_dist}")

    context.add_output_metadata({
        "size": len(train_df),
        "date_start": str(train_df['time'].min()),
        "date_end": str(train_df['time'].max()),
        "green_count": int(train_dist.get('green', 0)),
        "yellow_count": int(train_dist.get('yellow', 0)),
        "orange_count": int(train_dist.get('orange', 0)),
    })

    return train_df


@dg.asset(
    key_prefix="h2s",
    group_name="h2s_training_data",
    description="Validation data (20% time-based split)",
    partitions_def=monthly_training_partitions,
    ins={
        "relabeled_training_data": dg.AssetIn(key=_KEY("relabeled_training_data")),
    },
    config_schema={
        "validation_split": dg.Field(
            float,
            default_value=0.2,
            description="Fraction of data for validation (default: 0.2 = 20%)"
        ),
    }
)
def validation_data(
    context: dg.AssetExecutionContext,
    relabeled_training_data: pd.DataFrame
) -> pd.DataFrame:
    """Extract validation portion of data using time-based split.

    Returns last validation_split% of time-sorted data.
    """
    validation_split = context.op_config["validation_split"]

    # Sort by time
    df = relabeled_training_data.sort_values('time').reset_index(drop=True)

    # Calculate split index
    split_idx = int(len(df) * (1 - validation_split))
    val_df = df.iloc[split_idx:].copy()

    context.log.info(f"Validation set: {len(val_df)} rows ({val_df['time'].min()} to {val_df['time'].max()})")

    val_dist = val_df['h2s_category'].value_counts()
    context.log.info(f"Validation set categories:\n{val_dist}")

    context.add_output_metadata({
        "size": len(val_df),
        "date_start": str(val_df['time'].min()),
        "date_end": str(val_df['time'].max()),
        "green_count": int(val_dist.get('green', 0)),
        "yellow_count": int(val_dist.get('yellow', 0)),
        "orange_count": int(val_dist.get('orange', 0)),
    })

    return val_df


# ==============================================================================
# PHASE 2: Model Training
# ==============================================================================

@dg.asset(
    key_prefix="h2s",
    group_name="h2s_model_training",
    kinds={"xgboost", "ml"},
    description="Train XGBoost model with 5-fold time-series cross-validation. Partition variant controls preprocessing (xgboost_base=class weights only, xgboost_smote=SMOTE oversampling).",
    partitions_def=model_run_partitions,
    ins={
        "training_data": dg.AssetIn(
            key=dg.AssetKey(["h2s", "training_data"]),
            partition_mapping=dg.MultiToSingleDimensionPartitionMapping("month")
        ),
    },
    config_schema={
        "n_folds": dg.Field(int, default_value=5, description="Number of CV folds"),
        "n_estimators": dg.Field(int, default_value=100, description="Number of boosting rounds"),
        "max_depth": dg.Field(int, default_value=6, description="Maximum tree depth"),
        "learning_rate": dg.Field(float, default_value=0.1, description="Learning rate"),
        "use_class_weights": dg.Field(bool, default_value=True, description="Auto-balance class weights"),
        "hazard_weight_multiplier": dg.Field(float, default_value=3.0,
            description="Extra weight multiplier applied to orange and yellow classes"),
    }
)
def trained_model_cv(
    context: dg.AssetExecutionContext,
    training_data: pd.DataFrame
) -> Dict:
    """Train XGBoost classifier with time-series cross-validation.

    Uses basic numerical features from raw data.
    Advanced feature engineering (cyclical encoding, interactions) will be added later.

    Returns:
        Dict with:
            - 'model': Trained XGBoost model
            - 'cv_metrics': List of metrics per fold
            - 'feature_importance': Dict of feature importances
            - 'training_metadata': Hyperparameters and training info
    """
    import xgboost as xgb
    import traceback as tb

    context.log.info("=" * 60)
    context.log.info("STARTING trained_model_cv asset")
    context.log.info("=" * 60)

    # Derive variant from partition key
    multi_key = context.partition_key
    month_key = multi_key.keys_by_dimension["month"]
    variant = multi_key.keys_by_dimension["variant"]
    month_str = pd.to_datetime(month_key).strftime("%Y_%m")
    use_smote = variant == "xgboost_smote"
    use_rf    = variant == "random_forest"

    context.log.info(f"Partition: month={month_key}, variant={variant}")
    context.log.info(f"SMOTE enabled: {use_smote}, Random Forest: {use_rf}")

    # Extract training data
    train_df = training_data.copy()

    context.log.info(f"Raw training data shape: {train_df.shape}")
    context.log.info(f"Available columns: {list(train_df.columns)}")

    # Full feature set — uses all pre-computed columns available in modeldata_h2s_nofill.parquet.
    # Falls back gracefully to whatever subset is present in train_df.
    basic_features = [
        # Core weather
        'temperature_2m', 'relative_humidity_2m', 'dewpoint_2m',
        'precipitation', 'surface_pressure', 'cloud_cover',
        # Wind
        'wind_speed_10m', 'wind_direction_10m', 'wind_gusts_10m',
        'wind_direction_sin', 'wind_direction_cos',
        'wind_speed_10m_avg_2h', 'wind_speed_10m_avg_3h', 'wind_speed_10m_avg_4h',
        'wind_gusts_10m_max_2h', 'wind_gusts_10m_max_3h', 'wind_gusts_10m_max_4h',
        # Tidal / flow
        'Flow (m^3/s)--Border', 'tide_height', 'tidal_state_encoded',
        # Encoded categoricals
        'wind_direction_categorical_encoded',
        # Interaction features
        'wind_temp_interaction', 'humidity_temp_interaction',
    ]

    # Check which features are available
    available_features = [f for f in basic_features if f in train_df.columns]
    missing_features = set(basic_features) - set(available_features)

    if missing_features:
        context.log.warning(f"Missing features (will skip): {missing_features}")

    context.log.info(f"Using {len(available_features)} features: {available_features}")

    # Extract features and target
    X_train = train_df[available_features].copy()
    y_train = train_df['h2s_category'].copy()

    # Reset indices to ensure clean 0-based indexing for CV
    X_train = X_train.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)

    context.log.info(f"Training data: {len(X_train)} samples, {len(available_features)} features")
    context.log.info(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")

    # Create label mapping based on classes ACTUALLY present in training data
    # This handles case where new thresholds eliminate some classes (e.g., no orange samples)
    unique_classes = sorted(y_train.unique())
    label_map = {class_name: idx for idx, class_name in enumerate(unique_classes)}

    context.log.info(f"Classes present in training data: {unique_classes}")
    context.log.info(f"Label mapping: {label_map}")

    # Log class distribution
    class_dist = y_train.value_counts()
    for class_name in unique_classes:
        count = class_dist.get(class_name, 0)
        pct = (count / len(y_train)) * 100
        context.log.info(f"  {class_name}: {count} samples ({pct:.1f}%)")

    # Warn if missing expected classes
    expected_classes = {'green', 'orange', 'yellow'}
    missing_classes = expected_classes - set(unique_classes)
    if missing_classes:
        context.log.warning(f"⚠️  Missing classes in training data: {missing_classes}")
        context.log.warning(f"   This may be due to new H2S thresholds (Orange ≥30 ppb)")
        context.log.warning(f"   Model will only be able to predict: {unique_classes}")

    # Train with cross-validation — dispatch to RF or XGBoost based on variant
    if use_rf:
        model, cv_metrics = train_random_forest_with_cv(
            X_train=X_train,
            y_train=y_train,
            label_map=label_map,
            n_folds=context.op_config['n_folds'],
            n_estimators=context.op_config['n_estimators'],
            max_depth=context.op_config['max_depth'] or None,
            use_class_weights=context.op_config['use_class_weights'],
            use_smote=use_smote,
            random_state=42,
            logger=context.log,
            hazard_multiplier=context.op_config['hazard_weight_multiplier'],
        )
    else:
        model, cv_metrics = train_model_with_cv(
            X_train=X_train,
            y_train=y_train,
            label_map=label_map,
            n_folds=context.op_config['n_folds'],
            n_estimators=context.op_config['n_estimators'],
            max_depth=context.op_config['max_depth'],
            learning_rate=context.op_config['learning_rate'],
            use_class_weights=context.op_config['use_class_weights'],
            use_smote=use_smote,
            random_state=42,
            logger=context.log,
            hazard_multiplier=context.op_config['hazard_weight_multiplier'],
        )

    # Log CV results
    cv_summary = calculate_cv_summary(cv_metrics)
    context.log.info(f"Cross-validation complete:")
    context.log.info(f"  Mean Balanced Accuracy: {cv_summary['balanced_accuracy_mean']:.3f} ± {cv_summary['balanced_accuracy_std']:.3f}")

    # Log per-class recall (only for classes present in training data)
    for class_name in unique_classes:
        recall_key = f'recall_{class_name}_mean'
        recall_std_key = f'recall_{class_name}_std'
        if recall_key in cv_summary:
            context.log.info(f"  Mean {class_name.capitalize()} Recall: {cv_summary[recall_key]:.3f} ± {cv_summary[recall_std_key]:.3f}")

    # Get feature importance
    feature_importance = get_feature_importance(model, available_features)
    top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
    context.log.info(f"Top 5 features: {[f[0] for f in top_features]}")

    # Training metadata
    class_dist = y_train.value_counts().to_dict()
    training_metadata = {
        'trained_at': datetime.now().isoformat(),
        'variant': variant,
        'month': month_key,
        'n_samples': len(train_df),
        'n_features': len(available_features),
        'class_distribution': class_dist,
        'hyperparameters': {
            'n_folds': context.op_config['n_folds'],
            'n_estimators': context.op_config['n_estimators'],
            'max_depth': context.op_config['max_depth'],
            'learning_rate': context.op_config['learning_rate'],
            'use_class_weights': context.op_config['use_class_weights'],
            'hazard_weight_multiplier': context.op_config['hazard_weight_multiplier'],
            'use_smote': use_smote,
        },
        'cv_mean_balanced_accuracy': cv_summary['balanced_accuracy_mean'],
        'cv_mean_orange_recall': cv_summary.get('recall_orange_mean', 0.0),  # May be 0 if no orange class
        'label_map': label_map,
        'classes_present': unique_classes,
    }

    # Metadata
    metadata = {
        "n_folds": context.op_config['n_folds'],
        "mean_balanced_accuracy": float(cv_summary['balanced_accuracy_mean']),
        "std_balanced_accuracy": float(cv_summary['balanced_accuracy_std']),
        "top_feature": top_features[0][0] if top_features else "unknown",
        "classes_present": list(unique_classes),
        "n_classes": len(unique_classes),
    }

    # Add orange recall only if orange class exists
    if 'recall_orange_mean' in cv_summary:
        metadata["mean_orange_recall"] = float(cv_summary['recall_orange_mean'])

    context.add_output_metadata(metadata)

    model_filename = "model.json" if hasattr(model, 'save_model') else "model.joblib"

    return {
        'model': model,
        'cv_metrics': cv_metrics,
        'feature_importance': feature_importance,
        'training_metadata': training_metadata,
        'variant': variant,
        'month_str': month_str,
        'feature_names': available_features,
        'label_map': label_map,
        'model_filename': model_filename,
    }


@dg.asset(
    key_prefix="h2s",
    group_name="h2s_training_export",
    required_resource_keys={"s3"},
    kinds={"s3", "json"},
    description="Export training metrics to S3",
    partitions_def=model_run_partitions,
    ins={
        "trained_model_cv": dg.AssetIn(key=_KEY("trained_model_cv")),
    },
)
def model_training_metrics(
    context: dg.AssetExecutionContext,
    trained_model_cv: Dict
) -> None:
    """Export training metrics and CV results to S3.

    S3 Path: tijuana/forecast/models/training/{YYYY_MM}/{variant}/
    """
    s3_resource = context.resources.s3
    month_str = trained_model_cv['month_str']
    variant = trained_model_cv['variant']
    base_path = f"{TRAINING_PATH}/{month_str}/{variant}"

    context.log.info(f"Exporting training artifacts to S3: {base_path}")

    # Export trained model file
    model = trained_model_cv['model']
    if hasattr(model, 'save_model'):
        # XGBoost
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp:
            tmp_path = tmp.name
            model.save_model(tmp_path)
            with open(tmp_path, 'rb') as f:
                model_bytes = f.read()
            os.unlink(tmp_path)
        model_filename = "model.json"
        content_type = 'application/json'
    else:
        # sklearn (RandomForest, etc.) — use joblib
        import joblib
        with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as tmp:
            tmp_path = tmp.name
            joblib.dump(model, tmp_path)
            with open(tmp_path, 'rb') as f:
                model_bytes = f.read()
            os.unlink(tmp_path)
        model_filename = "model.joblib"
        content_type = 'application/octet-stream'

    model_filename = trained_model_cv['model_filename']
    s3_resource.putFile(
        model_bytes,
        f"{base_path}/{model_filename}",
        bucket=s3_resource.S3_BUCKET,
        content_type=content_type,
    )
    context.log.info(f"✓ Saved model to {base_path}/{model_filename} ({len(model_bytes) / 1024:.1f} KB)")

    # Export CV metrics
    cv_metrics_json = json.dumps(trained_model_cv['cv_metrics'], indent=2)
    s3_resource.putFile_text(
        data=cv_metrics_json,
        path=f"{base_path}/cv_metrics.json",
        bucket=s3_resource.S3_BUCKET,
        content_type='application/json'
    )

    # Export training metadata
    metadata_json = json.dumps(trained_model_cv['training_metadata'], indent=2)
    s3_resource.putFile_text(
        data=metadata_json,
        path=f"{base_path}/training_metadata.json",
        bucket=s3_resource.S3_BUCKET,
        content_type='application/json'
    )

    # Export feature importance
    importance_json = json.dumps(trained_model_cv['feature_importance'], indent=2)
    s3_resource.putFile_text(
        data=importance_json,
        path=f"{base_path}/feature_importance.json",
        bucket=s3_resource.S3_BUCKET,
        content_type='application/json'
    )

    # Create metadata for training metrics
    training_metadata = store_assets.objectMetadata(
        name="H2S Model Training Metrics",
        description=f"Cross-validation metrics and feature importance from {month_str}/{variant} training run",
        variableMeasured=["Balanced Accuracy", "Precision", "Recall", "F1 Score", "Feature Importance"]
    )
    training_metadata.distribution = [
        {"format": "json", "url": f"{base_path}/cv_metrics.json"},
        {"format": "json", "url": f"{base_path}/training_metadata.json"},
        {"format": "json", "url": f"{base_path}/feature_importance.json"},
    ]
    store_assets.metadata_to_s3(training_metadata, f"{base_path}/training_metrics", s3_resource)

    context.log.info(f"✓ Exported training artifacts to {base_path}")

    context.add_output_metadata({
        "s3_path": base_path,
        "artifacts_exported": 4,  # 3 data files + 1 metadata
    })


@dg.asset(
    key_prefix="h2s",
    group_name="h2s_model_training",
    required_resource_keys={"s3"},
    kinds={"visualization", "s3"},
    description="Generate and export feature importance visualization",
    partitions_def=model_run_partitions,
    ins={
        "trained_model_cv": dg.AssetIn(key=_KEY("trained_model_cv")),
    },
)
def feature_importance_analysis(
    context: dg.AssetExecutionContext,
    trained_model_cv: Dict,
) -> None:
    """Generate feature importance comparison visualization."""
    from h2s.predictor.visualizations import generate_feature_importance

    s3_resource = context.resources.s3
    month_str = trained_model_cv['month_str']
    variant = trained_model_cv['variant']

    context.log.info(f"Generating feature importance visualization for {variant}...")

    # Generate plot — use the new model's own feature list, not the production model's
    new_model = trained_model_cv['model']
    prep_info = {'feature_cols': trained_model_cv['feature_names']}

    model_name = f"{variant} ({type(new_model).__name__})"
    plot_bytes = generate_feature_importance(new_model, prep_info, top_n=15, model_name=model_name)

    # Upload to training path
    s3_path = f"{TRAINING_PATH}/{month_str}/{variant}/feature_importance.png"
    plot_data = plot_bytes.read()
    s3_resource.putFile(plot_data, s3_path, bucket=s3_resource.S3_BUCKET, content_type='image/png')

    # Also write to forecast latest path so dashboards pick it up
    latest_path = f"latest/{LATEST_FORECAST_DATA}/visualizations/feature_importance_{variant}.png"
    s3_resource.putFile(plot_data, latest_path, bucket=s3_resource.S3_BUCKET, content_type='image/png')

    # Create metadata for visualization
    viz_metadata = store_assets.objectMetadata(
        name="H2S Feature Importance Visualization",
        description=f"Feature importance plot from {month_str}/{variant} model training",
        variableMeasured=["Feature Importance Scores"]
    )
    viz_metadata.distribution = [
        {"format": "png", "url": s3_path},
    ]
    store_assets.metadata_to_s3(viz_metadata, f"{TRAINING_PATH}/{month_str}/{variant}/feature_importance", s3_resource)

    context.log.info(f"✓ Uploaded feature importance to {s3_path}")

    context.add_output_metadata({
        "s3_path": s3_path,
        "top_features": list(trained_model_cv['feature_importance'].keys())[:5],
    })


# ============================================================================
# PHASE 3: MODEL VALIDATION (Assets 8-10)
# ============================================================================
# Compare new trained model vs current production model on held-out validation set


@dg.asset(
    key_prefix="h2s",
    group_name="h2s_model_validation",
    kinds={"ml", "validation"},
    description="Generate predictions on validation set using newly trained model",
    partitions_def=model_run_partitions,
    ins={
        "trained_model_cv": dg.AssetIn(key=_KEY("trained_model_cv")),
        "validation_data": dg.AssetIn(
            key=dg.AssetKey(["h2s", "validation_data"]),
            partition_mapping=dg.MultiToSingleDimensionPartitionMapping("month")
        ),
    },
)
def validation_predictions(
    context: dg.AssetExecutionContext,
    trained_model_cv: Dict,
    validation_data: pd.DataFrame,
) -> pd.DataFrame:
    """Generate predictions on validation set using newly trained model.

    Uses the trained model from CV to predict on the held-out validation set.
    Returns predictions in same format as h2s_predictions asset.

    Returns:
        DataFrame with columns: time, site_name, predicted_category,
        probability_green, probability_yellow, probability_orange,
        confidence, alert, actual_category (ground truth)
    """
    context.log.info("=" * 60)
    context.log.info("GENERATING VALIDATION PREDICTIONS")
    context.log.info("=" * 60)

    # Extract validation data and new model
    val_df = validation_data.copy()
    new_model = trained_model_cv['model']
    feature_names = trained_model_cv['feature_names']
    label_map = trained_model_cv['label_map']
    reverse_label_map = {v: k for k, v in label_map.items()}

    context.log.info(f"Validation samples: {len(val_df)}")
    context.log.info(f"Using {len(feature_names)} features from trained model")

    # Use the same raw features the model was trained on — no re-preprocessing needed
    X_val = val_df[feature_names].copy()

    # Generate predictions
    y_pred = new_model.predict(X_val)
    y_pred_proba = new_model.predict_proba(X_val)

    # Build predictions DataFrame
    predictions_df = pd.DataFrame({
        'time': val_df['time'].values,
        'site_name': val_df['site_name'].values,
        'predicted_category': [reverse_label_map[pred] for pred in y_pred],
        'probability_green': y_pred_proba[:, 0],
        'probability_orange': y_pred_proba[:, 1],
        'probability_yellow': y_pred_proba[:, 2],
        'confidence': y_pred_proba.max(axis=1),
        'alert': [reverse_label_map[pred] in ['orange', 'yellow'] for pred in y_pred],
        'actual_category': val_df['h2s_category'].values,
    })

    # Calculate validation accuracy
    correct = (predictions_df['predicted_category'] == predictions_df['actual_category']).sum()
    accuracy = correct / len(predictions_df)

    # Count alerts
    n_alerts = predictions_df['alert'].sum()
    n_orange = (predictions_df['predicted_category'] == 'orange').sum()
    n_yellow = (predictions_df['predicted_category'] == 'yellow').sum()

    context.log.info(f"Validation Accuracy: {accuracy:.1%} ({correct}/{len(predictions_df)})")
    context.log.info(f"Alerts: {n_alerts} ({n_orange} orange, {n_yellow} yellow)")

    context.add_output_metadata({
        "n_samples": len(predictions_df),
        "validation_accuracy": float(accuracy),
        "n_alerts": int(n_alerts),
        "n_orange_predictions": int(n_orange),
        "n_yellow_predictions": int(n_yellow),
    })

    return predictions_df


@dg.asset(
    key_prefix="h2s",
    group_name="h2s_model_validation",
    required_resource_keys={"s3"},
    kinds={"ml", "validation"},
    description="Compute new model metrics on the validation set",
    partitions_def=model_run_partitions,
    ins={
        "validation_predictions": dg.AssetIn(key=_KEY("validation_predictions")),
        "validation_data": dg.AssetIn(
            key=dg.AssetKey(["h2s", "validation_data"]),
            partition_mapping=dg.MultiToSingleDimensionPartitionMapping("month")
        ),
    },
)
def validation_report(
    context: dg.AssetExecutionContext,
    validation_predictions: pd.DataFrame,
    validation_data: pd.DataFrame,
) -> Dict:
    """Evaluate the newly trained model on the held-out validation set.

    Returns new-model metrics only.  Comparison against the current production
    model is handled downstream by model_comparison_report (eager automation).

    Returns:
        Dict with new model metrics and validation metadata
    """
    context.log.info("=" * 60)
    context.log.info("NEW MODEL VALIDATION REPORT")
    context.log.info("=" * 60)

    s3_resource = context.resources.s3
    multi_key = context.partition_key
    month_str = pd.to_datetime(multi_key.keys_by_dimension["month"]).strftime("%Y_%m")
    variant = multi_key.keys_by_dimension["variant"]

    val_df = validation_data.copy()

    # Create label mapping based on classes ACTUALLY present (matches training)
    unique_classes = sorted(val_df['h2s_category'].unique())
    label_map = {class_name: idx for idx, class_name in enumerate(unique_classes)}

    context.log.info(f"Validation classes: {unique_classes}")
    context.log.info(f"Label mapping: {label_map}")

    # === NEW MODEL METRICS ===
    y_true = val_df['h2s_category'].map(label_map).values
    y_pred_new = validation_predictions['predicted_category'].map(label_map).values

    new_metrics = calculate_metrics(y_true, y_pred_new, class_names=unique_classes)

    context.log.info("\n📊 NEW MODEL METRICS:")
    context.log.info(f"  Balanced Accuracy: {new_metrics['balanced_accuracy']:.3f}")

    for class_name in unique_classes:
        if f'recall_{class_name}' in new_metrics:
            context.log.info(f"  {class_name.capitalize()} Recall: {new_metrics[f'recall_{class_name}']:.3f}")
            context.log.info(f"  {class_name.capitalize()} Precision: {new_metrics[f'precision_{class_name}']:.3f}")
            context.log.info(f"  {class_name.capitalize()} F1: {new_metrics[f'f1_{class_name}']:.3f}")

    # === EXPORT TO S3 ===
    validation_report_json = {
        'timestamp': datetime.now().isoformat(),
        'validation_period': month_str,
        'variant': variant,
        'validation_samples': len(val_df),
        'new_model_metrics': new_metrics,
    }

    s3_path = f"{TRAINING_PATH}/{month_str}/{variant}/validation_report.json"
    s3_resource.putFile(
        json.dumps(validation_report_json, indent=2).encode('utf-8'),
        s3_path,
        bucket=s3_resource.S3_BUCKET,
        content_type='application/json'
    )

    context.log.info(f"✓ Uploaded validation report to {s3_path}")

    context.add_output_metadata({
        "s3_path": s3_path,
        "new_balanced_accuracy": float(new_metrics['balanced_accuracy']),
        "new_orange_recall": float(new_metrics.get('recall_orange', 0.0)),
    })

    return validation_report_json


@dg.asset(
    key_prefix="h2s",
    group_name="h2s_model_validation",
    required_resource_keys={"s3"},
    kinds={"ml", "visualization"},
    description="Compare new vs current production model; triggers automatically when both are ready",
    partitions_def=model_run_partitions,
    ins={
        "validation_report": dg.AssetIn(key=_KEY("validation_report")),
        "h2s_model_artifacts": dg.AssetIn(key=dg.AssetKey(["h2s", "h2s_model_artifacts"])),
        "validation_data": dg.AssetIn(
            key=dg.AssetKey(["h2s", "validation_data"]),
            partition_mapping=dg.MultiToSingleDimensionPartitionMapping("month")
        ),
    },
    auto_materialize_policy=dg.AutoMaterializePolicy.eager(),
)
def model_comparison_report(
    context: dg.AssetExecutionContext,
    validation_report: Dict,
    h2s_model_artifacts,
    validation_data: pd.DataFrame,
) -> Dict:
    """Compare new model vs current production model and generate visualization.

    Runs automatically (eager) once both validation_report and h2s_model_artifacts
    are materialized — no hard dependency from the training job.

    Returns:
        Dict with comparison results, quality gates, and approval recommendation
    """
    context.log.info("=" * 60)
    context.log.info("MODEL COMPARISON REPORT")
    context.log.info("=" * 60)

    s3_resource = context.resources.s3
    month_str = validation_report['validation_period']
    variant = validation_report['variant']

    val_df = validation_data.copy()
    unique_classes = sorted(val_df['h2s_category'].unique())
    label_map = {class_name: idx for idx, class_name in enumerate(unique_classes)}

    new_metrics = validation_report['new_model_metrics']

    # === CURRENT MODEL METRICS ===
    # Production model expects a 'date' column for temporal feature engineering
    y_true = val_df['h2s_category'].map(label_map).values
    val_for_prod = val_df.rename(columns={'time': 'date'})
    val_preprocessed = h2s_model_artifacts.preprocess_data(val_for_prod)
    current_predictions = h2s_model_artifacts.predict(val_preprocessed)
    y_pred_current = current_predictions['predicted_category'].map(label_map).values

    current_metrics = calculate_metrics(y_true, y_pred_current, class_names=unique_classes)

    context.log.info(f"\n📊 NEW MODEL:     balanced_acc={new_metrics['balanced_accuracy']:.3f}")
    context.log.info(f"📊 CURRENT MODEL: balanced_acc={current_metrics['balanced_accuracy']:.3f}")

    # === MODEL COMPARISON ===
    new_metrics_with_defaults = {
        **new_metrics,
        'recall_orange': new_metrics.get('recall_orange', 0.0),
        'precision_orange': new_metrics.get('precision_orange', 0.0),
    }
    current_metrics_with_defaults = {
        **current_metrics,
        'recall_orange': current_metrics.get('recall_orange', 0.0),
        'precision_orange': current_metrics.get('precision_orange', 0.0),
    }

    approval_recommended, comparison_details = compare_models(
        new_metrics=new_metrics_with_defaults,
        current_metrics=current_metrics_with_defaults,
        min_balanced_acc_delta=-0.05,
        min_orange_recall_delta=-0.05,
        min_orange_precision_delta=-0.10,
    )

    if 'orange' not in unique_classes:
        context.log.warning("⚠️  Orange class not present in validation data — orange quality gates may not be meaningful")

    context.log.info("\n🚦 QUALITY GATES:")
    for gate_name, gate_info in comparison_details['quality_gates'].items():
        status = "✓ PASS" if gate_info['passed'] else "✗ FAIL"
        context.log.info(f"  {status} - {gate_name}: {gate_info['actual']:.3f} (threshold: {gate_info['threshold']:.3f})")

    context.log.info("\n" + "=" * 60)
    context.log.info("✓ RECOMMENDATION: APPROVE DEPLOYMENT" if approval_recommended else "✗ RECOMMENDATION: REJECT DEPLOYMENT")
    context.log.info("=" * 60)

    # === VISUALIZATION ===
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('Model Comparison Report', fontsize=16, fontweight='bold')

    # Subplot 1: Confusion Matrix — New Model
    ax1 = axes[0, 0]
    cm_new = np.array(new_metrics['confusion_matrix'])
    sns.heatmap(
        cm_new, annot=True, fmt='d', cmap='Blues',
        xticklabels=['Green', 'Orange', 'Yellow'],
        yticklabels=['Green', 'Orange', 'Yellow'],
        ax=ax1
    )
    ax1.set_title('New Model - Confusion Matrix', fontweight='bold')
    ax1.set_ylabel('True Label')
    ax1.set_xlabel('Predicted Label')

    # Subplot 2: Confusion Matrix — Current Model
    ax2 = axes[0, 1]
    cm_current = np.array(current_metrics['confusion_matrix'])
    sns.heatmap(
        cm_current, annot=True, fmt='d', cmap='Oranges',
        xticklabels=['Green', 'Orange', 'Yellow'],
        yticklabels=['Green', 'Orange', 'Yellow'],
        ax=ax2
    )
    ax2.set_title('Current Model - Confusion Matrix', fontweight='bold')
    ax2.set_ylabel('True Label')
    ax2.set_xlabel('Predicted Label')

    # Subplot 3: Per-Class Metrics Comparison
    ax3 = axes[1, 0]
    classes = ['green', 'orange', 'yellow']
    metrics_to_plot = ['precision', 'recall', 'f1']
    x = np.arange(len(classes))
    width = 0.12
    for i, metric in enumerate(metrics_to_plot):
        new_values = [new_metrics.get(f'{metric}_{cls}', 0.0) for cls in classes]
        current_values = [current_metrics.get(f'{metric}_{cls}', 0.0) for cls in classes]
        offset = (i - 1) * width
        ax3.bar(x + offset - width/2, new_values, width, label=f'New {metric.capitalize()}', alpha=0.8)
        ax3.bar(x + offset + width/2, current_values, width, label=f'Current {metric.capitalize()}', alpha=0.6)
    ax3.set_ylabel('Score')
    ax3.set_title('Per-Class Metrics Comparison', fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels([c.capitalize() for c in classes])
    ax3.legend(fontsize=8, ncol=2)
    ax3.set_ylim([0, 1.0])
    ax3.grid(axis='y', alpha=0.3)

    # Subplot 4: Summary text
    ax4 = axes[1, 1]
    ax4.axis('off')
    summary_text = "VALIDATION SUMMARY\n" + "=" * 40 + "\n\n"
    summary_text += f"Validation Samples: {validation_report['validation_samples']}\n\n"
    summary_text += "BALANCED ACCURACY\n"
    summary_text += f"  New:     {new_metrics['balanced_accuracy']:.3f}\n"
    summary_text += f"  Current: {current_metrics['balanced_accuracy']:.3f}\n"
    summary_text += f"  Delta:   {new_metrics['balanced_accuracy'] - current_metrics['balanced_accuracy']:+.3f}\n\n"
    summary_text += "ORANGE RECALL (Critical Metric)\n"
    nr = new_metrics.get('recall_orange', 0.0)
    cr = current_metrics.get('recall_orange', 0.0)
    summary_text += f"  New:     {nr:.3f}\n"
    summary_text += f"  Current: {cr:.3f}\n"
    summary_text += f"  Delta:   {nr - cr:+.3f}\n\n"
    summary_text += "QUALITY GATES\n"
    for gate_name, gate_info in comparison_details['quality_gates'].items():
        summary_text += f"  {'✓' if gate_info['passed'] else '✗'} {gate_name}\n"
    summary_text += "\n" + "=" * 40 + "\n"
    summary_text += "✓ APPROVED FOR DEPLOYMENT" if approval_recommended else "✗ NOT RECOMMENDED FOR DEPLOYMENT"
    ax4.text(0.05, 0.95, summary_text,
             transform=ax4.transAxes, fontsize=10, verticalalignment='top',
             fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout()
    plot_bytes = io.BytesIO()
    plt.savefig(plot_bytes, format='png', dpi=150, bbox_inches='tight')
    plot_bytes.seek(0)
    plt.close()

    # Upload visualization and comparison report to S3
    s3_path = f"{TRAINING_PATH}/{month_str}/{variant}/model_comparison.png"
    s3_resource.putFile(plot_bytes.read(), s3_path, bucket=s3_resource.S3_BUCKET, content_type='image/png')
    context.log.info(f"✓ Uploaded model comparison visualization to {s3_path}")

    comparison_report_json = {
        'timestamp': datetime.now().isoformat(),
        'validation_period': month_str,
        'variant': variant,
        'validation_samples': validation_report['validation_samples'],
        'approval_recommended': approval_recommended,
        'comparison_details': comparison_details,
        'new_model_metrics': new_metrics,
        'current_model_metrics': current_metrics,
        'visualization_path': s3_path,
    }

    report_s3_path = f"{TRAINING_PATH}/{month_str}/{variant}/model_comparison_report.json"
    s3_resource.putFile(
        json.dumps(comparison_report_json, indent=2).encode('utf-8'),
        report_s3_path,
        bucket=s3_resource.S3_BUCKET,
        content_type='application/json',
    )

    context.add_output_metadata({
        "s3_path": s3_path,
        "approval_recommended": approval_recommended,
        "balanced_acc_delta": float(comparison_details['metric_differences']['balanced_accuracy']),
        "orange_recall_delta": float(comparison_details['metric_differences'].get('recall_orange', 0.0)),
    })

    return comparison_report_json


# ============================================================================
# PHASE 4: MODEL DEPLOYMENT (Assets 11-13)
# ============================================================================
# Manual approval gate → Archive current model → Deploy to production


@dg.asset(
    key_prefix="h2s",
    group_name="h2s_model_deployment",
    kinds={"ml", "deployment"},
    description="Manual approval gate for model deployment (BLOCKS auto-deployment)",
    partitions_def=model_run_partitions,
    ins={
        "model_comparison_report": dg.AssetIn(key=_KEY("model_comparison_report")),
    },
    config_schema={
        "approve_deployment": dg.Field(
            bool,
            default_value=False,
            description="Set to True to approve deployment. If False, raises Failure.",
        )
    },
)
def deployment_approval(
    context: dg.AssetExecutionContext,
    model_comparison_report: Dict,
) -> Dict:
    """Manual approval gate for model deployment.

    This asset BLOCKS automatic deployment by default. Human must review
    model_comparison_report in S3 and explicitly approve by setting
    approve_deployment=True in asset config.

    Quality Gates (from model_comparison_report):
    - Balanced accuracy: new >= current - 5%
    - Orange recall: new >= current - 5%
    - Orange precision: new >= current - 10%

    Raises:
        Failure: If approve_deployment=False (prevents downstream deployment)

    Returns:
        Dict with approval metadata
    """
    context.log.info("=" * 60)
    context.log.info("DEPLOYMENT APPROVAL GATE")
    context.log.info("=" * 60)

    approve = context.op_config.get("approve_deployment", False)
    recommendation = model_comparison_report['approval_recommended']

    context.log.info(f"\n📊 COMPARISON REPORT SUMMARY:")
    context.log.info(f"  Validation Samples: {model_comparison_report['validation_samples']}")
    context.log.info(f"  Automated Recommendation: {'✓ APPROVE' if recommendation else '✗ REJECT'}")

    # Show quality gates
    context.log.info(f"\n🚦 QUALITY GATES:")
    for gate_name, gate_info in model_comparison_report['comparison_details']['quality_gates'].items():
        status = "✓ PASS" if gate_info['passed'] else "✗ FAIL"
        context.log.info(
            f"  {status} - {gate_name}: {gate_info['actual']:.3f} "
            f"(threshold: {gate_info['threshold']:.3f})"
        )

    # Show metric deltas
    deltas = model_comparison_report['comparison_details']['metric_differences']
    context.log.info(f"\n📈 METRIC CHANGES (New - Current):")
    context.log.info(f"  Balanced Accuracy: {deltas['balanced_accuracy']:+.3f}")
    context.log.info(f"  Orange Recall:     {deltas['recall_orange']:+.3f}")
    context.log.info(f"  Orange Precision:  {deltas['precision_orange']:+.3f}")

    # Manual approval decision
    context.log.info("\n" + "=" * 60)
    if not approve:
        context.log.info("✗ DEPLOYMENT NOT APPROVED")
        context.log.info("=" * 60)
        context.log.info("\n⚠️  To approve deployment, rematerialize this asset with:")
        context.log.info("   approve_deployment: true")
        context.log.info("\n📋 Review comparison report at:")
        context.log.info(f"   s3://{model_comparison_report.get('visualization_path', 'N/A')}")

        raise dg.Failure(
            description="Deployment not approved. Set approve_deployment=True to proceed.",
            metadata={
                "automated_recommendation": recommendation,
                "approval_required": True,
                "comparison_report_path": model_comparison_report.get('visualization_path', 'N/A'),
            }
        )

    context.log.info("✓ DEPLOYMENT APPROVED")
    context.log.info("=" * 60)

    approval_metadata = {
        'approved_at': datetime.now().isoformat(),
        'approved_by': 'manual',
        'automated_recommendation': recommendation,
        'validation_period': model_comparison_report['validation_period'],
        'variant': model_comparison_report['variant'],
        'new_model_metrics': model_comparison_report['new_model_metrics'],
        'quality_gates_passed': all(
            gate['passed']
            for gate in model_comparison_report['comparison_details']['quality_gates'].values()
        ),
    }

    context.add_output_metadata({
        "approved": True,
        "approved_at": approval_metadata['approved_at'],
        "automated_recommendation": recommendation,
        "balanced_acc_delta": float(deltas['balanced_accuracy']),
        "orange_recall_delta": float(deltas['recall_orange']),
    })

    return approval_metadata



@dg.asset(
    key_prefix="h2s",
    group_name="h2s_model_deployment",
    required_resource_keys={"s3"},
    kinds={"ml", "deployment"},
    description="Deploy new trained model to production S3 paths",
    partitions_def=model_run_partitions,
    ins={
        "deployment_approval": dg.AssetIn(key=_KEY("deployment_approval")),
    },
)
def production_model_deployment(
    context: dg.AssetExecutionContext,
    deployment_approval: Dict,
) -> Dict:
    """Deploy newly trained model to production S3 paths.

    Loads the trained model from its S3 training path (saved by model_training_metrics)
    and copies it to the production path. To roll back, re-run approve_and_deploy_job
    for a previous month's partition.

    Replaces:
    - nestor_xgboost_weighted_model.json
    - deployment_metadata.json (new deployment record)

    Returns:
        Dict with deployment status and metadata
    """
    context.log.info("=" * 60)
    context.log.info("DEPLOYING NEW MODEL TO PRODUCTION")
    context.log.info("=" * 60)

    s3_resource = context.resources.s3
    timestamp = datetime.now().strftime("%Y_%m_%d_%H%M%S")

    # Resolve source model path from approval metadata
    validation_period = deployment_approval['validation_period']
    variant = deployment_approval['variant']
    source_model_path = f"{TRAINING_PATH}/{validation_period}/{variant}/model.json"

    # Production paths
    prod_model_path = f"{MODEL_PATH}/nestor_xgboost_weighted_model.json"
    prod_prep_path = f"{MODEL_PATH}/nestor_preprocessing_info.json"
    deployment_metadata_path = f"{MODEL_PATH}/deployment_metadata.json"

    # === LOAD NEW MODEL FROM S3 TRAINING PATH ===
    # Try XGBoost format first, then sklearn joblib
    for candidate_filename in ("model.json", "model.joblib"):
        source_model_path = f"{TRAINING_PATH}/{validation_period}/{variant}/{candidate_filename}"
        try:
            new_model_bytes = s3_resource.getFile(source_model_path, bucket=s3_resource.S3_BUCKET)
            context.log.info(f"✓ Loaded model from {source_model_path} ({len(new_model_bytes) / 1024:.1f} KB)")
            break
        except Exception:
            continue
    else:
        raise dg.Failure(f"Could not find trained model at {TRAINING_PATH}/{validation_period}/{variant}/ (tried model.json and model.joblib)")

    # === DEPLOY NEW MODEL ===
    try:
        # Upload new model to production path
        s3_resource.putFile(
            new_model_bytes,
            prod_model_path,
            bucket=s3_resource.S3_BUCKET,
            content_type='application/json'
        )

        context.log.info(f"✓ Deployed new model to {prod_model_path}")

        # Also write to variant-specific production path for multi-model forecast
        variant_model_path = f"{MODEL_PATH}/{deployment_approval['variant']}/{candidate_filename}"
        s3_resource.putFile(
            new_model_bytes,
            variant_model_path,
            bucket=s3_resource.S3_BUCKET,
            content_type='application/json'
        )
        context.log.info(f"✓ Deployed variant model to {variant_model_path}")

        # NOTE: preprocessing_info.json stays the same (same 20 features)
        # Only the trained model changes

    except Exception as e:
        context.log.error(f"✗ Deployment failed: {e}")
        context.log.error(f"⚠️  To roll back, re-run approve_and_deploy_job for a previous partition")
        raise dg.Failure(f"Model deployment failed: {e}")

    # === CREATE DEPLOYMENT METADATA ===
    deployment_metadata = {
        'deployed_at': datetime.now().isoformat(),
        'deployment_id': timestamp,
        'model_path': prod_model_path,
        'preprocessing_path': prod_prep_path,
        'source_model_path': f"{TRAINING_PATH}/{validation_period}/{variant}/{candidate_filename}",
        'approval_metadata': deployment_approval,
        'deployment_status': 'success',
    }

    # Upload deployment metadata
    s3_resource.putFile(
        json.dumps(deployment_metadata, indent=2).encode('utf-8'),
        deployment_metadata_path,
        bucket=s3_resource.S3_BUCKET,
        content_type='application/json'
    )

    context.log.info(f"✓ Uploaded deployment metadata to {deployment_metadata_path}")

    context.log.info("\n" + "=" * 60)
    context.log.info("✓ DEPLOYMENT COMPLETE")
    context.log.info("=" * 60)
    context.log.info(f"\n📦 New model deployed to: {prod_model_path}")
    context.log.info(f"📋 Deployment metadata: {deployment_metadata_path}")
    context.log.info(f"↩️  To roll back, re-run approve_and_deploy_job for a previous partition")

    context.add_output_metadata({
        "deployed_at": deployment_metadata['deployed_at'],
        "deployment_id": timestamp,
        "model_path": prod_model_path,
        "deployment_status": "success",
    })

    return deployment_metadata
