"""Multi-Station H2S Model Training Pipeline.

Replaces the single-model monthly training pipeline with per-station,
per-task auto-selected models (RF vs XGBoost vs Ensemble).

Produces 9 pickle files: 3 stations × 3 tasks (regression, >5ppb, >10ppb).
Uploaded to S3 at: tijuana/forecast/models/stations/{station_key}/{task}.pkl
"""

import io
import json
import pickle
from datetime import datetime, timezone

import dagster as dg
import numpy as np
import pandas as pd

from h2s.training.multi_station_trainer import (
    MODEL_FEATURES,
    STATION_PARTITION_MAP,
    STATIONS,
    TRAIN_FRACTION,
    prepare_multi_station_features,
    train_and_select,
)

STATION_PARTITIONS = dg.StaticPartitionsDefinition(
    partition_keys=list(STATION_PARTITION_MAP.keys())  # san_ysidro, nestor_bes, ib_civic_ctr
)

STATION_MODELS_S3_BASE = "tijuana/forecast/models/stations"

_KEY = lambda name: dg.AssetKey(["h2s", name])


# ==============================================================================
# Asset 1: Load and prepare training data (unpartitioned — shared across stations)
# ==============================================================================

@dg.asset(
    key_prefix="h2s",
    group_name="h2s_multi_station_training",
    required_resource_keys={"s3"},
    kinds={"python", "s3"},
    description="Multi-station training dataset loaded from S3, filtered and feature-engineered",
    config_schema={
        "local_fallback_path": dg.Field(
            str,
            default_value="data/modeldata_h2s_nofill.parquet",
            description="Local fallback path when S3 is unavailable (relative to project root)",
        ),
        "s3_bucket": dg.Field(
            str,
            default_value="resilentpublic",
            description="S3 bucket for training data (resilentpublic or test)",
        ),
    },
)
def multi_station_training_data(context: dg.AssetExecutionContext) -> pd.DataFrame:
    """Load training parquet from S3, filter to measured rows, engineer features."""
    s3 = context.resources.s3
    local_path = context.op_config["local_fallback_path"]
    bucket = context.op_config["s3_bucket"]
    s3_path = "latest/tijuana/forecast_data/modeldata_h2s_nofill.parquet"

    raw_df = None
    try:
        stream = s3.get_stream(path=s3_path, bucket=bucket)
        raw_df = pd.read_parquet(stream)
        context.log.info(f"✓ Loaded training data from S3 ({bucket}/{s3_path}): {len(raw_df)} rows")
    except Exception as e:
        context.log.warning(f"S3 load failed ({e}), falling back to local: {local_path}")
        raw_df = pd.read_parquet(local_path)
        context.log.info(f"✓ Loaded from local file: {len(raw_df)} rows, {len(raw_df.columns)} cols")

    df = prepare_multi_station_features(raw_df)

    context.log.info(f"✓ Feature engineering complete: {len(df)} clean rows")
    for site in df['site_name'].unique():
        ss = df[df['site_name'] == site]
        context.log.info(f"  {site}: {len(ss)} rows, >5ppb={ss['exceed_5'].mean()*100:.1f}%")

    context.add_output_metadata({
        "row_count": len(df),
        "stations": list(df['site_name'].unique()),
        "features": len(MODEL_FEATURES),
        "date_min": str(df['time'].min()),
        "date_max": str(df['time'].max()),
    })
    return df


# ==============================================================================
# Asset 2: Train per-station models (partitioned by station)
# ==============================================================================

@dg.asset(
    key_prefix="h2s",
    group_name="h2s_multi_station_training",
    partitions_def=STATION_PARTITIONS,
    kinds={"python", "ml"},
    description="Auto-trained models for one station: regression + >5ppb + >10ppb classifiers",
    ins={"multi_station_training_data": dg.AssetIn(key=_KEY("multi_station_training_data"))},
    config_schema={
        "ensemble_margin": dg.Field(
            float,
            default_value=0.01,
            description="AUC margin for ensembling classifiers (R² margin = 2×)",
        ),
    },
)
def per_station_trained_models(
    context: dg.AssetExecutionContext,
    multi_station_training_data: pd.DataFrame,
) -> dict:
    """Train regression + classifier models for the current station partition.

    Returns dict of {task_name: model_object} for the current station.
    """
    partition = context.partition_key  # e.g. 'san_ysidro'
    site_name = STATION_PARTITION_MAP[partition]
    ensemble_margin = context.op_config["ensemble_margin"]

    context.log.info(f"Training models for station: {site_name} (partition: {partition})")

    sdf = multi_station_training_data[
        multi_station_training_data['site_name'] == site_name
    ].copy().sort_values('time').reset_index(drop=True)

    if len(sdf) < 100:
        raise ValueError(f"Insufficient data for {site_name}: {len(sdf)} rows")

    X = sdf[MODEL_FEATURES].values
    y_cont = sdf['H2S'].values
    y_5 = sdf['exceed_5'].values
    y_10 = sdf['exceed_10'].values

    split = int(len(sdf) * TRAIN_FRACTION)
    Xtr, Xte = X[:split], X[split:]
    ytr_c, yte_c = y_cont[:split], y_cont[split:]
    ytr_5, yte_5 = y_5[:split], y_5[split:]
    ytr_10, yte_10 = y_10[:split], y_10[split:]

    context.log.info(f"  Records: {len(sdf):,} (train: {split:,}, test: {len(sdf)-split:,})")
    context.log.info(f"  Exceedance: >5={y_5.mean()*100:.1f}%, >10={y_10.mean()*100:.1f}%")

    models = {}
    task_defs = [
        ('regression', Xtr, Xte, ytr_c, yte_c),
        ('clf_5ppb',   Xtr, Xte, ytr_5, yte_5),
        ('clf_10ppb',  Xtr, Xte, ytr_10, yte_10),
    ]
    report_tasks = {}

    for task, Xtr_, Xte_, ytr_, yte_ in task_defs:
        context.log.info(f"  Training {task}...")
        model, choice, metrics = train_and_select(
            Xtr_, Xte_, ytr_, yte_, task, ensemble_margin=ensemble_margin
        )
        models[task] = model
        report_tasks[task] = {k: v for k, v in metrics.items() if k != 'feature_importance'}
        context.log.info(f"    Selected: {choice}")

    context.add_output_metadata({
        "station": site_name,
        "partition": partition,
        "n_train": int(split),
        "n_test": int(len(sdf) - split),
        "tasks": list(models.keys()),
        "algorithm_choices": {t: report_tasks[t].get('selected', '?') for t in report_tasks},
    })
    return models


# ==============================================================================
# Asset 3: Station training report (partitioned by station)
# ==============================================================================

@dg.asset(
    key_prefix="h2s",
    group_name="h2s_multi_station_training",
    partitions_def=STATION_PARTITIONS,
    required_resource_keys={"s3"},
    kinds={"json", "s3"},
    description="JSON training metrics report for one station, uploaded to S3",
    ins={
        "multi_station_training_data": dg.AssetIn(key=_KEY("multi_station_training_data")),
        "per_station_trained_models": dg.AssetIn(key=_KEY("per_station_trained_models")),
    },
    config_schema={
        "ensemble_margin": dg.Field(float, default_value=0.01),
    },
)
def station_training_report(
    context: dg.AssetExecutionContext,
    multi_station_training_data: pd.DataFrame,
    per_station_trained_models: dict,
) -> dict:
    """Generate and upload training metrics report for the current station."""
    from h2s.training.multi_station_trainer import eval_regressor, eval_classifier, get_feature_importance

    partition = context.partition_key
    site_name = STATION_PARTITION_MAP[partition]
    s3 = context.resources.s3
    ensemble_margin = context.op_config["ensemble_margin"]

    sdf = multi_station_training_data[
        multi_station_training_data['site_name'] == site_name
    ].sort_values('time').reset_index(drop=True)

    split = int(len(sdf) * TRAIN_FRACTION)
    X = sdf[MODEL_FEATURES].values
    Xte = X[split:]
    yte_c = sdf['H2S'].values[split:]
    yte_5 = sdf['exceed_5'].values[split:]
    yte_10 = sdf['exceed_10'].values[split:]

    tasks_metrics = {}
    for task, yte in [('regression', yte_c), ('clf_5ppb', yte_5), ('clf_10ppb', yte_10)]:
        model = per_station_trained_models[task]
        if task == 'regression':
            m = eval_regressor(model, Xte, yte)
        else:
            m = eval_classifier(model, Xte, yte)
        tasks_metrics[task] = {
            **m,
            'feature_importance': get_feature_importance(model),
        }

    report = {
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'station': site_name,
        'partition': partition,
        'n_records': len(sdf),
        'n_train': split,
        'n_test': len(sdf) - split,
        'features': MODEL_FEATURES,
        'ensemble_margin': ensemble_margin,
        'tasks': tasks_metrics,
    }

    # Upload to S3
    station_key = STATIONS[site_name]['key']
    report_path = f"{STATION_MODELS_S3_BASE}/{station_key}/training_report.json"
    try:
        report_bytes = json.dumps(report, indent=2, default=str).encode('utf-8')
        s3.putFile(report_bytes, report_path, bucket=s3.S3_BUCKET, content_type='application/json')
        context.log.info(f"✓ Uploaded training report to S3: {report_path}")
    except Exception as e:
        context.log.warning(f"Could not upload report to S3: {e}")

    context.add_output_metadata({
        "station": site_name,
        "regression_r2": tasks_metrics['regression'].get('R2'),
        "clf5_auc": tasks_metrics['clf_5ppb'].get('AUC'),
        "clf10_auc": tasks_metrics['clf_10ppb'].get('AUC'),
    })
    return report


# ==============================================================================
# Asset 4: Model deployment gate (manual approval + S3 upload)
# ==============================================================================

@dg.asset(
    key_prefix="h2s",
    group_name="h2s_multi_station_training",
    partitions_def=STATION_PARTITIONS,
    required_resource_keys={"s3"},
    kinds={"python", "s3"},
    description="Manual approval gate → upload station models to S3 production path",
    ins={"per_station_trained_models": dg.AssetIn(key=_KEY("per_station_trained_models"))},
    config_schema={
        "approve_deployment": dg.Field(
            bool,
            default_value=False,
            description="Set to True to approve and upload models to S3",
        ),
    },
)
def station_model_deployment(
    context: dg.AssetExecutionContext,
    per_station_trained_models: dict,
) -> dict:
    """Upload trained station models to S3 when deployment is approved.

    Set approve_deployment=True in asset config to upload.
    Models are written to: tijuana/forecast/models/stations/{station_key}/{task}.pkl
    """
    partition = context.partition_key
    site_name = STATION_PARTITION_MAP[partition]
    s3 = context.resources.s3
    approved = context.op_config["approve_deployment"]

    station_key = STATIONS[site_name]['key']
    base_path = f"{STATION_MODELS_S3_BASE}/{station_key}"

    if not approved:
        context.log.warning(
            f"Deployment NOT approved for {site_name}. "
            f"Set approve_deployment=True to upload models."
        )
        return {"status": "pending_approval", "station": site_name}

    context.log.info(f"Deploying models for {site_name} to S3: {base_path}")
    uploaded = {}
    for task, model in per_station_trained_models.items():
        s3_path = f"{base_path}/{task}.pkl"
        model_bytes = pickle.dumps(model)
        s3.putFile(model_bytes, s3_path, bucket=s3.S3_BUCKET, content_type='application/octet-stream')
        context.log.info(f"  ✓ Uploaded {task} → {s3_path}")
        uploaded[task] = s3_path

    # Write deployment metadata
    meta = {
        'deployed_at': datetime.now(timezone.utc).isoformat(),
        'station': site_name,
        'partition': partition,
        'models': uploaded,
        'features': MODEL_FEATURES,
    }
    meta_path = f"{base_path}/deployment_metadata.json"
    s3.putFile(
        json.dumps(meta, indent=2).encode('utf-8'),
        meta_path,
        bucket=s3.S3_BUCKET,
        content_type='application/json',
    )

    context.add_output_metadata({
        "status": "deployed",
        "station": site_name,
        "models_uploaded": list(uploaded.keys()),
        "s3_base_path": base_path,
    })
    return {"status": "deployed", "station": site_name, "models": uploaded}


# ==============================================================================
# Job definitions
# ==============================================================================

multi_station_training_job = dg.define_asset_job(
    name="multi_station_training_job",
    description="Train per-station H2S models for all stations",
    selection=dg.AssetSelection.assets(
        multi_station_training_data,
        per_station_trained_models,
        station_training_report,
    ),
    partitions_def=STATION_PARTITIONS,
    tags={"environment": "production", "pipeline": "h2s_multi_station_training"},
)

station_deployment_job = dg.define_asset_job(
    name="station_deployment_job",
    description="Deploy approved station models to S3 (set approve_deployment=True)",
    selection=dg.AssetSelection.assets(station_model_deployment),
    partitions_def=STATION_PARTITIONS,
    config={
        "ops": {
            "h2s__station_model_deployment": {
                "config": {"approve_deployment": True}
            }
        }
    },
    tags={"environment": "production", "pipeline": "h2s_deployment"},
)
