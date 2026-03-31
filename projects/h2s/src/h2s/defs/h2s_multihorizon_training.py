"""Multi-Horizon H2S Model Training Pipeline.

Trains 4 horizons × 3 stations × 3 tasks = 36 models.
Each horizon uses features that honestly reflect what's known at that lead time.

Models are stored as pickle files at:
  tijuana/forecast/models/multihorizon/{horizon}/{station_key}/{task}.pkl
"""

import io
import json
import pickle
from datetime import datetime, timezone

import dagster as dg
import numpy as np
import pandas as pd

from h2s.constants import MH_MODELS_S3_BASE
from h2s.training.multi_station_trainer import (
    eval_classifier,
    eval_regressor,
    get_feature_importance,
    train_and_select,
    TRAIN_FRACTION,
)
from h2s.training.multihorizon_trainer import (
    BASE_FEATURES,
    HORIZONS,
    HORIZON_NAMES,
    STATION_PARTITION_MAP,
    STATIONS,
    TASKS,
    build_horizon_features,
)

STATION_PARTITIONS = dg.StaticPartitionsDefinition(
    partition_keys=list(STATION_PARTITION_MAP.keys())
)

_KEY = lambda name: dg.AssetKey(["h2s", name])


# ==============================================================================
# Asset 1: Load and prepare training data (unpartitioned — shared across stations)
# ==============================================================================

@dg.asset(
    key_prefix="h2s",
    group_name="h2s_mh_training",
    required_resource_keys={"s3"},
    kinds={"python", "s3"},
    description="Multi-horizon training dataset loaded from S3 (pre-featurized parquet)",
    config_schema={
        "s3_bucket": dg.Field(
            str,
            default_value="resilentpublic",
            description="S3 bucket for training data",
        ),
    },
)
def mh_training_data(context: dg.AssetExecutionContext) -> pd.DataFrame:
    """Load training parquet from S3, filter to measured rows, add targets.

    The parquet already contains all 36 base features (pre-featurized upstream).
    We only filter, clean, and add target columns.
    """
    s3 = context.resources.s3
    bucket = context.op_config["s3_bucket"]
    s3_path = "latest/tijuana/forecast_data/modeldata_h2s_nofill.parquet"

    parquet_url = s3.get_presigned_url(path=s3_path, bucket=bucket)
    raw_df = pd.read_parquet(parquet_url)
    context.log.info(f"Loaded training data from S3 ({bucket}/{s3_path}): {len(raw_df)} rows")

    # Filter and clean
    df = raw_df.copy()
    df['time'] = pd.to_datetime(df['time'], utc=True)
    df = df[(df['h2s_measured'] == True) & (df['H2S'] <= 500)].copy()
    df = df.sort_values(['site_name', 'time']).reset_index(drop=True)
    df['H2S'] = df['H2S'].clip(lower=0)

    # Add targets
    df['exceed_5'] = (df['H2S'] > 5).astype(int)
    df['exceed_10'] = (df['H2S'] > 10).astype(int)

    context.log.info(f"Cleaned: {len(df)} rows")
    for site in df['site_name'].unique():
        ss = df[df['site_name'] == site]
        context.log.info(f"  {site}: {len(ss)} rows, >5ppb={ss['exceed_5'].mean()*100:.1f}%")

    context.add_output_metadata({
        "row_count": len(df),
        "stations": list(df['site_name'].unique()),
        "base_features": len(BASE_FEATURES),
        "date_min": str(df['time'].min()),
        "date_max": str(df['time'].max()),
    })
    return df


# ==============================================================================
# Asset 2: Train per-station multi-horizon models (partitioned by station)
# ==============================================================================

@dg.asset(
    key_prefix="h2s",
    group_name="h2s_mh_training",
    partitions_def=STATION_PARTITIONS,
    kinds={"python", "ml"},
    description="Train 12 models per station: 4 horizons × 3 tasks (regression + >5ppb + >10ppb)",
    ins={"mh_training_data": dg.AssetIn(key=_KEY("mh_training_data"))},
    config_schema={
        "ensemble_margin": dg.Field(
            float,
            default_value=0.01,
            description="AUC margin for ensembling (R² margin = 2×)",
        ),
    },
)
def mh_trained_models(
    context: dg.AssetExecutionContext,
    mh_training_data: pd.DataFrame,
) -> dict:
    """Train regression + classifier models for all 4 horizons at one station.

    Returns dict with:
      - 'models': {horizon: {task: model_object}}
      - 'horizon_features': {horizon: [feature_col_names]}
    """
    partition = context.partition_key
    site_name = STATION_PARTITION_MAP[partition]
    ensemble_margin = context.op_config["ensemble_margin"]

    context.log.info(f"Training MH models for {site_name} (partition: {partition})")

    sdf = mh_training_data[
        mh_training_data['site_name'] == site_name
    ].copy().sort_values('time').reset_index(drop=True)

    if len(sdf) < 100:
        raise ValueError(f"Insufficient data for {site_name}: {len(sdf)} rows")

    all_models = {}
    all_features = {}
    all_metrics = {}

    for hz_name in HORIZON_NAMES:
        hz_cfg = HORIZONS[hz_name]
        context.log.info(f"  Horizon {hz_name}: {hz_cfg['description']}")

        # Build horizon-specific features on top of pre-featurized data
        hz_df, feature_cols = build_horizon_features(sdf, hz_name, hz_cfg)
        hz_df = hz_df.dropna(subset=feature_cols).reset_index(drop=True)

        if len(hz_df) < 100:
            context.log.warning(f"    Only {len(hz_df)} rows after dropna, skipping")
            continue

        X = hz_df[feature_cols].values
        y_cont = hz_df['H2S'].values
        y_5 = hz_df['exceed_5'].values
        y_10 = hz_df['exceed_10'].values

        split = int(len(hz_df) * TRAIN_FRACTION)
        Xtr, Xte = X[:split], X[split:]

        context.log.info(f"    {len(hz_df)} rows (train:{split}, test:{len(hz_df)-split}), {len(feature_cols)} features")

        hz_models = {}
        hz_metrics = {}
        task_defs = [
            ('regression', y_cont[:split], y_cont[split:]),
            ('clf_5ppb',   y_5[:split],    y_5[split:]),
            ('clf_10ppb',  y_10[:split],   y_10[split:]),
        ]

        for task, ytr, yte in task_defs:
            context.log.info(f"    Training {task}...")
            model, choice, metrics = train_and_select(
                Xtr, Xte, ytr, yte, task, ensemble_margin=ensemble_margin
            )
            hz_models[task] = model
            hz_metrics[task] = {k: v for k, v in metrics.items() if k != 'feature_importance'}
            context.log.info(f"      Selected: {choice}")

        all_models[hz_name] = hz_models
        all_features[hz_name] = feature_cols
        all_metrics[hz_name] = hz_metrics

    context.add_output_metadata({
        "station": site_name,
        "partition": partition,
        "horizons_trained": list(all_models.keys()),
        "models_count": sum(len(m) for m in all_models.values()),
        "algorithm_choices": {
            hz: {t: all_metrics[hz][t].get('selected', '?') for t in all_metrics[hz]}
            for hz in all_metrics
        },
    })

    return {"models": all_models, "horizon_features": all_features}


# ==============================================================================
# Asset 3: Training report (partitioned by station)
# ==============================================================================

@dg.asset(
    key_prefix="h2s",
    group_name="h2s_mh_training",
    partitions_def=STATION_PARTITIONS,
    required_resource_keys={"s3"},
    kinds={"json", "s3"},
    description="JSON training metrics report for one station's MH models, uploaded to S3",
    ins={
        "mh_training_data": dg.AssetIn(key=_KEY("mh_training_data")),
        "mh_trained_models": dg.AssetIn(key=_KEY("mh_trained_models")),
    },
)
def mh_training_report(
    context: dg.AssetExecutionContext,
    mh_training_data: pd.DataFrame,
    mh_trained_models: dict,
) -> dict:
    """Generate and upload training metrics report for one station's MH models."""
    partition = context.partition_key
    site_name = STATION_PARTITION_MAP[partition]
    s3 = context.resources.s3
    station_key = STATIONS[site_name]['key']

    models = mh_trained_models['models']
    horizon_features = mh_trained_models['horizon_features']

    sdf = mh_training_data[
        mh_training_data['site_name'] == site_name
    ].sort_values('time').reset_index(drop=True)

    report = {
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'station': site_name,
        'partition': partition,
        'n_records': len(sdf),
        'horizons': {},
    }

    for hz_name in models:
        hz_cfg = HORIZONS[hz_name]
        hz_df, feature_cols = build_horizon_features(sdf, hz_name, hz_cfg)
        hz_df = hz_df.dropna(subset=feature_cols).reset_index(drop=True)

        split = int(len(hz_df) * TRAIN_FRACTION)
        X = hz_df[feature_cols].values
        Xte = X[split:]
        yte_c = hz_df['H2S'].values[split:]
        yte_5 = hz_df['exceed_5'].values[split:]
        yte_10 = hz_df['exceed_10'].values[split:]

        tasks_metrics = {}
        for task, yte in [('regression', yte_c), ('clf_5ppb', yte_5), ('clf_10ppb', yte_10)]:
            if task not in models[hz_name]:
                continue
            model = models[hz_name][task]
            if task == 'regression':
                m = eval_regressor(model, Xte, yte)
            else:
                m = eval_classifier(model, Xte, yte)
            fi = get_feature_importance(model)
            # Remap feature importance to horizon feature names
            fi_named = {}
            imp = getattr(model, 'feature_importances_', None)
            if imp is not None:
                idx = np.argsort(imp)[::-1][:10]
                fi_named = {feature_cols[i]: round(float(imp[i]), 4) for i in idx if i < len(feature_cols)}
            tasks_metrics[task] = {**m, 'feature_importance': fi_named}

        report['horizons'][hz_name] = {
            'n_features': len(feature_cols),
            'features': feature_cols,
            'n_train': split,
            'n_test': len(hz_df) - split,
            'tasks': tasks_metrics,
        }

    # Upload to S3
    report_path = f"{MH_MODELS_S3_BASE}/{station_key}/training_report.json"
    try:
        report_bytes = json.dumps(report, indent=2, default=str).encode('utf-8')
        s3.putFile(report_bytes, report_path, bucket=s3.S3_BUCKET, content_type='application/json')
        context.log.info(f"Uploaded training report to S3: {report_path}")
    except Exception as e:
        context.log.warning(f"Could not upload report to S3: {e}")

    # Summary metadata for Dagster UI
    meta = {"station": site_name}
    for hz in report['horizons']:
        tasks = report['horizons'][hz]['tasks']
        if 'regression' in tasks:
            meta[f"{hz}_regression_r2"] = tasks['regression'].get('R2')
        if 'clf_5ppb' in tasks:
            meta[f"{hz}_clf5_auc"] = tasks['clf_5ppb'].get('AUC')
    context.add_output_metadata(meta)

    return report


# ==============================================================================
# Asset 4: Model deployment gate (manual approval + S3 upload)
# ==============================================================================

@dg.asset(
    key_prefix="h2s",
    group_name="h2s_mh_training",
    partitions_def=STATION_PARTITIONS,
    required_resource_keys={"s3"},
    kinds={"python", "s3"},
    description="Manual approval gate -> upload MH station models to S3",
    ins={"mh_trained_models": dg.AssetIn(key=_KEY("mh_trained_models"))},
    config_schema={
        "approve_deployment": dg.Field(
            bool,
            default_value=False,
            description="Set to True to approve and upload models to S3",
        ),
    },
)
def mh_model_deployment(
    context: dg.AssetExecutionContext,
    mh_trained_models: dict,
) -> dict:
    """Upload trained MH models to S3 when deployment is approved.

    Models are written to: multihorizon/{horizon}/{station_key}/{task}.pkl
    Also writes horizon_features.json and deployment_metadata.json.
    """
    partition = context.partition_key
    site_name = STATION_PARTITION_MAP[partition]
    s3 = context.resources.s3
    approved = context.op_config["approve_deployment"]
    station_key = STATIONS[site_name]['key']

    if not approved:
        context.log.warning(
            f"Deployment NOT approved for {site_name}. "
            f"Set approve_deployment=True to upload models."
        )
        return {"status": "pending_approval", "station": site_name}

    models = mh_trained_models['models']
    horizon_features = mh_trained_models['horizon_features']

    context.log.info(f"Deploying MH models for {site_name} to S3")
    uploaded = {}
    for hz_name, hz_models in models.items():
        for task, model in hz_models.items():
            s3_path = f"{MH_MODELS_S3_BASE}/{hz_name}/{station_key}/{task}.pkl"
            model_bytes = pickle.dumps(model)
            s3.putFile(model_bytes, s3_path, bucket=s3.S3_BUCKET, content_type='application/octet-stream')
            context.log.info(f"  Uploaded {hz_name}/{task} -> {s3_path}")
            uploaded[f"{hz_name}/{task}"] = s3_path

    # Upload horizon features for this station
    feat_path = f"{MH_MODELS_S3_BASE}/{station_key}/horizon_features.json"
    feat_bytes = json.dumps(horizon_features, indent=2).encode('utf-8')
    s3.putFile(feat_bytes, feat_path, bucket=s3.S3_BUCKET, content_type='application/json')
    context.log.info(f"  Uploaded horizon_features.json -> {feat_path}")

    # Deployment metadata
    meta = {
        'deployed_at': datetime.now(timezone.utc).isoformat(),
        'station': site_name,
        'partition': partition,
        'models': uploaded,
        'horizons': list(models.keys()),
    }
    meta_path = f"{MH_MODELS_S3_BASE}/{station_key}/deployment_metadata.json"
    s3.putFile(
        json.dumps(meta, indent=2).encode('utf-8'),
        meta_path,
        bucket=s3.S3_BUCKET,
        content_type='application/json',
    )

    context.add_output_metadata({
        "status": "deployed",
        "station": site_name,
        "models_uploaded": len(uploaded),
        "horizons": list(models.keys()),
    })
    return {"status": "deployed", "station": site_name, "models": uploaded}


# ==============================================================================
# Job definitions
# ==============================================================================

mh_training_job = dg.define_asset_job(
    name="mh_training_job",
    description="Train multi-horizon H2S models for all stations (4 horizons x 3 tasks per station)",
    selection=dg.AssetSelection.assets(
        mh_training_data,
        mh_trained_models,
        mh_training_report,
    ),
    partitions_def=STATION_PARTITIONS,
    tags={"environment": "production", "pipeline": "h2s_mh_training"},
)

mh_deployment_job = dg.define_asset_job(
    name="mh_deployment_job",
    description="Deploy approved MH models to S3 (set approve_deployment=True)",
    selection=dg.AssetSelection.assets(mh_model_deployment),
    partitions_def=STATION_PARTITIONS,
    config={
        "ops": {
            "h2s__mh_model_deployment": {
                "config": {"approve_deployment": True}
            }
        }
    },
    tags={"environment": "production", "pipeline": "h2s_mh_deployment"},
)
