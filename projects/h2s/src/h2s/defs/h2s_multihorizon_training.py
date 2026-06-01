"""Multi-Horizon H2S Model Training Pipeline.

Trains 4 horizons × 3 stations × 4 tasks = 48 models. Each horizon uses
features that honestly reflect what's known at that lead time.

Models are streamed to S3 staging during training (one pickle at a time) and
promoted to their final paths via server-side copy on deployment:
  tijuana/forecast/models/multihorizon/{horizon}/{station_key}/{task}.pkl
"""

import gc
import json
import pickle
from datetime import datetime, timezone

import dagster as dg
import numpy as np
import pandas as pd
from minio.commonconfig import CopySource

from h2s.constants import MH_MODELS_S3_BASE, MH_STAGING_S3_BASE
from h2s.training.multi_station_trainer import (
    eval_classifier,
    eval_regressor,
    train_and_select,
    TRAIN_FRACTION,
)
from h2s.training.multihorizon_trainer import (
    BASE_FEATURES,
    HORIZONS,
    HORIZON_NAMES,
    STATION_PARTITION_MAP,
    STATIONS,
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

    parquet_url = s3.publicUrl(path=s3_path, bucket=bucket)
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
    df['exceed_30'] = (df['H2S'] > 30).astype(int)

    context.log.info(f"Cleaned: {len(df)} rows")
    for site in df['site_name'].unique():
        ss = df[df['site_name'] == site]
        context.log.info(
            f"  {site}: {len(ss)} rows, "
            f">5ppb={ss['exceed_5'].mean()*100:.1f}%, "
            f">30ppb={ss['exceed_30'].mean()*100:.1f}%"
        )

    context.add_output_metadata({
        "row_count": len(df),
        "stations": list(df['site_name'].unique()),
        "base_features": len(BASE_FEATURES),
        "date_min": str(df['time'].min()),
        "date_max": str(df['time'].max()),
        "exceed_30_pct": round(float(df['exceed_30'].mean()) * 100, 2),
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
    description="Train 16 models per station: 4 horizons × 4 tasks (regression + >5ppb + >10ppb + >30ppb)",
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

    Each model is pickled and uploaded to a per-run S3 staging prefix as soon
    as it is trained, then dropped from memory. Only metadata (S3 paths,
    eval metrics, feature lists) is returned — keeping the asset output small
    so Dagster's IO manager doesn't pin all 16 trained models in RAM.
    """
    partition = context.partition_key
    site_name = STATION_PARTITION_MAP[partition]
    station_key = STATIONS[site_name]['key']
    ensemble_margin = context.op_config["ensemble_margin"]
    s3 = context.resources.s3
    run_id = context.run_id
    staging_prefix = f"{MH_STAGING_S3_BASE}/{run_id}/{station_key}"

    context.log.info(f"Training MH models for {site_name} (partition: {partition}, run_id: {run_id})")
    context.log.info(f"Staging prefix: s3://{s3.S3_BUCKET}/{staging_prefix}")

    sdf = mh_training_data[
        mh_training_data['site_name'] == site_name
    ].copy().sort_values('time').reset_index(drop=True)

    if len(sdf) < 100:
        raise ValueError(f"Insufficient data for {site_name}: {len(sdf)} rows")

    staging_paths: dict = {}
    all_features: dict = {}
    all_metrics: dict = {}
    all_choices: dict = {}
    all_splits: dict = {}

    for hz_name in HORIZON_NAMES:
        hz_cfg = HORIZONS[hz_name]
        context.log.info(f"  Horizon {hz_name}: {hz_cfg['description']}")

        # Build long-format (origin, lead_hour) training rows. Targets are
        # H2S(origin + lead_hour) — i.e. an honest forecast at lead `h`.
        hz_df, feature_cols, targets = build_horizon_features(sdf, hz_name, hz_cfg)

        if len(hz_df) < 100:
            context.log.warning(f"    Only {len(hz_df)} rows after dropna, skipping")
            del hz_df, targets
            gc.collect()
            continue

        # Split by origin time so (origin, h=6) and (origin, h=7) never end up
        # on opposite sides of the split.
        unique_origins = np.sort(hz_df['origin_time'].unique())
        n_train_origins = max(1, int(len(unique_origins) * TRAIN_FRACTION))
        cutoff = unique_origins[n_train_origins - 1]
        train_mask = (hz_df['origin_time'] <= cutoff).values

        test_mask = ~train_mask

        X = hz_df[feature_cols].values
        Xtr, Xte = X[train_mask], X[test_mask]

        n_train = int(train_mask.sum())
        n_test = int(test_mask.sum())
        context.log.info(
            f"    {len(hz_df)} rows ({len(unique_origins)} origins, "
            f"train_origins:{n_train_origins}, "
            f"train:{n_train}, test:{n_test}), "

            f"{len(feature_cols)} features, lead_range={hz_cfg['lead_range']}"
        )

        task_defs = [

            ('regression', targets['y_reg'][train_mask], targets['y_reg'][test_mask]),
            ('clf_5ppb',   targets['y_5'][train_mask],   targets['y_5'][test_mask]),
            ('clf_10ppb',  targets['y_10'][train_mask],  targets['y_10'][test_mask]),
            ('clf_30ppb',  targets['y_30'][train_mask],  targets['y_30'][test_mask]),

        ]

        hz_paths: dict = {}
        hz_metrics: dict = {}
        hz_choices: dict = {}

        for task, ytr, yte in task_defs:
            context.log.info(f"    Training {task}...")
            model, choice, _ = train_and_select(
                Xtr, Xte, ytr, yte, task, ensemble_margin=ensemble_margin
            )
            context.log.info(f"      Selected: {choice}")

            # Evaluate on test set with the selected model and remap importance
            # to the horizon-specific feature names (train_and_select uses the
            # global MODEL_FEATURES list, which is wrong for MH features).
            if task == 'regression':
                eval_metrics = eval_regressor(model, Xte, yte)
            else:
                eval_metrics = eval_classifier(model, Xte, yte)

            fi_named: dict = {}
            imp = getattr(model, 'feature_importances_', None)
            if imp is not None:
                imp = np.asarray(imp)
                top_idx = np.argsort(imp)[::-1][:10]
                fi_named = {
                    feature_cols[i]: round(float(imp[i]), 4)
                    for i in top_idx
                    if i < len(feature_cols)
                }

            # Stream the pickled model straight to S3 staging.
            staging_path = f"{staging_prefix}/{hz_name}/{task}.pkl"
            model_bytes = pickle.dumps(model)
            s3.putFile(
                model_bytes,
                staging_path,
                bucket=s3.S3_BUCKET,
                content_type='application/octet-stream',
            )
            context.log.info(
                f"      Staged {hz_name}/{task} ({len(model_bytes) / 1e6:.1f} MB) -> {staging_path}"
            )

            hz_paths[task] = staging_path
            hz_metrics[task] = {**eval_metrics, 'selected': choice, 'feature_importance': fi_named}
            hz_choices[task] = choice

            # Free the model + bytes before the next task.
            del model, model_bytes
            gc.collect()

        staging_paths[hz_name] = hz_paths
        all_features[hz_name] = feature_cols
        all_metrics[hz_name] = hz_metrics
        all_choices[hz_name] = hz_choices
        all_splits[hz_name] = {'n_train': n_train, 'n_test': n_test}

        # Drop the horizon feature matrix before building the next one.
        del hz_df, X, Xtr, Xte, targets, train_mask, test_mask
        gc.collect()

    context.add_output_metadata({
        "station": site_name,
        "partition": partition,
        "run_id": run_id,
        "staging_prefix": staging_prefix,
        "horizons_trained": list(staging_paths.keys()),
        "models_count": sum(len(m) for m in staging_paths.values()),
        "algorithm_choices": all_choices,
    })

    return {
        "run_id": run_id,
        "station_key": station_key,
        "site_name": site_name,
        "models_s3": staging_paths,
        "horizon_features": all_features,
        "metrics": all_metrics,
        "splits": all_splits,
        "algorithm_choices": all_choices,
    }


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
    ins={"mh_trained_models": dg.AssetIn(key=_KEY("mh_trained_models"))},
)
def mh_training_report(
    context: dg.AssetExecutionContext,
    mh_trained_models: dict,
) -> dict:
    """Generate and upload training metrics report for one station's MH models.

    All test-set metrics and feature importances were computed inline by
    `mh_trained_models` and stored in its return dict, so this asset does not
    deserialize any model and does not rebuild the long-format feature
    matrices.
    """
    partition = context.partition_key
    site_name = STATION_PARTITION_MAP[partition]
    s3 = context.resources.s3
    station_key = STATIONS[site_name]['key']

    metrics_by_hz = mh_trained_models['metrics']
    horizon_features = mh_trained_models['horizon_features']
    splits = mh_trained_models.get('splits', {})

    report = {
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'station': site_name,
        'partition': partition,
        'run_id': mh_trained_models.get('run_id'),
        'horizons': {},
    }

    for hz_name, tasks_metrics in metrics_by_hz.items():
        feature_cols = horizon_features.get(hz_name, [])
        hz_split = splits.get(hz_name, {})
        report['horizons'][hz_name] = {
            'n_features': len(feature_cols),
            'features': feature_cols,
            'n_train': int(hz_split.get('n_train', 0)),
            'n_test': int(hz_split.get('n_test', 0)),
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
    """Promote staged MH models to their final S3 paths when approved.

    `mh_trained_models` already wrote each pickled model to a per-run staging
    prefix, so deployment is a sequence of S3 server-side copies — no model is
    ever held in this asset's memory. Final layout (unchanged):

      multihorizon/{horizon}/{station_key}/{task}.pkl
      multihorizon/{station_key}/horizon_features.json
      multihorizon/{station_key}/deployment_metadata.json
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

    staging_paths = mh_trained_models['models_s3']
    horizon_features = mh_trained_models['horizon_features']

    context.log.info(f"Deploying MH models for {site_name} via S3 server-side copy")
    client = s3.getClient()
    bucket = s3.S3_BUCKET
    uploaded: dict = {}
    for hz_name, hz_paths in staging_paths.items():
        for task, staging_path in hz_paths.items():
            final_path = f"{MH_MODELS_S3_BASE}/{hz_name}/{station_key}/{task}.pkl"
            client.copy_object(bucket, final_path, CopySource(bucket, staging_path))
            context.log.info(f"  Copied {staging_path} -> {final_path}")
            uploaded[f"{hz_name}/{task}"] = final_path

    feat_path = f"{MH_MODELS_S3_BASE}/{station_key}/horizon_features.json"
    feat_bytes = json.dumps(horizon_features, indent=2).encode('utf-8')
    s3.putFile(feat_bytes, feat_path, bucket=bucket, content_type='application/json')
    context.log.info(f"  Uploaded horizon_features.json -> {feat_path}")

    meta = {
        'deployed_at': datetime.now(timezone.utc).isoformat(),
        'station': site_name,
        'partition': partition,
        'run_id': mh_trained_models.get('run_id'),
        'models': uploaded,
        'horizons': list(staging_paths.keys()),
    }
    meta_path = f"{MH_MODELS_S3_BASE}/{station_key}/deployment_metadata.json"
    s3.putFile(
        json.dumps(meta, indent=2).encode('utf-8'),
        meta_path,
        bucket=bucket,
        content_type='application/json',
    )

    context.add_output_metadata({
        "status": "deployed",
        "station": site_name,
        "models_uploaded": len(uploaded),
        "horizons": list(staging_paths.keys()),
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
