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

from h2s.constants import (
    MODEL_FEATURES,
    MODEL_FEATURES_LEAN,
    STATION_MODELS_S3_BASE,
    STATION_PARTITION_MAP,
    STATIONS,
    TRAINING_SNAPSHOTS_PATH,
)
from h2s.training.multi_station_trainer import (
    TRAIN_FRACTION,
    prepare_multi_station_features,
    train_and_select,
)

# Per-station training trains two parallel variants per cycle:
#   - "evidence" (33 features, production default)
#   - "lean"     (19 features, deployed alongside as a "not overdetermined"
#                 demonstration — reviewers can load either model from S3)
# Lean model pickles carry a `_lean` suffix in the deployment dict and on
# the S3 path; metrics and features.json are keyed by variant in the report.
_VARIANTS: dict[str, list[str]] = {
    "evidence": MODEL_FEATURES,
    "lean": MODEL_FEATURES_LEAN,
}
_LEAN_SUFFIX = "_lean"

STATION_PARTITIONS = dg.StaticPartitionsDefinition(
    partition_keys=list(STATION_PARTITION_MAP.keys())  # san_ysidro, nestor_bes, ib_civic_ctr
)

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
    bucket = context.op_config["s3_bucket"]
    s3_path = "latest/tijuana/forecast_data/modeldata_h2s_nofill.parquet"

    # Load training data bytes from S3 (so we can snapshot the exact input used)
    raw_bytes = s3.getFile(path=s3_path, bucket=bucket)
    raw_df = pd.read_parquet(io.BytesIO(raw_bytes))
    context.log.info(f"✓ Loaded training data from S3 ({bucket}/{s3_path}): {len(raw_df)} rows")

    # Snapshot the exact parquet used for this training run to S3
    snapshot_ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    snapshot_path = f"{TRAINING_SNAPSHOTS_PATH}/{snapshot_ts}/modeldata_h2s_nofill.parquet"
    s3.putFile(
        raw_bytes,
        snapshot_path,
        bucket=s3.S3_BUCKET,
        content_type="application/octet-stream",
    )
    context.log.info(f"✓ Wrote training data snapshot to S3: {snapshot_path}")

    df = prepare_multi_station_features(raw_df)
    df.attrs["training_snapshot_s3_path"] = snapshot_path
    df.attrs["training_snapshot_bucket"] = s3.S3_BUCKET
    df.attrs["training_snapshot_source_bucket"] = bucket
    df.attrs["training_snapshot_source_path"] = s3_path
    df.attrs["training_snapshot_timestamp"] = snapshot_ts

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
        "training_snapshot_s3_path": snapshot_path,
        "training_snapshot_bucket": s3.S3_BUCKET,
    })
    return df


# ==============================================================================
# Asset 2: Train per-station models (partitioned by station)
# ==============================================================================

def _train_one_variant(
    context: dg.AssetExecutionContext,
    sdf: pd.DataFrame,
    features: list[str],
    split: int,
    ensemble_margin: float,
    variant_label: str,
) -> dict[str, tuple]:
    """Train regression + clf_5ppb + clf_10ppb for one feature set.

    Returns dict[task → (model, choice_str)] for the current station.
    The full metrics dict is recomputed downstream by station_training_report
    using the per-variant feature slice (so importance keys are correct).
    """
    X = sdf[features].values
    y_cont = sdf['H2S'].values
    y_5 = sdf['exceed_5'].values
    y_10 = sdf['exceed_10'].values

    Xtr, Xte = X[:split], X[split:]
    ytr_c, yte_c = y_cont[:split], y_cont[split:]
    ytr_5, yte_5 = y_5[:split], y_5[split:]
    ytr_10, yte_10 = y_10[:split], y_10[split:]

    result: dict[str, tuple] = {}
    for task, ytr_, yte_ in [
        ('regression', ytr_c, yte_c),
        ('clf_5ppb',   ytr_5, yte_5),
        ('clf_10ppb',  ytr_10, yte_10),
    ]:
        context.log.info(f"  [{variant_label}] Training {task}...")
        model, choice, _ = train_and_select(
            Xtr, Xte, ytr_, yte_, task, ensemble_margin=ensemble_margin
        )
        context.log.info(f"    [{variant_label}] {task} → {choice}")
        result[task] = (model, choice)
    return result


@dg.asset(
    key_prefix="h2s",
    group_name="h2s_multi_station_training",
    partitions_def=STATION_PARTITIONS,
    kinds={"python", "ml"},
    description="Auto-trained Evidence (33 feat) + Lean (19 feat) models per station",
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
    """Train Evidence and Lean variant models for the current station partition.

    Returns a flat dict with both variants:
      regression, clf_5ppb, clf_10ppb               ← Evidence (33 feat, production)
      regression_lean, clf_5ppb_lean, clf_10ppb_lean ← Lean    (19 feat, parallel)

    Lean is deployed alongside Evidence so reviewers can load either model
    from S3 and reproduce the comparison; see RESULTS.md for the
    not-overdetermined argument.
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

    split = int(len(sdf) * TRAIN_FRACTION)
    y_5 = sdf['exceed_5'].values
    y_10 = sdf['exceed_10'].values
    context.log.info(f"  Records: {len(sdf):,} (train: {split:,}, test: {len(sdf)-split:,})")
    context.log.info(f"  Exceedance: >5={y_5.mean()*100:.1f}%, >10={y_10.mean()*100:.1f}%")

    models: dict = {}
    choices: dict[str, dict[str, str]] = {}
    for variant, features in _VARIANTS.items():
        suffix = "" if variant == "evidence" else _LEAN_SUFFIX
        variant_results = _train_one_variant(
            context, sdf, features, split, ensemble_margin, variant
        )
        choices[variant] = {task: choice for task, (_, choice) in variant_results.items()}
        for task, (model, _) in variant_results.items():
            models[f"{task}{suffix}"] = model

    context.add_output_metadata({
        "station": site_name,
        "partition": partition,
        "n_train": int(split),
        "n_test": int(len(sdf) - split),
        "tasks": list(models.keys()),
        "variants": list(_VARIANTS.keys()),
        "algorithm_choices": choices,
    })
    return models


# ==============================================================================
# Asset 3: Station training report (partitioned by station)
# ==============================================================================

def _importance_for_features(model, feature_names: list[str], top_n: int = 10) -> dict:
    """Feature importance keyed by the variant's actual feature list.

    `multi_station_trainer.get_feature_importance` hardcodes MODEL_FEATURES,
    which would mis-label Lean models' importances. We re-do it here against
    the variant's own column order.
    """
    imp = getattr(model, "feature_importances_", None)
    if imp is None:
        return {}
    imp = np.asarray(imp)
    idx = np.argsort(imp)[::-1][:top_n]
    return {feature_names[i]: round(float(imp[i]), 4) for i in idx if i < len(feature_names)}


@dg.asset(
    key_prefix="h2s",
    group_name="h2s_multi_station_training",
    partitions_def=STATION_PARTITIONS,
    required_resource_keys={"s3"},
    kinds={"json", "s3"},
    description="JSON training metrics report for both Evidence + Lean variants per station",
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
    """Generate and upload training metrics report for both variants.

    Report shape:
      tasks: {
        evidence: { regression: {...}, clf_5ppb: {...}, clf_10ppb: {...} },
        lean:     { regression: {...}, clf_5ppb: {...}, clf_10ppb: {...} },
      }
      features: { evidence: [...33 cols...], lean: [...19 cols...] }
    """
    from h2s.training.multi_station_trainer import eval_regressor, eval_classifier

    partition = context.partition_key
    site_name = STATION_PARTITION_MAP[partition]
    s3 = context.resources.s3
    ensemble_margin = context.op_config["ensemble_margin"]

    sdf = multi_station_training_data[
        multi_station_training_data['site_name'] == site_name
    ].sort_values('time').reset_index(drop=True)

    split = int(len(sdf) * TRAIN_FRACTION)
    yte_c = sdf['H2S'].values[split:]
    yte_5 = sdf['exceed_5'].values[split:]
    yte_10 = sdf['exceed_10'].values[split:]

    tasks_metrics: dict[str, dict] = {}
    for variant, features in _VARIANTS.items():
        suffix = "" if variant == "evidence" else _LEAN_SUFFIX
        Xte = sdf[features].values[split:]
        variant_metrics: dict[str, dict] = {}
        for task_base, yte in [('regression', yte_c), ('clf_5ppb', yte_5), ('clf_10ppb', yte_10)]:
            model = per_station_trained_models[f"{task_base}{suffix}"]
            if task_base == 'regression':
                m = eval_regressor(model, Xte, yte)
            else:
                m = eval_classifier(model, Xte, yte)
            variant_metrics[task_base] = {
                **m,
                'feature_importance': _importance_for_features(model, features),
            }
        tasks_metrics[variant] = variant_metrics

    report = {
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'station': site_name,
        'partition': partition,
        'n_records': len(sdf),
        'n_train': split,
        'n_test': len(sdf) - split,
        'features': {variant: features for variant, features in _VARIANTS.items()},
        'ensemble_margin': ensemble_margin,
        'tasks': tasks_metrics,
        'training_snapshot': {
            's3_path': multi_station_training_data.attrs.get('training_snapshot_s3_path'),
            'bucket': multi_station_training_data.attrs.get('training_snapshot_bucket'),
            'source_bucket': multi_station_training_data.attrs.get('training_snapshot_source_bucket'),
            'source_path': multi_station_training_data.attrs.get('training_snapshot_source_path'),
            'timestamp': multi_station_training_data.attrs.get('training_snapshot_timestamp'),
        },
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
        "evidence_regression_r2": tasks_metrics['evidence']['regression'].get('R2'),
        "evidence_clf5_auc": tasks_metrics['evidence']['clf_5ppb'].get('AUC'),
        "evidence_clf10_auc": tasks_metrics['evidence']['clf_10ppb'].get('AUC'),
        "lean_regression_r2": tasks_metrics['lean']['regression'].get('R2'),
        "lean_clf5_auc": tasks_metrics['lean']['clf_5ppb'].get('AUC'),
        "lean_clf10_auc": tasks_metrics['lean']['clf_10ppb'].get('AUC'),
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
            default_value=True,
            description=(
                "Default True: running station_deployment_job IS the approval — "
                "models are uploaded to S3. Set to False for a dry run that "
                "loads + validates the trained models without writing to S3."
            ),
        ),
    },
)
def station_model_deployment(
    context: dg.AssetExecutionContext,
    per_station_trained_models: dict,
) -> dict:
    """Upload trained station models to S3 (default) or dry-run.

    By default, running station_deployment_job uploads the trained models to
    S3 — the act of launching the job IS the approval. Pass
    `approve_deployment=False` in the asset config to do a dry run that
    validates the upstream models without writing to S3 (returns
    `{"status": "dry_run", ...}`).

    Models are written to: tijuana/forecast/models/stations/{station_key}/{task}.pkl

    Both Evidence (33-feat, the production default — no suffix) and Lean
    (19-feat, suffix `_lean`) variants are uploaded each cycle. Schema files
    `features.json` and `features_lean.json` describe each variant's column
    order so a consumer can load `regression{_lean}.pkl` + `features{_lean}.json`
    and produce inferences end-to-end.
    """
    partition = context.partition_key
    site_name = STATION_PARTITION_MAP[partition]
    s3 = context.resources.s3
    approved = context.op_config["approve_deployment"]

    station_key = STATIONS[site_name]['key']
    base_path = f"{STATION_MODELS_S3_BASE}/{station_key}"

    if not approved:
        context.log.warning(
            f"Dry-run for {site_name} (approve_deployment=False). "
            f"Skipping S3 upload."
        )
        return {"status": "dry_run", "station": site_name}

    context.log.info(f"Deploying models for {site_name} to S3: {base_path}")

    # Upload each variant's pickles (Evidence's filenames are unsuffixed,
    # Lean's carry `_lean`). The dict already contains both sets keyed
    # appropriately by per_station_trained_models.
    uploaded: dict[str, str] = {}
    for task, model in per_station_trained_models.items():
        s3_path = f"{base_path}/{task}.pkl"
        s3.putFile(pickle.dumps(model), s3_path, bucket=s3.S3_BUCKET,
                   content_type='application/octet-stream')
        context.log.info(f"  ✓ Uploaded {task} → {s3_path}")
        uploaded[task] = s3_path

    # Write per-variant feature schema files (used by inference to match
    # the variant's column order).
    feature_files: dict[str, str] = {}
    for variant, features in _VARIANTS.items():
        suffix = "" if variant == "evidence" else _LEAN_SUFFIX
        feat_path = f"{base_path}/features{suffix}.json"
        s3.putFile(
            json.dumps(features, indent=2).encode('utf-8'),
            feat_path,
            bucket=s3.S3_BUCKET,
            content_type='application/json',
        )
        context.log.info(f"  ✓ Uploaded features{suffix}.json ({len(features)} features)")
        feature_files[variant] = feat_path

    # Deployment metadata describes both variants under `variants` keys so
    # downstream consumers can pick either path without guessing filenames.
    variants_meta: dict[str, dict] = {}
    for variant, features in _VARIANTS.items():
        suffix = "" if variant == "evidence" else _LEAN_SUFFIX
        variants_meta[variant] = {
            'features_path': feature_files[variant],
            'n_features': len(features),
            'models': {task: uploaded[f"{task}{suffix}"] for task in ('regression', 'clf_5ppb', 'clf_10ppb')},
        }

    meta = {
        'deployed_at': datetime.now(timezone.utc).isoformat(),
        'station': site_name,
        'partition': partition,
        'variants': variants_meta,
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
        "variants": list(_VARIANTS.keys()),
        "s3_base_path": base_path,
    })
    return {"status": "deployed", "station": site_name, "models": uploaded, "variants": list(_VARIANTS.keys())}


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
    description=(
        "Deploy station models to S3 — running this job IS the approval. "
        "Pass approve_deployment=False in run config for a dry run."
    ),
    selection=dg.AssetSelection.assets(station_model_deployment),
    partitions_def=STATION_PARTITIONS,
    tags={"environment": "production", "pipeline": "h2s_deployment"},
)
