"""Seed S3 with the hourly-pipeline starter model.

Trains a 3-class classifier (green / yellow / orange) for NESTOR - BES inline
from S3 training data and uploads it with matching preprocessing metadata, so
`forecast_prediction_job` can run on a fresh environment:

  tijuana/forecast/models/
    nestor_xgboost_weighted_model.json   <- pickled RF (loader detects pickle by magic byte)
    nestor_preprocessing_info.json       <- feature metadata (matches MODEL_FEATURES)
    deployment_metadata.json

What this job deliberately does NOT cover:

- Per-station daily models. Train and deploy them through the real pipeline:
    multi_station_training_job → station_deployment_job   (per partition)
  Those jobs produce both feature variants (evidence + lean), training
  reports, and training-data snapshots. This module used to carry a
  simplified duplicate of that trainer; the duplicate drifted every time the
  feature set changed, so it was removed.

- Hourly variant models (xgboost_base / xgboost_smote / random_forest).
  Populated by the legacy monthly training pipeline
  (monthly_model_training_job → approve_and_deploy_job). Until they exist,
  h2s_variant_predictions skips missing variants gracefully and the ensemble
  falls back to the primary model.

- Local pre-trained model files. Earlier versions preferred uploading from
  data/startmodels/ and data/models_v2/; those snapshots were trained on
  retired feature sets (43 and 37 features vs today's 33) and could never
  pass validation, so the upload paths were removed. Seeding always trains
  fresh against the current MODEL_FEATURES.
"""

import json
import pickle
from datetime import datetime, timezone

import dagster as dg
import pandas as pd

from h2s.constants import MODEL_FEATURES, MODEL_PATH
from h2s.training.multi_station_trainer import (
    TRAIN_FRACTION,
    prepare_multi_station_features,
)

TRAINING_DATA_S3_PATH = "latest/tijuana/forecast_data/modeldata_h2s_nofill.parquet"

# Preprocessing info uploaded alongside the hourly model. feature_cols MUST
# match the columns the model was trained on — both derive from
# MODEL_FEATURES here, so they cannot drift apart.
_PREPROCESSING_INFO = {
    "feature_cols": MODEL_FEATURES,
    "class_names": ["green", "orange", "yellow"],
    "site_name": "NESTOR - BES",
    "wind_cat_mapping": {"E": 0, "N": 1, "NE": 2, "NW": 3, "S": 4, "SE": 5, "SW": 6, "W": 7},
    "tidal_mapping": {"ebb": 0, "flood": 1, "slack": 2, "slack high": 3, "slack low": 4},
}


@dg.asset(
    key_prefix="h2s",
    group_name="h2s_seed",
    required_resource_keys={"s3"},
    kinds={"python", "s3"},
    description="Train + upload the hourly-pipeline starter model (NESTOR 3-class)",
    config_schema={
        "training_data_bucket": dg.Field(
            str,
            default_value="resilentpublic",
            description="S3 bucket holding the training parquet (resilentpublic or test)",
        ),
        "dry_run": dg.Field(
            bool,
            default_value=False,
            description="If True, train but log instead of uploading",
        ),
    },
)
def seed_models(context: dg.AssetExecutionContext) -> dict:
    """Train the hourly NESTOR 3-class model from S3 data and upload it.

    Per-station daily models are NOT seeded here — run
    multi_station_training_job → station_deployment_job after this.
    """
    from sklearn.ensemble import RandomForestClassifier

    s3 = context.resources.s3
    bucket = s3.S3_BUCKET
    data_bucket = context.op_config["training_data_bucket"]
    dry_run = context.op_config["dry_run"]

    context.log.info(f"Loading training data from S3 ({data_bucket}/{TRAINING_DATA_S3_PATH})...")
    url = s3.publicUrl(path=TRAINING_DATA_S3_PATH, bucket=data_bucket)
    raw_df = pd.read_parquet(url)
    context.log.info(f"Loaded {len(raw_df)} rows from S3")

    df = prepare_multi_station_features(raw_df, station='NESTOR - BES')
    if len(df) < 100:
        raise dg.Failure(f"Insufficient NESTOR - BES data ({len(df)} rows), cannot seed")

    # 3-class target matching class_names order: green=0, orange=1, yellow=2
    df = df.copy()
    df['h2s_class'] = 0  # green
    df.loc[df['H2S'] >= 5, 'h2s_class'] = 2   # yellow
    df.loc[df['H2S'] >= 30, 'h2s_class'] = 1  # orange

    X = df[MODEL_FEATURES].values
    y = df['h2s_class'].values
    split = int(len(df) * TRAIN_FRACTION)

    rf = RandomForestClassifier(
        n_estimators=500, max_depth=20, min_samples_leaf=5,
        max_features='sqrt', class_weight='balanced',
        n_jobs=-1, random_state=42,
    )
    rf.fit(X[:split], y[:split])
    context.log.info(
        f"Trained 3-class RF on {split} rows "
        f"(test: {len(df)-split}, {len(MODEL_FEATURES)} features)"
    )

    model_path = f"{MODEL_PATH}/nestor_xgboost_weighted_model.json"
    prep_path = f"{MODEL_PATH}/nestor_preprocessing_info.json"
    meta_path = f"{MODEL_PATH}/deployment_metadata.json"

    meta = {
        "deployed_at": datetime.now(timezone.utc).isoformat(),
        "primary_variant": "random_forest",
        "model_path": model_path,
        "preprocessing_path": prep_path,
        "trained_inline": True,
        "training_rows": len(df),
        "n_features": len(MODEL_FEATURES),
        "deployment_status": "seeded",
    }

    if dry_run:
        for path in (model_path, prep_path, meta_path):
            context.log.info(f"[DRY RUN] -> s3://{bucket}/{path}")
    else:
        s3.putFile(pickle.dumps(rf), model_path, bucket=bucket)
        context.log.info(f"Trained model -> s3://{bucket}/{model_path}")
        prep_bytes = json.dumps(_PREPROCESSING_INFO, indent=2).encode("utf-8")
        s3.putFile(prep_bytes, prep_path, bucket=bucket, content_type="application/json")
        context.log.info(f"Preprocessing info -> s3://{bucket}/{prep_path}")
        s3.putFile(json.dumps(meta, indent=2).encode("utf-8"), meta_path, bucket=bucket, content_type="application/json")
        context.log.info(f"Deployment metadata -> s3://{bucket}/{meta_path}")

    uploaded = [model_path, prep_path, meta_path]
    summary = {
        "status": "dry_run" if dry_run else "seeded",
        "training_rows": len(df),
        "n_features": len(MODEL_FEATURES),
        "files_uploaded": len(uploaded),
        "paths": uploaded,
    }

    context.add_output_metadata({
        "status": summary["status"],
        "training_rows": len(df),
        "n_features": len(MODEL_FEATURES),
        "files_uploaded": len(uploaded),
    })
    return summary


seed_models_job = dg.define_asset_job(
    name="seed_models_job",
    description=(
        "Bootstrap a fresh environment: train + upload the hourly starter model. "
        "Run multi_station_training_job → station_deployment_job afterwards for "
        "the per-station daily models."
    ),
    selection=dg.AssetSelection.assets(seed_models),
    tags={"environment": "production", "pipeline": "h2s_seed"},
)
