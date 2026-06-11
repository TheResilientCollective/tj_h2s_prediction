"""Seed S3 with all models needed by the H2S pipelines.

For each phase, if local pre-trained model files exist they are uploaded directly.
If local files are NOT found, models are trained inline from S3 training data
and then uploaded — so S3 always ends up with usable models.

Phase 1 — Hourly pipeline (NESTOR - BES 3-class classifier):
  tijuana/forecast/models/
    nestor_xgboost_weighted_model.json   <- production model (pickle RF)
    nestor_preprocessing_info.json       <- feature metadata
    deployment_metadata.json

Phase 2 — Daily per-station models (regression + >5ppb + >10ppb per station).
  Seeds the Evidence variant (33 feat); subsequent multi_station_training_job
  runs add the Lean variant alongside.
  tijuana/forecast/models/stations/
    {STATION}/{regression,clf_5ppb,clf_10ppb}_evidence.pkl
    {STATION}/features_evidence.json
"""

import json
import pickle
from datetime import datetime, timezone
from pathlib import Path

import dagster as dg
import pandas as pd

from h2s.constants import (
    MODEL_FEATURES,
    MODEL_PATH,
    STATION_KEYS,
    STATION_MODELS_S3_BASE,
)
from h2s.training.multi_station_trainer import (
    TRAIN_FRACTION,
    prepare_multi_station_features,
    train_and_select,
)

# Resolve the repo root (projects/h2s/src/h2s/defs -> repo root)
_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parents[4]
_STARTMODELS = _REPO_ROOT / "data" / "startmodels"
_MODELS_V2_DIR = _REPO_ROOT / "data" / "models_v2"

PRIMARY_VARIANT = "random_forest"
VARIANTS = {
    "xgboost_base": "model.json",
    "xgboost_smote": "model.json",
    "random_forest": "model.joblib",
}
TASK_FILE_MAP = {
    'regression': 'model_reg',
    'clf_5ppb':   'model_clf5',
    'clf_10ppb':  'model_clf10',
}

TRAINING_DATA_S3_PATH = "latest/tijuana/forecast_data/modeldata_h2s_nofill.parquet"

# Preprocessing info for the Phase 1 hourly model
_PREPROCESSING_INFO = {
    "feature_cols": MODEL_FEATURES,
    "class_names": ["green", "orange", "yellow"],
    "site_name": "NESTOR - BES",
    "wind_cat_mapping": {"E": 0, "N": 1, "NE": 2, "NW": 3, "S": 4, "SE": 5, "SW": 6, "W": 7},
    "tidal_mapping": {"ebb": 0, "flood": 1, "slack": 2, "slack high": 3, "slack low": 4},
}


def _find_latest_models_dir() -> Path | None:
    """Return the most recent date-stamped dir under data/models_v2/, or None."""
    if not _MODELS_V2_DIR.exists():
        return None
    dirs = sorted([d for d in _MODELS_V2_DIR.iterdir() if d.is_dir()], reverse=True)
    return dirs[0] if dirs else None


# =========================================================================
# Inline training helpers (used when local model files don't exist)
# =========================================================================

def _load_training_data(s3, bucket, context):
    """Load training parquet from S3 for inline model training."""
    context.log.info(f"Loading training data from S3 ({bucket}/{TRAINING_DATA_S3_PATH})...")
    url = s3.publicUrl(path=TRAINING_DATA_S3_PATH, bucket=bucket)
    raw_df = pd.read_parquet(url)
    context.log.info(f"Loaded {len(raw_df)} rows from S3")
    return raw_df


def _train_and_seed_phase1(s3, bucket, context, raw_df, uploaded, dry_run):
    """Train a 3-class hourly model for NESTOR - BES and upload to S3."""
    from sklearn.ensemble import RandomForestClassifier

    context.log.info("Phase 1: Training hourly model from S3 data (no local startmodels)")

    df = prepare_multi_station_features(raw_df, station='NESTOR - BES')
    if len(df) < 100:
        context.log.error(f"Insufficient NESTOR - BES data ({len(df)} rows), cannot train Phase 1")
        return

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
    context.log.info(f"Trained 3-class RF on {split} rows (test: {len(df)-split})")

    model_path = f"{MODEL_PATH}/nestor_xgboost_weighted_model.json"
    prep_path = f"{MODEL_PATH}/nestor_preprocessing_info.json"

    if dry_run:
        context.log.info(f"[DRY RUN] trained model -> s3://{bucket}/{model_path}")
        context.log.info(f"[DRY RUN] preprocessing info -> s3://{bucket}/{prep_path}")
    else:
        s3.putFile(pickle.dumps(rf), model_path, bucket=bucket)
        context.log.info(f"Trained model -> s3://{bucket}/{model_path}")
        prep_bytes = json.dumps(_PREPROCESSING_INFO, indent=2).encode("utf-8")
        s3.putFile(prep_bytes, prep_path, bucket=bucket, content_type="application/json")
        context.log.info(f"Preprocessing info -> s3://{bucket}/{prep_path}")
    uploaded.extend([model_path, prep_path])

    meta = {
        "deployed_at": datetime.now(timezone.utc).isoformat(),
        "primary_variant": "random_forest",
        "model_path": model_path,
        "preprocessing_path": prep_path,
        "trained_inline": True,
        "training_rows": len(df),
        "deployment_status": "seeded",
    }
    meta_path = f"{MODEL_PATH}/deployment_metadata.json"
    if not dry_run:
        s3.putFile(json.dumps(meta, indent=2).encode("utf-8"), meta_path, bucket=bucket, content_type="application/json")
    uploaded.append(meta_path)


def _train_and_seed_phase2(s3, bucket, context, raw_df, uploaded, dry_run):
    """Train per-station models (regression + classifiers) and upload to S3."""
    context.log.info("Phase 2: Training per-station models from S3 data (no local models_v2)")

    df = prepare_multi_station_features(raw_df)

    for site_name, station_key in STATION_KEYS.items():
        sdf = df[df['site_name'] == site_name].sort_values('time').reset_index(drop=True)
        if len(sdf) < 100:
            context.log.warning(f"Insufficient data for {site_name} ({len(sdf)} rows), skipping")
            continue

        base_path = f"{STATION_MODELS_S3_BASE}/{station_key}"
        X = sdf[MODEL_FEATURES].values
        split = int(len(sdf) * TRAIN_FRACTION)
        station_uploaded = {}

        context.log.info(f"Training {site_name}: {len(sdf)} rows (train:{split}, test:{len(sdf)-split})")

        # Seed only the Evidence variant — daily pipeline reads `_evidence`
        # paths. The subsequent multi_station_training_job adds Lean.
        for task, y_col in [('regression', 'H2S'), ('clf_5ppb', 'exceed_5'), ('clf_10ppb', 'exceed_10')]:
            y = sdf[y_col].values
            model, choice, _ = train_and_select(X[:split], X[split:], y[:split], y[split:], task)
            s3_path = f"{base_path}/{task}_evidence.pkl"
            if dry_run:
                context.log.info(f"[DRY RUN] {site_name}/{task}_evidence ({choice}) -> s3://{bucket}/{s3_path}")
            else:
                s3.putFile(pickle.dumps(model), s3_path, bucket=bucket)
                context.log.info(f"  {task}_evidence ({choice}) -> s3://{bucket}/{s3_path}")
            station_uploaded[f"{task}_evidence"] = s3_path
            uploaded.append(s3_path)

        # Store feature list (ensures inference matches training shape)
        feat_path = f"{base_path}/features_evidence.json"
        if not dry_run:
            s3.putFile(json.dumps(MODEL_FEATURES, indent=2).encode("utf-8"), feat_path, bucket=bucket, content_type="application/json")
            context.log.info(f"  features_evidence.json ({len(MODEL_FEATURES)} features) -> s3://{bucket}/{feat_path}")
        uploaded.append(feat_path)

        station_meta = {
            "deployed_at": datetime.now(timezone.utc).isoformat(),
            "station": site_name,
            "station_key": station_key,
            "trained_inline": True,
            "training_rows": len(sdf),
            "models": station_uploaded,
            "deployment_status": "seeded",
        }
        meta_path = f"{base_path}/deployment_metadata.json"
        if not dry_run:
            s3.putFile(json.dumps(station_meta, indent=2).encode("utf-8"), meta_path, bucket=bucket, content_type="application/json")
        uploaded.append(meta_path)


# =========================================================================
# Dagster asset + job
# =========================================================================

@dg.asset(
    key_prefix="h2s",
    group_name="h2s_seed",
    required_resource_keys={"s3"},
    kinds={"python", "s3"},
    description="Upload or train all models to S3: hourly pipeline + per-station + multi-horizon",
    config_schema={
        "startmodels_dir": dg.Field(
            str,
            default_value=str(_STARTMODELS),
            description="Directory containing hourly pipeline starter models",
        ),
        "station_models_dir": dg.Field(
            str,
            default_value="",
            description="Directory containing per-station .pkl files. Leave empty to auto-detect latest data/models_v2/{date}/",
        ),
        "primary_variant": dg.Field(
            str,
            default_value=PRIMARY_VARIANT,
            description="Variant to install as the production hourly model",
        ),
        "dry_run": dg.Field(
            bool,
            default_value=False,
            description="If True, log what would be uploaded without actually uploading",
        ),
    },
)
def seed_models(context: dg.AssetExecutionContext) -> dict:
    """Upload or train all models for both pipelines.

    For each phase: if local model files exist, upload them.
    If not, train models from S3 training data and upload those.
    """
    s3 = context.resources.s3
    bucket = s3.S3_BUCKET
    startmodels_dir = Path(context.op_config["startmodels_dir"])
    primary_variant = context.op_config["primary_variant"]
    dry_run = context.op_config["dry_run"]

    station_models_dir_cfg = context.op_config["station_models_dir"]
    if station_models_dir_cfg:
        station_models_dir: Path | None = Path(station_models_dir_cfg)
    else:
        station_models_dir = _find_latest_models_dir()

    uploaded = []

    # Lazy-load training data (only fetched from S3 when a phase needs inline training)
    _training_cache = {}

    def _ensure_training_data():
        if 'df' not in _training_cache:
            _training_cache['df'] = _load_training_data(s3, bucket, context)
        return _training_cache['df']

    def _upload(local_path: Path, s3_path: str, content_type: str = "application/octet-stream") -> bool:
        if not local_path.exists():
            context.log.warning(f"Skipping (not found): {local_path}")
            return False
        if dry_run:
            context.log.info(f"[DRY RUN] {local_path.name} -> s3://{bucket}/{s3_path}")
            uploaded.append(s3_path)
            return True
        data = local_path.read_bytes()
        s3.putFile(data, s3_path, bucket=bucket, content_type=content_type)
        context.log.info(f"{local_path.name} -> s3://{bucket}/{s3_path} ({len(data):,} bytes)")
        uploaded.append(s3_path)
        return True

    # =========================================================================
    # Phase 1: Hourly pipeline models
    # =========================================================================
    context.log.info("Phase 1: Seeding hourly pipeline models")

    local_p1_valid = False
    if startmodels_dir.exists():
        # Validate local model feature count matches current MODEL_FEATURES
        prep_file = startmodels_dir / primary_variant / "nestor_preprocessing_info.json"
        if prep_file.exists():
            local_prep = json.loads(prep_file.read_text())
            local_n = len(local_prep.get('feature_cols', []))
            if local_n != len(MODEL_FEATURES):
                context.log.warning(
                    f"Phase 1: Local model has {local_n} features, "
                    f"current MODEL_FEATURES has {len(MODEL_FEATURES)}. Training fresh model."
                )
            else:
                local_p1_valid = True
        else:
            local_p1_valid = True  # no prep info to validate, trust the local files

    if local_p1_valid and startmodels_dir.exists():
        # Upload from validated local files
        primary_dir = startmodels_dir / primary_variant
        _upload(primary_dir / VARIANTS.get(primary_variant, "model.joblib"), f"{MODEL_PATH}/nestor_xgboost_weighted_model.json")
        _upload(primary_dir / "nestor_preprocessing_info.json", f"{MODEL_PATH}/nestor_preprocessing_info.json", "application/json")

        for variant, model_filename in VARIANTS.items():
            variant_dir = startmodels_dir / variant
            _upload(variant_dir / model_filename, f"{MODEL_PATH}/{variant}/{model_filename}")
            variant_prep = variant_dir / "nestor_preprocessing_info.json"
            if variant_prep.exists():
                _upload(variant_prep, f"{MODEL_PATH}/{variant}/nestor_preprocessing_info.json", "application/json")

        hourly_meta = {
            "deployed_at": datetime.now(timezone.utc).isoformat(),
            "primary_variant": primary_variant,
            "model_path": f"{MODEL_PATH}/nestor_xgboost_weighted_model.json",
            "preprocessing_path": f"{MODEL_PATH}/nestor_preprocessing_info.json",
            "seeded_variants": list(VARIANTS.keys()),
            "deployment_status": "seeded",
        }
        hourly_meta_bytes = json.dumps(hourly_meta, indent=2).encode("utf-8")
        hourly_meta_path = f"{MODEL_PATH}/deployment_metadata.json"
        if dry_run:
            context.log.info(f"[DRY RUN] deployment_metadata.json -> s3://{bucket}/{hourly_meta_path}")
        else:
            s3.putFile(hourly_meta_bytes, hourly_meta_path, bucket=bucket, content_type="application/json")
            context.log.info(f"deployment_metadata.json -> s3://{bucket}/{hourly_meta_path}")
        uploaded.append(hourly_meta_path)
    else:
        _train_and_seed_phase1(s3, bucket, context, _ensure_training_data(), uploaded, dry_run)

    # =========================================================================
    # Phase 2: Daily per-station models
    # =========================================================================
    context.log.info("Phase 2: Seeding per-station daily pipeline models")

    local_p2_valid = False
    if station_models_dir is not None and station_models_dir.exists():
        # Validate local models match current feature set before uploading
        test_pkl = next(station_models_dir.glob("*.pkl"), None)
        if test_pkl:
            test_model = pickle.loads(test_pkl.read_bytes())
            n_features = getattr(test_model, 'n_features_in_', None)
            if n_features is not None and n_features != len(MODEL_FEATURES):
                context.log.warning(
                    f"Phase 2: Local models have {n_features} features, "
                    f"current MODEL_FEATURES has {len(MODEL_FEATURES)}. Training fresh models."
                )
            else:
                local_p2_valid = True

    if local_p2_valid:
        # Upload from validated local files
        context.log.info(f"Station models source: {station_models_dir}")

        for site_name, station_key in STATION_KEYS.items():
            base_path = f"{STATION_MODELS_S3_BASE}/{station_key}"
            station_uploaded = {}

            for task, file_prefix in TASK_FILE_MAP.items():
                local_file = station_models_dir / f"{file_prefix}_{station_key}.pkl"
                # Seed only the Evidence variant — see Phase 2 docstring.
                s3_path = f"{base_path}/{task}_evidence.pkl"
                if _upload(local_file, s3_path):
                    station_uploaded[f"{task}_evidence"] = s3_path

            # Store feature list (matches validated local models)
            feat_path = f"{base_path}/features_evidence.json"
            if not dry_run:
                s3.putFile(json.dumps(MODEL_FEATURES, indent=2).encode("utf-8"), feat_path, bucket=bucket, content_type="application/json")
            uploaded.append(feat_path)

            station_meta = {
                "deployed_at": datetime.now(timezone.utc).isoformat(),
                "station": site_name,
                "station_key": station_key,
                "source_dir": str(station_models_dir),
                "models": station_uploaded,
                "deployment_status": "seeded",
            }
            station_meta_bytes = json.dumps(station_meta, indent=2).encode("utf-8")
            station_meta_path = f"{base_path}/deployment_metadata.json"
            if dry_run:
                context.log.info(f"[DRY RUN] {site_name} deployment_metadata.json -> s3://{bucket}/{station_meta_path}")
            else:
                s3.putFile(station_meta_bytes, station_meta_path, bucket=bucket, content_type="application/json")
                context.log.info(f"{site_name} deployment_metadata.json -> s3://{bucket}/{station_meta_path}")
            uploaded.append(station_meta_path)
    else:
        _train_and_seed_phase2(s3, bucket, context, _ensure_training_data(), uploaded, dry_run)

    trained_inline = 'df' in _training_cache  # True if any phase needed inline training

    summary = {
        "status": "dry_run" if dry_run else "seeded",
        "primary_variant": primary_variant,
        "station_models_source": str(station_models_dir),
        "files_uploaded": len(uploaded),
        "paths": uploaded,
        "trained_inline": trained_inline,
    }

    context.add_output_metadata({
        "status": summary["status"],
        "primary_variant": primary_variant,
        "station_models_source": str(station_models_dir),
        "files_uploaded": len(uploaded),
        "trained_inline": trained_inline,
    })

    return summary


seed_models_job = dg.define_asset_job(
    name="seed_models_job",
    description="Seed S3 with all models: upload local files or train inline from S3 training data",
    selection=dg.AssetSelection.assets(seed_models),
    tags={"environment": "production", "pipeline": "h2s_seed"},
)
