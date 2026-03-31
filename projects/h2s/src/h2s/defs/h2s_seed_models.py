"""Seed S3 with all models needed by the H2S pipelines.

For each phase, if local pre-trained model files exist they are uploaded directly.
If local files are NOT found, models are trained inline from S3 training data
and then uploaded — so S3 always ends up with usable models.

Phase 1 — Hourly pipeline (NESTOR - BES 3-class classifier):
  tijuana/forecast/models/
    nestor_xgboost_weighted_model.json   <- production model (pickle RF)
    nestor_preprocessing_info.json       <- feature metadata
    deployment_metadata.json

Phase 2 — Daily per-station models (regression + >5ppb + >10ppb per station):
  tijuana/forecast/models/stations/
    {STATION}/regression.pkl, clf_5ppb.pkl, clf_10ppb.pkl

Phase 3 — Multi-horizon models (4 horizons x 3 stations x 3 tasks = 36 models):
  tijuana/forecast/models/multihorizon/
    {horizon}/{STATION}/{task}.pkl
"""

import json
import pickle
from datetime import datetime, timezone
from pathlib import Path

import dagster as dg
import pandas as pd

from h2s.constants import MODEL_PATH, MH_MODELS_S3_BASE
from h2s.training.multi_station_trainer import (
    MODEL_FEATURES,
    TRAIN_FRACTION,
    prepare_multi_station_features,
    train_and_select,
)
from h2s.training.multihorizon_trainer import (
    HORIZONS,
    TASKS as MH_TASKS,
    build_horizon_features,
)

# Resolve the repo root (projects/h2s/src/h2s/defs -> repo root)
_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parents[4]
_STARTMODELS = _REPO_ROOT / "data" / "startmodels"
_MODELS_V2_DIR = _REPO_ROOT / "data" / "models_v2"
_MODELS_MH_DIR = _REPO_ROOT / "data" / "models_mh"

PRIMARY_VARIANT = "random_forest"
VARIANTS = {
    "xgboost_base": "model.json",
    "xgboost_smote": "model.json",
    "random_forest": "model.joblib",
}

STATION_MODELS_S3_BASE = "tijuana/forecast/models/stations"
STATIONS = {
    'SAN YSIDRO':   'SAN_YSIDRO',
    'NESTOR - BES': 'NESTOR__BES',
    'IB CIVIC CTR': 'IB_CIVIC_CTR',
}
TASK_FILE_MAP = {
    'regression': 'model_reg',
    'clf_5ppb':   'model_clf5',
    'clf_10ppb':  'model_clf10',
}

MH_HORIZON_NAMES = ['0_6h', '6_24h', '24_48h', '48_72h']

TRAINING_DATA_S3_PATH = "latest/tijuana/forecast_data/modeldata_h2s_nofill.parquet"

# Preprocessing info for the Phase 1 hourly model
_PREPROCESSING_INFO = {
    "feature_cols": MODEL_FEATURES,
    "class_names": ["green", "orange", "yellow"],
    "site_name": "NESTOR - BES",
    "wind_cat_mapping": {"E": 0, "N": 1, "NE": 2, "NW": 3, "S": 4, "SE": 5, "SW": 6, "W": 7},
    "tidal_mapping": {"ebb": 0, "flood": 1, "slack": 2, "slack high": 3, "slack low": 4},
}


def _find_latest_mh_dir() -> Path | None:
    """Return data/models_mh/ if it exists and has .pkl files, else None."""
    if not _MODELS_MH_DIR.exists():
        return None
    if any(_MODELS_MH_DIR.glob("*.pkl")):
        return _MODELS_MH_DIR
    return None


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
    url = s3.get_presigned_url(path=TRAINING_DATA_S3_PATH, bucket=bucket)
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

    for site_name, station_key in STATIONS.items():
        sdf = df[df['site_name'] == site_name].sort_values('time').reset_index(drop=True)
        if len(sdf) < 100:
            context.log.warning(f"Insufficient data for {site_name} ({len(sdf)} rows), skipping")
            continue

        base_path = f"{STATION_MODELS_S3_BASE}/{station_key}"
        X = sdf[MODEL_FEATURES].values
        split = int(len(sdf) * TRAIN_FRACTION)
        station_uploaded = {}

        context.log.info(f"Training {site_name}: {len(sdf)} rows (train:{split}, test:{len(sdf)-split})")

        for task, y_col in [('regression', 'H2S'), ('clf_5ppb', 'exceed_5'), ('clf_10ppb', 'exceed_10')]:
            y = sdf[y_col].values
            model, choice, _ = train_and_select(X[:split], X[split:], y[:split], y[split:], task)
            s3_path = f"{base_path}/{task}.pkl"
            if dry_run:
                context.log.info(f"[DRY RUN] {site_name}/{task} ({choice}) -> s3://{bucket}/{s3_path}")
            else:
                s3.putFile(pickle.dumps(model), s3_path, bucket=bucket)
                context.log.info(f"  {task} ({choice}) -> s3://{bucket}/{s3_path}")
            station_uploaded[task] = s3_path
            uploaded.append(s3_path)

        # Store feature list (ensures inference matches training shape)
        feat_path = f"{base_path}/features.json"
        if not dry_run:
            s3.putFile(json.dumps(MODEL_FEATURES, indent=2).encode("utf-8"), feat_path, bucket=bucket, content_type="application/json")
            context.log.info(f"  features.json ({len(MODEL_FEATURES)} features) -> s3://{bucket}/{feat_path}")
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


def _train_and_seed_phase3(s3, bucket, context, raw_df, uploaded, dry_run):
    """Train 36 multi-horizon models and upload to S3."""
    context.log.info("Phase 3: Training multi-horizon models from S3 data (no local models_mh)")

    # MH uses pre-featurized parquet — only filter/clean + add targets
    df = raw_df.copy()
    df['time'] = pd.to_datetime(df['time'], utc=True)
    df = df[(df['h2s_measured'] == True) & (df['H2S'] <= 500)].copy()
    df = df.sort_values(['site_name', 'time']).reset_index(drop=True)
    df['H2S'] = df['H2S'].clip(lower=0)
    df['exceed_5'] = (df['H2S'] > 5).astype(int)
    df['exceed_10'] = (df['H2S'] > 10).astype(int)

    all_horizon_features = {}

    for site_name, station_key in STATIONS.items():
        sdf = df[df['site_name'] == site_name].copy().sort_values('time').reset_index(drop=True)
        if len(sdf) < 100:
            context.log.warning(f"MH: Insufficient data for {site_name} ({len(sdf)} rows), skipping")
            continue

        context.log.info(f"MH: Training {site_name} ({len(sdf)} rows)")

        for hz_name in MH_HORIZON_NAMES:
            hz_cfg = HORIZONS[hz_name]
            hz_df, feature_cols = build_horizon_features(sdf, hz_name, hz_cfg)
            hz_df = hz_df.dropna(subset=feature_cols).reset_index(drop=True)

            if len(hz_df) < 100:
                context.log.warning(f"  {hz_name}: only {len(hz_df)} rows after dropna, skipping")
                continue

            X = hz_df[feature_cols].values
            split = int(len(hz_df) * TRAIN_FRACTION)

            for task, y_col in [('regression', 'H2S'), ('clf_5ppb', 'exceed_5'), ('clf_10ppb', 'exceed_10')]:
                y = hz_df[y_col].values
                model, choice, _ = train_and_select(X[:split], X[split:], y[:split], y[split:], task)
                s3_path = f"{MH_MODELS_S3_BASE}/{hz_name}/{station_key}/{task}.pkl"
                if dry_run:
                    context.log.info(f"[DRY RUN] {hz_name}/{station_key}/{task} ({choice})")
                else:
                    s3.putFile(pickle.dumps(model), s3_path, bucket=bucket)
                    context.log.info(f"  {hz_name}/{station_key}/{task} ({choice})")
                uploaded.append(s3_path)

            all_horizon_features.setdefault(station_key, {})[hz_name] = feature_cols

    # Upload horizon features per station
    for station_key, hz_feats in all_horizon_features.items():
        feat_path = f"{MH_MODELS_S3_BASE}/{station_key}/horizon_features.json"
        if not dry_run:
            s3.putFile(json.dumps(hz_feats, indent=2).encode("utf-8"), feat_path, bucket=bucket, content_type="application/json")
        uploaded.append(feat_path)

    # Deployment metadata
    mh_meta = {
        "deployed_at": datetime.now(timezone.utc).isoformat(),
        "trained_inline": True,
        "horizons": MH_HORIZON_NAMES,
        "stations": list(STATIONS.values()),
        "tasks": list(MH_TASKS),
        "deployment_status": "seeded",
    }
    mh_meta_path = f"{MH_MODELS_S3_BASE}/deployment_metadata.json"
    if not dry_run:
        s3.putFile(json.dumps(mh_meta, indent=2).encode("utf-8"), mh_meta_path, bucket=bucket, content_type="application/json")
    uploaded.append(mh_meta_path)


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
        "mh_models_dir": dg.Field(
            str,
            default_value="",
            description="Directory with pre-trained MH .pkl models. Leave empty to auto-detect data/models_mh/",
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

    mh_dir_cfg = context.op_config["mh_models_dir"]
    if mh_dir_cfg:
        mh_dir: Path | None = Path(mh_dir_cfg)
    else:
        mh_dir = _find_latest_mh_dir()

    uploaded = []

    # Check if any phase needs inline training (to load training data once)
    need_train_p1 = not startmodels_dir.exists()
    need_train_p2 = station_models_dir is None or not station_models_dir.exists()
    need_train_p3 = mh_dir is None or not mh_dir.exists()

    raw_training_df = None
    if need_train_p1 or need_train_p2 or need_train_p3:
        raw_training_df = _load_training_data(s3, bucket, context)

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

    if startmodels_dir.exists():
        # Upload from local files
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
        _train_and_seed_phase1(s3, bucket, context, raw_training_df, uploaded, dry_run)

    # =========================================================================
    # Phase 2: Daily per-station models
    # =========================================================================
    context.log.info("Phase 2: Seeding per-station daily pipeline models")

    if station_models_dir is not None and station_models_dir.exists():
        # Upload from local files
        context.log.info(f"Station models source: {station_models_dir}")

        for site_name, station_key in STATIONS.items():
            base_path = f"{STATION_MODELS_S3_BASE}/{station_key}"
            station_uploaded = {}

            for task, file_prefix in TASK_FILE_MAP.items():
                local_file = station_models_dir / f"{file_prefix}_{station_key}.pkl"
                s3_path = f"{base_path}/{task}.pkl"
                if _upload(local_file, s3_path):
                    station_uploaded[task] = s3_path

            # Store feature list (ensures inference matches training shape)
            feat_path = f"{base_path}/features.json"
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
        _train_and_seed_phase2(s3, bucket, context, raw_training_df, uploaded, dry_run)

    # =========================================================================
    # Phase 3: Multi-horizon models
    # =========================================================================
    context.log.info("Phase 3: Seeding multi-horizon models")

    if mh_dir and mh_dir.exists():
        # Upload from local files
        context.log.info(f"MH models source: {mh_dir}")

        for hz_name in MH_HORIZON_NAMES:
            for station_key in STATIONS.values():
                for task in MH_TASKS:
                    local_file = mh_dir / f"best_{hz_name}_{task}_{station_key}.pkl"
                    s3_path = f"{MH_MODELS_S3_BASE}/{hz_name}/{station_key}/{task}.pkl"
                    _upload(local_file, s3_path)

        _upload(
            mh_dir / "horizon_features.json",
            f"{MH_MODELS_S3_BASE}/horizon_features.json",
            "application/json",
        )
        _upload(
            mh_dir / "training_report_mh.json",
            f"{MH_MODELS_S3_BASE}/training_report_mh.json",
            "application/json",
        )

        mh_meta = {
            "deployed_at": datetime.now(timezone.utc).isoformat(),
            "source_dir": str(mh_dir),
            "horizons": MH_HORIZON_NAMES,
            "stations": list(STATIONS.values()),
            "tasks": list(MH_TASKS),
            "deployment_status": "seeded",
        }
        mh_meta_bytes = json.dumps(mh_meta, indent=2).encode("utf-8")
        mh_meta_path = f"{MH_MODELS_S3_BASE}/deployment_metadata.json"
        if dry_run:
            context.log.info(f"[DRY RUN] deployment_metadata.json -> s3://{bucket}/{mh_meta_path}")
        else:
            s3.putFile(mh_meta_bytes, mh_meta_path, bucket=bucket, content_type="application/json")
            context.log.info(f"MH deployment_metadata.json -> s3://{bucket}/{mh_meta_path}")
        uploaded.append(mh_meta_path)
    else:
        _train_and_seed_phase3(s3, bucket, context, raw_training_df, uploaded, dry_run)

    summary = {
        "status": "dry_run" if dry_run else "seeded",
        "primary_variant": primary_variant,
        "station_models_source": str(station_models_dir),
        "mh_models_source": str(mh_dir) if mh_dir else "trained_inline",
        "files_uploaded": len(uploaded),
        "paths": uploaded,
        "trained_inline": {
            "phase1": need_train_p1,
            "phase2": need_train_p2,
            "phase3": need_train_p3,
        },
    }

    context.add_output_metadata({
        "status": summary["status"],
        "primary_variant": primary_variant,
        "station_models_source": str(station_models_dir),
        "files_uploaded": len(uploaded),
        "trained_inline": str([f"P{i}" for i, v in enumerate([need_train_p1, need_train_p2, need_train_p3], 1) if v]),
    })

    return summary


seed_models_job = dg.define_asset_job(
    name="seed_models_job",
    description="Seed S3 with all models: upload local files or train inline from S3 training data",
    selection=dg.AssetSelection.assets(seed_models),
    tags={"environment": "production", "pipeline": "h2s_seed"},
)