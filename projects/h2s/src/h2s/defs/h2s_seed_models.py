"""Seed S3 with all models needed by the H2S pipelines.

Phase 1 — Hourly pipeline models (from data/startmodels/):
  tijuana/forecast/models/
    nestor_xgboost_weighted_model.json   <- production model (random_forest joblib)
    nestor_preprocessing_info.json       <- shared preprocessing metadata
    deployment_metadata.json             <- deployment provenance
    xgboost_base/model.json              <- variant model
    xgboost_smote/model.json             <- variant model
    random_forest/model.joblib           <- variant model

Phase 2 — Daily per-station models (from data/models_v2/{date}/):
  tijuana/forecast/models/stations/
    {STATION}/regression.pkl
    {STATION}/clf_5ppb.pkl
    {STATION}/clf_10ppb.pkl
    {STATION}/deployment_metadata.json
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import dagster as dg

from h2s.constants import MODEL_PATH, MH_MODELS_S3_BASE

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
MH_TASKS = ['regression', 'clf_5ppb', 'clf_10ppb']


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


@dg.asset(
    key_prefix="h2s",
    group_name="h2s_seed",
    required_resource_keys={"s3"},
    kinds={"python", "s3"},
    description="Upload all local models to S3: hourly pipeline (startmodels/) + daily per-station (models_v2/)",
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
    """Upload all models to S3 for both pipelines.

    Phase 1: Hourly pipeline models from startmodels/ (nestor_xgboost_weighted_model.json + variants)
    Phase 2: Daily per-station models from models_v2/ (regression.pkl, clf_5ppb.pkl, clf_10ppb.pkl per station)
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
        context.log.info(f"✓ {local_path.name} -> s3://{bucket}/{s3_path} ({len(data):,} bytes)")
        uploaded.append(s3_path)
        return True

    # =========================================================================
    # Phase 1: Hourly pipeline models
    # =========================================================================
    context.log.info("Phase 1: Seeding hourly pipeline models")

    if not startmodels_dir.exists():
        raise FileNotFoundError(f"Startmodels directory not found: {startmodels_dir}")

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
        context.log.info(f"✓ deployment_metadata.json -> s3://{bucket}/{hourly_meta_path}")
    uploaded.append(hourly_meta_path)

    # =========================================================================
    # Phase 2: Daily per-station models
    # =========================================================================
    context.log.info("Phase 2: Seeding per-station daily pipeline models")

    if station_models_dir is None:
        raise FileNotFoundError(
            f"No per-station models found under {_MODELS_V2_DIR}. "
            "Run scripts/train_station_models.py first, or set station_models_dir in config."
        )
    if not station_models_dir.exists():
        raise FileNotFoundError(f"Station models directory not found: {station_models_dir}")

    context.log.info(f"Station models source: {station_models_dir}")

    for site_name, station_key in STATIONS.items():
        base_path = f"{STATION_MODELS_S3_BASE}/{station_key}"
        station_uploaded = {}

        for task, file_prefix in TASK_FILE_MAP.items():
            local_file = station_models_dir / f"{file_prefix}_{station_key}.pkl"
            s3_path = f"{base_path}/{task}.pkl"
            if _upload(local_file, s3_path):
                station_uploaded[task] = s3_path

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
            context.log.info(f"✓ {site_name} deployment_metadata.json -> s3://{bucket}/{station_meta_path}")
        uploaded.append(station_meta_path)

    # =========================================================================
    # Phase 3: Multi-horizon models
    # =========================================================================
    mh_dir_cfg = context.op_config["mh_models_dir"]
    if mh_dir_cfg:
        mh_dir: Path | None = Path(mh_dir_cfg)
    else:
        mh_dir = _find_latest_mh_dir()

    if mh_dir and mh_dir.exists():
        context.log.info(f"Phase 3: Seeding multi-horizon models from {mh_dir}")

        for hz_name in MH_HORIZON_NAMES:
            for station_key in STATIONS.values():
                for task in MH_TASKS:
                    # Expected naming: best_{horizon}_{task}_{station}.pkl
                    local_file = mh_dir / f"best_{hz_name}_{task}_{station_key}.pkl"
                    s3_path = f"{MH_MODELS_S3_BASE}/{hz_name}/{station_key}/{task}.pkl"
                    _upload(local_file, s3_path)

        # Upload horizon_features.json
        _upload(
            mh_dir / "horizon_features.json",
            f"{MH_MODELS_S3_BASE}/horizon_features.json",
            "application/json",
        )

        # Upload training report
        _upload(
            mh_dir / "training_report_mh.json",
            f"{MH_MODELS_S3_BASE}/training_report_mh.json",
            "application/json",
        )

        # Deployment metadata
        mh_meta = {
            "deployed_at": datetime.now(timezone.utc).isoformat(),
            "source_dir": str(mh_dir),
            "horizons": MH_HORIZON_NAMES,
            "stations": list(STATIONS.values()),
            "tasks": MH_TASKS,
            "deployment_status": "seeded",
        }
        mh_meta_bytes = json.dumps(mh_meta, indent=2).encode("utf-8")
        mh_meta_path = f"{MH_MODELS_S3_BASE}/deployment_metadata.json"
        if dry_run:
            context.log.info(f"[DRY RUN] deployment_metadata.json -> s3://{bucket}/{mh_meta_path}")
        else:
            s3.putFile(mh_meta_bytes, mh_meta_path, bucket=bucket, content_type="application/json")
            context.log.info(f"✓ MH deployment_metadata.json -> s3://{bucket}/{mh_meta_path}")
        uploaded.append(mh_meta_path)
    else:
        context.log.info("Phase 3: No multi-horizon models found, skipping")

    summary = {
        "status": "dry_run" if dry_run else "seeded",
        "primary_variant": primary_variant,
        "station_models_source": str(station_models_dir),
        "mh_models_source": str(mh_dir) if mh_dir else "none",
        "files_uploaded": len(uploaded),
        "paths": uploaded,
    }

    context.add_output_metadata({
        "status": summary["status"],
        "primary_variant": primary_variant,
        "station_models_source": str(station_models_dir),
        "files_uploaded": len(uploaded),
    })

    return summary


seed_models_job = dg.define_asset_job(
    name="seed_models_job",
    description="Seed S3 with all models: hourly pipeline (startmodels/) + daily per-station (models_v2/)",
    selection=dg.AssetSelection.assets(seed_models),
    tags={"environment": "production", "pipeline": "h2s_seed"},
)
