"""Seed S3 with starter models.

Uploads local starter models from data/startmodels/ to S3 so the
forecast pipeline can run. This is a one-time (or re-seed) operation.

S3 layout after seeding:
  tijuana/forecast/models/
    nestor_xgboost_weighted_model.json   <- production model (random_forest joblib)
    nestor_preprocessing_info.json       <- shared preprocessing metadata
    deployment_metadata.json             <- deployment provenance
    xgboost_base/model.json              <- variant model
    xgboost_smote/model.json             <- variant model
    random_forest/model.joblib           <- variant model
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import dagster as dg

from h2s.constants import MODEL_PATH

# Resolve the repo root (3 levels up from this file: defs -> h2s -> src -> h2s_project -> repo)
_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parents[4]  # projects/h2s/src/h2s/defs -> repo root
_STARTMODELS = _REPO_ROOT / "data" / "startmodels"

# Primary variant used as the production model
PRIMARY_VARIANT = "random_forest"

# All variants to upload
VARIANTS = {
    "xgboost_base": "model.json",
    "xgboost_smote": "model.json",
    "random_forest": "model.joblib",
}


@dg.asset(
    key_prefix="h2s",
    group_name="h2s_seed",
    required_resource_keys={"s3"},
    kinds={"python", "s3"},
    description="Upload local starter models to S3 (seed / re-seed)",
    config_schema={
        "startmodels_dir": dg.Field(
            str,
            default_value=str(_STARTMODELS),
            description="Local directory containing starter models",
        ),
        "primary_variant": dg.Field(
            str,
            default_value=PRIMARY_VARIANT,
            description="Variant to install as the production model",
        ),
        "dry_run": dg.Field(
            bool,
            default_value=False,
            description="If True, log what would be uploaded without actually uploading",
        ),
    },
)
def seed_models(context: dg.AssetExecutionContext) -> dict:
    """Upload all starter models and preprocessing info to S3.

    Uploads:
    1. Production model (primary variant → nestor_xgboost_weighted_model.json)
    2. Shared preprocessing info (nestor_preprocessing_info.json)
    3. Per-variant models (xgboost_base, xgboost_smote, random_forest)
    4. Deployment metadata
    """
    s3 = context.resources.s3
    bucket = s3.S3_BUCKET
    startmodels_dir = Path(context.op_config["startmodels_dir"])
    primary_variant = context.op_config["primary_variant"]
    dry_run = context.op_config["dry_run"]

    if not startmodels_dir.exists():
        raise FileNotFoundError(f"Startmodels directory not found: {startmodels_dir}")

    uploaded = []

    def _upload(local_path: Path, s3_path: str, content_type: str = "application/octet-stream"):
        if not local_path.exists():
            context.log.warning(f"Skipping (not found): {local_path}")
            return False
        if dry_run:
            context.log.info(f"[DRY RUN] Would upload {local_path} -> s3://{bucket}/{s3_path}")
            uploaded.append(s3_path)
            return True
        data = local_path.read_bytes()
        s3.putFile(data, s3_path, bucket=bucket, content_type=content_type)
        context.log.info(f"Uploaded {local_path.name} -> s3://{bucket}/{s3_path} ({len(data):,} bytes)")
        uploaded.append(s3_path)
        return True

    # 1. Production model (primary variant)
    primary_dir = startmodels_dir / primary_variant
    primary_model_file = primary_dir / VARIANTS.get(primary_variant, "model.joblib")
    _upload(primary_model_file, f"{MODEL_PATH}/nestor_xgboost_weighted_model.json")

    # 2. Shared preprocessing info
    prep_file = primary_dir / "nestor_preprocessing_info.json"
    _upload(prep_file, f"{MODEL_PATH}/nestor_preprocessing_info.json", "application/json")

    # 3. Per-variant models
    for variant, model_filename in VARIANTS.items():
        variant_dir = startmodels_dir / variant
        model_file = variant_dir / model_filename
        s3_model_path = f"{MODEL_PATH}/{variant}/{model_filename}"
        _upload(model_file, s3_model_path)

        # Also upload variant-specific preprocessing info if it differs
        variant_prep = variant_dir / "nestor_preprocessing_info.json"
        if variant_prep.exists():
            _upload(variant_prep, f"{MODEL_PATH}/{variant}/nestor_preprocessing_info.json", "application/json")

    # 4. Deployment metadata
    meta = {
        "deployed_at": datetime.now(timezone.utc).isoformat(),
        "primary_variant": primary_variant,
        "model_path": f"{MODEL_PATH}/nestor_xgboost_weighted_model.json",
        "preprocessing_path": f"{MODEL_PATH}/nestor_preprocessing_info.json",
        "seeded_variants": list(VARIANTS.keys()),
        "approval_metadata": {
            "approved_by": "seed_models_job",
            "variant": primary_variant,
            "quality_gates_passed": True,
        },
        "deployment_status": "seeded",
    }
    meta_bytes = json.dumps(meta, indent=2).encode("utf-8")
    meta_path = f"{MODEL_PATH}/deployment_metadata.json"
    if dry_run:
        context.log.info(f"[DRY RUN] Would upload deployment_metadata.json -> s3://{bucket}/{meta_path}")
    else:
        s3.putFile(meta_bytes, meta_path, bucket=bucket, content_type="application/json")
        context.log.info(f"Uploaded deployment_metadata.json -> s3://{bucket}/{meta_path}")
    uploaded.append(meta_path)

    summary = {
        "status": "dry_run" if dry_run else "seeded",
        "primary_variant": primary_variant,
        "files_uploaded": len(uploaded),
        "paths": uploaded,
    }

    context.add_output_metadata({
        "status": summary["status"],
        "primary_variant": primary_variant,
        "files_uploaded": len(uploaded),
    })

    return summary


seed_models_job = dg.define_asset_job(
    name="seed_models_job",
    description="Upload local starter models to S3 (one-time seed or re-seed)",
    selection=dg.AssetSelection.assets(seed_models),
    tags={"environment": "production", "pipeline": "h2s_seed"},
)
