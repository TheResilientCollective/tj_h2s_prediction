#!/usr/bin/env python3
"""Seed S3 with starter models (standalone script, no Dagster required).

Usage:
    cd projects/h2s
    uv run python scripts/seed_models_to_s3.py          # upload all
    uv run python scripts/seed_models_to_s3.py --dry-run # preview only

Requires .env with S3 credentials.
"""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# Load .env
env_path = Path(__file__).resolve().parent.parent / ".env"
if env_path.exists():
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

from h2s.resources.minio import S3Resource
from h2s.constants import MODEL_PATH

REPO_ROOT = Path(__file__).resolve().parents[3]
STARTMODELS = REPO_ROOT / "data" / "startmodels"

PRIMARY_VARIANT = "random_forest"
VARIANTS = {
    "xgboost_base": "model.json",
    "xgboost_smote": "model.json",
    "random_forest": "model.joblib",
}


def main():
    dry_run = "--dry-run" in sys.argv

    s3 = S3Resource(
        S3_BUCKET=os.environ["S3_BUCKET"],
        S3_ADDRESS=os.environ["S3_ADDRESS"],
        S3_PORT=os.environ["S3_PORT"],
        S3_USE_SSL=os.environ.get("S3_USE_SSL", "true").lower() == "true",
        S3_ACCESS_KEY=os.environ["S3_ACCESS_KEY"],
        S3_SECRET_KEY=os.environ["S3_SECRET_KEY"],
    )
    bucket = s3.S3_BUCKET

    if not STARTMODELS.exists():
        print(f"ERROR: {STARTMODELS} not found")
        sys.exit(1)

    uploaded = []

    def upload(local_path: Path, s3_path: str, content_type: str = "application/octet-stream"):
        if not local_path.exists():
            print(f"  SKIP (not found): {local_path}")
            return
        if dry_run:
            print(f"  [DRY RUN] {local_path.name} -> s3://{bucket}/{s3_path}")
        else:
            data = local_path.read_bytes()
            s3.putFile(data, s3_path, bucket=bucket, content_type=content_type)
            print(f"  OK {local_path.name} -> s3://{bucket}/{s3_path} ({len(data):,} bytes)")
        uploaded.append(s3_path)

    print(f"Seeding S3 with starter models {'(DRY RUN)' if dry_run else ''}")
    print(f"  Source: {STARTMODELS}")
    print(f"  Target: s3://{bucket}/{MODEL_PATH}/")
    print()

    # 1. Production model
    primary_dir = STARTMODELS / PRIMARY_VARIANT
    print(f"1. Production model ({PRIMARY_VARIANT}):")
    upload(primary_dir / VARIANTS[PRIMARY_VARIANT], f"{MODEL_PATH}/nestor_xgboost_weighted_model.json")

    # 2. Preprocessing info
    print("2. Preprocessing info:")
    upload(primary_dir / "nestor_preprocessing_info.json", f"{MODEL_PATH}/nestor_preprocessing_info.json", "application/json")

    # 3. Variant models
    print("3. Variant models:")
    for variant, model_filename in VARIANTS.items():
        variant_dir = STARTMODELS / variant
        upload(variant_dir / model_filename, f"{MODEL_PATH}/{variant}/{model_filename}")
        prep = variant_dir / "nestor_preprocessing_info.json"
        if prep.exists():
            upload(prep, f"{MODEL_PATH}/{variant}/nestor_preprocessing_info.json", "application/json")

    # 4. Deployment metadata
    print("4. Deployment metadata:")
    meta = {
        "deployed_at": datetime.now(timezone.utc).isoformat(),
        "primary_variant": PRIMARY_VARIANT,
        "model_path": f"{MODEL_PATH}/nestor_xgboost_weighted_model.json",
        "preprocessing_path": f"{MODEL_PATH}/nestor_preprocessing_info.json",
        "seeded_variants": list(VARIANTS.keys()),
        "approval_metadata": {
            "approved_by": "seed_models_script",
            "variant": PRIMARY_VARIANT,
            "quality_gates_passed": True,
        },
        "deployment_status": "seeded",
    }
    meta_bytes = json.dumps(meta, indent=2).encode("utf-8")
    meta_path = f"{MODEL_PATH}/deployment_metadata.json"
    if dry_run:
        print(f"  [DRY RUN] deployment_metadata.json -> s3://{bucket}/{meta_path}")
    else:
        s3.putFile(meta_bytes, meta_path, bucket=bucket, content_type="application/json")
        print(f"  OK deployment_metadata.json -> s3://{bucket}/{meta_path}")
    uploaded.append(meta_path)

    print(f"\nDone. {len(uploaded)} files {'would be uploaded' if dry_run else 'uploaded'}.")


if __name__ == "__main__":
    main()
