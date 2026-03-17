#!/usr/bin/env python3
"""Upload H2S model files from data/startmodels/ to S3.

Structure of data/startmodels/:
    xgboost_base/
        model.json                     → MODEL_PATH/xgboost_base/model.json
        nestor_preprocessing_info.json → used as primary preprocessing info
    xgboost_smote/
        model.json                     → MODEL_PATH/xgboost_smote/model.json
        nestor_preprocessing_info.json
    random_forest/
        model.joblib                   → MODEL_PATH/random_forest/model.joblib
        nestor_preprocessing_info.json

The --primary flag (default: xgboost_smote) also copies that variant's model to
MODEL_PATH/nestor_xgboost_weighted_model.json (the path h2s_model_artifacts loads).

Usage:
    cd projects/h2s && uv run python ../../upload_models_to_s3.py
    cd projects/h2s && uv run python ../../upload_models_to_s3.py --primary xgboost_base
    cd projects/h2s && uv run python ../../upload_models_to_s3.py --primary random_forest
"""

import os
import sys
import argparse
from pathlib import Path

# Add h2s project to path
h2s_path = str(Path(__file__).parent / "projects/h2s/src")
if h2s_path not in sys.path:
    sys.path.insert(0, h2s_path)

from h2s.resources.minio import S3Resource

BASE_DIR = Path(__file__).parent
STARTMODELS_DIR = BASE_DIR / "data" / "startmodels"
S3_MODEL_BASE = "tijuana/forecast/models"

VARIANTS = {
    "xgboost_base":  ("model.json",   "application/json"),
    "xgboost_smote": ("model.json",   "application/json"),
    "random_forest": ("model.joblib", "application/octet-stream"),
}


def load_env():
    env_file = BASE_DIR / "projects" / "h2s" / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def make_s3():
    return S3Resource(
        S3_BUCKET=os.environ.get("S3_BUCKET", "test"),
        S3_ADDRESS=os.environ["S3_ADDRESS"],
        S3_PORT=os.environ["S3_PORT"],
        S3_USE_SSL=os.environ.get("S3_USE_SSL", "true").lower() == "true",
        S3_ACCESS_KEY=os.environ["S3_ACCESS_KEY"],
        S3_SECRET_KEY=os.environ["S3_SECRET_KEY"],
    )


def upload(s3, local_path: Path, s3_path: str, content_type: str):
    data = local_path.read_bytes()
    s3.putFile(data=data, path=s3_path, bucket=s3.S3_BUCKET, content_type=content_type)
    print(f"  ✓ {s3_path}  ({len(data)/1024:.1f} KB)")


def main():
    parser = argparse.ArgumentParser(description="Upload H2S models from data/startmodels/ to S3")
    parser.add_argument(
        "--primary",
        default="xgboost_smote",
        choices=list(VARIANTS),
        help="Variant to deploy as the primary production model (default: xgboost_smote)",
    )
    parser.add_argument(
        "--skip-variants",
        action="store_true",
        help="Only upload the primary model + preprocessing info, skip individual variant paths",
    )
    args = parser.parse_args()

    load_env()

    print("=" * 70)
    print("H2S Model Upload  —  data/startmodels/ → S3")
    print("=" * 70)
    print(f"Source: {STARTMODELS_DIR}")
    print(f"Primary model variant: {args.primary}")
    print()

    # Validate source directory
    if not STARTMODELS_DIR.exists():
        print(f"✗ data/startmodels/ not found. Run the download section first.")
        sys.exit(1)

    s3 = make_s3()
    print(f"Connected: {s3.S3_ADDRESS}:{s3.S3_PORT}  bucket={s3.S3_BUCKET}")
    print()

    # --- Upload preprocessing info (from primary variant) ---
    prep_local = STARTMODELS_DIR / args.primary / "nestor_preprocessing_info.json"
    if not prep_local.exists():
        print(f"✗ Preprocessing info not found: {prep_local}")
        sys.exit(1)

    print("Uploading preprocessing info...")
    upload(s3, prep_local, f"{S3_MODEL_BASE}/nestor_preprocessing_info.json", "application/json")
    print()

    # --- Upload primary production model ---
    primary_filename, primary_ct = VARIANTS[args.primary]
    primary_local = STARTMODELS_DIR / args.primary / primary_filename
    if not primary_local.exists():
        print(f"✗ Primary model not found: {primary_local}")
        sys.exit(1)

    print(f"Uploading primary production model ({args.primary} → nestor_xgboost_weighted_model.json)...")
    upload(s3, primary_local, f"{S3_MODEL_BASE}/nestor_xgboost_weighted_model.json", primary_ct)
    print()

    # --- Upload variant models ---
    if not args.skip_variants:
        print("Uploading variant models...")
        for variant, (filename, ct) in VARIANTS.items():
            local = STARTMODELS_DIR / variant / filename
            if local.exists():
                upload(s3, local, f"{S3_MODEL_BASE}/{variant}/{filename}", ct)
            else:
                print(f"  ⚠ skipped {variant}/{filename}  (not found locally)")
        print()

    print("=" * 70)
    print("✓ Upload complete")
    print("=" * 70)
    print()
    print(f"Primary model:  s3://{s3.S3_BUCKET}/{S3_MODEL_BASE}/nestor_xgboost_weighted_model.json")
    print(f"Preprocessing:  s3://{s3.S3_BUCKET}/{S3_MODEL_BASE}/nestor_preprocessing_info.json")
    if not args.skip_variants:
        for variant, (filename, _) in VARIANTS.items():
            print(f"Variant:        s3://{s3.S3_BUCKET}/{S3_MODEL_BASE}/{variant}/{filename}")
    print()


if __name__ == "__main__":
    main()
