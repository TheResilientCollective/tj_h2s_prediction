#!/usr/bin/env python3
"""Upload H2S model files to S3.

Uploads:
1. nestor_xgboost_weighted_model.json (4.4 MB)
2. nestor_preprocessing_info.json (converted from .pkl)

To: s3://test/tijuana/forecast/models/
"""

import os
import sys
from io import BytesIO

# Add h2s project to path
h2s_path = "/Users/valentin/development/dev_resilient/tj_h2s_prediction/projects/h2s/src"
if h2s_path not in sys.path:
    sys.path.insert(0, h2s_path)

from h2s.resources.minio import S3Resource

# Paths
base_dir = "/Users/valentin/development/dev_resilient/tj_h2s_prediction"
model_file = f"{base_dir}/nestor_xgboost_weighted_model.json"
prep_file = f"{base_dir}/nestor_preprocessing_info.json"

# S3 paths
s3_model_path = "tijuana/forecast/models/nestor_xgboost_weighted_model.json"
s3_prep_path = "tijuana/forecast/models/nestor_preprocessing_info.json"

def main():
    print("="*80)
    print("H2S Model Upload to S3")
    print("="*80)

    # Check files exist
    if not os.path.exists(model_file):
        print(f"✗ Error: Model file not found: {model_file}")
        sys.exit(1)

    if not os.path.exists(prep_file):
        print(f"✗ Error: Preprocessing file not found: {prep_file}")
        print("  Did you run convert_preprocessing.py?")
        sys.exit(1)

    print(f"✓ Found model file: {os.path.getsize(model_file) / 1024 / 1024:.1f} MB")
    print(f"✓ Found preprocessing file: {os.path.getsize(prep_file)} bytes")
    print()

    # Initialize S3 resource
    print("Connecting to S3...")
    s3 = S3Resource(
        S3_BUCKET=os.getenv('S3_BUCKET', 'test'),
        S3_ADDRESS=os.getenv('S3_ADDRESS'),
        S3_PORT=os.getenv('S3_PORT'),
        S3_USE_SSL=os.getenv('S3_USE_SSL', 'true').lower() == 'true',
        S3_ACCESS_KEY=os.getenv('S3_ACCESS_KEY'),
        S3_SECRET_KEY=os.getenv('S3_SECRET_KEY'),
    )

    print(f"✓ Connected to S3: {s3.S3_ADDRESS}:{s3.S3_PORT}")
    print(f"  Bucket: {s3.S3_BUCKET}")
    print()

    # Upload model
    print(f"Uploading model to S3...")
    with open(model_file, 'rb') as f:
        model_data = BytesIO(f.read())
        s3.putFile(
            data=model_data.read(),
            path=s3_model_path,
            bucket=s3.S3_BUCKET,
            content_type='application/json'
        )
    print(f"✓ Uploaded: {s3_model_path}")

    # Upload preprocessing info
    print(f"Uploading preprocessing metadata to S3...")
    with open(prep_file, 'rb') as f:
        prep_data = BytesIO(f.read())
        s3.putFile(
            data=prep_data.read(),
            path=s3_prep_path,
            bucket=s3.S3_BUCKET,
            content_type='application/json'
        )
    print(f"✓ Uploaded: {s3_prep_path}")

    print()
    print("="*80)
    print("✓ Upload Complete!")
    print("="*80)
    print(f"\nModel files now available at:")
    print(f"  s3://{s3.S3_BUCKET}/{s3_model_path}")
    print(f"  s3://{s3.S3_BUCKET}/{s3_prep_path}")
    print()
    print("You can now run:")
    print("  cd projects/h2s && uv run dg dev")
    print()

if __name__ == "__main__":
    main()
