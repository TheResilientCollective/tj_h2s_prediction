#!/bin/bash
# Load environmental data from S3 and run the full prediction pipeline.
# Requires: .env file at repo root with S3 credentials.
# Requires: h2s_model_artifacts already materialized (run materialize_artifacts.sh first).
set -e
cd "$(dirname "$0")/.."
set -a; source .env; set +a

echo "Loading environmental data from S3 and running prediction pipeline..."
uv run dg launch --assets raw_environmental_data+
echo "✓ Done"
