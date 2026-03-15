#!/bin/bash
# Load model artifacts from S3 into Dagster (h2s_model_artifacts asset).
# Requires: .env file at repo root with S3 credentials.
set -e
cd "$(dirname "$0")/.."
set -a; source .env; set +a

echo "Loading model artifacts from S3..."
uv run dg launch --assets h2s_model_artifacts
echo "✓ Done"
