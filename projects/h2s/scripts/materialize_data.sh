#!/bin/bash
# Load environmental data from S3 and run the full prediction pipeline.
# Requires: .env file at repo root with S3 credentials.
# Requires: h2s_model_artifacts already materialized (run materialize_artifacts.sh first).
set -e
cd "$(dirname "$0")/.."
set -a; source .env; set +a

echo "Loading environmental data from S3 and running prediction pipeline..."
uv run dg launch -m h2s.definitions \
  --assets "h2s/raw_environmental_data,h2s/preprocessed_features,h2s/h2s_predictions,h2s/h2s_alerts,h2s/predictions_export,h2s/confusion_matrix_viz,h2s/model_comparison_viz,h2s/prediction_timeline_viz"
echo "✓ Done"
