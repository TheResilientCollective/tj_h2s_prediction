#!/bin/bash
# Run prediction pipeline with local data (no S3 environmental data required).
# Model is still loaded from S3 — run materialize_artifacts.sh first.
# Requires: .env file at repo root with S3 credentials.
set -e
cd "$(dirname "$0")/.."
set -a; source .env; set +a

DATA_PATH="$(cd ../../data && pwd)/modeldata_h2s_nofill.parquet"

if [ ! -f "$DATA_PATH" ]; then
  echo "ERROR: Local data file not found: $DATA_PATH" >&2
  exit 1
fi

echo "================================================"
echo "  H2S Pipeline — LOCAL DATA MODE"
echo "================================================"
echo "  Data: $DATA_PATH"
echo ""

echo "[1/2] Loading model artifacts from S3..."
uv run dg launch -m h2s.definitions --assets "h2s/h2s_model_artifacts"
echo ""

echo "[2/2] Running prediction pipeline with local data..."
# Write a temporary config with the resolved absolute path
TMPCONFIG=$(mktemp /tmp/h2s_local_config_XXXXXX.yaml)
trap "rm -f $TMPCONFIG" EXIT

cat > "$TMPCONFIG" <<EOF
ops:
  h2s/raw_environmental_data:
    config:
      use_local_data: true
      local_data_path: "$DATA_PATH"
EOF

uv run dg launch -m h2s.definitions \
  --assets "h2s/raw_environmental_data,h2s/preprocessed_features,h2s/h2s_predictions,h2s/h2s_alerts,h2s/predictions_export,h2s/confusion_matrix_viz,h2s/model_comparison_viz,h2s/prediction_timeline_viz" \
  --config "$TMPCONFIG"

echo ""
echo "✓ Pipeline complete (local data mode)"
echo "To run with live S3 data: bash scripts/materialize_data.sh"
