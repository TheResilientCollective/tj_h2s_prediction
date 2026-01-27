#!/bin/bash
# Materialize H2S pipeline using LOCAL test data (no S3 required for environmental data)
# Model is still loaded from S3

set -a
source ../../../.env
set +a

cd "$(dirname "$0")/.."

echo "================================================"
echo "Materializing H2S Pipeline with LOCAL TEST DATA"
echo "================================================"
echo ""
echo "Configuration:"
echo "  - Model: FROM S3 (tijuana/forecast/models/)"
echo "  - Environmental Data: FROM LOCAL (data/latest.csv)"
echo "  - H2S Actuals: Already in latest.csv"
echo ""

# Materialize with local test data config
uv run dg launch \
  --select "raw_environmental_data+" \
  --config config_local_test.yaml

echo ""
echo "✓ Pipeline materialized with local test data"
echo ""
echo "To use PRODUCTION mode (S3 data), run: bash scripts/materialize_data.sh"