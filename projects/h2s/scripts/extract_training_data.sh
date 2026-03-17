#!/bin/bash
# Extract and prepare monthly training data for a given month.
#
# Usage:
#   bash scripts/extract_training_data.sh [MONTH]
#
# MONTH - ISO month string, default: current month (e.g. 2026-03-01)
#
# Assets materialized:
#   monthly_training_data → relabeled_training_data → data_quality_report
#   → training_data → validation_data
#
# Run this before train_models.sh.
set -e
cd "$(dirname "$0")/.."
set -a; source .env; set +a

MONTH="${1:-$(date +%Y-%m-01)}"

echo "================================================"
echo "  H2S Training Data Extraction"
echo "================================================"
echo "  Month: $MONTH"
echo ""

uv run dg launch -m h2s.definitions \
  --job h2s/monthly_data_extraction_job \
  --partition "$MONTH"

echo ""
echo "✓ Data extraction complete for $MONTH"
echo "Next step: bash scripts/train_models.sh $MONTH"
