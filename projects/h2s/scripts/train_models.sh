#!/bin/bash
# Run the monthly model training pipeline for a given month and variant.
#
# Usage:
#   bash scripts/train_models.sh [MONTH] [VARIANT]
#
# MONTH   - ISO month string, default: current month (e.g. 2026-03-01)
# VARIANT - one of: xgboost_base | xgboost_smote | random_forest | all
#           default: all  (runs all three variants)
#
# Steps run: trained_model_cv → model_training_metrics → feature_importance_analysis
#            → validation_predictions → validation_report → model_comparison_report
#
# Requires: .env at repo root, and monthly data already extracted for MONTH
#           (run extract_training_data.sh first if needed).
set -e
cd "$(dirname "$0")/.."
set -a; source .env; set +a

MONTH="${1:-$(date +%Y-%m-01)}"
VARIANT="${2:-all}"

VARIANTS=("xgboost_base" "xgboost_smote" "random_forest")

if [ "$VARIANT" != "all" ]; then
  VARIANTS=("$VARIANT")
fi

echo "================================================"
echo "  H2S Model Training"
echo "================================================"
echo "  Month:    $MONTH"
echo "  Variants: ${VARIANTS[*]}"
echo ""

for V in "${VARIANTS[@]}"; do
  echo "--- Training variant: $V ---"
  uv run dg launch -m h2s.definitions \
    --job monthly_model_training_job \
    --partition "month=$MONTH|variant=$V"
  echo ""
done

echo "✓ Training complete"
echo "Review validation reports in S3 at: tijuana/forecast/models/training/$(echo $MONTH | tr -d -)/..."
echo "When ready to deploy: bash scripts/deploy_model.sh $MONTH <variant>"
