#!/bin/bash
# Deploy an approved model variant to production.
#
# Usage:
#   bash scripts/deploy_model.sh MONTH VARIANT
#
# MONTH   - ISO month string (e.g. 2026-03-01)
# VARIANT - one of: xgboost_base | xgboost_smote | random_forest
#
# This triggers: deployment_approval → archived_previous_model → production_model_deployment
#
# WARNING: This overwrites the production model in S3. Review validation reports first.
set -e
cd "$(dirname "$0")/.."
set -a; source .env; set +a

MONTH="${1}"
VARIANT="${2}"

if [ -z "$MONTH" ] || [ -z "$VARIANT" ]; then
  echo "Usage: bash scripts/deploy_model.sh MONTH VARIANT" >&2
  echo "  MONTH   e.g. 2026-03-01" >&2
  echo "  VARIANT e.g. xgboost_base | xgboost_smote | random_forest" >&2
  exit 1
fi

echo "================================================"
echo "  H2S Model Deployment"
echo "================================================"
echo "  Month:   $MONTH"
echo "  Variant: $VARIANT"
echo ""
echo "WARNING: This will overwrite the production model in S3."
read -p "Continue? [y/N] " CONFIRM
if [[ "$CONFIRM" != "y" && "$CONFIRM" != "Y" ]]; then
  echo "Aborted."
  exit 0
fi
echo ""

uv run dg launch \
  --job deploy_approved_model_job \
  --partition "month=$MONTH|variant=$VARIANT"

echo ""
echo "✓ Deployment complete: $VARIANT ($MONTH) is now production"
