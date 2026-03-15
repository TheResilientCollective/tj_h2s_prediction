# H2S Pipeline Scripts

Helper scripts for running Dagster jobs. All scripts load `.env` from the repo root automatically.

---

## Prediction Pipeline

### `materialize_artifacts.sh`
Load the production model from S3.
```bash
bash scripts/materialize_artifacts.sh
```
Materializes: `h2s_model_artifacts`

### `materialize_data.sh`
Load live environmental data from S3 and run the full prediction pipeline.
```bash
bash scripts/materialize_data.sh
```
Materializes: `raw_environmental_data` and all downstream assets (preprocessed_features → h2s_predictions → h2s_alerts → visualizations → export)
**Requires:** `materialize_artifacts.sh` run first, S3 environmental data available.

### `materialize_local_test.sh`
Run the prediction pipeline using local data (no S3 environmental data required).
```bash
bash scripts/materialize_local_test.sh
```
Uses: `data/modeldata_h2s_nofill.parquet` (resolved automatically)
**Requires:** `materialize_artifacts.sh` run first (model still loaded from S3).

---

## Training Pipeline

The training pipeline has three phases that run in sequence:

```
extract_training_data.sh → train_models.sh → (review reports) → deploy_model.sh
```

### `extract_training_data.sh`
Extract and prepare training/validation data for a given month.
```bash
bash scripts/extract_training_data.sh [MONTH]
# e.g.
bash scripts/extract_training_data.sh 2026-03-01
```
Materializes: `monthly_training_data → relabeled_training_data → data_quality_report → training_data → validation_data`
Default MONTH: current month.

### `train_models.sh`
Train and validate model variants for a given month.
```bash
bash scripts/train_models.sh [MONTH] [VARIANT]
# e.g. train all three variants:
bash scripts/train_models.sh 2026-03-01

# or train a single variant:
bash scripts/train_models.sh 2026-03-01 xgboost_smote
```
**VARIANT** options: `xgboost_base` | `xgboost_smote` | `random_forest` | `all` (default)
Materializes: `trained_model_cv → model_training_metrics → feature_importance_analysis → validation_predictions → validation_report → model_comparison_report`
**Requires:** `extract_training_data.sh` run first for the same MONTH.

### `deploy_model.sh`
Deploy an approved model variant to production (overwrites S3 production model).
```bash
bash scripts/deploy_model.sh MONTH VARIANT
# e.g.
bash scripts/deploy_model.sh 2026-03-01 xgboost_smote
```
Prompts for confirmation before deploying.
Materializes: `deployment_approval → archived_previous_model → production_model_deployment`
**Requires:** `train_models.sh` complete and validation reports reviewed.

---

## Model Variants

| Variant | Description |
|---|---|
| `xgboost_base` | XGBoost with class weights only |
| `xgboost_smote` | XGBoost with SMOTE oversampling on hazard classes |
| `random_forest` | Random Forest with balanced class weights + SMOTE |

---

## Schedules (automatic)

The following run automatically on the 1st of each month:

| Schedule | Time (UTC) | Job |
|---|---|---|
| `monthly_data_schedule` | 02:00 | `monthly_data_extraction_job` |
| `monthly_model_training_schedule` | 04:00 | `monthly_model_training_job` (all variants) |

Manual deployment (`deploy_model.sh`) is always required after reviewing reports.

---

## Quick Reference: Full Monthly Retraining

```bash
# 1. Extract data for the month
bash scripts/extract_training_data.sh 2026-03-01

# 2. Train all variants
bash scripts/train_models.sh 2026-03-01

# 3. Review validation reports in S3:
#    tijuana/forecast/models/training/2026_03/<variant>/validation_report.*

# 4. Deploy the best variant
bash scripts/deploy_model.sh 2026-03-01 xgboost_smote

# 5. Refresh predictions
bash scripts/materialize_artifacts.sh
bash scripts/materialize_data.sh
```
