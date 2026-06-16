# Validation and Accuracy Reporting

This guide covers how to validate H2S prediction models, track accuracy metrics, and generate accuracy reports. There are two approaches: **Dagster pipeline jobs** for continuous validation and **backfill scripts** for historical backfills and one-off analysis.

## Overview

The validation system measures model performance by comparing predictions against observed H2S values. It supports two strategies:

1. **Dagster Jobs** — Continuous daily validation integrated with forecast pipelines
2. **Backfill Scripts** — Standalone scripts for historical backfills and one-off validation runs

## Data Flow

```
Observation Data (modeldata_h2s_nofill.parquet)
    ↓
[backfill_validation.py OR Dagster daily_validation_job]
    ↓
Validation Metrics (metrics.json per date)
    ├── daily/{YYYY-MM-DD}/metrics.json
    ├── daily_station/{YYYY-MM-DD}/daily_station/metrics.json
    └── ... (per-pipeline variants)
    ↓
[backfill_accuracy_reports.py OR Dagster accuracy_reporting_job]
    ↓
Accuracy Reports (scorecards)
    ├── accuracy_reports/daily/{YYYY-MM-DD}/scorecard.json
    ├── accuracy_reports/rolling/{7d,30d,90d}/scorecard.json
    ├── accuracy_reports/monthly/{YYYY-MM}/scorecard.json
    └── accuracy_reports/latest.json
```

## Backfill Scripts (Recommended for Historical Data)

The backfill scripts are standalone Python scripts for validating historical observations and generating accuracy reports.

### `backfill_validation.py`

Generates validation metrics by running models against historical observations. Supports three modes:

#### Mode 1: Forecast Mode (default)
Compares existing hourly predictions on S3 against observations.

```bash
cd projects/h2s
source .env
uv run python scripts/backfill_validation.py
```

**Use case:** When you have stored forecast predictions from the forecast_prediction_job and want to validate them against actual observations.

#### Mode 2: Hindcast Mode
Runs the production hourly NESTOR-BES model (nestor_xgboost_weighted_model.json) against historical observations.

```bash
uv run python scripts/backfill_validation.py --hindcast
uv run python scripts/backfill_validation.py --hindcast --start 2026-01-01 --end 2026-06-15
```

**Use case:** Evaluate the production hourly model's skill on historical data (oracle/one-step inputs).

**Output location:** `tijuana/forecast/validation/{date}/metrics.json`

#### Mode 3: Daily Station Mode
Runs per-station models (all 3 stations: SAN_YSIDRO, NESTOR__BES, IB_CIVIC_CTR) against historical observations.

```bash
uv run python scripts/backfill_validation.py --daily-station
uv run python scripts/backfill_validation.py --daily-station --start 2026-03-01 --end 2026-06-15
```

**Use case:** Validate the deployed per-station models and populate the accuracy reporting pipeline with multi-station metrics.

**Output location:** `tijuana/forecast/validation/{date}/daily_station/metrics.json`

#### Options

```bash
--start DATE            Start date (YYYY-MM-DD), default: 2026-03-01
--end DATE              End date (YYYY-MM-DD), default: today
--dry-run               Show what would be done without writing to S3
--overwrite             Overwrite existing metrics files
```

### `backfill_accuracy_reports.py`

Generates accuracy scorecard reports from validation metrics. Requires metrics to exist in S3 (from backfill_validation.py).

```bash
uv run python scripts/backfill_accuracy_reports.py
uv run python scripts/backfill_accuracy_reports.py --start 2026-03-01 --end 2026-06-15
```

**Output structure:**
```
s3://{bucket}/tijuana/forecast/accuracy_reports/
├── daily/{YYYY-MM-DD}/scorecard.json      # Daily metrics for each date
├── rolling/7d/scorecard.json               # 7-day rolling window
├── rolling/30d/scorecard.json              # 30-day rolling window
├── rolling/90d/scorecard.json              # 90-day rolling window
├── monthly/{YYYY-MM}/scorecard.json        # Monthly summary
└── latest.json                              # Latest rollup (single source of truth)
```

#### Options

```bash
--start DATE            Start date (YYYY-MM-DD), default: 2026-03-01
--end DATE              End date (YYYY-MM-DD), default: today
--dry-run               Show what would be done without writing
```

### Complete Workflow Example

To backfill validation for a 3-month period and generate accuracy reports:

```bash
cd projects/h2s
source .env

# 1. Generate validation metrics from per-station models
echo "Backfilling validation metrics..."
uv run python scripts/backfill_validation.py --daily-station \
  --start 2026-03-01 --end 2026-06-15

# 2. Generate accuracy reports
echo "Generating accuracy reports..."
uv run python scripts/backfill_accuracy_reports.py \
  --start 2026-03-01 --end 2026-06-15

echo "Done! Check S3 at: s3://{bucket}/tijuana/forecast/accuracy_reports/"
```

## Dagster Pipeline Jobs

For continuous validation integrated with forecast pipelines, use Dagster jobs.

### Daily Validation Job

The `daily_validation_job` creates metrics for the previous day by comparing that day's forecast predictions against actual observations.

**Natural accumulation workflow:**
1. **Day 1**: Forecast job runs → stores predictions
2. **Day 2**: Validation job runs → compares Day 1 predictions vs. Day 1 actuals → writes metrics.json
3. **Days 3+**: Repeat daily
4. **Day 8+**: Monthly dashboard generates successfully

**Schedule:** Daily at 8 AM UTC (materialized YESTERDAY's partition)

```bash
cd projects/h2s

# Run validation for a specific date
uv run dg launch --job daily_validation_job --partition 2026-06-15

# Run full validation with monthly dashboard
# (requires >0 days of metrics available)
uv run dg launch --job daily_validation_job --partition 2026-06-15
```

### Validation Store + Skill Curves

The `station_forecast_validation_rebuild_job` joins all stored product predictions against measured H2S and computes per-lead-hour skill curves.

```bash
# Rebuild validation store from ALL stored product runs
uv run dg launch --job station_forecast_validation_rebuild_job
```

**Outputs:**
- `tijuana/forecast/validation_store/validation.parquet` — consolidated, rebuildable parquet
- `tijuana/forecast/validation_store/skill_curves.parquet` — per-(product, variant, lead-hour) metrics
- `tijuana/forecast/validation_store/skill_report.json` — summary statistics

## Metrics Schema

### Daily Metrics (v2 schema)

```json
{
  "schema_version": 2,
  "date": "2026-06-15",
  "pipeline": "daily_station",
  "generated_at": "2026-06-16T10:30:00+00:00",
  "sites": {
    "SAN_YSIDRO": {
      "n_predictions": 24,
      "n_matched_observations": 24,
      "match_rate": 1.0,
      "balanced_accuracy": 0.823,
      "false_alarm_rate": 0.045,
      "confusion_matrix": [[18, 2, 0], [1, 2, 0], [0, 0, 1]],
      "class_metrics": {
        "green": {"precision": 0.90, "recall": 0.95, "f1": 0.92},
        "yellow": {"precision": 0.50, "recall": 0.50, "f1": 0.50},
        "orange": {"precision": 1.00, "recall": 0.50, "f1": 0.67}
      }
    },
    "NESTOR__BES": { ... },
    "IB_CIVIC_CTR": { ... }
  }
}
```

### Accuracy Scorecard

```json
{
  "period": "2026-06-09_2026-06-15",
  "window_type": "7d",
  "generated_at": "2026-06-16T10:35:00+00:00",
  "summary": {
    "n_days": 7,
    "n_predictions": 504,
    "n_matched_observations": 504,
    "balanced_accuracy": 0.512,
    "orange_recall": 0.667,
    "orange_precision": 0.400,
    "false_alarm_rate": 0.023
  },
  "sites": {
    "SAN_YSIDRO": { ... },
    "NESTOR__BES": { ... },
    "IB_CIVIC_CTR": { ... }
  }
}
```

## Production vs. Development

The validation system reads from the bucket configured in your `.env`:

- **Development:** `S3_BUCKET=test` (default)
- **Production:** `S3_BUCKET=resilentpublic`

To validate production data:

```bash
# Validate production bucket
S3_BUCKET=resilentpublic uv run python scripts/backfill_validation.py --daily-station
S3_BUCKET=resilentpublic uv run python scripts/backfill_accuracy_reports.py
```

Metrics and reports are stored in the respective bucket's accuracy_reports path.

## Model Loading

The backfill scripts automatically handle model variants:

1. **Try suffixed variant names first** (current deployment):
   - `regression_evidence.pkl` (Evidence variant — 33 features)
   - `regression_lean.pkl` (Lean variant — 19 features)

2. **Fall back to un-suffixed legacy names** (backwards compatibility):
   - `regression.pkl` (legacy single-variant deployment)

Features are loaded from:
- `features_evidence.json` or `features_lean.json` (variant-specific)
- `features.json` (legacy)

## Troubleshooting

### "No predictions generated — skipping"
The models weren't found in S3. Check:
```bash
# List available models
s3cmd ls s3://test/tijuana/forecast/models/stations/
```

Models must be deployed first:
```bash
cd projects/h2s
uv run dg launch --job station_model_training_job --partition san_ysidro,nestor_bes,ib_civic_ctr
uv run dg launch --job station_model_deployment_job --partition san_ysidro,nestor_bes,ib_civic_ctr
```

### "No validation metrics found"
Ensure metrics exist in S3:
```bash
# Check daily_station metrics
s3cmd ls s3://test/tijuana/forecast/validation/2026-06-15/daily_station/
```

If empty, run backfill_validation.py first:
```bash
uv run python scripts/backfill_validation.py --daily-station --start 2026-06-15 --end 2026-06-15
```

### Low accuracy scores
Check:
1. **Model quality** — Compare against station_models_backtest.py results
2. **Data quality** — Observation data may be sparse or noisy
3. **Feature alignment** — Ensure feature columns match deployment schema

Run backtest to isolate model vs. pipeline issues:
```bash
uv run python scripts/station_models_backtest.py \
  --data ../../data/modeldata_h2s_nofill.parquet \
  --output ./output/station_backtest/
```

## When to Use What

| Use Case | Tool |
|----------|------|
| Validate per-station models on historical data | `backfill_validation.py --daily-station` |
| Validate hourly production model on historical data | `backfill_validation.py --hindcast` |
| Generate accuracy reports from existing metrics | `backfill_accuracy_reports.py` |
| Test model skill with oracle (one-step) inputs | `station_models_backtest.py` |
| Continuous daily validation (integrated) | Dagster `daily_validation_job` |
| Multi-lead skill curves (recursive) | Dagster `station_forecast_validation_rebuild_job` |

## See Also

- CLAUDE.md — Operational runbooks and architecture overview
- projects/h2s/tests/ — Validation test suite
- accuracy_reporting_pipeline.py — Dagster job definitions for accuracy reporting