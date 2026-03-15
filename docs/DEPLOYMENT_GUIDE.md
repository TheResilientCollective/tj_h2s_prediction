# H2S Prediction System - Deployment Guide

## Overview

This system generates H2S predictions for NESTOR - BES using a trained XGBoost model, orchestrated via a Dagster pipeline with S3 integration.

**Performance:**
- Orange Detection: 61.3% (84/137 events caught)
- Yellow Detection: 46.1% (113/245 events caught)
- Balanced Accuracy: 63.1%
- False Alarm Rate: 5.4%

---

## Quick Start

### 1. Environment Configuration

Create a `.env` file in `projects/h2s/`:

```bash
S3_BUCKET=test
S3_ADDRESS=oss.resilientservice.mooo.com
S3_PORT=443
S3_USE_SSL=true
S3_ACCESS_KEY=your_access_key
S3_SECRET_KEY=your_secret_key

# Optional
PUBLIC_BUCKET=test
LATEST_BASEPATH=latest/
```

### 2. Install & Start Dagster UI

```bash
cd projects/h2s
uv sync
uv run dg dev
```

The UI is available at http://localhost:3000. Schedules are activated there — see [Automated Schedules](#automated-schedules).

### 3. Validate Definitions Load

```bash
uv run dg check defs
uv run dg list defs
```

---

## Automated Schedules

All four schedules start in `RUNNING` state (`default_status=RUNNING`). They can be paused or resumed individually in the Dagster UI under **Automation → Schedules**.

| Schedule Name | Cron | UTC Time | Job |
|---|---|---|---|
| `forecast_prediction_schedule` | `0 */6 * * *` | 00:00, 06:00, 12:00, 18:00 | `forecast_prediction_job` |
| `daily_validation_schedule` | `0 8 * * *` | Daily at 08:00 | `daily_validation_job` |
| `monthly_data_schedule` | `0 2 1 * *` | 1st of month at 02:00 | `monthly_data_extraction_job` |
| `monthly_model_training_schedule` | `0 4 1 * *` | 1st of month at 04:00 | `monthly_model_training_job` |

**No cron setup is needed** — Dagster manages all scheduling internally.

### Enabling/Disabling Schedules

In the Dagster UI:
1. Go to **Automation → Schedules**
2. Click the toggle next to any schedule to pause or resume it
3. Click the schedule name to view run history and upcoming ticks

---

## Manual Runs

### Run the full forecast pipeline

```bash
uv run dg launch --job forecast_prediction_job
```

### Run the daily validation report

```bash
uv run dg launch --job daily_validation_job
```

### Materialize individual assets

```bash
# Load new environmental data from S3
uv run dg launch --assets raw_environmental_data

# Run prediction only (requires model + preprocessed data)
uv run dg launch --assets h2s_predictions

# Export predictions to S3
uv run dg launch --assets predictions_export
```

### Helper scripts (auto-load .env)

```bash
bash scripts/materialize_artiacts.sh   # Load model artifacts from S3
bash scripts/materialize_data.sh       # Load environmental data
```

---

## S3 Output Structure

```
s3://test/
├── tijuana/forecast/
│   ├── models/                               # Pre-trained model
│   │   ├── nestor_xgboost_weighted_model.json
│   │   └── nestor_preprocessing_info.json
│   ├── predictions/                          # Timestamped predictions
│   │   └── YYYY-MM-DD_HH/
│   │       ├── h2s_predictions.csv
│   │       ├── h2s_predictions.json
│   │       └── h2s_predictions.metadata.json
│   └── validation/                           # Daily validation reports
│       └── YYYY-MM-DD/validation_report.json
└── latest/tijuana/
    ├── weather_forecast/latest.csv           # Input data (from openmeteo pipeline)
    └── forecast_data/                        # Latest predictions (overwritten each run)
        ├── h2s_predictions.csv
        ├── h2s_predictions.json
        └── visualizations/
```

---

## Input Data Requirements

The pipeline reads weather and tidal data from `latest/tijuana/weather_forecast/latest.csv` in S3 (written by the openmeteo pipeline). For local testing, place data at `projects/h2s/data/latest.csv`.

### Required Columns

**Weather Data:**
- `temperature_2m`: Temperature at 2m (°C)
- `wind_speed_10m`: Wind speed at 10m (m/s)
- `wind_direction_10m`: Wind direction (degrees)
- `wind_gusts_10m`: Wind gusts (m/s)
- `precipitation`: Precipitation (mm)
- `relative_humidity_2m`: Relative humidity (%)
- `surface_pressure`: Surface pressure (hPa)
- `cloud_cover`: Cloud cover (%)
- `dewpoint_2m`: Dewpoint temperature (°C)

**Tidal/Flow Data:**
- `flow_rate_cms`: Water flow rate (m³/s)
- `tide_height_m`: Tide height (m)
- `tidal_state`: Tidal state (`flood`, `ebb`, `slack`, `slack low`, `slack high`)

**Categorical:**
- `wind_direction_categorical`: `N`, `NE`, `E`, `SE`, `S`, `SW`, `W`, `NW`

**Temporal:**
- `time`: Timestamp (ISO format: `2024-01-15T12:00:00Z`)

**Optional:**
- `site_name`: Site name (will filter to NESTOR - BES if present)
- `H2S`: Actual H2S value (used for validation only)

### Example Input Row

```csv
time,temperature_2m,wind_speed_10m,wind_direction_10m,wind_gusts_10m,precipitation,relative_humidity_2m,surface_pressure,cloud_cover,dewpoint_2m,wind_direction_categorical,flow_rate_cms,tide_height_m,tidal_state
2024-01-15T12:00:00Z,15.2,3.5,180,5.2,0.0,65,1013.2,25,8.1,S,125.5,1.2,flood
```

---

## Output Format

Predictions are exported to S3 as CSV and JSON. Each row contains all input columns plus:

| Column | Type | Description |
|---|---|---|
| `predicted_category` | string | `green`, `yellow`, or `orange` |
| `probability_green` | float | Model confidence in green (0–1) |
| `probability_yellow` | float | Model confidence in yellow (0–1) |
| `probability_orange` | float | Model confidence in orange (0–1) |
| `confidence` | float | Max probability (model certainty) |
| `alert` | bool | `True` if yellow or orange |

**Categories:**
- `green`: H2S < 5 ppb (safe)
- `yellow`: H2S 5–30 ppb (caution, monitor)
- `orange`: H2S ≥ 30 ppb (alert, take action)

**Confidence levels:**
- `> 0.7`: High confidence
- `0.5–0.7`: Moderate confidence
- `< 0.5`: Low confidence (uncertain)

---

## Threshold Tuning

Default thresholds: orange = 0.33, yellow = 0.33.

| Priority | Orange Threshold | Yellow Threshold | Orange Recall | False Positives |
|---|---|---|---|---|
| **Balanced** (default) | 0.33 | 0.33 | 61% | 5.4% |
| **Sensitive** | 0.25 | 0.30 | 70% | 10% |
| **Very Sensitive** | 0.20 | 0.25 | 75% | 15% |
| **Conservative** | 0.40 | 0.40 | 55% | 3% |
| **Very Conservative** | 0.50 | 0.50 | 45% | 1.5% |

Thresholds are configured in `projects/h2s/src/h2s/defs/h2s_pipeline.py`. **Recommendation:** Start with defaults, monitor for 2–4 weeks, then adjust.

---

## Monitoring & Maintenance

### Dagster UI run history

- Go to **Runs** in the Dagster UI to see all past and current runs
- Each run shows asset materialization status, logs, and timing
- Failed runs surface the full traceback in the run log

### Daily validation reports

`daily_validation_schedule` runs every morning at 08:00 UTC and writes a JSON report to `tijuana/forecast/validation/YYYY-MM-DD/`. Reports compare prior predictions against actual H2S measurements.

**Alert thresholds to watch:**
- Orange recall drops below 55%
- False positive rate exceeds 10%
- Balanced accuracy drops below 60%

### Retraining workflow

Retraining runs automatically on the 1st of each month:

1. `monthly_data_schedule` (02:00 UTC) — extracts and prepares training data
2. `monthly_model_training_schedule` (04:00 UTC) — trains all model variants in parallel
3. Review validation reports in `tijuana/forecast/models/training/` (S3)
4. Manually trigger `deploy_approved_model_job` in Dagster UI for the approved variant

The deployment step requires a human approval gate and is never triggered automatically.

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'h2s'`**
```bash
# Ensure you're in the right directory and run via uv
cd projects/h2s
uv sync
uv run dg dev
```

**`Validation error for S3Resource: Input should be a valid string`**
- S3 config in `definitions.py` must use `EnvVar('S3_BUCKET')`, not `os.getenv()`
- Verify `.env` file exists in `projects/h2s/` and all required keys are set

**`AttributeError: 'bytes' object has no attribute 'read'`**
- `S3Resource.getFile()` returns raw bytes, not a file-like object
- Pass bytes directly; do not call `.read()` on the result

**Assets not appearing in Dagster UI**
```bash
uv run dg list defs --json   # Verify assets are registered
```
- Assets must be explicitly imported and registered in `definitions.py`

**`KeyError: 'temperature_2m' not found`**
- Check column names in the input CSV against the Required Columns list above
- Column names are case-sensitive and must match exactly

**`ValueError: unknown categorical value`**
- `wind_direction_categorical` accepts only: `N`, `NE`, `E`, `SE`, `S`, `SW`, `W`, `NW`
- `tidal_state` accepts only: `flood`, `ebb`, `slack`, `slack low`, `slack high`

---

## Performance Expectations

**Strengths:**
- Detects 61.3% of critical orange events (H2S ≥ 30 ppb)
- Provides 1–3 hour advance warning
- Low false alarm rate (5.4%)
- Runs automatically every 6 hours

**Limitations:**
- Misses ~39% of orange events
- Cannot detect sudden spikes (< 1 hour)
- Requires input sensor data to be current and accurate
- Should supplement, not replace, direct H2S monitoring

**Good for:** Early warning, maintenance planning, trend analysis, reducing operator workload

**Not suitable for:** Emergency response, regulatory compliance as sole measure, life-safety decisions without verification

---

## Version History

**v2.0** (March 2026)
- Migrated to Dagster pipeline with automated scheduling
- S3-backed model loading and prediction export
- Added daily validation reports and monthly retraining workflow
- Preprocessing metadata converted from `.pkl` to `.json`

**v1.0** (December 2025)
- Initial production release
- XGBoost model with 61.3% orange recall
- Trained on 9,631 NESTOR - BES samples
- 20 engineered features, balanced class weighting

---

## Related Documentation

- `README.md` — Quick start and usage examples
- `NESTOR_BES_H2S_Forecasting_Report.md` — Full technical report
- `Complete_Model_Testing_Summary.md` — All algorithms tested
- `projects/h2s/tests/README.md` — Test documentation

---

*Last updated: March 2026*
