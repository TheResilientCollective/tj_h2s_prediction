# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is an H2S (Hydrogen Sulfide) prediction system for the Tijuana River region, covering three monitoring stations: IB_CIVIC_CTR, NESTOR__BES, and SAN_YSIDRO. The repository contains two implementations:

1. **Standalone Python scripts** (`src/`) - Original prediction scripts for direct usage
2. **Dagster orchestration pipeline** (`projects/h2s/`) - Production data pipeline with S3 integration

The system predicts H2S levels in three categories:
- **Green:** H2S < 5 ppb (safe)
- **Yellow:** 5 ≤ H2S < 30 ppb (caution)
- **Orange:** H2S ≥ 30 ppb (alert)

**Model Performance (hourly pipeline):** 61.3% orange detection rate, 5.4% false alarm rate.

## Project Structure

```
tj_h2s_prediction/
├── src/                          # Standalone prediction scripts
│   ├── predict_h2s.py           # Main prediction script
│   ├── batch_predict.py         # Batch processing
│   └── generate_visualizations.py
├── multihorizon/                 # Local MH training/forecast scripts
│   ├── train_multihorizon.py
│   ├── forecast_multihorizon.py
│   └── horizon_features.json
├── data/
│   ├── models_mh/               # Trained MH model pkl files (local)
│   ├── models_v2/               # Trained per-station models (local)
│   └── startmodels/             # Seed models for initial S3 upload
├── projects/h2s/                 # Dagster orchestration project
│   ├── src/h2s/
│   │   ├── definitions.py       # Dagster definitions (asset + job registration)
│   │   ├── constants.py         # S3 path constants + shared utilities
│   │   ├── defs/
│   │   │   ├── h2s_pipeline.py          # Hourly forecast pipeline (14 assets)
│   │   │   ├── h2s_daily_pipeline.py    # Daily analysis: source attribution + station forecasts
│   │   │   ├── h2s_dispersion_pipeline.py  # Dispersion modeling: Lagrangian + Gaussian + HYSPLIT
│   │   │   ├── h2s_multi_station_training.py  # Per-station model training (partitioned)
│   │   │   ├── h2s_multihorizon_training.py   # MH model training (partitioned, STOPPED)
│   │   │   ├── h2s_multihorizon_pipeline.py   # MH forecast pipeline (STOPPED)
│   │   │   ├── h2s_training_pipeline.py       # Legacy single-model training pipeline
│   │   │   ├── h2s_seed_models.py             # Seed models job for initial S3 upload
│   │   │   └── h2s_schedules.py               # All schedules and job definitions
│   │   ├── predictor/
│   │   │   ├── h2s_predictor.py  # H2SPredictor class with S3 loading
│   │   │   └── visualizations.py # Plot generators returning BytesIO
│   │   ├── dispersion/
│   │   │   ├── lagrangian.py    # Backward particle tracking + source attribution
│   │   │   ├── gaussian.py      # Forward Gaussian plume model
│   │   │   └── hysplit_controls.py  # HYSPLIT CONTROL file generation
│   │   ├── training/
│   │   │   ├── feature_builder.py       # ensure_base_features() for 43-feature set
│   │   │   ├── model_trainer.py         # train_and_select() for XGBoost/RF
│   │   │   ├── multi_station_trainer.py # Per-station training logic
│   │   │   ├── multihorizon_trainer.py  # MH training + EnsembleRegressor/Classifier
│   │   │   ├── relabeling.py            # H2S threshold relabeling
│   │   │   └── validation.py            # Model validation utilities
│   │   ├── resources/
│   │   │   ├── minio.py         # S3Resource
│   │   │   └── slack.py         # SlackAlertResource
│   │   └── utils/
│   │       └── store_assets.py  # S3 storage utilities
│   ├── scripts/                 # Helper scripts
│   │   ├── train_station_models.py  # Local per-station training
│   │   └── seed_models_to_s3.py     # Seed models to S3 (also available as Dagster job)
│   ├── tests/                   # Test suite
│   │   ├── conftest.py
│   │   ├── test_h2s_pipeline.py
│   │   ├── test_predictor.py
│   │   ├── test_s3_integration.py
│   │   └── README.md
│   ├── pytest.ini
│   └── pyproject.toml
├── nestor_xgboost_weighted_model.json  # 4.2 MB trained model (root copy)
└── nestor_preprocessing_info.json      # Feature metadata (JSON, not pickle)
```

## Operational Runbooks

### Initial Installation

Run once when deploying to a new environment:

```bash
cd projects/h2s
uv sync
cp .env.example .env   # fill in S3 credentials

# 1. Seed S3 with starter models (hourly pipeline + per-station daily pipeline)
uv run dg launch --job seed_models_job

# 2. Run hourly forecast pipeline
uv run dg launch --job forecast_prediction_job

# 3. Run daily analysis (source attribution + station forecasts + dashboard)
uv run dg launch --job daily_analysis_job
```

`seed_models_job` uploads:
- Hourly pipeline models from `data/startmodels/` → `tijuana/forecast/models/`
- Per-station daily models from `data/models_v2/` → `tijuana/forecast/models/stations/`

### Rebuilding Models (new training data available)

No approval gate is required — `station_deployment_job` acts as the explicit approval step.

```bash
cd projects/h2s

# 1. Train per-station models (partitioned: san_ysidro, nestor_bes, ib_civic_ctr)
#    Runs multi_station_training_data → per_station_trained_models → station_training_report
uv run dg launch --job multi_station_training_job

# 2. Review training metrics in Dagster UI (station_training_report asset metadata)

# 3. Deploy approved models to S3 (this IS the approval — running this job means you approve)
uv run dg launch --job station_deployment_job

# 4. Run daily analysis — it will re-load fresh models from S3
uv run dg launch --job daily_analysis_job
```

**Important:** `multi_station_training_job` stores models in Dagster's IO only.
`station_deployment_job` uploads them to S3 where the daily pipeline reads from.
Running `daily_analysis_job` after training but before deployment will use the previously deployed models.

### Running the Forecast Pipelines

```bash
cd projects/h2s

# Hourly H2S prediction (auto-runs every 6h via forecast_prediction_schedule)
uv run dg launch --job forecast_prediction_job

# Daily source attribution + station forecasts + dashboard (auto-runs daily at 8am)
uv run dg launch --job daily_analysis_job

# Dispersion modeling: 72h Gaussian forward forecast + alert check (auto-runs every 6h)
uv run dg launch --job dispersion_forecast_job

# Dispersion modeling: Weekly Lagrangian source attribution (Monday 02:30 UTC, STOPPED by default)
uv run dg launch --job dispersion_inversion_job
```

### Re-executing a Failed daily_analysis_job

If `daily_analysis_job` fails partway through, use **"Re-execute all"** in the Dagster UI — not "Re-execute failed steps". Re-executing only the failed step reads `multi_station_model_artifacts` from a stale IO cache and will fail again. Running all steps re-loads models fresh from S3.

### Dispersion Pipeline Operations

**Forward forecast** (`dispersion_forecast_job`, runs every 6h):
- Loads latest emission rates from S3 (or uses calibrated defaults: east=20, west=10, south=137 g/s)
- Runs 72h Gaussian plume model using FORECAST meteorology
- Checks next 6h for threshold crossings (30 ppb watch, 100 ppb critical)
- Sends Slack alert if thresholds exceeded
- Uploads HYSPLIT forward CONTROL bundle to S3 (no execution)

**Source attribution** (`dispersion_inversion_job`, weekly Monday 02:30 UTC, STOPPED):
- Runs Lagrangian backward particle tracking over inversion window (default: Feb 1 - Apr 1 2026)
- Computes ensemble source fractions from 16 candidate sources
- Groups sources into east/west/south zones, derives emission rates (g/s)
- Uploads emission_rates.json to S3 for use by forward forecasts
- Generates HYSPLIT backward CONTROL bundle (no execution)

**HYSPLIT bundles**: Download from S3 (`tijuana/dispersion/hysplit/{backward|forward}_bundle_latest.zip`), unzip, and run `bash run_hysplit_*.sh` in a HYSPLIT container or submit to NOAA READY.

## Common Commands

### Dagster Development

```bash
# Navigate to Dagster project
cd projects/h2s

# Install dependencies
uv sync

# Check definitions (validate assets load correctly)
uv run dg check defs

# List all assets and resources
uv run dg list defs

# Start Dagster UI (default: http://localhost:3000)
uv run dg dev

# Materialize a specific asset
uv run dg launch --assets h2s/h2s_predictions
```

### Training Scripts

```bash
cd projects/h2s

# Train per-station models locally (outputs to data/models_v2/YYYYMMDD/)
uv run python scripts/train_station_models.py \
  --obs ../../data/modeldata_h2s_nofill.parquet \
  --models ../../data/models_v2/$(date +%Y%m%d)

# Then seed to S3 via Dagster:
uv run dg launch --job seed_models_job
```

### Standalone Scripts

```bash
# Single prediction
python src/predict_h2s.py --input data.csv --output predictions.csv

# Batch processing
python src/batch_predict.py --input-dir ./data --output-dir ./predictions

# Alerts only (filter out green predictions)
python src/predict_h2s.py --input data.csv --output alerts.csv --filter-alerts

# Adjust sensitivity (lower threshold = more sensitive)
python src/predict_h2s.py --input data.csv --output predictions.csv --orange-threshold 0.25
```

### Testing

```bash
cd projects/h2s

# Install test dependencies
uv sync

# Run all tests
uv run pytest

# Run specific test file
uv run pytest tests/test_h2s_pipeline.py -v

# Run tests with coverage
uv run pytest --cov=h2s --cov-report=html

# Skip S3 integration tests (if credentials not available)
uv run pytest -m "not s3"

# Run only fast tests
uv run pytest -m "not slow"

# Stop on first failure
uv run pytest -x
```

### Testing S3 Integration

```bash
cd projects/h2s
uv run python -c "
from h2s.predictor.h2s_predictor import H2SPredictor
from h2s.resources.minio import S3Resource
import os

s3 = S3Resource(
    S3_BUCKET=os.getenv('S3_BUCKET'),
    S3_ADDRESS=os.getenv('S3_ADDRESS'),
    S3_PORT=os.getenv('S3_PORT'),
    S3_USE_SSL=os.getenv('S3_USE_SSL', 'true').lower() == 'true',
    S3_ACCESS_KEY=os.getenv('S3_ACCESS_KEY'),
    S3_SECRET_KEY=os.getenv('S3_SECRET_KEY'),
)

predictor = H2SPredictor.from_s3(
    s3,
    'tijuana/forecast/models/nestor_xgboost_weighted_model.json',
    'tijuana/forecast/models/nestor_preprocessing_info.json'
)
print(f'Model loaded: {len(predictor.feature_cols)} features')
"
```

## Architecture

### Active Pipelines

**`forecast_prediction_job`** (every 6h) — hourly H2S forecast for NESTOR-BES
```
h2s_model_artifacts → preprocessed_features → h2s_predictions → h2s_alerts → slack_alerts
                                                               → h2s_variant_predictions → h2s_ensemble_predictions
                                                               → predictions_export
h2s_model_artifacts → feature_importance_viz / confusion_matrix_viz / model_comparison_viz
                    → prediction_timeline_viz / cross_correlation_viz
```

**`daily_analysis_job`** (every 6h) — multi-station source attribution + 48h forecasts
```
multi_station_model_artifacts → source_attribution → daily_station_forecasts → daily_dashboard_viz
                                                                              → daily_summary_json
```

**`mh_forecast_job`** (every 6h, currently STOPPED) — 72h multi-horizon forecast
```
mh_model_artifacts → mh_observation_state → mh_forecasts → mh_dashboard_viz
                                                          → mh_summary_export → mh_slack_alerts
```

**`dispersion_inversion_job`** (weekly Monday 02:30 UTC, STOPPED by default) — backward source attribution
```
lagrangian_source_attribution → emission_rate_inversion → hysplit_controls_generation (backward CONTROL bundle)
```

**`dispersion_forecast_job`** (every 6h, RUNNING) — forward Gaussian plume forecast
```
emission_rate_inversion → gaussian_forward_forecast → dispersion_alert_check
                                                    → hysplit_controls_generation (forward CONTROL bundle)
```

### S3 Path Conventions

```
s3://test/
├── tijuana/forecast/
│   ├── models/
│   │   ├── nestor_xgboost_weighted_model.json  # hourly pipeline model
│   │   ├── nestor_preprocessing_info.json       # 43-feature preprocessing metadata
│   │   ├── deployment_metadata.json
│   │   ├── xgboost_base/model.json              # variants
│   │   ├── xgboost_smote/model.json
│   │   ├── random_forest/model.joblib
│   │   ├── stations/{station_key}/              # IB_CIVIC_CTR, NESTOR__BES, SAN_YSIDRO
│   │   │   ├── clf_5ppb.pkl
│   │   │   ├── clf_10ppb.pkl
│   │   │   └── regression.pkl
│   │   └── multihorizon/{horizon}/{station_key}/{task}.pkl  # 36 MH models
│   ├── output/YYYY-MM-DD_HH/                   # Timestamped hourly predictions
│   └── multihorizon/{date}/forecast_mh.csv     # MH forecast output
├── tijuana/dispersion/
│   ├── lagrangian/
│   │   ├── ensemble.json                        # Source attribution ensemble (16 candidate sources)
│   │   └── footprint_ensemble.parquet           # Ensemble footprint heatmap (lat × lon)
│   ├── emission_rates.json                      # Per-zone Q (east/west/south in g/s)
│   ├── hysplit/
│   │   ├── backward_bundle_{run_tag}.zip        # HYSPLIT backward CONTROL bundle
│   │   ├── backward_bundle_latest.zip
│   │   ├── forward_bundle_{run_tag}.zip         # HYSPLIT forward CONTROL bundle
│   │   └── forward_bundle_latest.zip
│   └── forward_forecast_{run_tag}.json          # Gaussian 72h plume forecast
└── latest/tijuana/
    ├── weather_forecast/latest.csv              # Input (from openmeteo.py)
    ├── tides/latest.csv
    ├── streamflow/latest.csv
    ├── dispersion/forward_forecast_latest.json  # Latest Gaussian forecast
    └── forecast_data/
        ├── h2s_predictions.{csv,json}
        ├── daily_summary.json
        ├── modeldata_h2s.csv                    # Historical H2S measurements
        └── visualizations/
```

### Key Design Decisions

**Why JSON instead of pickle for preprocessing?**
- S3-friendly (human-readable, portable, secure)
- Eliminates sklearn version warnings
- Uses dict lookups instead of LabelEncoder objects

**Why copy code from resilient_workflows_public?**
- Avoids sys.path manipulation and import issues
- Simplified `store_assets.py` without heavy dependencies
- Self-contained S3Resource in `h2s/resources/minio.py`

**Why tempfile for XGBoost model loading?**
- XGBoost requires file path (not BytesIO)
- `S3Resource.getFile()` returns raw bytes — write to tempfile, load, delete

**Why `EnsembleRegressor`/`EnsembleClassifier` in `multihorizon_trainer.py`?**
- Pickle deserialization requires the class to be importable from a stable module path
- Defined in `h2s.training.multihorizon_trainer` — do not move or rename

**Why FORECAST_DATA_PATH for Gaussian forward forecast?**
- `gaussian_forward_forecast` uses forecast meteorology (model_forecast.parquet), not observations
- This is the operational forecast use case — predicting future H2S based on weather forecasts
- Lagrangian inversion uses OBS_DATA_PATH for backward attribution on historical events

**Why 2-hour backward integration time for Lagrangian inversion?**
- Valley-scale sources: 1-7 km from sensor (travel time: 8-37 min @ 3 m/s wind)
- 6-hour integration was 10× too long (particles travel 64 km, miss local sources)
- 2-hour integration (21 km reach) is appropriate for Tijuana River Valley scale
- Critical fix: 6h gave east=0%, 2h gives east=46% (east sources now correctly detected)

**Current emission rates (from 2-hour Lagrangian inversion, Feb-Apr 2026):**
- East: 76.1 g/s (Stewart's Drain, Silva Drain, TJ crossing: 45.6% of total)
- West: 33.7 g/s (Tijuana Beach Outlet, Oneonta Slough: 20.2% of total)
- South: 57.2 g/s (Goat Canyon, Smugglers Gulch: 34.3% of total)
- Total: 167 g/s (conserved from March 13 2026 calibration event)

**Wind speed dependency (critical finding):**
- H2S strongly anti-correlated with wind speed (r = -0.246)
- Low wind (0-1 m/s): mean H2S = 49.9 ppb (weak dilution)
- High wind (>5 m/s): mean H2S = 6.4 ppb (strong dilution)
- Current Lagrangian model uses fixed diffusion (sigma_u=0.3) — should be wind-dependent (sigma ~ U^0.5)
- See WIND_SPEED_DEPENDENCY.md for recommended parameterization

**Why upload HYSPLIT bundles but not execute?**
- HYSPLIT requires ~20 GB GDAS meteorology files and specialized container environment
- Bundles are generated as CONTROL files + shell scripts, uploaded to S3
- User downloads and executes in local HYSPLIT container or submits to NOAA READY server
- Keeps Dagster pipeline lightweight and portable

## Environment Configuration

Create `.env` file (see `env.example`):

```bash
# S3/MinIO Configuration (required for Dagster pipeline)
S3_BUCKET=test
S3_ADDRESS=oss.resilientservice.mooo.com
S3_PORT=443
S3_USE_SSL=true
S3_ACCESS_KEY=your_access_key
S3_SECRET_KEY=your_secret_key

# Optional: Latest path configuration
PUBLIC_BUCKET=test
LATEST_BASEPATH=latest/

# Slack alerting
SLACK_TOKEN=xoxb-...
SLACK_CHANNEL=#h2s-alerts
SLACK_CHANNEL_FAILURES=#h2s-failures

# Deployment context
DAGSTER_DEPLOYMENT=local     # or production
ENV_LABEL=DEV                # shown in dashboard titles and Slack alerts
SCHED_HOSTNAME=sched         # for Dagster UI URL in failure alerts
HOST=local
```

**Dagster uses `EnvVar` for S3 and Slack credentials** - environment variables are loaded at runtime, not at definitions.py import time.

## Model Files

**Location:** Root directory and S3

- `nestor_xgboost_weighted_model.json` - 4.2 MB trained XGBoost classifier (hourly pipeline)
- `nestor_preprocessing_info.json` - metadata (feature names, class mappings)

**Features (43 total) — built by `feature_builder.py`:**
- Weather: temperature_2m, wind_speed_10m, wind_direction_10m, relative_humidity_2m, surface_pressure, precipitation, cloud_cover, dewpoint_2m
- Wind rolling averages (2h, 3h, 4h) + gusts rolling max
- Cyclical encodings: hour_sin/cos, month_sin/cos, wind_direction_sin/cos
- Flow: flow_rate_cms, flow_log, flow_low, flow_high, flow_lag_6h, flow_rolling_24h
- H2S lags: h2s_lag_1h/3h/6h, h2s_rolling_6h/24h
- SBIWTP: sbiwtp_flow_mgd, sbiwtp_anomaly, sbiwtp_deficit, etc.
- Stability/regime: is_night, source_regime, stable_atm
- Encoded: wind_direction_cat_encoded, tidal_state_encoded

**Classes:** ['green', 'orange', 'yellow']

## Daily Partitions and Validation Metrics

### Partition System

**Forecast and validation jobs use daily partitions** (start_date=2026-01-01, timezone=UTC):

```bash
# Run forecast for specific date
uv run dg launch --job forecast_prediction_job --partition 2026-04-02

# Run validation for specific date
uv run dg launch --job daily_validation_metrics_job --partition 2026-04-02

# Run full validation with monthly dashboard (requires >0 days of metrics)
uv run dg launch --job daily_validation_job --partition 2026-04-02
```

**Jobs:**
- `forecast_prediction_job` — Generates predictions for a date (uses forecast data from that date)
- `daily_validation_metrics_job` — Creates metrics.json only (for backfilling)
- `daily_validation_job` — Creates metrics.json + monthly dashboard (fails if zero metrics days available)

### Validation Metrics Accumulation

**Natural accumulation workflow** (recommended):

1. **Day 1**: Forecast runs → predictions stored to S3
2. **Day 2**: Validation runs → compares Day 1 predictions vs Day 1 actuals → creates metrics.json
3. **Days 3-7**: Repeat daily
4. **Day 8+**: Monthly dashboard generates successfully (uses last 30 days of metrics)

**Daily schedules:**
- `forecast_prediction_schedule`: Every 6 hours (00, 06, 12, 18 UTC) → materializes TODAY's partition
- `daily_validation_schedule`: Daily at 8 AM UTC → materializes YESTERDAY's partition

**Important:** Validation requires predictions and observations to have matching timestamps. The current system:
- ✅ Works for daily production runs (forecast uses today's data, validation uses yesterday's data)
- ❌ Cannot backfill historical validations (forecast data not partitioned by date)

### Historical Backfills (Future Enhancement)

**Current limitation:** `preprocessed_features` loads from `latest/tijuana/forecast_data/model_forecast.parquet` which always contains the most recent forecast, not historical forecasts. Backfilling partition `2026-03-26` loads today's forecast, generates predictions with today's timestamps, then validation finds zero matches with March 26 observations.

**Solution:** Partition forecast data by generation date. See `projects/h2s/FORECAST_DATA_PARTITIONING.md` for detailed implementation guide to enable true historical backfills.

### Metrics Storage

```
s3://test/tijuana/forecast/validation/
  2026-04-01/
    metrics.json          # Daily metrics (balanced accuracy, confusion matrix, FAR)
    confusion_matrix.png  # Visualization
    model_comparison.png
  2026-04-02/
    metrics.json
    ...
```

**metrics.json structure:**
```json
{
  "date": "2026-04-01",
  "site": "NESTOR__BES",
  "n_predictions": 462,
  "n_matched": 450,
  "match_rate": 0.974,
  "balanced_accuracy": 0.856,
  "false_alarm_rate": 0.034,
  "class_metrics": {
    "green": {"precision": 0.92, "recall": 0.95, "f1": 0.93},
    "yellow": {"precision": 0.78, "recall": 0.71, "f1": 0.74},
    "orange": {"precision": 0.88, "recall": 0.81, "f1": 0.84}
  },
  "confusion_matrix": [[240, 12, 3], [15, 145, 8], [2, 5, 20]]
}
```

## Troubleshooting

**"ModuleNotFoundError: No module named 'h2s'"**
- Ensure you're in `projects/h2s/` and run commands with `uv run`
- Check `uv sync` completed successfully

**"Validation error for S3Resource: Input should be a valid string"**
- S3 config must use `EnvVar('S3_BUCKET')` not `os.getenv('S3_BUCKET')`
- Dagster definitions.py already uses EnvVar correctly

**"AttributeError: 'bytes' object has no attribute 'read'"**
- `S3Resource.getFile()` returns raw bytes, not BytesIO
- Use `model_bytes` directly, not `model_bytes.read()`

**"Assets not appearing in Dagster UI"**
- Assets must be explicitly registered in `definitions.py`
- Check `uv run dg list defs --json` to see if assets are loaded
- Verify `from h2s.defs.h2s_pipeline import ...` in definitions.py

**"Too many false alarms / missing events"**
- Adjust thresholds in standalone scripts: `--orange-threshold 0.25` (more sensitive) or `0.40` (less sensitive)
- Default: 0.33 (61% detection, 5.4% false positives)

## Input Data Requirements

CSV must include these columns:
- `time` - Timestamp
- `temperature_2m`, `wind_speed_10m`, `wind_direction_10m`, `relative_humidity_2m`
- `surface_pressure`, `precipitation`, `cloud_cover`
- `wind_direction_categorical` - Cardinal direction (N, NE, E, etc.)
- `flow_rate_cms`, `tide_height_m`, `tidal_state` - Tidal data

See README.md for complete column list.

## Related Documentation

- `README.md` - Quick start, usage examples, model details
- `DEPLOYMENT_GUIDE.md` - Complete API reference, integration examples
- `NESTOR_BES_H2S_Forecasting_Report.md` - Technical report
- `Complete_Model_Testing_Summary.md` - Model evaluation
