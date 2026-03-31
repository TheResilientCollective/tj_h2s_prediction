
# H2S Prediction System — Tijuana River Valley (NESTOR - BES)

ML-based forecasting of H2S levels at the NESTOR - BES wastewater treatment site, orchestrated via Dagster with S3 integration.

---

## Model Performance

| Metric | Value |
|--------|-------|
| Orange Detection Rate | **61.3%** (84/137 critical events) |
| Yellow Detection Rate | 46.1% |
| Balanced Accuracy | 63.1% |
| False Alarm Rate | **5.4%** |
| Algorithm | XGBoost (3-class, weighted) |
| Features | 20 engineered features |
| Training Data | 9,631 samples, Nov 2023 – Jan 2025 |

**Categories:**
- 🟢 Green: H2S < 5 ppb (safe)
- 🟡 Yellow: 5 ≤ H2S < 30 ppb (caution)
- 🟠 Orange: H2S ≥ 30 ppb (alert)

### Threshold Tuning

| Setting | Orange Threshold | Orange Recall | False Positives |
|---------|-----------------|---------------|-----------------|
| Conservative | 0.40 | ~55% | ~3% |
| **Default** | **0.33** | **61%** | **5.4%** |
| Sensitive | 0.25 | ~70% | ~10% |
| Very Sensitive | 0.20 | ~75% | ~15% |

---

## Initial Installation

Run once when deploying to a new environment:

```bash
cd projects/h2s
uv sync
cp .env.example .env   # fill in S3 credentials

# 1. Seed S3 with all starter models (hourly + per-station daily pipeline)
uv run dg launch --job seed_models_job

# 2. Run the hourly forecast pipeline
uv run dg launch --job forecast_prediction_job

# 3. Run the daily analysis (source attribution + station forecasts + dashboard)
uv run dg launch --job daily_analysis_job
```

`seed_models_job` uploads starter models for both pipelines:
- `data/startmodels/` → `tijuana/forecast/models/` (hourly pipeline)
- `data/models_v2/` → `tijuana/forecast/models/stations/` (daily per-station pipeline)

---

## Rebuilding Models

Run when new training data is available. No separate approval gate — running `station_deployment_job` is the approval.

```bash
cd projects/h2s

# 1. Train per-station models (partitioned by station: san_ysidro, nestor_bes, ib_civic_ctr)
uv run dg launch --job multi_station_training_job

# 2. Review metrics in the Dagster UI (station_training_report asset)

# 3. Deploy to S3 — running this job IS the approval
uv run dg launch --job station_deployment_job

# 4. Re-run daily analysis with new models
uv run dg launch --job daily_analysis_job
```

> **Note:** `multi_station_training_job` stores trained models in Dagster's IO only. `station_deployment_job` uploads them to S3, where the daily pipeline reads from. Always run deployment before expecting updated forecasts.

---

## Running the Forecast Pipelines

```bash
cd projects/h2s

# Hourly H2S predictions (also runs automatically every 6h)
uv run dg launch --job forecast_prediction_job

# Daily source attribution + station forecasts + Slack summary (also runs automatically at 8am)
uv run dg launch --job daily_analysis_job
```

### Automated Schedules

| Schedule | Cron | Description |
|----------|------|-------------|
| `forecast_prediction_schedule` | `0 */6 * * *` | Full prediction pipeline every 6 hours |
| `daily_validation_schedule` | `0 8 * * *` | Compare yesterday's predictions vs actuals |
| `monthly_data_schedule` | `0 2 1 * *` | Extract monthly training data |
| `monthly_model_training_schedule` | `0 4 1 * *` | Retrain model variants |

Both forecast and validation schedules start in `RUNNING` state and activate automatically when Dagster starts.

### Dagster UI

```bash
cd projects/h2s
uv run dg dev   # http://localhost:3000
```

---

## S3 Path Structure

```
s3://test/
├── tijuana/forecast/
│   ├── models/                              # Pre-trained model files
│   │   ├── nestor_xgboost_weighted_model.json
│   │   ├── nestor_preprocessing_info.json
│   │   └── training/modeldata_h2s.parquet   # Historical training data
│   ├── predictions/
│   │   └── YYYY-MM-DD_HH/                   # Per 6-hourly run
│   │       └── h2s_predictions.{csv,json}
│   └── validation/
│       └── YYYY-MM-DD/                      # Daily validation report
│           ├── confusion_matrix.png
│           ├── model_comparison.png
│           ├── prediction_timeline.png
│           └── daily_predictions_combined.{csv,json}
└── latest/tijuana/
    ├── weather_forecast/latest.csv          # Input data (from openmeteo)
    └── forecast_data/                       # Latest predictions
        └── h2s_predictions.{csv,json}
```

---

## Testing

All tests live in `projects/h2s/tests/`. Run from `projects/h2s/`:

```bash
cd projects/h2s
uv sync
```

### Run the pipeline logic tests (no S3 required)

```bash
uv run pytest tests/test_h2s_pipeline.py -v
```

Tests asset logic in isolation — preprocessing, prediction output format, probability validation, alert filtering — using mocked dependencies. This is the fastest way to verify pipeline correctness without any infrastructure.

### Run all unit tests (no S3 required)

```bash
uv run pytest -m "not s3" -v
```

### Run a specific test class or function

```bash
# All tests in a class
uv run pytest tests/test_h2s_pipeline.py::TestPreprocessedFeatures -v

# Single test
uv run pytest tests/test_h2s_pipeline.py::TestPreprocessedFeatures::test_creates_temporal_features -v
```

### Run S3 integration tests (requires `.env`)

```bash
# Requires S3_BUCKET, S3_ADDRESS, S3_ACCESS_KEY, S3_SECRET_KEY in .env
uv run pytest tests/test_s3_integration.py -v
```

### Run all tests with coverage

```bash
uv run pytest --cov=h2s --cov-report=html
# Open htmlcov/index.html
```

### Test files

| File | Description | Requires S3 |
|------|-------------|-------------|
| `test_h2s_pipeline.py` | Asset logic (preprocessing, predictions, alerts) | No |
| `test_predictor.py` | H2SPredictor class unit tests | No |
| `test_asset_materialization.py` | Dagster asset materialization with mocked resources | No |
| `test_training_pipeline.py` | Training pipeline asset logic | No |
| `test_s3_integration.py` | S3 upload/download, model loading from S3 | Yes |

### Test markers

```bash
uv run pytest -m "not slow"       # Skip slow tests
uv run pytest -m integration      # Integration tests only
uv run pytest -m s3                # S3 tests only
uv run pytest -x                   # Stop on first failure
```

---

## Environment Configuration

Create `projects/h2s/.env`:

```bash
S3_BUCKET=test
S3_ADDRESS=oss.resilientservice.mooo.com
S3_PORT=443
S3_USE_SSL=true
S3_ACCESS_KEY=your_access_key
S3_SECRET_KEY=your_secret_key
```

---

## Project Structure

```
tj_h2s_prediction/
├── projects/h2s/                  # Dagster pipeline (primary)
│   ├── src/h2s/
│   │   ├── definitions.py         # Asset, job, schedule registration
│   │   ├── constants.py           # S3 path constants
│   │   ├── defs/
│   │   │   ├── h2s_pipeline.py    # Prediction pipeline assets
│   │   │   ├── h2s_training_pipeline.py  # Monthly retraining assets
│   │   │   └── h2s_schedules.py   # Jobs and schedules
│   │   ├── predictor/
│   │   │   ├── h2s_predictor.py   # H2SPredictor class
│   │   │   └── visualizations.py  # Plot generators
│   │   ├── resources/minio.py     # S3Resource
│   │   └── utils/store_assets.py  # S3 storage utilities
│   ├── scripts/
│   │   └── train_station_models.py  # Train RF/XGB models per station locally
│   └── tests/                     # Test suite
├── src/                           # Standalone prediction scripts
│   ├── h2s_daily_analysis.py      # Source attribution + 48h forecast
│   ├── predict_h2s.py             # Single-file prediction
│   ├── batch_predict.py           # Multi-file batch prediction
│   └── generate_visualizations.py # Feature importance & comparison plots
├── nestor_xgboost_weighted_model.json  # 4.2 MB trained model
└── nestor_preprocessing_info.json      # Feature metadata
```

---

## Standalone Scripts

Scripts in `src/` and `projects/h2s/scripts/` for training, prediction, and visualization.

```bash
# Train per-station models locally (outputs to data/models_v2/YYYYMMDD/)
cd projects/h2s
uv run python scripts/train_station_models.py \
  --obs ../../data/modeldata_h2s_nofill.parquet \
  --models ../../data/models_v2/$(date +%Y%m%d)
# Then seed to S3: uv run dg launch --job seed_models_job

# Single-file prediction
python src/predict_h2s.py --input data.csv --models ./models --output ./results

# Batch prediction (all stations)
python src/batch_predict.py --obs data/model_forecast.csv --models ./models --output ./output

# Daily source attribution + 48h forecast
python src/h2s_daily_analysis.py --obs data/modeldata_h2s_nofill.parquet --forecast model_forecast.parquet --models ./models --output ./output

# Generate analysis plots (all stations)
python src/generate_visualizations.py --models ./models --output ./reports
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError: No module named 'h2s'` | Run `uv sync` from `projects/h2s/`, use `uv run` prefix |
| `Validation error for S3Resource` | Use `EnvVar('S3_BUCKET')` not `os.getenv()` in definitions |
| Assets not in Dagster UI | Check `uv run dg list defs --json`; verify registration in `definitions.py` |
| Training job fails with FileNotFoundError | Upload training data to `tijuana/forecast/models/training/modeldata_h2s.parquet` in S3 |
| Too many false alarms | Increase threshold: `--orange-threshold 0.40` |
| Missing too many events | Decrease threshold: `--orange-threshold 0.25` |
