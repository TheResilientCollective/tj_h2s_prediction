# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is an H2S (Hydrogen Sulfide) prediction system for the NESTOR - BES wastewater treatment site. The repository contains two implementations:

1. **Standalone Python scripts** (`src/`) - Original prediction scripts for direct usage
2. **Dagster orchestration pipeline** (`projects/h2s/`) - Production data pipeline with S3 integration

The system uses an XGBoost classification model to predict H2S levels in three categories:
- **Green:** H2S < 5 ppb (safe)
- **Yellow:** 5 ≤ H2S < 15 ppb (caution)
- **Orange:** H2S ≥ 15 ppb (alert)

**Model Performance:** 61.3% orange detection rate, 5.4% false alarm rate.

## Project Structure

```
tj_h2s_prediction/
├── src/                          # Standalone prediction scripts
│   ├── predict_h2s.py           # Main prediction script
│   ├── batch_predict.py         # Batch processing
│   └── generate_visualizations.py
├── projects/h2s/                 # Dagster orchestration project
│   ├── src/h2s/
│   │   ├── definitions.py       # Dagster definitions (asset registration)
│   │   ├── defs/
│   │   │   └── h2s_pipeline.py  # 7 pipeline assets (data → prediction → export)
│   │   ├── predictor/
│   │   │   ├── h2s_predictor.py # H2SPredictor class with S3 loading
│   │   │   └── visualizations.py # Plot generators returning BytesIO
│   │   ├── resources/
│   │   │   └── minio.py         # S3Resource (copied from resilient_workflows_public)
│   │   └── utils/
│   │       └── store_assets.py  # S3 storage utilities (simplified)
│   ├── scripts/                 # Helper scripts for asset materialization
│   │   ├── materialize_artiacts.sh  # Load model from S3
│   │   └── materialize_data.sh      # Load environmental data
│   ├── tests/                   # Test suite
│   │   ├── conftest.py         # Pytest configuration and fixtures
│   │   ├── test_h2s_pipeline.py     # Asset logic tests
│   │   ├── test_predictor.py        # H2SPredictor unit tests
│   │   ├── test_s3_integration.py   # S3 operations tests (requires credentials)
│   │   └── README.md           # Test documentation
│   ├── pytest.ini               # Pytest configuration
│   └── pyproject.toml           # Dependencies (dagster, xgboost, minio, pytest, etc.)
├── nestor_xgboost_weighted_model.json  # 4.2 MB trained model
├── nestor_preprocessing_info.json      # Feature metadata (JSON, not pickle)
├── convert_preprocessing.py            # One-time pickle→JSON converter
└── upload_models_to_s3.py             # Upload models to S3
```

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
uv run dg launch --assets h2s_predictions

# Helper scripts (automatically load .env)
bash scripts/materialize_artiacts.sh   # Load model from S3
bash scripts/materialize_data.sh       # Load environmental data
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

### Model Management

```bash
# Convert preprocessing from pickle to JSON (one-time, already done)
python convert_preprocessing.py

# Upload models to S3 (requires .env with S3 credentials)
cd projects/h2s && uv run python ../../upload_models_to_s3.py
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

### Dagster Pipeline Flow

The pipeline consists of 7 assets organized into 4 groups:

**1. Model Management (`h2s_model`)**
- `h2s_model_artifacts` - Loads XGBoost model from S3 at `tijuana/forecast/models/`
  - Uses `H2SPredictor.from_s3()` with tempfile workaround (XGBoost requires file path)
  - Preprocessing metadata is JSON (not pickle) for S3 compatibility

**2. Data Ingestion (`h2s_prediction`)**
- `raw_environmental_data` - Loads weather/tidal data from S3 with local fallback
  - Primary: `latest/tijuana/weather_forecast/latest.csv` (from openmeteo.py asset)
  - Fallback: Local `/data/latest.csv` for testing

**3. Prediction Pipeline (`h2s_prediction`)**
- `preprocessed_features` - Applies feature engineering
  - Cyclical encoding (sin/cos for hour, wind direction)
  - Interaction features (wind×temp, humidity×temp)
  - Categorical encoding using dict lookups (not sklearn LabelEncoder)
- `h2s_predictions` - Generates predictions with probabilities
  - Returns: predicted_category, probability_green/yellow/orange, confidence, alert
- `h2s_alerts` - Filters to orange/yellow predictions only

**4. Visualization & Export (`h2s_visualization`, `h2s_export`)**
- `feature_importance_viz` - Generates and uploads plot to S3
  - Returns BytesIO (not file) for direct S3 upload
  - Stored in both timestamped and `latest/` paths
- `predictions_export` - Exports predictions as CSV/JSON to S3
  - Uses `store_assets.store_dataframe_to_s3()` utility
  - Dual storage: `tijuana/forecast/output/` (timestamped) + `latest/tijuana/forecast_data/`

### S3 Path Conventions

Following `resilient_workflows_public` patterns:

```
s3://test/
├── tijuana/forecast/
│   ├── models/                         # Pre-trained model (uploaded once)
│   │   ├── nestor_xgboost_weighted_model.json
│   │   └── nestor_preprocessing_info.json
│   └── output/                         # Timestamped predictions
│       ├── YYYY-MM-DD_HH/h2s_predictions.{csv,json,metadata.json}
│       └── visualizations/YYYY-MM-DD/feature_importance.png
└── latest/
    └── tijuana/
        ├── weather_forecast/latest.csv  # Input data (from openmeteo.py)
        └── forecast_data/              # Latest predictions
            ├── h2s_predictions.{csv,json}
            └── visualizations/feature_importance.png
```

### Key Design Decisions

**Why JSON instead of pickle for preprocessing?**
- S3-friendly (human-readable, portable, secure)
- Eliminates sklearn version warnings
- Uses dict lookups instead of LabelEncoder objects

**Why copy code from resilient_workflows_public?**
- Avoids sys.path manipulation and import issues
- Simplified `store_assets.py` without heavy dependencies (geopandas, pydantic_schemaorg)
- Self-contained S3Resource in `h2s/resources/minio.py`

**Why tempfile for XGBoost model loading?**
- XGBoost requires file path (not BytesIO)
- `S3Resource.getFile()` returns raw bytes
- Tempfile written, model loaded, tempfile deleted

**Asset dependency chain:**
```
h2s_model_artifacts ────────────┐
                                ├─→ preprocessed_features ─→ h2s_predictions ──┬─→ h2s_alerts
raw_environmental_data ─────────┘                                             │
                                                                               └─→ predictions_export
h2s_model_artifacts ─────────────────────────────────────────────────────────────→ feature_importance_viz
```

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
```

**Dagster uses `EnvVar` for S3 credentials** - environment variables are loaded at runtime, not at definitions.py import time.

## Model Files

**Location:** Root directory and S3

- `nestor_xgboost_weighted_model.json` - 4.2 MB trained XGBoost classifier
- `nestor_preprocessing_info.json` - 948 bytes metadata (feature names, class mappings)

**Features (20 total):**
- Weather: temperature_2m, wind_speed_10m, wind_direction_10m, relative_humidity_2m, surface_pressure, precipitation, etc.
- Tidal: flow_rate_cms, tide_height_m, tidal_state
- Derived: hour_sin, hour_cos, wind_direction_sin, wind_direction_cos, wind_temp_interaction, humidity_temp_interaction
- Encoded: wind_direction_cat_encoded, tidal_state_encoded

**Classes:** ['green', 'orange', 'yellow']

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
