# Tijuana River H2S Forecasting System

This system uses machine learning prediction models to forecast hydrogen sulfide (H2S) levels at wastewater treatment sites near the Tijuana River, enabling proactive alerts for hazardous conditions.

## Implementation Status

### ✅ Current Implementation (v2.0)

**Hourly Forecast Pipeline** (`forecast_prediction_job`, every 6h)
* Single production XGBoost model (NESTOR-BES site) loaded from S3
* 3-category prediction: green (<5 ppb), yellow (5-30 ppb), orange (≥30 ppb)
* Variant predictions: `xgboost_base`, `xgboost_smote`, `random_forest` + ensemble
* 5 visualization types: feature importance, confusion matrix, model comparison, timeline, cross-correlation
* Slack alerts for yellow/orange predictions
* S3 export (timestamped + latest paths)
* Daily validation report (predictions vs actuals)

**Daily Analysis Pipeline** (`daily_analysis_job`, every 6h)
* Multi-station model loading (3 stations × 3 models = 9 models from S3)
* Source attribution — last 7 days wind bearing + Gaussian plume analysis
* 48h forward predictions per station (IB_CIVIC_CTR, NESTOR__BES, SAN_YSIDRO)
* 5-row dashboard PNG visualization
* JSON summary export for web dashboards

**Multi-Horizon Forecast Pipeline** (`mh_forecast_job`, every 6h, currently STOPPED)
* 4 time horizons: 0-6h, 6-24h, 24-48h, 48-72h
* Per-horizon honest feature sets (fresh vs stale lags)
* Regression + binary classification (5 ppb and 10 ppb thresholds) per horizon per station
* Dashboard viz + CSV/JSON export
* Slack alerts

**Model Training**
* Monthly retraining with `MonthlyPartitionsDefinition` (cumulative data)
* Multi-station training partitioned by station (3 partitions)
* Multi-horizon training partitioned by station (currently STOPPED — pending validation)
* Model seeding workflow (`seed_models_job`) for initial S3 population

**Infrastructure**
* Dagster orchestration with schedules, jobs, sensors
* Slack `run_failure` sensor for pipeline errors
* S3/MinIO storage for all artifacts
* `ENV_LABEL` environment variable for dev/prod distinction

---

### 🔄 In Progress / Needs Validation
* **Multi-horizon pipeline activation** — models trained and deployed to S3, schedules currently STOPPED pending live validation
* **Dashboard accuracy** — daily_analysis_job and mh_forecast_job both run every 6h; confirm if daily_analysis should be daily instead

---

### 📋 Future Roadmap
* Public-facing dashboard (Netlify deployment — static site reading from S3 `latest/` paths)
* SMS/webhook alert delivery
* Historical performance API
* Automated model A/B testing
* Real-time streaming predictions

---

## Automation

### Dagster Infrastructure
* Use dagster assets, resources, schedules and sensors
* Existing resources:
  * `projects/h2s/src/h2s/resources/minio.py` - S3/MinIO client
  * `projects/h2s/src/h2s/resources/slack.py` - Slack alert resource
* Utility class to store data in S3:
  * `projects/h2s/src/h2s/utils/store_assets.py`
* Environment configuration:
  * `.env` - S3 credentials, Slack token, and configuration

### Schedules (all in `h2s_schedules.py`)
| Schedule | Cron | Status |
|---|---|---|
| `forecast_prediction_schedule` | Every 6h | RUNNING |
| `daily_analysis_schedule` | Every 6h | RUNNING |
| `daily_validation_schedule` | 8 AM UTC daily | RUNNING |
| `monthly_data_schedule` | 2 AM on 1st of month | RUNNING |
| `monthly_model_training_schedule` | 4 AM on 1st of month | RUNNING |
| `multi_station_training_schedule` | 2 AM on 1st of month | RUNNING |
| `mh_training_schedule` | 3 AM on 1st of month | STOPPED |
| `mh_forecast_schedule` | Every 6h | STOPPED |

---

## Pipeline Architecture

### Jobs and Asset Dependency Chains

**`forecast_prediction_job`** (6-hourly)
```
h2s_model_artifacts → preprocessed_features → h2s_predictions → h2s_alerts → slack_alerts
                                                               → h2s_variant_predictions → h2s_ensemble_predictions
                                                               → predictions_export
h2s_model_artifacts → feature_importance_viz
                    → confusion_matrix_viz
                    → model_comparison_viz
                    → prediction_timeline_viz
                    → cross_correlation_viz
```

**`daily_analysis_job`** (6-hourly)
```
multi_station_model_artifacts → source_attribution → daily_station_forecasts → daily_dashboard_viz
                                                                              → daily_summary_json
```

**`mh_forecast_job`** (6-hourly, STOPPED)
```
mh_model_artifacts → mh_observation_state → mh_forecasts → mh_dashboard_viz
                                                          → mh_summary_export
                                                          → mh_slack_alerts
```

**`multi_station_training_job`** (monthly, partitioned by station)
```
multi_station_training_data → per_station_trained_models → station_training_report
```
Then manually: `station_deployment_job` → uploads to S3

**`mh_training_job`** (monthly, partitioned by station, STOPPED)
```
mh_training_data → mh_trained_models → mh_training_report
```
Then manually: `mh_deployment_job` → uploads to S3

**`monthly_data_extraction_job`** + **`monthly_model_training_job`** (legacy single-model pipeline, kept for reference)

---

## Data Sources

### Input Data (S3)
Retrieve data using the MinIO resource (named `s3`) and `store_assets` methods.

**Environmental Forecasts** (Updated hourly in production):
* **Weather:** `{S3_BUCKET}/latest/tijuana/weather_forecast/latest.csv`
  * Source: OpenMeteo API
  * **Columns (11 total):** date, temperature_2m, wind_speed_10m, wind_direction_10m, relative_humidity_2m, precipitation, surface_pressure, cloud_cover, visibility, dewpoint_2m, site_name
* **Tidal data:** `{S3_BUCKET}/latest/tijuana/tides/latest.csv`
  * Columns: time, tide_height, tidal_state
* **Flow data:** `{S3_BUCKET}/latest/tijuana/streamflow/latest.csv`
  * Columns: time, flow_rate_cms, Flow (m^3/s)--Border

**Historical H2S Measurements** (For validation and reanalysis):
* Path: `{S3_BUCKET}/latest/tijuana/forecast_data/modeldata_h2s.csv`
* Updated: Hourly when in production
* **Columns (24 total):**
  * **Core:** time, site_name, H2S, h2s_measured
  * **Weather (current):** temperature_2m, wind_speed_10m, wind_direction_10m, wind_gusts_10m, precipitation, relative_humidity_2m, surface_pressure, cloud_cover, dewpoint_2m
  * **Wind (derived):** wind_direction_categorical, wind_direction_sin, wind_direction_cos
  * **Wind (rolling averages):** wind_speed_10m_avg_2h, wind_speed_10m_avg_3h, wind_speed_10m_avg_4h
  * **Wind gusts (rolling max):** wind_gusts_10m_max_2h, wind_gusts_10m_max_3h, wind_gusts_10m_max_4h
  * **Hydrological:** Flow (m^3/s)--Border, tide_height, tidal_state

### Model Storage (S3)
```
tijuana/forecast/models/
  nestor_xgboost_weighted_model.json    # production hourly model
  nestor_preprocessing_info.json        # 43-feature preprocessing metadata
  deployment_metadata.json              # deployment provenance
  xgboost_base/model.json               # variant
  xgboost_smote/model.json              # variant
  random_forest/model.joblib            # variant
  stations/
    {station_key}/                      # IB_CIVIC_CTR, NESTOR__BES, SAN_YSIDRO
      clf_5ppb.pkl                      # binary classifier (>5 ppb)
      clf_10ppb.pkl                     # binary classifier (>10 ppb)
      regression.pkl                    # regression
  multihorizon/
    {horizon}/                          # 0_6h, 6_24h, 24_48h, 48_72h
      {station_key}/
        clf_5ppb.pkl
        clf_10ppb.pkl
        regression.pkl
    {station_key}/horizon_features.json
```

### Output Storage (S3)
* **Predictions (timestamped):**
  * `{S3_BUCKET}/tijuana/forecast/output/{YYYY-MM-DD_HH}/h2s_predictions.{csv,json,metadata.json}`
* **Predictions (latest):**
  * `{S3_BUCKET}/latest/tijuana/forecast_data/h2s_predictions.{csv,json}`
* **Visualizations:**
  * `{S3_BUCKET}/tijuana/forecast/output/visualizations/{YYYY-MM-DD}/` (timestamped)
  * `{S3_BUCKET}/latest/tijuana/forecast_data/visualizations/` (latest)
* **Daily analysis:**
  * `{S3_BUCKET}/latest/tijuana/forecast_data/daily_summary.json`
* **Multi-horizon forecast:**
  * `{S3_BUCKET}/tijuana/forecast/multihorizon/{date}/forecast_mh.csv`
  * `{S3_BUCKET}/latest/tijuana/forecast_data/multihorizon/` (latest)

---

## Modeling

### Current Models

**Hourly Forecast Model (NESTOR-BES)**
* **Algorithm:** XGBoost Classifier
* **Features (43 total):** Weather, wind rolling averages/gusts, cyclical encodings, flow derivatives, H2S lags, SBIWTP features, stability/regime features
* **Classes:** green (<5 ppb), yellow (5-30 ppb), orange (≥30 ppb)
* **Performance:** 61.3% orange detection, 5.4% false alarm rate

**Per-Station Models (3 stations)**
* **Stations:** IB_CIVIC_CTR, NESTOR__BES, SAN_YSIDRO
* **Tasks per station:** regression, clf_5ppb (binary >5 ppb), clf_10ppb (binary >10 ppb)
* **Source:** Trained via `multi_station_training_job`, deployed via `station_deployment_job`

**Multi-Horizon Models (3 stations × 4 horizons × 3 tasks = 36 models)**
* **Horizons:** 0-6h, 6-24h, 24-48h, 48-72h
* **Honest feature sets:** each horizon only uses lags/stats available at that lead time
* **Ensemble:** `EnsembleRegressor` / `EnsembleClassifier` (importable from `multihorizon_trainer.py`)
* **Status:** Trained and deployed to S3; forecast pipeline currently STOPPED pending validation

---

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
ENV_LABEL=DEV                # shown in dashboard titles
SCHED_HOSTNAME=sched         # for Dagster UI URL in failure alerts
HOST=local
```

**Dagster uses `EnvVar` for S3 and Slack credentials** - loaded at runtime, not at definitions.py import time.

---

## Operations & Runbooks

### Initial Deployment

```bash
cd projects/h2s
uv sync
cp .env.example .env   # fill in credentials

# 1. Seed S3 with starter models
uv run dg launch --job seed_models_job

# 2. Run hourly forecast pipeline
uv run dg launch --job forecast_prediction_job

# 3. Run daily analysis
uv run dg launch --job daily_analysis_job
```

### Rebuilding Per-Station Models

```bash
# 1. Train (partitioned by station)
uv run dg launch --job multi_station_training_job

# 2. Review station_training_report in Dagster UI

# 3. Deploy to S3 (this IS the approval gate)
uv run dg launch --job station_deployment_job
```

### Activating Multi-Horizon Pipeline

```bash
# 1. Verify models exist in S3 (should already be there from training)
# 2. Test forecast manually
uv run dg launch --job mh_forecast_job

# 3. If results look good, enable schedules in Dagster UI:
#    - mh_forecast_schedule (STOPPED → RUNNING)
#    - mh_training_schedule (STOPPED → RUNNING)
```

### Re-executing Failed daily_analysis_job

Use **"Re-execute all"** in the Dagster UI — not "Re-execute failed steps". Re-executing only failed steps reads `multi_station_model_artifacts` from stale IO cache. Re-executing all steps reloads models fresh from S3.

---

## System Dependencies

### Runtime Environment
* **Python:** 3.10 or higher
* **Dagster:** 1.x (latest stable)
* **XGBoost:** 2.x
* **Key libraries:** pandas, numpy, matplotlib, seaborn, minio, scikit-learn, dagster-slack

### External Services
* **S3/MinIO:** oss.resilientservice.mooo.com
* **OpenMeteo:** Hourly weather forecast API
* **Slack:** Alert delivery (two channels: alerts + failures)
* **Netlify:** (Future) Public dashboard hosting
