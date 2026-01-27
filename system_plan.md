# Tijuana River H2S Forecasting System

This system uses machine learning prediction models to forecast hydrogen sulfide (H2S) levels at the NESTOR-BES wastewater treatment site, enabling proactive alerts for hazardous conditions.

## Implementation Status

### ✅ Current Implementation (v1.0)
* Single pre-trained XGBoost model (NESTOR-BES site only)
* 3-category prediction: green (<5 ppb), yellow (5-15 ppb), orange (≥15 ppb)
* Hourly forecast processing from S3
* S3 storage for predictions and visualizations
* 4 visualization types: feature importance, confusion matrix, model comparison, timeline
* Production/test mode configuration (fail-fast, no silent fallbacks)
* Dagster pipeline with 11 assets
* Model performance: 61.3% orange detection rate, 5.4% false alarm rate

### 🔄 In Progress
* Monthly model retraining workflow
* Automated alerting system
* Performance monitoring dashboard

### 📋 Future Roadmap
* Multi-site support (expand beyond NESTOR-BES)
* Enhanced threshold models (separate predictions for >5 ppb and >30 ppb)
* Public-facing dashboard (Netlify deployment)
* SMS/webhook alert delivery
* Automated model A/B testing
* Historical performance API
* Real-time streaming predictions

---

## Automation

### Dagster Infrastructure
* Use dagster assets, resources, schedules and sensors
* Existing resources:
  * `projects/h2s/src/h2s/resources/minio.py` - S3/MinIO client
* Utility class to store data in S3:
  * `projects/h2s/src/h2s/utils/store_assets.py`
* Environment configuration:
  * `.env` - S3 credentials and configuration

### Scheduling
* Hourly prediction runs (aligned with forecast updates)
* Monthly model retraining (1st of each month)
* Daily validation runs (compare predictions vs actuals)
* Weekly performance report generation

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
* **Current production model:**
  * Model: `{S3_BUCKET}/tijuana/forecast/models/nestor_xgboost_weighted_model.json` (4.2 MB)
  * Preprocessing: `{S3_BUCKET}/tijuana/forecast/models/nestor_preprocessing_info.json` (948 bytes)
* **Latest model symlink:**
  * `{S3_BUCKET}/latest/tijuana/forecast/models/` (points to current production)
* **Archived models:**
  * `{S3_BUCKET}/tijuana/forecast/models/archive/{YYYY_MM}/`
  * Includes: model files, preprocessing metadata, training metrics, validation results

### Output Storage (S3)
* **Predictions (timestamped):**
  * `{S3_BUCKET}/tijuana/forecast/output/{YYYY-MM-DD_HH}/h2s_predictions.{csv,json,metadata.json}`
  * **Output columns:** All input features plus:
    * predicted_category (green/yellow/orange)
    * probability_green (0-1)
    * probability_yellow (0-1)
    * probability_orange (0-1)
    * confidence (max probability, 0-1)
    * alert (boolean: True for yellow/orange)
* **Predictions (latest):**
  * `{S3_BUCKET}/latest/tijuana/forecast_data/h2s_predictions.{csv,json}`
  * Same column structure as timestamped predictions
* **Visualizations (timestamped):**
  * `{S3_BUCKET}/tijuana/forecast/output/visualizations/{YYYY-MM-DD}/`
  * Files: feature_importance.png, confusion_matrix.png, model_comparison.png, prediction_timeline.png
* **Visualizations (latest):**
  * `{S3_BUCKET}/latest/tijuana/forecast_data/visualizations/`
* **Performance metrics:**
  * `{S3_BUCKET}/tijuana/forecast/validation/{YYYY_MM}/`
  * Monthly validation reports, confusion matrices, performance trends

---

## Modeling

### Current Model
* **Site:** NESTOR-BES (code should support multi-site selection for future)
* **Algorithm:** XGBoost Classifier (gradient boosted decision trees)
* **Features (20 total):**
  * Weather: temperature_2m, wind_speed_10m, wind_direction_10m, relative_humidity_2m, surface_pressure, precipitation, cloud_cover, dewpoint_2m
  * Tidal: flow_rate_cms, tide_height, tidal_state
  * Temporal: hour_sin, hour_cos, day_of_week, month
  * Derived: wind_direction_sin, wind_direction_cos, wind_temp_interaction, humidity_temp_interaction
  * Encoded: wind_direction_cat_encoded, tidal_state_encoded

### H2S Categories (Current Implementation)
* **Green:** H2S < 5 ppb (safe)
* **Yellow:** 5 ≤ H2S < 15 ppb (caution)
* **Orange:** H2S ≥ 15 ppb (alert)

### Future Model Enhancements
Expand predictions to test if we can improve performance:

**Single Category Models:**
* Binary yellow threshold: >5 ppb
* Binary orange threshold: >30 ppb (current)

**Multi-Category Models:**
* 3-class (current): green (0-5), yellow (5-30), orange (≥30)
* Fine-grained: 5 ppb increments

---

## Model Management & Versioning

### Model Artifacts
* **Naming convention:** `nestor_xgboost_{YYYY_MM}_v{version}.json`
* **Metadata includes:**
  * Training date and data range
  * Performance metrics (balanced accuracy, precision, recall per class)
  * Feature list and importance scores
  * Hyperparameters
  * Training/validation split details

### Deployment Strategy
* **Champion/Challenger approach:**
  * Keep previous model as fallback
  * A/B testing for new models (10% traffic initially)
  * Gradual rollout based on performance
* **Rollback capability:**
  * Automatic rollback if new model underperforms
  * Manual rollback via configuration change

### Model Retraining
**Triggers:**
* Monthly scheduled retraining (1st of each month)
* Performance degradation detection (accuracy drops >5%)
* Significant data drift detected
* Manual trigger for emergency updates

**Process:**
1. Extract training data from previous month
2. Train model with cross-validation
3. Generate validation metrics and visualizations
4. Store in archive with timestamp
5. Manual review and approval before production deployment
6. Deploy to production with A/B testing

---

## Concept & Workflow

### Real-Time Forecasting
1. Hourly weather forecast updated in S3
2. Dagster sensor detects new forecast data
3. Pipeline materializes predictions
4. Hazardous predictions (yellow/orange) trigger alerts
5. Results stored in S3 with visualizations

### Monthly Validation
1. At end of month, rerun predictions using actual environmental data (modeldata_h2s.csv)
2. Compare predictions vs actual H2S measurements
3. Generate confusion matrix and performance statistics
4. Update model performance dashboard
5. Trigger retraining if performance degraded

### Historical Validation
For each monthly model:
* **September 2025 model:** Validate on October 2025 - Present
* **December 2025 model:** Validate on January 2026 - Present
* Track performance decay over time
* Identify seasonal patterns in model accuracy

---

## Data Quality & Monitoring

### Input Data Validation
* **Schema validation:**
  * Check for required columns before prediction
  * Validate data types and ranges
  * Ensure timestamp format consistency
* **Completeness checks:**
  * Flag missing sensor readings
  * Detect gaps in time series (>2 hours)
  * Alert on incomplete forecasts
* **Anomaly detection:**
  * Statistical outliers in environmental variables
  * Physically impossible values (temp >60°C, flow <0)
  * Sudden jumps inconsistent with history

### Data Freshness Monitoring
* **Alerts triggered when:**
  * S3 forecast data is stale (>2 hours old)
  * modeldata_h2s.csv not updated in >4 hours
  * Missing data for current hour
* **Monitoring dashboard shows:**
  * Last update timestamp for each data source
  * Data completeness percentage (24h rolling)
  * Source availability status

### Feature Drift Monitoring
* **Track over time:**
  * Distribution of input features (mean, std, quantiles)
  * Correlation structure between features
  * Occurrence rates of rare conditions
* **Alert on significant shifts:**
  * KL divergence >0.1 from training distribution
  * New feature value ranges not seen in training
  * Changing correlations between predictors

---

## Alerting & Notifications

### Real-Time Alerts
**Trigger conditions:**
* Orange level predictions (H2S ≥15 ppb)
* Critical: >30 ppb sustained for 2+ consecutive hours
* Yellow alerts if confidence >80%

**Alert content:**
* Predicted H2S level and category
* Confidence score / probability distribution
* Contributing factors (e.g., "High flow + low wind")
* Forecast duration: next 1hr, 3hr, 6hr
* Map with affected areas
* Recommended actions

**Delivery channels:**
* Email to registered recipients
* SMS for critical alerts (>30 ppb)
* Webhook to external monitoring systems
* Public dashboard notification banner

**Alert recipients:**
* Environmental health officials (County)
* Site operators at NESTOR-BES
* Emergency response coordinators
* Public dashboard (optional, for yellow/orange only)

### System Alerts
* Model prediction failures
* Data source unavailability
* Pipeline execution errors
* Performance degradation detected

---

## Operations & Monitoring

### SLAs (Service Level Agreements)
* **Prediction latency:** <5 minutes from forecast availability to prediction completion
* **Data freshness:** Forecasts <1 hour old, actuals <2 hours old
* **System uptime:** 99.5% (excluding scheduled maintenance)
* **Alert delivery:** <2 minutes from prediction to notification

### Monitoring Dashboards
**Real-time metrics:**
* Current prediction status and latest forecast
* Prediction counts by category (green/yellow/orange) - 24h rolling
* Alert delivery success rate
* Data pipeline health (asset materialization status)

**Performance metrics:**
* Model accuracy trends (monthly validation)
* Precision/Recall by category
* Confusion matrix (updated monthly)
* Feature importance stability

**System health:**
* Asset execution success rate
* S3 sync status and latency
* Error rates by component
* Resource utilization (CPU, memory, storage)

### Logging
* **Prediction logs:** All predictions logged to S3 with full inputs and outputs
* **Asset execution logs:** Dagster captures all asset run details
* **Alert logs:** Delivery confirmations, failures, recipient acknowledgments
* **Error logs:** Structured logging with context for debugging
* **Retention:** 90 days in active storage, 1 year in archive

---

## Testing & Quality Assurance

### Unit Tests
* **Predictor class:**
  * `test_predictor.py` - preprocessing, prediction, thresholds
  * Model loading from S3 and local
  * Feature engineering correctness
* **Visualization functions:**
  * `test_visualizations.py` - plot generation, BytesIO output
  * Merge conflict handling
  * Empty data edge cases
* **S3 integration:**
  * `test_s3_integration.py` - upload, download, streaming (with mocks)

### Integration Tests
* **Full pipeline execution:**
  * `test_h2s_pipeline.py` - asset dependencies, data flow
  * Test with local data: `use_local_data=True` for CI/CD
  * Validate outputs in expected S3 paths
* **End-to-end scenarios:**
  * Forecast ingestion → prediction → visualization → export
  * Historical validation workflow

### Performance Tests
* **Model inference:**
  * Latency: <1 second per prediction
  * Batch throughput: 4000+ samples in <10 seconds
* **Pipeline execution:**
  * Full materialization: <3 minutes
  * Memory usage: <2 GB peak

### Validation Tests
* **Monthly model performance:**
  * Automated validation against actual data
  * Confusion matrix generation
  * Precision/Recall tracking by category
  * Statistical significance testing
* **Regression tests:**
  * Ensure predictions remain consistent for fixed inputs
  * No degradation in test set performance

---

## Error Handling & Recovery

### Data Source Failures
* **S3 unavailable:**
  * Production mode: FAIL immediately (no fallback)
  * Alert operator with clear error message
  * Provide instructions for test mode if needed
* **Forecast data missing:**
  * Log error with timestamp and expected path
  * Skip prediction for current hour
  * Alert if consecutive failures (>2 hours)
* **Corrupt or invalid data:**
  * Log problematic data for investigation
  * Attempt validation and cleanup
  * Skip current run, wait for next scheduled execution

### Model Failures
* **Model file missing/corrupt:**
  * Alert operator immediately
  * Attempt to use previous model version from archive
  * Prevent predictions until resolved
* **Prediction errors (runtime):**
  * Log full input data causing error
  * Continue with next batch if batch processing
  * Alert after 3 consecutive failures

### Recovery Procedures
* **Automatic retry:**
  * Exponential backoff (1min, 2min, 5min)
  * Maximum 3 retry attempts per run
  * Different retry strategies for transient vs persistent errors
* **Manual intervention:**
  * Documented runbook for common failures
  * Clear escalation path for critical issues
  * Access to logs and debugging tools
* **Backfill capability:**
  * Script to rerun predictions for missed time periods
  * Validates against actual data if available
  * Stores backfilled results with metadata flag

---

## Public Dashboard (Netlify Deployment)

### Features
* **Current status:**
  * Live H2S forecast for next 24 hours
  * Current category (green/yellow/orange) with color coding
  * Last update timestamp
* **Historical view:**
  * Predictions vs actuals comparison chart
  * Accuracy metrics over time
  * Filter by date range and category
* **Model comparison:**
  * Monthly model performance side-by-side
  * Interactive selection of model version
  * Visualizations for each model (confusion matrix, feature importance)
* **Environmental context:**
  * Weather conditions (temp, wind, humidity)
  * Tidal state and flow rates
  * Interactive timeline with all variables
* **Data access:**
  * Downloadable prediction data (CSV/JSON)
  * API endpoint for programmatic access (future)
  * Historical archive access

### Technology Stack
* **Frontend:**
  * Static site generator: Next.js or Gatsby
  * UI framework: React with Tailwind CSS
  * Charts: D3.js or Recharts
* **Data source:**
  * Fetch from S3 `latest/` paths
  * Client-side data processing
  * No backend required (fully static)
* **Deployment:**
  * Netlify hosting with CDN
  * Auto-rebuild on new predictions (webhook trigger from Dagster)
  * Branch previews for development

### User Experience
* **Mobile-responsive design**
  * Touch-friendly interface
  * Optimized for phones and tablets
* **Color-coded alerts:**
  * Green: Safe, subtle background
  * Yellow: Caution, amber highlights
  * Orange: Alert, prominent warning
* **Accessibility:**
  * WCAG 2.1 AA compliance
  * Screen reader support
  * Keyboard navigation
* **Performance:**
  * Initial load <2 seconds
  * Interactive within 3 seconds
  * Lazy loading for historical data

### Visualization Details
* **Actual H2S levels:** Line chart with measured values
* **Prediction indicators:**
  * Green: No marker (baseline)
  * Yellow: Dot at 10 ppb threshold
  * Orange: Dot at 15 ppb threshold
* **Confidence bands:** Shaded area showing prediction uncertainty
* **Environmental overlay:** Toggle to show wind, temp, flow on same chart

---

## System Dependencies

### Runtime Environment
* **Python:** 3.10 or higher
* **Dagster:** 1.x (latest stable)
* **XGBoost:** 2.x
* **Key libraries:**
  * pandas, numpy - data processing
  * matplotlib, seaborn - visualization
  * minio - S3 client
  * scikit-learn - metrics and preprocessing

### Data Sources
* **S3/MinIO storage:**
  * Endpoint: oss.resilientservice.mooo.com
  * Bucket: test (or configured in .env)
* **OpenMeteo weather forecast API:**
  * Hourly forecasts for Tijuana region
  * Free tier with rate limiting
* **Tidal data stream:**
  * NOAA tidal predictions
  * Updated daily
* **Flow measurements:**
  * Border monitoring station
  * Real-time measurements from USGS/IBWC

### External Services
* **Email/SMS provider:** (To be configured)
  * SendGrid, Twilio, or AWS SNS
  * API keys in .env
* **Netlify:**
  * Dashboard hosting
  * Build hooks for auto-deployment
* **Monitoring (future):**
  * Datadog or Grafana for metrics
  * Sentry for error tracking

---

## Configuration Files

### Environment Variables (.env)
```bash
# S3/MinIO Configuration
S3_BUCKET=test
S3_ADDRESS=oss.resilientservice.mooo.com
S3_PORT=443
S3_USE_SSL=true
S3_ACCESS_KEY=your_access_key
S3_SECRET_KEY=your_secret_key

# Optional paths
PUBLIC_BUCKET=test
LATEST_BASEPATH=latest/

# Alert configuration (future)
ALERT_EMAIL_RECIPIENTS=alerts@example.com
ALERT_SMS_RECIPIENTS=+1234567890
SMTP_SERVER=smtp.example.com
```

### Dagster Configuration
* **dagster.yaml:** Resource and execution configuration
* **config_local_test.yaml:** Test mode configuration for local development
* **Asset configs:** Per-asset settings for materialization

---

## Future Enhancements

### Short Term (3 months)
* Complete monthly retraining automation
* Implement basic email alerting
* Add performance monitoring dashboard
* Expand test coverage to >80%

### Medium Term (6 months)
* Public dashboard deployment (Netlify)
* SMS alert integration
* Multi-site model support (pilot 2nd site)
* API for external integrations

### Long Term (12 months)
* Real-time streaming predictions (sub-minute latency)
* Advanced ensemble models
* Explainable AI features (SHAP values)
* Mobile app with push notifications
* Integration with regional air quality systems
* Automated model optimization (AutoML)
