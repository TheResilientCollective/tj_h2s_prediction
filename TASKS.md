# H2S Forecasting System - Task Tracker

**Last Updated:** 2026-04-02
**Source:** `system_plan.md` - Keep synchronized when plan changes

## Legend
- 🔴 High Priority
- 🟡 Medium Priority
- 🟢 Low Priority
- ✅ Completed
- 🔄 In Progress
- ⏸️ Blocked

---

## ✅ Completed

### Epic 1: Monthly Model Retraining Workflow ✅
**Effort:** Large (4-6 weeks)
**Dependencies:** None
**Reference:** [Model Management & Versioning](system_plan.md#model-management--versioning)
**Status:** COMPLETED (Jan 2026)

#### Phase 1: Data Extraction & Preparation ✅
- [x] **Create training data extraction asset**
  - [x] Add `monthly_training_data` asset to pipeline
  - [x] Query previous month's environmental data from S3
  - [x] Filter to NESTOR-BES site
  - [x] Merge with actual H2S measurements from modeldata_h2s.csv
  - [x] Validate data quality (completeness, outliers)
  - [x] Export to S3: `tijuana/forecast/training_data/{YYYY_MM}/`

#### Phase 2: Model Training Pipeline ✅
- [x] **Implement cross-validation training**
  - [x] Create `trained_model_cv` asset (depends on training_data_extraction)
  - [x] Implement 5-fold time-series aware cross-validation
  - [x] Track performance metrics per fold (accuracy, precision, recall)
  - [x] Generate feature importance analysis
  - [x] Save best model with hyperparameters

- [x] **Add model versioning & archiving**
  - [x] Implement naming convention: `nestor_xgboost_{YYYY_MM}_v{version}.json`
  - [x] Store preprocessing metadata as JSON
  - [x] Archive to S3: `tijuana/forecast/models/archive/{YYYY_MM}/`
  - [x] Include training metadata: date, data range, metrics, hyperparameters

#### Phase 3: Validation & Deployment ✅
- [x] **Create validation workflow**
  - [x] Test new model on held-out validation set
  - [x] Compare against current production model
  - [x] Generate validation report with confusion matrix
  - [x] Store validation results in S3

- [x] **Implement deployment automation**
  - [x] Add manual approval step for production deployment
  - [x] Copy approved model to `latest/tijuana/forecast/models/`
  - [x] Update model version in Dagster configuration
  - [x] Archive previous production model

#### Phase 4: Scheduling & Automation ✅
- [x] **Add Dagster schedule**
  - [x] Create monthly schedule (1st of each month, 2 AM)
  - [x] Configure schedule to run full retraining pipeline
  - [x] Add success/failure notifications

**Acceptance Criteria:** ✅ ALL MET
- [x] Models trained automatically on 1st of each month
- [x] Performance metrics logged to S3 for every training run
- [x] Model archives include all metadata and artifacts
- [x] Manual approval required before production deployment
- [x] Previous model backed up before replacement

**Implementation Details:**
- **14 Assets Created:**
  - Phase 1: `monthly_training_data`, `relabeled_training_data`, `data_quality_report`, `training_data`, `validation_data`
  - Phase 2: `trained_model_cv`, `model_training_metrics`, `feature_importance_analysis`
  - Phase 3: `validation_predictions`, `validation_report`, `model_comparison_report`
  - Phase 4: `deployment_approval`, `archived_previous_model`, `production_model_deployment`
- **2 Jobs:** `monthly_retraining_job`, `deploy_approved_model_job`
- **1 Schedule:** `monthly_retraining_schedule` (cron: `0 2 1 * *`)
- **Supporting Modules:** `model_trainer.py`, `relabeling.py`, `validation.py`

---

### Epic 0: Update H2S Category Thresholds to Client Spec ✅
**Effort:** Small (1-2 days)
**Dependencies:** None
**Reference:** [H2S Categories](system_plan.md#h2s-categories-current-implementation)
**Status:** COMPLETED (Jan 2026)

#### Threshold Update
**Old Thresholds:**
- Green: <5 ppb
- Yellow: 5-15 ppb
- Orange: ≥15 ppb

**New Client Specification (IMPLEMENTED):**
- Green: <5 ppb
- Yellow: 5-30 ppb
- Orange: ≥30 ppb

#### Code Changes Completed ✅
- [x] **Update visualization functions** (`projects/h2s/src/h2s/predictor/visualizations.py`)
  - [x] Update `categorize_h2s()` in `generate_confusion_matrix()` (line 91-97)
  - [x] Update `categorize_h2s()` in `generate_confusion_matrix_with_metrics()` (line 170-176)
  - [x] Update `categorize_h2s()` in `generate_model_comparison()` (line 271-277)
  - [x] Timeline visualization threshold markers (note: actual code doesn't have markers)

- [x] **Training pipeline updated**
  - [x] Created `h2s/training/relabeling.py` with new thresholds
  - [x] `relabeled_training_data` asset applies new categories
  - [x] Monthly retraining workflow uses new thresholds
  - [x] Model retraining ready to use new labels

- [x] **Update documentation**
  - [x] system_plan.md updated (visualization threshold markers fixed)
  - [N/A] README.md, DEPLOYMENT_GUIDE.md don't exist in repository

- [x] **Testing**
  - [x] test_training_pipeline.py uses new thresholds
  - [x] All relabeling tests pass (6/6 tests)
  - [x] Threshold verification tests complete

**Acceptance Criteria:** ✅ ALL MET
- [x] All code uses new thresholds: Yellow 5-30 ppb, Orange ≥30 ppb
- [x] Training pipeline configured for new thresholds
- [x] Visualizations use correct categorization
- [x] Tests pass with new thresholds (verified Jan 27, 2026)
- [x] Documentation updated

**Implementation Details:**
- **Core Module:** `h2s/training/relabeling.py` - Central threshold logic
- **Assets Updated:** All 3 visualization functions in `visualizations.py`
- **Tests:** 6 tests verify correct categorization
- **Next Step:** Run monthly retraining job to create model trained on new labels

**Note:** Existing production model (trained Jan 15) still functional but should be retrained with new labels for optimal class weights and decision boundaries. Monthly retraining pipeline is ready and configured.

---

### Infrastructure: S3 Path Constants & Metadata Support ✅
**Effort:** Small (1 day)
**Dependencies:** None
**Status:** COMPLETED (Jan 27, 2026)

#### Objectives
Migrate all S3 uploads to use centralized path constants and store_assets utility for consistent metadata file generation.

#### Implementation ✅
- [x] **Create centralized constants module**
  - [x] Created `h2s/constants.py` with path constants:
    - `MODEL_PATH = 'tijuana/forecast/models'`
    - `TRAINING_PATH = 'tijuana/forecast/models/training'`
    - `ARCHIVE_PATH = 'tijuana/forecast/models/archive'`
    - `PREDICTIONS_PATH = 'tijuana/forecast/predictions'`
    - `OUTPUT_PATH = 'tijuana/forecast/output'`
    - `LATEST_FORECAST_DATA = 'tijuana/forecast_data'`

- [x] **Update prediction pipeline** (`h2s_pipeline.py`)
  - [x] Import constants module
  - [x] Update `predictions_export` to use `PREDICTIONS_PATH`
  - [x] Update all visualization paths to use constants
  - [x] Verify definitions load successfully

- [x] **Update training pipeline** (`h2s_training_pipeline.py`)
  - [x] Import constants module
  - [x] Update `model_training_metrics` to use `TRAINING_PATH` + store_assets
  - [x] Update `feature_importance_analysis` to use `TRAINING_PATH` + store_assets
  - [x] Update `archived_previous_model` to use `ARCHIVE_PATH`
  - [x] Update `production_model_deployment` to use `MODEL_PATH`

- [x] **Add metadata file generation**
  - [x] `model_training_metrics` → `.metadata.json` (training metrics bundle)
  - [x] `feature_importance_analysis` → `.metadata.json` (PNG visualization)
  - [x] `predictions_export` → `.metadata.json` (predictions CSV/JSON)

**Acceptance Criteria:** ✅ ALL MET
- [x] All S3 paths use centralized constants
- [x] No hardcoded paths in asset code
- [x] Metadata files created alongside data files
- [x] Dagster definitions load without errors
- [x] Ready for schema.org enhancement

**S3 Structure:**
```
tijuana/forecast/
├── models/                    # Production models
├── models/training/{YYYY_MM}/ # Training artifacts + metadata
├── models/archive/{YYYY_MM}/  # Archived models
├── predictions/               # Predictions + metadata (NEW!)
└── output/visualizations/     # Visualizations
```

**Benefits:**
- ✅ Consistent path management across all assets
- ✅ Metadata files ready for schema.org enhancement
- ✅ Proper separation: predictions, models, training, archives
- ✅ `store_assets` integration for 3 key assets

**Next Step:** Add full schema.org metadata with `pydantic_schemaorg` when ready

---

### Multi-Station Training & Daily Analysis Pipeline ✅
**Status:** COMPLETED (Mar 2026)

- [x] Per-station models: 3 stations × 3 tasks = 9 models (clf_5ppb, clf_10ppb, regression)
- [x] `multi_station_training_job` partitioned by station (IB_CIVIC_CTR, NESTOR__BES, SAN_YSIDRO)
- [x] `station_deployment_job` — uploads approved models to S3
- [x] `daily_analysis_job` — source attribution (wind bearing + Gaussian plume), 48h station forecasts, 5-row dashboard PNG, JSON summary
- [x] Model seeding workflow (`seed_models_job`)

---

### Epic 2: Automated Alerting System ✅
**Status:** COMPLETED (Slack) — Email/SMS remain future work

**Implemented:**
- [x] `slack_alerts` asset in hourly forecast pipeline
- [x] `mh_slack_alerts` asset in multi-horizon pipeline
- [x] `SlackAlertResource` (`resources/slack.py`)
- [x] `slack_on_run_failure` sensor for pipeline errors
- [x] Two Slack channels: `SLACK_CHANNEL` (alerts) + `SLACK_CHANNEL_FAILURES` (pipeline errors)
- [x] Pacific time display in alert messages

**Remaining (future):**
- [ ] Email delivery (SMTP/SendGrid)
- [ ] SMS for critical alerts (Twilio)

---

### Epic 7: Multi-Site Model Support ✅
**Status:** COMPLETED (Mar 2026) — 3 stations implemented

- [x] IB_CIVIC_CTR, NESTOR__BES, SAN_YSIDRO
- [x] Per-station training pipeline with partitioning
- [x] Per-station 48h forecasts in `daily_station_forecasts`
- [x] Source attribution identifies which station is at risk

---

## 🔄 In Progress

### Multi-Horizon Pipeline Validation 🔴
**Status:** Models trained and deployed to S3; forecast job created but schedules STOPPED

- [x] 36 models trained (4 horizons × 3 stations × 3 tasks)
- [x] `mh_forecast_job` pipeline complete with dashboard + Slack alerts
- [ ] Run `mh_forecast_job` manually and validate output quality
- [ ] Enable `mh_forecast_schedule` (STOPPED → RUNNING) once validated
- [ ] Enable `mh_training_schedule` (STOPPED → RUNNING) once validated

---

### Epic 3: Performance Monitoring Dashboard 🟡
**Status:** Forecast dashboard exists (`daily_dashboard_viz`); metrics tracking not implemented

**Done:**
- [x] `daily_dashboard_viz` — 5-row PNG: source attribution, station forecasts, weather context
- [x] `daily_summary_json` — JSON for web dashboards

**Remaining:**
- [ ] Accuracy metrics tracking (predictions vs actuals over time) — **See Epic 15 below**
- [ ] Model performance trends (precision/recall by category) — **See Epic 15 below**
- [ ] System health indicators (pipeline success rates, data freshness)
- [ ] Public-facing dashboard (Netlify — see Epic 5)

---

### Epic 15: Forecast Accuracy Metrics & Validation 🔴
**Effort:** Small-Medium (1-2 weeks)
**Dependencies:** None (enhances existing `daily_validation_report`)
**Status:** PLANNING — Implementation plan ready for review

#### Objectives
Compare hourly H2S predictions against actual observations from `modeldata_h2s_nofill.parquet` to track forecast accuracy over time. Generate daily metrics and monthly performance dashboards.

updated truncated h2s:
s3://test/latest/tijuana/sd_apcd_air/h2s/apcd_h2s_latest.csv
https://oss.resilientservice.mooo.com/test/latest/tijuana/sd_apcd_air/h2s/apcd_h2s_latest.parquet
full h2s dataset:
s3://test/latest/tijuana/sd_apcd_air/h2s_all/h2s_all.parquet
https://oss.resilientservice.mooo.com/test/latest/tijuana/sd_apcd_air/h2s_all/h2s_all.parquet
Examples are in the test bucket but the actual data source is the `OBS_DATA_PATH` constant in `h2s/constants.py`.

#### Component 1: Fix & Enhance `daily_validation_report` Asset ✅
**File:** `projects/h2s/src/h2s/defs/h2s_pipeline.py` (lines 846-1033)

- [x] **Fix actuals data loading**
  - [x] Change path from `tijuana/forecast/actuals/latest.csv` (broken) to `OBS_DATA_PATH` constant
  - [x] Load parquet format: `pd.read_parquet(s3.get_presigned_url(path=OBS_DATA_PATH))`
  - [x] Filter to NESTOR-BES: handle both `NESTOR__BES` and `NESTOR - BES` variants
  - [x] Convert observation time from Pacific to UTC: `actuals['time'].dt.tz_convert('UTC')`
  - [x] **FAIL HARD if data missing** (no graceful skips)

- [x] **Calculate numerical metrics**
  - [x] Import `calculate_metrics()` from `h2s.training.validation`
  - [x] Categorize actual H2S values (green <5, yellow <30, orange ≥30)
  - [x] Merge predictions + actuals on time column
  - [x] Call `calculate_metrics(y_true, y_pred, class_names)`
  - [x] Calculate false alarm rate using `calculate_false_alarm_rate()`
  - [x] **FAIL HARD if merge produces 0 matches**

- [x] **Store metrics.json to S3**
  - [x] Create metrics JSON structure:
    - `date`, `timestamp`, `site`
    - `n_predictions`, `n_matched`, `match_rate`
    - `balanced_accuracy`
    - `class_metrics` (precision, recall, F1, support per class)
    - `confusion_matrix` (3×3 array)
    - `false_alarm_rate`
  - [x] Upload to S3: `{VALIDATION_PATH}/{YYYY-MM-DD}/metrics.json`

#### Component 2: New `monthly_performance_viz` Asset ✅
**File:** `projects/h2s/src/h2s/defs/h2s_pipeline.py` (lines 1052-1204)

- [x] **Load 30-day metrics history**
  - [x] Iterate through last 30 days
  - [x] Load each `metrics.json` from S3
  - [x] Parse JSON and store in list with date objects
  - [x] **FAIL HARD if <7 days of data** (need minimum for trends)

- [x] **Create 4-panel performance dashboard**
  - [x] **Panel 1:** Aggregate confusion matrix (30-day sum, normalized %)
    - Annotate cells with percentage + raw counts
    - Color scale: RdYlGn_r (red = poor, green = good)
  - [x] **Panel 2:** Daily balanced accuracy trend line
    - Reference line at 0.61 (target performance)
    - Date axis with rotation
  - [x] **Panel 3:** Per-class recall trends (3 lines)
    - Orange, yellow, green recall over 30 days
    - Color-coded by category
  - [x] **Panel 4:** Daily false alarm rate trend
    - Reference line at 0.054 (5.4% target)
    - Orange predicted when actually green

- [x] **Upload to S3**
  - [x] Save as BytesIO PNG (150 dpi, 16×12 figure)
  - [x] Timestamped: `{VALIDATION_PATH}/monthly/{YYYY-MM}/performance_dashboard.png`
  - [x] Latest: `latest/{LATEST_FORECAST}/visualizations/performance_dashboard.png`

#### Component 3: Job & Schedule Wiring ✅

- [x] **Update `daily_validation_job`** (`h2s_schedules.py`)
  - [x] Add `monthly_performance_viz` to asset selection
  - [x] Both assets run together at 8 AM UTC daily

- [x] **Register in definitions.py**
  - [x] Import `monthly_performance_viz` from `h2s_pipeline`
  - [x] Add to assets list (line ~152)

#### Edge Cases & Error Handling ✅

- [x] **No predictions for yesterday** → FAIL HARD (already handled in existing code)
- [x] **No observation data** → **FAIL HARD** (raises ValueError)
- [x] **No H2S column** → **FAIL HARD** (raises ValueError with column list)
- [x] **No metrics history** → **FAIL HARD** (raises ValueError)
- [x] **Partial history (<7 days)** → **FAIL HARD** (need minimum 7 days for trends)
- [x] **Partial history (7-30 days)** → Works with available days (logs missing count)
- [x] **Site name variants** → Filter both `NESTOR__BES` and `NESTOR - BES`
- [x] **Timezone mismatches** → Convert to UTC before merge
- [x] **Zero matches after merge** → **FAIL HARD** (raises ValueError)
- [x] **All NaN H2S values** → **FAIL HARD** (raises ValueError)

#### Testing

- [ ] **Unit tests** (`test_h2s_pipeline.py`)
  - [ ] Test metrics JSON structure
  - [ ] Test categorization function (green/yellow/orange)
  - [ ] Test timezone conversion logic
  - [ ] Mock S3 `getFile()` for observation data

- [ ] **Integration test**
  - [ ] Run `uv run dg launch --job daily_validation_job`
  - [ ] Verify `metrics.json` appears in S3 `validation/{date}/`
  - [ ] Verify `performance_dashboard.png` uploads (timestamped + latest)
  - [ ] Check logs for warnings/errors

**Acceptance Criteria:**
- [ ] `daily_validation_report` loads actuals from correct S3 path (parquet)
- [ ] Metrics.json generated daily with all required fields
- [ ] Monthly dashboard shows 4 panels (confusion matrix, balanced accuracy, recall trends, false alarms)
- [ ] Dashboard runs daily showing trailing 30-day window
- [ ] Graceful handling of missing data (no hard failures)
- [ ] All visualizations upload to both timestamped and latest paths

**S3 Output Structure:**
```
tijuana/forecast/validation/
├── YYYY-MM-DD/
│   ├── metrics.json                    (NEW)
│   ├── daily_predictions_combined.csv
│   ├── confusion_matrix.png
│   ├── model_comparison.png
│   └── prediction_timeline.png
└── monthly/YYYY-MM/
    └── performance_dashboard.png        (NEW)

latest/tijuana/forecast/visualizations/
└── performance_dashboard.png            (NEW - always current)
```

#### Design Questions for Review

**Scheduling:**
- [ ] **Q1:** Run `monthly_performance_viz` daily (trailing 30d window) or monthly only?
  - **Recommendation:** Daily (low overhead, always current)
  - **Alternative:** Monthly on 1st at 10 AM UTC

**Scope:**
- [ ] **Q2:** Validate only NESTOR-BES or extend to all 3 stations?
  - **Current:** NESTOR-BES only (matches hourly pipeline)
  - **Future:** Could add per-station validation for daily_analysis_job predictions

**Dashboard:**
- [ ] **Q3:** Is 30-day window appropriate? (Could make configurable: 7/30/90 days)
- [ ] **Q4:** Are 4 panels sufficient or add more? (precision trends, support counts, hourly breakdown)
- [ ] **Q5:** Should false alarm rate be "orange when green" or "orange when green OR yellow"?

**Alerting:**
- [ ] **Q6:** Add automated Slack alerts for metric degradation?
  - Example: "⚠️ Balanced accuracy dropped to 55% yesterday (target: 61%)"
  - Would add after metrics calculation in `daily_validation_report`

**Multi-Horizon:**
- [ ] **Q7:** Add similar validation for multi-horizon pipeline (3 stations × 4 horizons)?
  - Would be separate asset (`mh_validation_report`) due to different structure

**Implementation Files:**
- `projects/h2s/src/h2s/defs/h2s_pipeline.py` — modify `daily_validation_report`, add `monthly_performance_viz`
- `projects/h2s/src/h2s/defs/h2s_schedules.py` — update `daily_validation_job` selection
- `projects/h2s/src/h2s/definitions.py` — register new asset
- `projects/h2s/src/h2s/constants.py` — already has `OBS_DATA_PATH`, `VALIDATION_PATH`
- `projects/h2s/src/h2s/training/validation.py` — already has `calculate_metrics()`, `calculate_false_alarm_rate()`

---

## 📅 Short Term (3 months)

### Epic 4: Expand Test Coverage to >80% 🔴
**Effort:** Medium (3-4 weeks)
**Dependencies:** None
**Reference:** [Testing & Quality Assurance](system_plan.md#testing--quality-assurance)

#### Unit Tests
- [ ] **Predictor class tests** (`test_predictor.py`)
  - [ ] Test preprocessing with various input data formats
  - [ ] Test prediction output format and types
  - [ ] Test threshold adjustments (orange_threshold, yellow_threshold)
  - [ ] Test model loading from S3 (with mocks)
  - [ ] Test model loading from local files
  - [ ] Test feature engineering correctness
  - [ ] Test categorical encoding mappings

- [ ] **Visualization function tests** (`test_visualizations.py`)
  - [ ] Test all 4 visualization functions return BytesIO
  - [ ] Test with empty/null data (graceful degradation)
  - [ ] Test merge conflict handling
  - [ ] Test plot dimensions and file sizes
  - [ ] Test matplotlib figure cleanup (no memory leaks)

- [ ] **S3 integration tests** (`test_s3_integration.py`)
  - [ ] Test file upload/download with mocks
  - [ ] Test streaming operations
  - [ ] Test error handling (connection failures, missing files)
  - [ ] Test retry logic

#### Integration Tests
- [ ] **Pipeline execution tests** (`test_h2s_pipeline.py`)
  - [ ] Test full pipeline with local test data
  - [ ] Test asset dependencies resolve correctly
  - [ ] Validate outputs in expected S3 paths (with mocks)
  - [ ] Test error propagation between assets

- [ ] **End-to-end tests**
  - [ ] Test: Forecast ingestion → prediction → visualization → export
  - [ ] Test: Historical validation workflow
  - [ ] Test: Alert generation pipeline

#### Performance Tests
- [ ] **Model inference benchmarks**
  - [ ] Measure latency for single prediction (<1 second)
  - [ ] Measure batch throughput (4000+ samples in <10 seconds)
  - [ ] Profile memory usage during prediction

- [ ] **Pipeline performance tests**
  - [ ] Measure full materialization time (<3 minutes)
  - [ ] Track memory usage during pipeline runs (<2 GB peak)

#### Coverage & CI/CD
- [ ] **Set up code coverage tracking**
  - [ ] Configure pytest-cov
  - [ ] Generate HTML coverage reports
  - [ ] Set minimum coverage threshold: 80%

- [ ] **Add CI/CD pipeline**
  - [ ] Run tests on every commit
  - [ ] Fail build if coverage drops below 80%
  - [ ] Run tests with local data (use_local_data=True)

**Acceptance Criteria:**
- [ ] Overall test coverage >80%
- [ ] All critical paths have tests
- [ ] CI/CD pipeline running successfully
- [ ] Tests run in <5 minutes
- [ ] No flaky tests

---

## 📅 Medium Term (6 months)

### Epic 5: Public Dashboard Deployment (Netlify) 🟡
**Effort:** Large (6-8 weeks)
**Dependencies:** Performance monitoring dashboard
**Reference:** [Public Dashboard](system_plan.md#public-dashboard-netlify-deployment)

#### Phase 1: Technology Stack Setup
- [ ] **Choose and configure framework**
  - [ ] Decide: Next.js vs Gatsby
  - [ ] Set up project repository
  - [ ] Configure build pipeline
  - [ ] Add UI framework (React + Tailwind CSS)
  - [ ] Add charting library (D3.js or Recharts)

#### Phase 2: Core Features
- [ ] **Current Status View**
  - [ ] Fetch latest predictions from S3
  - [ ] Display H2S forecast for next 24 hours
  - [ ] Show current category with color coding
  - [ ] Display last update timestamp
  - [ ] Add auto-refresh (every 5 minutes)

- [ ] **Historical View**
  - [ ] Fetch historical predictions from S3
  - [ ] Create predictions vs actuals comparison chart
  - [ ] Add date range filter
  - [ ] Display accuracy metrics over time
  - [ ] Filter by category (green/yellow/orange)

- [ ] **Model Comparison View**
  - [ ] List available monthly models
  - [ ] Display performance metrics side-by-side
  - [ ] Show visualizations for selected model
  - [ ] Interactive model version selection

- [ ] **Environmental Context**
  - [ ] Display weather conditions (temp, wind, humidity)
  - [ ] Show tidal state and flow rates
  - [ ] Interactive timeline with all variables
  - [ ] Toggle environmental overlay on predictions

#### Phase 3: Data Access & API
- [ ] **Downloadable Data**
  - [ ] Add CSV export button for predictions
  - [ ] Add JSON export option
  - [ ] Implement date range selection for exports
  - [ ] Add historical archive access

#### Phase 4: UX & Accessibility
- [ ] **Mobile Responsive Design**
  - [ ] Optimize for phones and tablets
  - [ ] Touch-friendly interface
  - [ ] Responsive charts and tables

- [ ] **Accessibility**
  - [ ] WCAG 2.1 AA compliance
  - [ ] Screen reader support
  - [ ] Keyboard navigation
  - [ ] Color contrast validation

- [ ] **Performance Optimization**
  - [ ] Initial load <2 seconds
  - [ ] Interactive within 3 seconds
  - [ ] Lazy loading for historical data
  - [ ] Image optimization for visualizations

#### Phase 5: Deployment & Automation
- [ ] **Netlify Setup**
  - [ ] Connect GitHub repository
  - [ ] Configure build settings
  - [ ] Set up environment variables
  - [ ] Configure custom domain (if needed)

- [ ] **Auto-Rebuild Integration**
  - [ ] Add webhook trigger from Dagster
  - [ ] Trigger rebuild on new predictions
  - [ ] Set up branch previews for development

**Acceptance Criteria:**
- [ ] Dashboard accessible publicly via Netlify
- [ ] All views functional and tested
- [ ] Mobile responsive on iOS and Android
- [ ] WCAG 2.1 AA compliant
- [ ] Auto-updates when new predictions available
- [ ] Performance metrics met (load <2s, interactive <3s)

---

### Epic 6: SMS Alert Integration 🟡
**Effort:** Small (1-2 weeks)
**Dependencies:** Automated alerting system
**Reference:** [Alerting & Notifications](system_plan.md#alerting--notifications)

#### Setup & Configuration
- [ ] **Choose SMS provider**
  - [ ] Evaluate Twilio, AWS SNS, or similar
  - [ ] Create account and get API credentials
  - [ ] Test SMS delivery

- [ ] **Configure SMS settings**
  - [ ] Add SMS provider credentials to .env
  - [ ] Define SMS recipient list (phone numbers)
  - [ ] Set up SMS templates (160 character limit)

#### Implementation
- [ ] **Create SMS alert logic**
  - [ ] Modify alert_detection asset to trigger SMS
  - [ ] SMS only for critical alerts (>30 ppb)
  - [ ] Format message: "CRITICAL H2S ALERT: 35 ppb predicted at NESTOR-BES. Confidence: 85%"
  - [ ] Add error handling and retry logic

- [ ] **Add delivery tracking**
  - [ ] Log SMS delivery status to S3
  - [ ] Track delivery success rate
  - [ ] Alert if SMS delivery fails repeatedly

**Acceptance Criteria:**
- [ ] SMS sent for critical predictions (>30 ppb)
- [ ] Message includes key info within 160 chars
- [ ] Delivery success rate >95%
- [ ] No spam (max 1 SMS per hour for same condition)

---

### Epic 8: API for External Integrations 🟡
**Effort:** Medium (3-4 weeks)
**Dependencies:** None
**Reference:** [Future Roadmap](system_plan.md#future-roadmap)

#### Phase 1: API Design
- [ ] **Define API endpoints**
  - [ ] GET `/api/predictions/latest` - Latest predictions
  - [ ] GET `/api/predictions/{date}` - Predictions for specific date
  - [ ] GET `/api/alerts/latest` - Recent alerts
  - [ ] GET `/api/models` - Available model versions
  - [ ] GET `/api/performance/{model_id}` - Model metrics

- [ ] **Choose framework**
  - [ ] FastAPI (recommended for Python)
  - [ ] Flask (simpler option)
  - [ ] API Gateway + Lambda (serverless)

#### Phase 2: Implementation
- [ ] **Build API server**
  - [ ] Set up FastAPI project
  - [ ] Implement endpoints
  - [ ] Add data fetching from S3
  - [ ] Add caching layer (Redis or in-memory)

- [ ] **Add authentication**
  - [ ] Implement API key authentication
  - [ ] Rate limiting (100 requests/hour)
  - [ ] Usage tracking per API key

#### Phase 3: Documentation & Deployment
- [ ] **Create API documentation**
  - [ ] Auto-generate OpenAPI/Swagger docs
  - [ ] Add usage examples
  - [ ] Document rate limits and quotas

- [ ] **Deploy API**
  - [ ] Choose hosting (AWS, Heroku, or similar)
  - [ ] Set up CI/CD pipeline
  - [ ] Add monitoring and logging

**Acceptance Criteria:**
- [ ] API endpoints functional and tested
- [ ] Authentication working
- [ ] Documentation published
- [ ] API available 99.5% uptime
- [ ] Response time <500ms for all endpoints

---

## 📅 Long Term (12+ months)

### Epic 9: Real-Time Streaming Predictions 🟢
**Effort:** Extra Large (8-12 weeks)
**Dependencies:** Performance monitoring, API
**Reference:** [Future Roadmap](system_plan.md#future-roadmap)

- [ ] **Set up streaming infrastructure**
  - [ ] Choose streaming platform (Kafka, AWS Kinesis, or Pub/Sub)
  - [ ] Implement data stream ingestion
  - [ ] Add stream processing logic

- [ ] **Modify prediction pipeline for streaming**
  - [ ] Refactor predictor for sub-minute latency
  - [ ] Add real-time preprocessing
  - [ ] Implement streaming output to dashboards

- [ ] **Add real-time alerting**
  - [ ] Trigger alerts on streaming predictions
  - [ ] WebSocket updates to dashboard
  - [ ] Push notifications to mobile apps

**Acceptance Criteria:**
- [ ] Predictions generated within 10 seconds of new data
- [ ] Sub-minute end-to-end latency
- [ ] Real-time updates on dashboard

---

### Epic 10: Advanced Ensemble Models 🟢
**Effort:** Large (6-8 weeks)
**Dependencies:** Monthly retraining workflow
**Reference:** [Future Roadmap](system_plan.md#future-roadmap)

- [ ] **Experiment with multiple model types**
  - [ ] Train Random Forest baseline
  - [ ] Train LightGBM variant
  - [ ] Train Neural Network (LSTM for time series)
  - [ ] Compare performance vs XGBoost

- [ ] **Build ensemble architecture**
  - [ ] Implement voting ensemble
  - [ ] Implement stacking ensemble
  - [ ] Compare ensemble vs single models

- [ ] **Deploy best ensemble**
  - [ ] A/B test ensemble vs current XGBoost
  - [ ] Deploy if performance improves >5%

**Acceptance Criteria:**
- [ ] 3+ models trained and evaluated
- [ ] Ensemble shows improvement over single model
- [ ] Production deployment successful

---

### Epic 11: Explainable AI Features (SHAP Values) 🟢
**Effort:** Medium (3-4 weeks)
**Dependencies:** None
**Reference:** [Future Roadmap](system_plan.md#future-roadmap)

- [ ] **Integrate SHAP library**
  - [ ] Add SHAP to dependencies
  - [ ] Generate SHAP values for predictions
  - [ ] Create SHAP visualization functions

- [ ] **Add explanations to predictions**
  - [ ] Show top 3 contributing features per prediction
  - [ ] Generate SHAP waterfall plots
  - [ ] Add feature contribution to alert emails

- [ ] **Dashboard integration**
  - [ ] Add "Why this prediction?" section
  - [ ] Interactive SHAP plots
  - [ ] Feature importance comparison

**Acceptance Criteria:**
- [ ] SHAP values generated for all predictions
- [ ] Visualizations clear and understandable
- [ ] Integrated into dashboard and alerts

---

### Epic 12: Mobile App with Push Notifications 🟢
**Effort:** Extra Large (12-16 weeks)
**Dependencies:** API, Real-time streaming
**Reference:** [Future Roadmap](system_plan.md#future-roadmap)

- [ ] **Mobile app development**
  - [ ] Choose platform (React Native for cross-platform)
  - [ ] Design mobile UI/UX
  - [ ] Implement core features:
    - [ ] Current predictions view
    - [ ] Historical trends
    - [ ] Alert history
    - [ ] Settings (notification preferences)

- [ ] **Push notification setup**
  - [ ] Integrate Firebase Cloud Messaging (FCM)
  - [ ] Implement notification service
  - [ ] Add notification preferences (alert levels, quiet hours)

- [ ] **App deployment**
  - [ ] Submit to Apple App Store
  - [ ] Submit to Google Play Store
  - [ ] Set up crash reporting (Sentry)

**Acceptance Criteria:**
- [ ] App available on iOS and Android
- [ ] Push notifications functional
- [ ] <100ms response time for API calls
- [ ] 4.5+ star rating

---

### Epic 13: Integration with Regional Air Quality Systems 🟢
**Effort:** Medium (4-6 weeks)
**Dependencies:** API
**Reference:** [Future Roadmap](system_plan.md#future-roadmap)

- [ ] **Identify integration partners**
  - [ ] Contact County environmental health
  - [ ] Identify existing air quality monitoring systems
  - [ ] Define data sharing agreements

- [ ] **Build integrations**
  - [ ] Add webhook endpoints for data push
  - [ ] Implement data format conversions
  - [ ] Add API endpoints for partner systems to pull data

- [ ] **Testing & validation**
  - [ ] Test data flow end-to-end
  - [ ] Validate data quality
  - [ ] Monitor integration health

**Acceptance Criteria:**
- [ ] Data shared with 1+ regional systems
- [ ] Integration stable and reliable
- [ ] Data quality validated

---

### Epic 14: Automated Model Optimization (AutoML) 🟢
**Effort:** Large (6-8 weeks)
**Dependencies:** Monthly retraining workflow
**Reference:** [Future Roadmap](system_plan.md#future-roadmap)

- [ ] **Choose AutoML framework**
  - [ ] Evaluate: AutoGluon, H2O.ai, TPOT
  - [ ] Test with H2S dataset
  - [ ] Compare vs manual tuning

- [ ] **Integrate AutoML into pipeline**
  - [ ] Add AutoML as optional training mode
  - [ ] Configure search space and budget
  - [ ] Track experiment results

- [ ] **Automate hyperparameter tuning**
  - [ ] Bayesian optimization for XGBoost
  - [ ] Grid search for ensemble weights
  - [ ] Early stopping for efficiency

**Acceptance Criteria:**
- [ ] AutoML finds models competitive with manual tuning
- [ ] Training time <24 hours
- [ ] Results reproducible and tracked

---

## Task Management

### How to Use This File

1. **Update regularly** - Mark tasks complete as work finishes
2. **Add details** - Expand subtasks as needed for your workflow
3. **Track blockers** - Use ⏸️ to mark blocked tasks
4. **Sync with system_plan.md** - Update when plan changes

### Quick Stats

- **Completed:** Epic 0 (thresholds), Epic 1 (monthly retraining), Infrastructure, Multi-Station + Daily Analysis, Epic 2 (Slack alerting), Epic 7 (multi-site)
- **In Progress:** Multi-horizon validation, Epic 3 (dashboard metrics), Epic 15 (forecast accuracy metrics - PLANNING)
- **Short Term (3 mo):** Epic 4 (test coverage)
- **Medium Term (6 mo):** Epic 5 (Netlify dashboard), Epic 6 (SMS), Epic 8 (API)
- **Long Term (12+ mo):** Epics 9–14 (streaming, AutoML, mobile, etc.)

### Next Actions

1. 🔴 Review and implement Epic 15 (Forecast Accuracy Metrics) — answers design questions, then implement
2. 🔴 Validate and activate multi-horizon pipeline (`mh_forecast_job`)
3. 🔴 Expand test coverage to >80% (Epic 4)
4. 🟡 Public Netlify dashboard (Epic 5)
