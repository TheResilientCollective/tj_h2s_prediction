# H2S Forecasting System - Task Tracker

**Last Updated:** 2026-01-27
**Source:** `system_plan.md` - Keep synchronized when plan changes

## Legend
- 🔴 High Priority
- 🟡 Medium Priority
- 🟢 Low Priority
- ✅ Completed
- 🔄 In Progress
- ⏸️ Blocked

---

## 🔄 In Progress

### Epic 0: Update H2S Category Thresholds to Client Spec 🔴
**Effort:** Small (1-2 days)
**Dependencies:** None
**Reference:** [H2S Categories](system_plan.md#h2s-categories-current-implementation)
**URGENT:** Current implementation does not match client specifications

#### Current vs Client Thresholds
**Current Implementation:**
- Green: <5 ppb
- Yellow: 5-15 ppb
- Orange: ≥15 ppb

**Client Specification (NEW):**
- Green: <5 ppb
- Yellow: 5-30 ppb
- Orange: ≥30 ppb

#### Code Changes Required
- [ ] **Update visualization functions** (`projects/h2s/src/h2s/predictor/visualizations.py`)
  - [ ] Update `categorize_h2s()` in `generate_confusion_matrix()` (line 91-97)
  - [ ] Update `categorize_h2s()` in `generate_confusion_matrix_with_metrics()` (line 170-176)
  - [ ] Update `categorize_h2s()` in `generate_model_comparison()` (line 271-277)
  - [ ] Update timeline visualization threshold markers (lines 445-446)
    - Change: Yellow dot from 10 ppb to 15 ppb
    - Change: Orange dot from 15 ppb to 30 ppb

- [ ] **Retrain model with new thresholds**
  - [ ] Relabel training data with new categories
  - [ ] Retrain XGBoost model
  - [ ] Validate performance with new thresholds
  - [ ] Upload new model to S3

- [ ] **Update documentation**
  - [ ] Update README.md category definitions
  - [ ] Update DEPLOYMENT_GUIDE.md
  - [ ] Update NESTOR_BES_H2S_Forecasting_Report.md
  - [ ] Update system_plan.md "Current Implementation" section

- [ ] **Testing**
  - [ ] Update test fixtures with new thresholds
  - [ ] Update expected values in test_predictor.py
  - [ ] Update test_visualizations.py assertions
  - [ ] Verify confusion matrix uses new categories

**Acceptance Criteria:**
- [ ] All code uses new thresholds: Yellow 5-30 ppb, Orange ≥30 ppb
- [ ] Model retrained with new category labels
- [ ] Visualizations show correct threshold markers
- [ ] Tests pass with new thresholds
- [ ] Documentation updated throughout

**Note:** This is a breaking change that requires model retraining. Coordinate deployment to avoid confusion.

---

### Epic 1: Monthly Model Retraining Workflow 🔴
**Effort:** Large (4-6 weeks)
**Dependencies:** None
**Reference:** [Model Management & Versioning](system_plan.md#model-management--versioning)

#### Phase 1: Data Extraction & Preparation
- [ ] **Create training data extraction asset**
  - [ ] Add `training_data_extraction` asset to pipeline
  - [ ] Query previous month's environmental data from S3
  - [ ] Filter to NESTOR-BES site
  - [ ] Merge with actual H2S measurements from modeldata_h2s.csv
  - [ ] Validate data quality (completeness, outliers)
  - [ ] Export to S3: `tijuana/forecast/training_data/{YYYY_MM}/`

#### Phase 2: Model Training Pipeline
- [ ] **Implement cross-validation training**
  - [ ] Create `model_training` asset (depends on training_data_extraction)
  - [ ] Implement 5-fold time-series aware cross-validation
  - [ ] Track performance metrics per fold (accuracy, precision, recall)
  - [ ] Generate feature importance analysis
  - [ ] Save best model with hyperparameters

- [ ] **Add model versioning & archiving**
  - [ ] Implement naming convention: `nestor_xgboost_{YYYY_MM}_v{version}.json`
  - [ ] Store preprocessing metadata as JSON
  - [ ] Archive to S3: `tijuana/forecast/models/archive/{YYYY_MM}/`
  - [ ] Include training metadata: date, data range, metrics, hyperparameters

#### Phase 3: Validation & Deployment
- [ ] **Create validation workflow**
  - [ ] Test new model on held-out validation set
  - [ ] Compare against current production model
  - [ ] Generate validation report with confusion matrix
  - [ ] Store validation results in S3

- [ ] **Implement deployment automation**
  - [ ] Add manual approval step for production deployment
  - [ ] Copy approved model to `latest/tijuana/forecast/models/`
  - [ ] Update model version in Dagster configuration
  - [ ] Archive previous production model

#### Phase 4: Scheduling & Automation
- [ ] **Add Dagster schedule**
  - [ ] Create monthly schedule (1st of each month, 2 AM)
  - [ ] Configure schedule to run full retraining pipeline
  - [ ] Add success/failure notifications

**Acceptance Criteria:**
- [ ] Models trained automatically on 1st of each month
- [ ] Performance metrics logged to S3 for every training run
- [ ] Model archives include all metadata and artifacts
- [ ] Manual approval required before production deployment
- [ ] Previous model backed up before replacement

---

### Epic 2: Automated Alerting System 🔴
**Effort:** Medium (2-3 weeks)
**Dependencies:** None
**Reference:** [Alerting & Notifications](system_plan.md#alerting--notifications)

#### Phase 1: Email Configuration
- [ ] **Set up email infrastructure**
  - [ ] Add SMTP settings to .env file
    ```bash
    ALERT_EMAIL_RECIPIENTS=alerts@example.com,ops@example.com
    SMTP_SERVER=smtp.gmail.com
    SMTP_PORT=587
    SMTP_USERNAME=your-email@gmail.com
    SMTP_PASSWORD=app-specific-password
    ```
  - [ ] Choose email provider (SendGrid, Gmail SMTP, or AWS SES)
  - [ ] Test email connectivity

- [ ] **Create email templates**
  - [ ] Design HTML email template with H2S branding
  - [ ] Add sections: prediction level, confidence, contributing factors
  - [ ] Include forecast duration (1hr, 3hr, 6hr)
  - [ ] Add recommended actions section
  - [ ] Create plain-text fallback version

#### Phase 2: Alert Detection Logic
- [ ] **Create alert detection asset**
  - [ ] Add `alert_detection` asset (depends on h2s_predictions)
  - [ ] Implement trigger conditions:
    - [ ] Orange predictions (H2S ≥15 ppb)
    - [ ] Critical sustained (>30 ppb for 2+ consecutive hours)
    - [ ] Yellow predictions with high confidence (>80%)
  - [ ] Add cooldown logic (no duplicate alerts within 1 hour)
  - [ ] Track alert history in local state or S3

- [ ] **Add contributing factors analysis**
  - [ ] Identify top 3 environmental factors from feature importance
  - [ ] Format as human-readable text (e.g., "High flow + low wind speed")
  - [ ] Include current values and thresholds

#### Phase 3: Alert Delivery & Tracking
- [ ] **Implement email delivery**
  - [ ] Create email client with error handling
  - [ ] Send alerts to recipients from .env
  - [ ] Add retry logic (3 attempts with exponential backoff)
  - [ ] Log delivery success/failure

- [ ] **Add delivery tracking**
  - [ ] Log all alerts to S3: `tijuana/forecast/alerts/{YYYY-MM-DD}/`
  - [ ] Include: timestamp, prediction data, recipients, delivery status
  - [ ] Create alert summary asset (daily digest)

#### Phase 4: Testing & Monitoring
- [ ] **Test alert system**
  - [ ] Create test alert trigger (manual button in Dagster UI)
  - [ ] Verify email formatting and delivery
  - [ ] Test cooldown logic
  - [ ] Confirm no duplicate alerts

- [ ] **Add monitoring**
  - [ ] Track alert delivery success rate
  - [ ] Monitor alert frequency (alerts/day)
  - [ ] Create dashboard for alert history

**Acceptance Criteria:**
- [ ] Emails sent for orange/critical predictions
- [ ] Alert content includes prediction level, confidence, and contributing factors
- [ ] Delivery success rate >95%
- [ ] No duplicate alerts within 1 hour for same prediction
- [ ] All alerts logged to S3

---

### Epic 3: Performance Monitoring Dashboard 🔴
**Effort:** Medium (2-3 weeks)
**Dependencies:** None
**Reference:** [Operations & Monitoring](system_plan.md#operations--monitoring)

#### Phase 1: Metrics Collection
- [ ] **Add metrics tracking assets**
  - [ ] Create `daily_metrics` asset
  - [ ] Track prediction counts by category (green/yellow/orange)
  - [ ] Calculate accuracy metrics (requires actual H2S data)
  - [ ] Monitor data pipeline health (asset success/failure rates)
  - [ ] Store metrics to S3: `tijuana/forecast/metrics/{YYYY-MM-DD}/`

- [ ] **Track model performance**
  - [ ] Compare predictions vs actuals (when available)
  - [ ] Calculate daily precision/recall by category
  - [ ] Track confusion matrix updates
  - [ ] Monitor feature importance stability

#### Phase 2: Dashboard Creation
- [ ] **Choose dashboard framework**
  - [ ] Option 1: Dagster built-in asset metadata
  - [ ] Option 2: Streamlit dashboard
  - [ ] Option 3: Grafana + data exports
  - [ ] Make decision based on team familiarity

- [ ] **Build core dashboard views**
  - [ ] **Real-time view:**
    - [ ] Current prediction status
    - [ ] Latest forecast timestamp
    - [ ] 24h rolling prediction counts by category
    - [ ] Alert delivery success rate
  - [ ] **Performance view:**
    - [ ] Monthly accuracy trends
    - [ ] Precision/Recall by category charts
    - [ ] Confusion matrix heatmap
    - [ ] Feature importance comparison
  - [ ] **System health view:**
    - [ ] Asset execution success rates
    - [ ] S3 sync status and latency
    - [ ] Error rates by component
    - [ ] Data freshness indicators

#### Phase 3: Alerts & Automation
- [ ] **Add dashboard alerts**
  - [ ] Alert when accuracy drops >5%
  - [ ] Alert when data becomes stale (>2 hours)
  - [ ] Alert on pipeline failures

- [ ] **Schedule dashboard updates**
  - [ ] Auto-refresh every 5 minutes
  - [ ] Daily summary email
  - [ ] Weekly performance report

**Acceptance Criteria:**
- [ ] Dashboard displays real-time prediction metrics
- [ ] Historical performance trends visible
- [ ] System health indicators accurate
- [ ] Dashboard accessible to team
- [ ] Auto-updates without manual intervention

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

### Epic 7: Multi-Site Model Support 🟡
**Effort:** Large (4-6 weeks)
**Dependencies:** Monthly retraining workflow
**Reference:** [Modeling](system_plan.md#modeling)

#### Phase 1: Data Preparation
- [ ] **Identify pilot site (Site 2)**
  - [ ] Confirm data availability for 2nd site
  - [ ] Validate data quality and completeness
  - [ ] Align with NESTOR-BES data format

- [ ] **Refactor data pipeline for multi-site**
  - [ ] Add site_name parameter to data assets
  - [ ] Update S3 paths to include site identifier
  - [ ] Modify raw_environmental_data to filter by site

#### Phase 2: Model Architecture
- [ ] **Implement site-specific models**
  - [ ] Train separate model for Site 2
  - [ ] Store models with site identifier: `{site}_xgboost_*.json`
  - [ ] Update predictor to load site-specific models

- [ ] **Add site selection to pipeline**
  - [ ] Add site configuration to asset configs
  - [ ] Support parallel predictions for multiple sites
  - [ ] Separate S3 output paths by site

#### Phase 3: Visualization & Reporting
- [ ] **Update visualizations for multi-site**
  - [ ] Add site name to all plots
  - [ ] Support site comparison views
  - [ ] Update dashboard to show multiple sites

**Acceptance Criteria:**
- [ ] Models trained for 2 sites (NESTOR-BES + Site 2)
- [ ] Predictions generated independently per site
- [ ] Separate S3 storage paths per site
- [ ] Dashboard shows site selector
- [ ] Code architecture supports easy addition of Site 3+

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

- **Total Epics:** 15
- **In Progress:** 4 epics (includes 1 URGENT)
- **Short Term (3 mo):** 1 epic
- **Medium Term (6 mo):** 4 epics
- **Long Term (12+ mo):** 6 epics

### Next Actions

Priority tasks to start immediately:
1. 🔴 **URGENT:** Update H2S category thresholds to client spec (Epic 0)
2. 🔴 Monthly model retraining workflow (Epic 1)
3. 🔴 Automated alerting system (Epic 2)
4. 🔴 Performance monitoring dashboard (Epic 3)
