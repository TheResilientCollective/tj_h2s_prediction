# Calibration & Model Selection Strategy

## Overview

The H2S prediction system uses a **multi-stage calibration loop** to keep models fresh, compare algorithms, and ensure operational readiness without introducing risk.

**Key Principle:** Approval is explicit (human runs deployment job), not automatic.

---

## Training & Selection Pipeline

### Stage 1: Multi-Model Training (station_model_training_job)

For each station (3 total) and task (4 total: regression + 3 classifiers):

#### A. Prepare Features
```python
df = prepare_multi_station_features(
    raw_observations,
    station='NESTOR__BES'
)
# Returns: Evidence (33 features) + Lean (19 features) variants
```

#### B. Split Data
```python
# 80/20 train/test split
# Time-series aware (no random shuffle to preserve temporal structure)
X_train, X_test = train_test_split(X, test_size=0.2, shuffle=False)
y_train, y_test = ...
```

#### C. Train RF + XGBoost for Each Task
```python
for variant in ['Evidence', 'Lean']:
    for task in ['regression', 'clf_5ppb', 'clf_10ppb', 'clf_30ppb']:
        # Train both models
        rf_model = train_rf(X_train, y_train)
        xgb_model = train_xgb(X_train, y_train)
        
        # Evaluate both
        rf_metrics = eval_model(rf_model, X_test, y_test)
        xgb_metrics = eval_model(xgb_model, X_test, y_test)
        
        # Select or ensemble
        best_model, choice, algorithm_choices = train_and_select(
            X_train, X_test, y_train, y_test,
            task=task
        )
```

#### D. Auto-Select or Ensemble
For each task:

**Selection Metric:**
- **Regression:** recall_30 (How many >30 ppb events caught?)
- **Classifiers:** AUC (Ranking quality at 0.5 threshold)

**Decision Logic:**
```python
margin = ensemble_margin[selection_metric]

if abs(score_rf - score_xgb) < margin:
    # Scores too close — ensemble
    weight_rf = max(score_rf, 0.0) / (score_rf + score_xgb)
    model = EnsembleRegressor(rf, xgb, weight_a=weight_rf)
    choice = 'Ensemble'
elif score_rf > score_xgb:
    model = rf
    choice = 'RandomForest'
else:
    model = xgb
    choice = 'XGBoost'
```

**Default Margins:**
| Task | Metric | Margin | Rationale |
|------|--------|--------|-----------|
| regression | recall_30 | 0.02 (2 pp) | Tight; obvious wins dominate |
| clf_5ppb | AUC | 0.01 (0.01 points) | High precision for 5 ppb |
| clf_10ppb | AUC | 0.01 | High precision for 10 ppb |
| clf_30ppb | AUC | 0.01 | High precision for watch threshold |

### Stage 2: Immutable Archive

Every training run creates an immutable S3 archive:

```
s3://resilientpublic/tijuana/forecast/models/archive/stations/
  ├── NESTOR__BES/
  │   ├── 20260624T123456Z-abc123def/  ← version tag = timestamp + git SHA
  │   │   ├── regression_evidence.pkl
  │   │   ├── clf_5ppb_evidence.pkl
  │   │   ├── clf_10ppb_evidence.pkl
  │   │   ├── clf_30ppb_evidence.pkl
  │   │   ├── regression_lean.pkl
  │   │   ├── clf_5ppb_lean.pkl
  │   │   ├── clf_10ppb_lean.pkl
  │   │   ├── clf_30ppb_lean.pkl
  │   │   ├── features_evidence.json
  │   │   ├── features_lean.json
  │   │   ├── training_report.json  ← metrics + algorithm_choices
  │   │   └── archive_metadata.json  ← immutable record
  │   └── 20260618T090000Z-xyz789/  ← older version (never overwritten)
  ├── IB_CIVIC_CTR/
  └── SAN_YSIDRO/
```

**Why Immutable?** 
- Audit trail: Any forecast can be replayed against the models that produced it
- No accidental overwrites
- Promotes and rollbacks reference specific versions

### Stage 3: Training Report (Slack Notification)

After training completes, a comparison report posts to Slack:

```
🏋️ NESTOR__BES Model Training Complete (2026-06-24)

REGRESSION (recall_30 selection):
  RandomForest: recall_30=0.68, MAE=3.45
  XGBoost:     recall_30=0.71, MAE=3.32 ✓
  → Selected: XGBoost (0.03 pp better, outside margin)

CLF_5PPB (AUC selection):
  RandomForest: AUC=0.952, Recall=0.891
  XGBoost:     AUC=0.954, Recall=0.902
  → Selected: Ensemble (0.002 pp diff, within 0.01 margin)

CLF_10PPB (AUC selection):
  RandomForest: AUC=0.962, Recall=0.832
  XGBoost:     AUC=0.963, Recall=0.841
  → Selected: Ensemble (0.001 pp diff, within 0.01 margin)

CLF_30PPB (AUC selection):
  RandomForest: AUC=0.970, Recall=0.953
  XGBoost:     AUC=0.971, Recall=0.951
  → Selected: Ensemble (0.001 pp diff, within 0.01 margin)

📊 View archive: /archive/NESTOR__BES/20260624T123456Z-abc123def/

✅ Ready for deployment (run station_model_deployment_job to approve)
OR
🔄 Promote an older version (run promote_station_models_job --version ...)
```

---

## Stage 4: Explicit Deployment Approval

**Job:** `station_model_deployment_job`

**Input:** Select archived version (default = latest training run)

**Action:**
1. Load models from archive
2. Validate checksums (integrity check)
3. Upload to production S3:
   ```
   s3://resilientpublic/tijuana/forecast/models/stations/NESTOR__BES/Evidence/regression.pkl
   s3://resilientpublic/tijuana/forecast/models/stations/NESTOR__BES/Evidence/clf_5ppb.pkl
   ... (all 8 per-variant models)
   ```
4. Stamp `deployment_metadata.json` with active model_version
5. Slack notification: "Models deployed for NESTOR__BES, IB_CIVIC_CTR, SAN_YSIDRO"

**Gate:** Operator explicitly runs this job. If they don't, production models stay unchanged.

---

## Stage 5: Forecasting with Deployed Models

**Job:** `station_forecast_job` (runs every 6h)

```python
# Load production models
predictor = H2SPredictor.from_s3(s3, model_path, feature_schema_path)

# Run recursive forecasting
for station in stations:
    for variant in ['Evidence', 'Lean']:
        products = run_products(
            feature_frame=forecast_features,
            h2s_history=recent_observations,
            models=variant_models,
            feature_cols=feature_schema
        )
        # Emit: nowcast/nearcast/forecast rows (p5, p10, p30, h2s_pred)
```

---

## Model Selection Algorithm Deep Dive

### Why Ensemble?

The ensemble approach (`EnsembleRegressor`, `EnsembleClassifier`) combines RF and XGBoost when they score similarly:

**Regression Ensemble:**
```python
class EnsembleRegressor:
    def __init__(self, model_a, model_b, weight_a=0.5):
        self.model_a = model_a  # RF or XGB
        self.model_b = model_b  # XGB or RF
        self.weight_a = weight_a
        self.weight_b = 1.0 - weight_a

    def predict(self, X):
        # Weighted average of predictions
        return self.weight_a * self.model_a.predict(X) + self.weight_b * self.model_b.predict(X)
```

**Classifier Ensemble:**
```python
class EnsembleClassifier:
    def predict_proba(self, X):
        # Weighted average of probability estimates
        return self.weight_a * self.model_a.predict_proba(X) + self.weight_b * self.model_b.predict_proba(X)
```

**Weights:** Proportional to metric score (e.g., both have recall_30 ∈ [0, 1], weights ∝ recall scores).

**Rationale:** 
- Reduces variance (bagging effect)
- Combines RF robustness + XGB gradient-boosting precision
- Avoids overfitting to one algorithm
- Typical ensemble improvement: +0.5–1.0 pp on metrics

### Example: When Ensemble Wins

```
Regression task for NESTOR__BES (Evidence variant):

RF recall_30:   0.685
XGB recall_30:  0.698
Diff:           0.013  ← Within 0.02 margin

Decision: Ensemble
  weight_rf = 0.685 / (0.685 + 0.698) = 0.495
  weight_xgb = 0.698 / (0.685 + 0.698) = 0.505
  
  prediction = 0.495 * rf.predict(X) + 0.505 * xgb.predict(X)
  expected recall_30 ≈ (0.495 * 0.685) + (0.505 * 0.698) ≈ 0.692
  → Improvement from individual models (±0.7 pp)
```

---

## Calibration Loop: Feedback & Iteration

### Daily Cycle

1. **Station Forecast Job** (every 6h)
   - Run models on latest meteorological forecast
   - Emit products to S3

2. **Validation Store** (overnight)
   - Compare forecast leads to newly observed H2S
   - Accumulate in `validation_store/validation.parquet`

3. **Skill Curves** (daily)
   - Per (product, lead, variant): recall, precision, Spearman, MAE
   - Stored in `skill_curves.parquet`

4. **Performance Report** (weekly, Mon 14:00 UTC, currently STOPPED)
   - Asymmetric rubric: weight dangerous misses heavily
   - Scatter plot + skill-by-hour + verdict board
   - Plain-language narrative (ResilientLLM webhook)

### Monthly/Quarterly Recalibration

1. **Identify Drift:**
   - Review weekly performance reports
   - Compare current-month recall to historical baseline
   - If recall drops >5 pp, flag for retraining

2. **Retrain Models:**
   - Run `station_model_training_job` with latest data
   - Auto-select/ensemble new models
   - Review metrics in Slack report

3. **A/B Test or Promote:**
   - Run `promote_station_models_job` to move to production
   - Or hold in archive for more data

### Backfill for Historical Analysis

**Scenario:** "Let's see how this model would have performed 3 months ago."

```bash
# Generate archive for that month
cd projects/h2s
uv run python scripts/backfill_validation.py \
  --month 2026-03 \
  --station NESTOR__BES \
  --archive-version 20260301T000000Z-somehash

# Runs all forecasts with that archived model
# Computes validation & skill curves
# Stores under backtest/backfill_202603_20260624T123456Z-abc123def/
```

---

## Algorithm Choices Metadata

Every archived model carries `algorithm_choices` documenting what was auto-selected:

```json
{
  "training_date": "2026-06-24T12:34:56Z",
  "station": "NESTOR__BES",
  "algorithm_choices": {
    "regression": {
      "variant": "Evidence",
      "selected": "XGBoost",
      "selection_metric": "recall_30",
      "selection_value_rf": 0.685,
      "selection_value_xgb": 0.698,
      "margin": 0.02,
      "ensemble": false,
      "feature_importance": {
        "h2s_lag_1h": 0.124,
        "wind_speed_10m": 0.087,
        ...
      }
    },
    "clf_5ppb": {
      "variant": "Evidence",
      "selected": "Ensemble",
      "selection_metric": "auc",
      "selection_value_rf": 0.952,
      "selection_value_xgb": 0.954,
      "margin": 0.01,
      "ensemble_weight_rf": 0.495,
      "ensemble_weight_xgb": 0.505,
      "feature_importance": { ... }
    },
    ...
  }
}
```

**Use:** Replay any forecast and know exactly which algorithm produced it.

---

## Tuning Knobs

### 1. Selection Margin

**Location:** `multi_station_trainer.py`, `_DEFAULT_MARGINS` dict

```python
_DEFAULT_MARGINS = {
    "recall_5": 0.02,      # 2 pp tolerance
    "recall_10": 0.02,
    "recall_30": 0.02,
    "recall_100": 0.02,
    "r2": 0.02,            # 0.02 R² tolerance
    "auc": 0.01,           # 0.01 AUC tolerance
}
```

**Effect:** 
- Larger margin → more ensembles (averages variance)
- Smaller margin → more single-model picks (stronger signal needed)

**Recommendation:** Default (0.01–0.02) is appropriate. Don't tune without A/B test data.

### 2. Class Weights (Imbalanced Data)

**Location:** `model_trainer.py`, `calculate_class_weights`

```python
class_weights = {}
for class_name, class_idx in label_map.items():
    count = class_counts.get(class_name, 1)
    w = total / (n_classes * count)  # Inverse frequency
    if class_name in HAZARD_CLASSES:
        w *= hazard_multiplier  # 3.0 default
    weights[class_idx] = w
```

**Default hazard_multiplier:** 3.0 (yellow and orange get 3× weight vs green)

**Effect:** Reduces false negatives on alerts, at cost of more false positives.

**Tuning:** Increase to 5–10 if alert misses are critical; decrease to 1–2 if false alarms dominate complaints.

### 3. SMOTE for Sparse Classes

**Location:** `multi_station_trainer.py`, `use_smote_on_minority` flag

**Current Status:** OFF by default for regression; OFF for all classifiers.

**Reason:** SMOTE backtest degraded recall for clf_30ppb (unexpected; possible artifact creation).

**To Enable:**
```python
model, choice, metrics = train_and_select(
    X_train, X_test, y_train, y_test,
    task='clf_30ppb',
    use_smote_on_minority=True
)
```

**Recommendation:** Only use with careful ablation (separate validation fold to measure effect).

### 4. Selection Metric for Regression

**Location:** `train_and_select(..., selection_metric='recall_30')`

**Options:**
- `'recall_5'` — Optimize for catching all caution events
- `'recall_10'` — Optimize for yellow-high boundary
- `'recall_30'` — Optimize for watch level (current default)
- `'recall_100'` — Optimize for extreme events
- `'r2'` — Optimize for magnitude fit (legacy)

**Tradeoff:**
- recall_30 → more sensitive to extremes, may miss intermediate H2S
- r2 → better bulk fit, may miss high-end recall

**Recommendation:** Keep recall_30 (operational watch level) unless analysis shows it misses Tier 2 (10 ppb) events.

---

## Validation Against Real Observations

### Skill Curves (forecast_validation_rebuild_job)

For each (product, lead_hour, variant, station):

```python
# Load all forecast runs for this product/variant/station
products = load_products(product='nowcast', variant='Evidence', station='NESTOR__BES')

# Match forecasts to observations
validation = join_to_observations(products)

# Compute metrics
skill = {
    'n': n_matched,
    'spearman': spearman_correlation(y_true, y_pred),
    'mae': mean_absolute_error(y_true, y_pred),
    'recall_5': recall_at_threshold(y_true, y_pred, 5),
    'recall_10': recall_at_threshold(y_true, y_pred, 10),
    'recall_30': recall_at_threshold(y_true, y_pred, 30),
}
```

**Interpretation:**
- Spearman ≈ 0.8–0.9 for nowcast (skill high)
- Spearman ≈ 0.5–0.7 for nearcast (skill moderate)
- Spearman ≈ 0.2–0.4 for forecast (skill bounded by exogenous information)

**Use:** Confirms that offline training metrics translate to real forecasts.

---

## When to Retrain

**Automatic Trigger (Currently Manual):**

1. **Drift Detection:**
   - Skill curves show >5 pp drop in recall @ any level
   - FAR (false alarm rate) exceeds 10%

2. **New Data Available:**
   - 30+ new observation days since last training
   - Or seasonal change (spring/summer/fall/winter transition)

3. **Feature Changes:**
   - New meteorological source added
   - SBIWTP operations changed
   - Sensor recalibration

**Manual Approval:** Operator reviews metrics and decides to retrain. No automatic deployment.

---

## Next Steps for Users

1. **Review Training Reports:** Check Slack notifications after `station_model_training_job`
2. **Compare Metrics:** Use training_report.json to assess new models vs archived versions
3. **Run Deployment:** Execute `station_model_deployment_job` to approve new models
4. **Monitor Forecasts:** Watch skill curves; flag drift for recalibration
5. **Backfill Analysis:** Use `backfill_validation.py` to test old models on new data
