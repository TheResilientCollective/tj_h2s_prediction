# Regression Model — H2S Magnitude Prediction

## Overview

The regression model predicts actual H2S concentration (ppb) for each lead hour in a 24-hour forecast. It is the foundation upon which probability classifiers are built.

**Purpose:** Provide honest magnitude prediction across all hazard levels (green through orange).

**Algorithm:** XGBoost or RandomForest, auto-selected per training session.

**Variants:** Evidence (33 features) and Lean (19 features), trained
independently. **Lean is the operational primary** (`PRIMARY_VARIANT = "lean"`);
Evidence runs alongside as a cross-check.

**Stations:** One model per station (SAN_YSIDRO, NESTOR__BES, IB_CIVIC_CTR).

---

## Input Features (production 33-feature Evidence set)

The exact production feature list is `CORE_FEATURES` / `MODEL_FEATURES` in
`constants.py`, promoted from the 2026-06-10 "Berry" feature-trim ablation
(PR #27/#28). It is built by `prepare_multi_station_features` /
`ensure_base_features`. **Note there is no raw `Flow (m^3/s)` column,
`flow_log`/`flow_low`/`flow_high`, or SBIWTP channel in the production set** —
those live only in `MODEL_FEATURES_LEGACY` (44-feature), retained for legacy
preprocessing and not used for new training.

### Weather — raw (10 features)

| Feature | Units | Notes |
|---------|-------|-------|
| temperature_2m | °C | 2-meter air temperature (top exogenous Spearman ≈ 0.33) |
| wind_speed_10m | m/s | 10-meter wind speed |
| wind_gusts_10m | m/s | 10-meter wind gusts |
| relative_humidity_2m | % | 2-meter relative humidity |
| surface_pressure | Pa | Surface pressure |
| precipitation | mm | Hourly precipitation |
| cloud_cover | % | Total cloud cover |
| dewpoint_2m | °C | 2-meter dewpoint |
| wind_direction_sin | — | sin(wind_direction) cyclic encoding |
| wind_direction_cos | — | cos(wind_direction) cyclic encoding |

(Wind direction enters **only** through sin/cos; the raw `wind_direction_10m`
degrees column is not a model feature.)

### Wind rolling aggregates (4 features)

| Feature | Window | Notes |
|---------|--------|-------|
| wind_speed_10m_avg_2h | 2-hour mean | (4 h aggregates dropped in PR #27 as tree-redundant) |
| wind_speed_10m_avg_3h | 3-hour mean | |
| wind_gusts_10m_max_2h | 2-hour max | |
| wind_gusts_10m_max_3h | 3-hour max | |

### Tide (2 features)

| Feature | Notes |
|---------|-------|
| tide_height | Predicted tide height (m) |
| tidal_state_encoded | Flood/ebb state, encoded integer |

### Time cyclicals & regime (4 features)

| Feature | Notes |
|---------|-------|
| hour_sin, hour_cos | Diurnal cycle |
| month_sin, month_cos | Seasonal cycle |

### Derived regime / interactions (5 features)

| Feature | Notes |
|---------|-------|
| is_night | Day/night flag (absorbs hour-of-day signal) |
| source_regime | is_night × wind-direction quadrant — source upwind of sensor |
| stable_atm | Atmospheric stability proxy (88% recall on >100 ppb hours in calibration) |
| wind_x_stable_atm | wind_speed × stable_atm interaction |
| wind_temp_interaction | wind × temperature interaction |
| humidity_temp_interaction | humidity × temperature interaction |

(That is 6 rows; `is_night`, `source_regime`, `stable_atm` are the regime core
and `wind_x_stable_atm`, `wind_temp_interaction`, `humidity_temp_interaction` the
interactions — together 6 features.)

### H2S autoregressive lags (5 features)

| Feature | Notes |
|---------|-------|
| h2s_lag_1h | H2S 1 hour ago (autoregressive ceiling ≈ 0.70 Spearman) |
| h2s_lag_3h | H2S 3 hours ago |
| h2s_lag_6h | H2S 6 hours ago |
| h2s_rolling_6h | 6-hour rolling mean (consistently top-3 XGB importance) |
| h2s_rolling_24h | 24-hour rolling mean (baseline/regime level) |

**Note:** During recursive forecasting these lags are updated with predicted H2S
at each lead — the "autoregressive" behavior.

### Flow (2 features)

| Feature | Notes |
|---------|-------|
| flow_lag_6h | Tijuana River border flow, 6 h ago |
| flow_rolling_24h | 24-hour rolling mean flow |

Total: 10 + 4 + 2 + 4 + 6 + 5 + 2 = **33 features**.

---

## Feature Counts

| Variant | Count | Definition |
|---------|-------|------------|
| Evidence | 33 | `CORE_FEATURES` (all features above) |
| Lean | 19 | Evidence **minus** the 14 below |

**Lean = Evidence minus** (`MODEL_FEATURES_LEAN` in `constants.py`):
- Interactions: `wind_temp_interaction`, `humidity_temp_interaction`, `wind_x_stable_atm`
- Wind rolling: `wind_speed_10m_avg_2h/3h`, `wind_gusts_10m_max_2h/3h`
- Lower-importance weather: `dewpoint_2m`, `surface_pressure`, `cloud_cover`,
  `precipitation`, `wind_gusts_10m`
- Time cyclicals: `hour_sin`, `hour_cos` (is_night carries the hour-of-day signal)

So Lean keeps: temperature_2m, wind_speed_10m, wind_direction_sin/cos,
relative_humidity_2m, tide_height, tidal_state_encoded, month_sin/cos, is_night,
source_regime, stable_atm, h2s_lag_1h/3h/6h, h2s_rolling_6h/24h, flow_lag_6h,
flow_rolling_24h (**19 features**).

Lean did *not* pass the original ablation gate but is now the operational primary
(2026-06): it carries the headline forecasts and alerts, with Evidence shown
alongside. Both variants perform similarly; Lean is more interpretable and
less over-determined.

---

## Algorithm Details

### Model Factory (multi_station_trainer.py)

#### RandomForest Regressor
```python
RandomForestRegressor(
    n_estimators=500,
    max_depth=20,
    min_samples_leaf=5,
    max_features='sqrt',
    n_jobs=-1,
    random_state=42
)
```

#### XGBoost Regressor
```python
XGBRegressor(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1
)
```

### Training Pipeline

1. **Feature Engineering** (prepare_multi_station_features)
   - Load raw observations
   - Apply ensure_base_features (time/wind cyclicals, source_regime)
   - Compute per-site rolling windows (avoid cross-station contamination)
   - Filter: h2s_measured=True, H2S ≤ 500 ppb
   - Clip H2S to [0, ∞)

2. **Train/Test Split**
   - 80/20 split (time-series aware — no randomization)
   - RandomForest and XGBoost trained on same train/test fold

3. **Model Selection (train_and_select)**
   - Train both RF and XGBoost on train split
   - Evaluate both on test split
   - **Selection Metric:** recall_30 (default) — how many >30 ppb events are caught?
   - **Ensemble Decision:**
     - If |recall_30_rf - recall_30_xgb| < 0.02 (default margin), weighted ensemble (weights ∝ recall_30 scores)
     - Else pick the winner

4. **Final Model**
   - Train selected model (or ensemble) on full training set
   - Return model + algorithm_choices metadata

### Hyperparameters

| Parameter | RF | XGB | Rationale |
|-----------|----|----|-----------|
| n_estimators | 500 | 500 | Sufficient for convergence |
| max_depth | 20 | 6 | RF deeper (less prone to overfit); XGB shallow |
| learning_rate | N/A | 0.05 | Conservative; reduces overfit |
| subsample | N/A | 0.8 | Stochastic boosting reduces variance |
| colsample_bytree | N/A | 0.8 | Feature subsampling for robustness |
| min_samples_leaf | 5 | N/A | Prevent singleton leaves |
| reg_alpha, reg_lambda | N/A | 0.1, 1.0 | L1/L2 regularization for stability |

---

## Output & Recursive Usage

### Direct Output
```python
h2s_pred = model.predict(X)[0]  # ppb, clipped to [0, ∞)
```

### Recursive Forecasting (recursive.py)

For each lead hour (1 → 24):

1. **Exogenous Features** (caller provides for each lead)
   - Meteorology (temperature, wind, humidity, pressure, etc.)
   - Time (hour, month cyclicals)
   - Flow (Tijuana River discharge)
   - Tide (tidal state)

2. **Autoregressive Features** (engine updates from prior predictions)
   - h2s_lag_1h ← h2s_pred from lead t-1
   - h2s_lag_3h ← h2s_pred from lead t-3 (clamped if <3 steps)
   - h2s_lag_6h ← h2s_pred from lead t-6 (clamped if <6 steps)
   - h2s_rolling_6h ← mean(h2s_pred[t-6:t])
   - h2s_rolling_24h ← mean(h2s_pred[t-24:t])

3. **Prediction**
   ```python
   X = pd.DataFrame([row])[feature_cols]
   h2s_pred = model.predict(X)[0]
   ```

4. **Append to History**
   ```python
   h2s_history.append(h2s_pred)  # Available for next lead
   ```

---

## Performance Metrics

### Magnitude Metrics (Fit Quality)

| Metric | Formula | Interpretation |
|--------|---------|-----------------|
| MAE | Mean absolute error | Average prediction error (ppb) |
| RMSE | Root mean squared error | Penalizes large outliers |
| R² | Coefficient of determination | % variance explained |

### Threshold-Based Metrics (Operational Relevance)

At each operational threshold (5/10/30/100 ppb):

| Metric | Formula | Purpose |
|--------|---------|---------|
| Recall | TP / (TP + FN) | % of true events caught by cutting at threshold |
| Precision | TP / (TP + FP) | % of predictions above threshold that are real |
| n_positives | Count | How many observations exceed threshold |

**Rationale:** Magnitude R² can hide poor performance at extremes. Recall@30ppb directly measures "did we catch the watch-level events?" — the operational question.

---

## Example Training Report

```json
{
  "station": "NESTOR__BES",
  "variant": "Evidence",
  "task": "regression",
  "algorithm_selected": "XGBoost",
  "selection_metric": "recall_30",
  "training_date": "2026-06-24T12:34:56Z",
  "RF": {
    "MAE": 3.45,
    "RMSE": 6.78,
    "R2": 0.73,
    "recall_5": 0.82,
    "recall_10": 0.79,
    "recall_30": 0.68,
    "recall_100": 0.61,
    "n_positives_5": 1240,
    "n_positives_10": 860,
    "n_positives_30": 340,
    "n_positives_100": 45
  },
  "XGB": {
    "MAE": 3.32,
    "RMSE": 6.45,
    "R2": 0.75,
    "recall_5": 0.84,
    "recall_10": 0.81,
    "recall_30": 0.71,
    "recall_100": 0.65,
    "n_positives_5": 1240,
    "n_positives_10": 860,
    "n_positives_30": 340,
    "n_positives_100": 45
  },
  "selection_value_rf": 0.68,
  "selection_value_xgb": 0.71,
  "difference": 0.03,
  "ensemble_margin": 0.02,
  "decision": "XGBoost selected (>margin)",
  "feature_importance": {
    "h2s_lag_1h": 0.124,
    "wind_speed_10m": 0.087,
    "h2s_rolling_6h": 0.081,
    "hour_sin": 0.067,
    "temperature_2m": 0.059,
    "...": "..."
  }
}
```

---

## In Recursive Forecasting Context

When used in the recursive engine:

1. **Nowcast (Leads 1–3)**
   - h2s_lag_1h, h2s_lag_3h, h2s_lag_6h are mostly observed
   - Predictions are mostly "continuation" of recent trend
   - Skill is high (similar to observation-driven nowcasts)

2. **Nearcast (Leads 4–6)**
   - h2s_lag_3h begins to use model prediction at lead 4
   - Rolling windows start mixing predictions with actuals
   - Skill begins to decay

3. **Forecast (Leads 7–24)**
   - All h2s lags are predictions
   - Rolling windows are entirely predictions
   - Skill bounded by exogenous information (meteorology, flow, tide)
   - Use as risk-ranking tool, not ppb-truth

---

## Common Use Cases

### Use the Regression Directly
```python
from h2s.training.multi_station_trainer import train_and_select

model, choice, metrics = train_and_select(
    X_train, X_test, y_train, y_test,
    task='regression',
    selection_metric='recall_30'
)
print(f"Selected: {choice}")
predictions = model.predict(X_test)
```

### Use via H2SPredictor Class
```python
from h2s.predictor.h2s_predictor import H2SPredictor

predictor = H2SPredictor.from_s3(s3, model_path, feature_schema_path)
h2s_pred = predictor.predict(feature_row)
```

### Use in Recursive Forecasting
```python
from h2s.forecasting.recursive import run_products, VariantModels

models = VariantModels(regression=regression_model, clf_5ppb=clf5, clf_10ppb=clf10, clf_30ppb=clf30)
products_df = run_products(feature_frame, h2s_history, models, feature_cols)
```

---

## Troubleshooting

**"MAE very high, but R² looks reasonable"**
- Sign of heavy-tailed distribution. May be performing well at bulk but missing extremes.
- Check recall@30 and recall@100 to see if the issue is specifically high-end.

**"Recall at 30 ppb is low"**
- The model sees few >30 ppb events during training (class imbalance).
- Options: (1) Extend training window, (2) Use SMOTE oversampling, (3) Increase hazard class weight, (4) Fit separate high-alert model.

**"Predictions always cluster around the mean"**
- Model has learned "default to average" to minimize RMSE.
- Symptoms: Low MAE but poor coverage of extremes.
- Solution: Check feature importance — is the model using input features? Try stronger regularization (higher reg_lambda).

**"Evidence vs Lean variants score differently"**
- Normal; they use different feature sets. Track both independently.
- If Lean substantially outperforms Evidence, consider deprecating Evidence features as noise.
