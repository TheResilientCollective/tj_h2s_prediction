# Model Architecture — System Overview

## High-Level System Architecture

The H2S prediction system is built on three-layer hierarchy:

```
┌────────────────────────────────────────────────────────────┐
│                   FORECASTING LAYER                        │
│  Produces 24-hour products (nowcast/nearcast/forecast)    │
│  Input: Meteorological forecasts + last observed H2S      │
│  Output: h2s_pred, p5, p10, p30 per lead hour             │
└────────────────────────────────────────────────────────────┘
                            ▲
                            │ Uses
                            │
┌────────────────────────────────────────────────────────────┐
│                    MODEL LAYER                             │
│                                                            │
│  ┌──────────────────────────────────────────────────────┐ │
│  │ Regression Model (XGBoost/RF Ensemble)               │ │
│  │ Task: Predict H2S magnitude (ppb)                    │ │
│  │ Input: Meteorology + H2S lags/rolling features      │ │
│  │ Output: h2s_pred (0-500 ppb)                        │ │
│  └──────────────────────────────────────────────────────┘ │
│                                                            │
│  ┌─────────────────┬──────────────────┬──────────────────┐ │
│  │ Classifier >5   │ Classifier >10   │ Classifier >30   │ │
│  │ (P(H2S>5ppb))   │ (P(H2S>10ppb))   │ (P(H2S>30ppb))   │ │
│  │ Binary XGBoost/ │ Binary XGBoost/  │ Binary XGBoost/  │ │
│  │ RF Ensemble     │ RF Ensemble      │ RF Ensemble      │ │
│  │ AUC: 0.95+      │ AUC: 0.96+       │ AUC: 0.97+       │ │
│  │ (task-dependent)│ (task-dependent) │ (task-dependent) │ │
│  └─────────────────┴──────────────────┴──────────────────┘ │
│                                                            │
│  Per-Station (3) × Feature Variants (2) = 6 model sets    │
│  Each set: 1 regression + 3 classifiers = 4 models/set    │
│  Total: 6 × 4 = 24 production models deployed            │
└────────────────────────────────────────────────────────────┘
                            ▲
                            │ Trained by
                            │
┌────────────────────────────────────────────────────────────┐
│                   TRAINING LAYER                           │
│  station_model_training_job → station_deployment_job      │
│  Input: Historical observations (modeldata_h2s_nofill)    │
│  Output: 24 models = 3 stations × 2 variants × 4 tasks    │
│  Process: Auto-select RF/XGBoost/Ensemble per task        │
└────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. Regression Model (H2S Magnitude Prediction)

**Purpose:** Predict actual H2S concentration in ppb for 24-hour forecast horizon.

**Algorithm:** XGBoost or RandomForest, auto-selected per training session.

**Input Features (33 Evidence / 19 Lean):**
- Meteorology: temperature, wind speed/direction, humidity, pressure, cloud cover, precipitation
- Flow: Tijuana River flow rate (m³/s), flow lags
- H2S History: lags (1/3/6 h), rolling means (6/24 h)
- Derived: Time cyclicals (hour, month), wind cyclicals, stability regime, source regime
- Tide: tidal state

**Output:**
- Continuous prediction: h2s_pred (ppb), clipped to [0, ∞)
- Per-lead-hour in 24h forecasts

**Metric for Selection:**
- Default: recall_30 (how many actual >30 ppb events does the model catch?)
- Alternatives: recall_5, recall_10, recall_100, r2
- Rationale: Magnitude skill at the operational watch threshold (30 ppb) matters most for alert accuracy

**Performance Stats Tracked:**
- Magnitude: MAE, RMSE, R²
- Thresholds: recall, precision at 5/10/30/100 ppb

---

### 2. Binary Classifiers (Alert Threshold Probabilities)

**Purpose:** Estimate probability of exceeding each alert threshold.

**Three Parallel Tasks:**
1. **clf_5ppb** — P(H2S > 5 ppb) — Green/Yellow-Low boundary
2. **clf_10ppb** — P(H2S > 10 ppb) — Yellow-Low/Yellow-High boundary  
3. **clf_30ppb** — P(H2S > 30 ppb) — Yellow-High/Orange boundary (watch threshold)

**Algorithm:** Binary XGBoost or RandomForest, auto-selected per task/station.

**Input:** The **same** 33/19 feature columns as regression — *including* the H2S
autoregressive lags (`h2s_lag_1h/3h/6h`, `h2s_rolling_*`). The classifiers do not
read the regression's `h2s_pred` directly; in the recursive loop those lag columns
are filled with the regression's predictions, so the two heads share inputs but
predict independently.

**Output:** 
- Probability estimates: p5, p10, p30 ∈ [0, 1]
- Per-lead-hour in 24h forecasts
- **`p30` is NaN for all stations except NESTOR-BES** (`CLF_30PPB_STATIONS`
  gating — see below), and NaN for any genuinely missing classifier.

**Metric for Selection:**
- AUC (area under ROC curve) at default 0.5 threshold
- Rationale: Backward-compatible ranking metric; not tied to specific operational decision threshold

**Cascade Alert Logic** (off the primary Lean variant, NESTOR-BES):
- **Tier 1 (Nowcast):** If p5 > 0.5 at any lead 1–3, post alert
- **Tier 2 (Nearcast):** If p10 > 0.5 at any lead 4–6, post alert
- **Tier 3 (Forecast):** If p30 > 0.5 at any lead 7–24, post alert

**clf_30ppb station gating (2026-06):** `clf_30ppb` is trained + deployed for all
stations but its `p30` is emitted for **NESTOR-BES only**. IB Civic Center (~51
orange positives) and San Ysidro (~40, recall collapses to 0.16–0.43) suppress
`p30` to NaN and fall back to the ≥10 ppb yellow-high tier. See
[risk_classification.md](./risk_classification.md).

---

### 2b. Two-Head Agreement Layer (2026-06)

Between the model outputs and the operational headline tier sits
`classify_risk_agreement()`: it cross-checks the regression **magnitude** tier
against the classifier **probability** tier and reports only the level **both**
confirm as `risk`. A lone head becomes `risk_possible` (*provisional*), never
escalating the headline. Because gated stations have `p30 = NaN`, a magnitude-only
orange there is always provisional — gating and agreement interlock. Full detail in
[risk_classification.md](./risk_classification.md).

---

### 3. Recursive Forecasting Engine

**Purpose:** Use regression predictions as features for subsequent leads, creating chain-of-predictions.

**Location:** `forecasting/recursive.py`

**One Recursive Pass Produces All Three Products:**

```
t=0 (last actual):      H2S = 7.3 ppb  (observed)
       ↓
Lead 1 (t+1h):          use observed lags → predict h2s_pred_1 = 8.1 ppb → p5_1, p10_1, p30_1
       ↓
Lead 2 (t+2h):          use [7.3, 8.1, ...] → predict h2s_pred_2 = 8.5 ppb → p5_2, p10_2, p30_2
       ↓ ← lag_1h becomes prediction here
Lead 3 (t+3h):          use [7.3, 8.1, 8.5, ...] → predict h2s_pred_3 = 9.0 ppb
       ↓
Lead 4 (t+4h):          lag_3h now uses prediction; features drift into "nearcast"
       ↓
...continues to Lead 24 (forecast)
```

**Three Product Windows (Same Recursion):**
- **Nowcast:** Leads 1–3 (lags mostly actual)
- **Nearcast:** Leads 4–6 (lag_3h crosses to predictions)
- **Forecast:** Leads 7–24 (all short lags are predictions)

**Autoregressive Features Managed by Engine:**
- h2s_lag_1h, h2s_lag_3h, h2s_lag_6h
- h2s_rolling_6h, h2s_rolling_24h

**Exogenous Features (Caller-Provided):**
- All meteorological features (passed for each lead hour)
- Flow, tide, time, wind, interactions (same across all leads or weather-forecast-dependent)

**Honest Scope:** Skill decays with lead time. By lead 7–24, the model is using almost entirely its own predictions as features, capped by the exogenous information ceiling. Use forecast product as a risk-ranker, not ppb-truth.

---

### 4. Dispersion Model (Source Attribution)

**Purpose:** Determine where H2S is coming from and quantify emission rates.

**Two Complementary Approaches:**

#### A. Lagrangian Backward Inversion (Source Attribution)
- **Run:** `dispersion_inversion_job` (weekly Monday 02:30 UTC, STOPPED by default)
- **Window:** 2-hour backward integration (tuned for Tijuana Valley scale: 1–7 km reach)
- **Output:** Source fractions (ensemble of 16 candidate sources) → grouped into 3 zones (East/West/South)
- **Result:** Emission rates (g/s) per zone calibrated from 2-hour particle tracking
- **Wind-Dependent Diffusion:** σ ~ U^0.5 (calm winds sharpen attribution; strong winds diffuse it)

#### B. Gaussian Forward Forecast
- **Run:** `dispersion_forecast_job` (every 6h)
- **Input:** Latest emission rates + forecast meteorology (U, V wind)
- **Output:** 72-hour plume forecast; check next 6h for threshold crossings (30 ppb watch, 100 ppb critical)
- **Alert:** Posts to Slack if thresholds exceeded
- **Bundle:** Uploads HYSPLIT forward CONTROL files to S3 (not executed; user downloads and runs locally or via NOAA READY)

**Current Calibration (Mar 13 2026):**
- East: 87.3 g/s (52% of total; Dairy Mart Bridge dominant)
- West: 29.9 g/s (18% of total; Tijuana Beach Outlet, Oneonta Slough)
- South: 49.8 g/s (30% of total; Goat Canyon, Smugglers Gulch)
- **Total: 167 g/s**

---

## Data Flow: Training → Deployment → Forecasting

### Training Phase (station_model_training_job)

```
Raw Observations
(modeldata_h2s_nofill.parquet)
    ↓
Feature Engineering
(ensure_base_features + per-site rolling windows)
    ↓
80/20 Train/Test Split
(time-series aware)
    ↓
Train RF + XGBoost for Each Task
(regression, clf_5ppb, clf_10ppb, clf_30ppb)
    ↓
Auto-Select or Ensemble
(per task, station, variant)
    ↓
Archive Models + Training Report
(immutable S3 archive with version tag)
```

### Deployment Phase (station_model_deployment_job)

```
Archived Models
    ↓
Approval Gate (operator runs deployment job)
    ↓
Upload to Production S3
(tijuana/forecast/models/stations/{STATION}/{variant}/{task}.pkl)
    ↓
Stamp model_version in deployment_metadata.json
    ↓
Ready for Forecasting
```

### Forecasting Phase (station_forecast_job)

```
Latest Meteorological Forecast
    ↓
Load Production Models from S3
    ↓
Per-Station × Variant:
  - Recursive engine for leads 1..24
  - Regression: h2s_pred
  - Classifiers: p5, p10, p30
    ↓
Emit Products
(nowcast/nearcast/forecast rows)
    ↓
Store to S3
(tijuana/forecast/products/run_ts={timestamp}/products.parquet)
    ↓
Optional: Cascade Alert Check
(Tier 1–3 against primary Lean variant, NESTOR-BES)
```

---

## Model Deployment Topology

**Per-Station Model Set:**
```
Station: NESTOR__BES
  ├── Evidence Variant (33 features)
  │   ├── regression.pkl
  │   ├── clf_5ppb.pkl
  │   ├── clf_10ppb.pkl
  │   ├── clf_30ppb.pkl
  │   └── features_evidence.json (schema)
  ├── Lean Variant (19 features)
  │   ├── regression.pkl
  │   ├── clf_5ppb.pkl
  │   ├── clf_10ppb.pkl
  │   ├── clf_30ppb.pkl
  │   └── features_lean.json (schema)
  ├── deployment_metadata.json
  └── training_report.json
```

**S3 Storage:**
- **Production:** `s3://resilientpublic/tijuana/forecast/models/stations/{STATION}/{variant}/{task}.pkl`
- **Archive:** `s3://resilientpublic/tijuana/forecast/models/archive/stations/{STATION}/{version_tag}/`

**Version Control:**
- Every training run is archived with immutable version tag (YYYYMMDDTHHMMSSZ-{gitsha})
- `deployment_metadata.json` stamps the active model_version
- Every forecast row carries model_version; analysis can be replayed against archived models

---

## Key Design Decisions

1. **One Regression Model (magnitude head):** Produces h2s_pred; feeds the recursive lags and the magnitude side of two-head agreement
2. **Three Parallel Classifiers (probability head):** Each trained independently; allows flexibility in thresholds and algorithm selection
3. **Recursive Autoregression:** Simulates "all forecasted H2S as features" for honest forecast-tier skill assessment
4. **Two Feature Variants — Lean is primary (2026-06):** Lean (19, operational) and Evidence (33, cross-check) deployed in parallel; one constant (`PRIMARY_VARIANT`) re-points every surface
5. **Two-Head Agreement:** Headline tier requires both magnitude and probability heads to confirm; lone signals stay provisional (cuts false orange, no silent miss)
6. **Per-Station clf_30ppb Gating:** Orange probability emitted only where there are enough positives to trust it (NESTOR-BES); sparse stations fall back to ≥10 ppb
7. **Auto-Selection Strategy:** Per-task RF/XGBoost choice; ensemble if within margin (prevents overfitting to a single algorithm)
8. **Immutable Archive:** Enables audit trail and replay-from-archive for any past forecast
9. **Dispersion as Complement:** Lagrangian inversion + Gaussian forward provide source-to-sensor physics; H2S regression/classifiers are empirical alert engines

---

## Next Steps

- **[regression_model.md](./regression_model.md)** — Regression model details and performance
- **[classifier_models.md](./classifier_models.md)** — Classifier architecture and per-station performance
- **[risk_classification.md](./risk_classification.md)** — Two-head agreement, clf_30ppb gating, underfitting findings
- **[dispersion_model.md](./dispersion_model.md)** — Dispersion modeling and source attribution
- **[calibration.md](./calibration.md)** — Calibration loop and model selection strategy
