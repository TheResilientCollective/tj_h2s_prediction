# Binary Classifiers — Alert Threshold Probabilities

## Overview

Three parallel binary classifiers estimate the probability of exceeding each operational alert threshold:

- **clf_5ppb** — P(H2S > 5 ppb) — Green ↔ Yellow-Low boundary
- **clf_10ppb** — P(H2S > 10 ppb) — Yellow-Low ↔ Yellow-High boundary
- **clf_30ppb** — P(H2S > 30 ppb) — Yellow-High ↔ Orange boundary (watch threshold)

**Purpose:** Provide calibrated probability estimates for alert cascade logic and early-warning capabilities.

**Use Case:** When regression alone gives point estimate h2s_pred = 12 ppb, classifiers answer: "How confident are we that H2S will exceed 10 ppb? 0.85 probability — actionable Tier 2 alert."

**Output:** Probabilities p5, p10, p30 ∈ [0, 1], one per lead hour in 24h forecast.

---

## Why Separate Classifiers? (Not Just Regression Cut-Points)

**Key Insight:** A single regression model optimizes for magnitude fit (RMSE), not threshold discrimination.

**Example:** Two regressions with same MAE can have different threshold detection rates:
- Model A: Predicts 8.5 ppb when true is 10 ppb → misses the 10 ppb alert
- Model B: Predicts 10.2 ppb when true is 10 ppb → catches it

Dedicated classifiers train specifically to maximize P(predict>threshold | true>threshold) — the probability that matters operationally.

---

## Input Features

**Identical to the regression model** — the classifiers consume the *same* 33
(Evidence) / 19 (Lean) feature columns, **including the H2S autoregressive lags**
(`h2s_lag_1h/3h/6h`, `h2s_rolling_6h/24h`). See
[regression_model.md](./regression_model.md) for the exact column lists. The
operational primary is **Lean** (`PRIMARY_VARIANT = "lean"`).

**Note:** Classifiers do NOT read the regression's `h2s_pred` as a feature.
They use the same raw `h2s_lag_*` features which, in recursive forecasting, are
updated with the regression's predictions (the autoregressive chain). So they
"see" the regression's trajectory indirectly through the lags, but train and
predict independently.

---

## Algorithm & Training

### Model Factory (multi_station_trainer.py)

#### RandomForest Binary Classifier
```python
RandomForestClassifier(
    n_estimators=500,
    max_depth=20,
    min_samples_leaf=5,
    max_features='sqrt',
    class_weight='balanced',  # Auto-weight minority class
    n_jobs=-1,
    random_state=42
)
```

#### XGBoost Binary Classifier
```python
XGBClassifier(
    n_estimators=500,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    scale_pos_weight=(1 - pos_rate) / max(pos_rate, 0.01),  # Dynamic class weight
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
    eval_metric='logloss'
)
```

### Target Creation

For each task, create binary target:

```python
df['exceed_5'] = (df['H2S'] > 5).astype(int)   # Binary: 0 or 1
df['exceed_10'] = (df['H2S'] > 10).astype(int)
df['exceed_30'] = (df['H2S'] > 30).astype(int)
```

**Note:** Using strict inequality `>` (not `≥`) so 10.0 ppb exactly is NOT a positive.

### Training Procedure

1. **Per-Task Training** (independent, not stacked)
   - Load same feature-engineered DataFrame
   - Create task-specific target (exceed_5, exceed_10, or exceed_30)
   - Filter out NaN rows

2. **Class Balancing**
   - RF: `class_weight='balanced'` — auto-adjusts per class
   - XGB: `scale_pos_weight` tuned to (n_neg / n_pos) to penalize false negatives

3. **Optional SMOTE** (for sparse positive classes)
   - Only applied to clf_30ppb, only on TRAIN split (test untouched)
   - Flag: `use_smote_on_minority=True` in train_and_select
   - Currently OFF by default (SMOTE degraded recall in backtest; kept for opt-in)

4. **Model Selection** (train_and_select)
   - Train RF + XGBoost on same 80/20 split
   - **Selection Metric:** AUC (area under ROC curve)
   - Backward-compatible ranking metric; not tied to a specific decision threshold
   - **Ensemble Decision:**
     - If |AUC_rf - AUC_xgb| < 0.01 (default margin), weighted ensemble
     - Else pick winner

5. **Final Model**
   - Train selected model on full training set
   - Return model + algorithm_choices

---

## Probability Calibration

### What predict_proba Returns

```python
proba = model.predict_proba(X)[:, 1]  # P(H2S > threshold) ∈ [0, 1]
```

- **Column 0:** P(H2S ≤ threshold)
- **Column 1:** P(H2S > threshold) — the probability we use

### Calibration Quality

Good calibration means:
- When model predicts p=0.9, ~90% of those cases actually exceed the threshold
- When model predicts p=0.1, ~10% exceed the threshold

XGBoost with `objective='binary:logistic'` naturally produces calibrated probabilities. RandomForest also tends to be well-calibrated.

---

## Performance Metrics

### AUC (Area Under ROC Curve)

- **Range:** [0, 1]; 0.5 = random, 1.0 = perfect
- **Interpretation:** Probability that classifier ranks a random positive sample higher than a random negative
- **Operational Use:** Backward-compatible ranking metric; does NOT depend on threshold choice

### Binary Metrics at Fixed Threshold (0.5)

| Metric | Formula | Purpose |
|--------|---------|---------|
| Precision | TP / (TP + FP) | When model says ">threshold", how often is it right? |
| Recall | TP / (TP + FN) | How many true events does model catch? |
| F1 | 2 × (Precision × Recall) / (P + R) | Harmonic mean (balanced) |
| Brier | Mean((y_true - y_pred_proba)²) | Probability calibration loss |

---

## Per-Station Performance (in-sample 80/20 split, Lean variant)

The 5 ppb and 10 ppb classifiers are strong everywhere — positives are common
enough (9–22% base rate) that the baseline already works. clf_30ppb is the hard
one (0.6–4.6% base rate). Figures below are from the 2026-06-27 underfitting study
(`experiments/underfitting_results/`).

### clf_5ppb (P(H2S > 5 ppb)) — Tier 1

| Station | Base rate ≥5 | Baseline recall |
|---------|--------------|-----------------|
| NESTOR-BES | 21.6% | ~0.83 |
| IB Civic Ctr | 9.3% | ~0.68–0.75 |
| San Ysidro | 12.1% | ~0.70–0.78 |

### clf_10ppb (P(H2S > 10 ppb)) — Tier 2

| Station | Base rate ≥10 | Baseline recall |
|---------|---------------|-----------------|
| NESTOR-BES | 12.0% | strong |
| IB Civic Ctr | 4.0% | strong |
| San Ysidro | 4.5% | strong |

### clf_30ppb (P(H2S > 30 ppb)) — ⚠️ Gated to NESTOR-BES only

This is where the underfitting lives. **As of 2026-06 `p30` is emitted for
NESTOR-BES only** (`CLF_30PPB_STATIONS = {"NESTOR - BES"}`); the other stations'
`p30` is suppressed to NaN and they fall back to ≥10 ppb (yellow-high) as their
top alert.

| Station | AUC | Orange positives | Baseline recall | p30 emitted? |
|---------|-----|------------------|-----------------|--------------|
| NESTOR-BES | ~0.97 | ~344–550 (4.6%) | 0.77 (→0.86 night-trained) | ✅ Yes |
| IB Civic Ctr | ~0.96 | ~51–84 (0.9%) | 0.70 | ❌ Gated off (NaN) |
| San Ysidro | ~0.97–0.98 | ~40–85 (0.6%) | **0.16–0.43** (collapses) | ❌ Gated off (NaN) |

**Why San Ysidro collapses:** ~40 positives in a sea of negatives (0.6% base
rate). AUC stays high (ranking is fine) but at any fixed operating point recall is
unstable — more noise than signal. The 6/27/2026 Berry Elementary miss (103–219
ppb overnight, predicted green) is the canonical failure.

**The fix that shipped:** rather than emit an untrustworthy orange call, the
revision (a) gates `p30` off for the sparse stations, and (b) requires **two-head
agreement** so a magnitude-only orange surfaces as *provisional*, not a hard
alert. See [risk_classification.md](./risk_classification.md).

**The fix in progress:** a **pooled cross-station ≥30 model** lifts San Ysidro
recall from 0.16 → 0.64–0.73 (default) / 0.84 (@0.25 cutoff) in experiments. When
it lands, re-enabling a station = adding its name to `CLF_30PPB_STATIONS`.

**SMOTE** was evaluated for clf_30ppb and *degraded* OOS recall (AUC was already
0.96–0.98), so it is OFF by default (opt-in via `enable_smote_clf_30ppb`).

---

## Usage in Cascade Alert System

### Cascade Alert Logic (cascade_alerts_job)

The cascade evaluates three tiers off the **primary variant** (Lean —
`TRIGGER_VARIANT = PRIMARY_VARIANT`), NESTOR-BES station. Evidence is reported
alongside but never gates.

```python
# Load latest forecast products, filter to the primary (Lean) variant, NESTOR-BES
nestor = products[(products['variant'] == 'lean') & (products['station'] == 'NESTOR__BES')]

# Tier 1: Nowcast p5 ; Tier 2: Nearcast p10 ; Tier 3: Forecast p30
tier1 = (nestor.query('1 <= lead_hour <= 3')['p5']  > 0.5).any()
tier2 = (nestor.query('4 <= lead_hour <= 6')['p10'] > 0.5).any()
tier3 = (nestor.query('7 <= lead_hour <= 24')['p30'] > 0.5).any()
```

### Cascade Thresholds (CASCADE_TRIGGERS in constants.py)

| Tier | Product | Metric | Cutoff | Window (lead h) |
|------|---------|--------|--------|-----------------|
| 1 | Nowcast | p5 | 0.5 | 1–3 |
| 2 | Nearcast | p10 | 0.5 | 4–6 |
| 3 | Forecast | p30 | 0.5 | 7–24 |

The cascade cutoffs are all 0.5. **Note this differs from the two-head agreement
cutoffs** used by the daily-pipeline risk tier (`probability_risk_tier`):
`PROB_5_ALERT = 0.5`, `PROB_10_ALERT = 0.5`, `PROB_30_ALERT = 0.25`. The cascade
is the bellwether trigger; the agreement rubric is the per-row headline classifier.

---

## Probability vs Magnitude Prediction

### Key Relationship in Recursive Forecasting

```
┌─────────────────────────────────────────────────┐
│ Regression predicts h2s_pred (ppb)              │
│ This becomes h2s_lag_1h for the next hour       │
└─────────────────────────────────────────────────┘
                    ▲
                    │ Uses
                    │
┌─────────────────────────────────────────────────┐
│ Classifiers receive same h2s_lag features       │
│ But independently predict P(>5), P(>10), P(>30) │
│                                                 │
│ Example:                                        │
│   h2s_pred = 8.5 ppb  ← regression output      │
│   p5 = 0.92           ← clf_5ppb output        │
│   p10 = 0.35          ← clf_10ppb output       │
│   p30 = 0.02          ← clf_30ppb output       │
│                                                 │
│ Interpretation: Confident H2S > 5 (alert),     │
│ less sure about 10 ppb, very unlikely 30 ppb   │
└─────────────────────────────────────────────────┘
```

### Do Classifiers Use h2s_pred as Input?

**Short Answer:** No.

**Detailed Answer:** Classifiers use the same raw features as regression (meteorology, lags, etc.), including h2s_lag_1h/3h/6h. During recursive forecasting, those lags are populated with the regression's h2s_pred. So classifiers indirectly "see" the regression's predictions, but they do not explicitly read h2s_pred as a feature.

**Why not pass h2s_pred directly?** 
- Would create explicit dependency on regression quality
- Allows classifiers to learn independent threshold discrimination
- Simplifies model management (3 independent classifiers, not 3 dependent chains)

---

## Training Report Example

```json
{
  "station": "NESTOR__BES",
  "variant": "Evidence",
  "task": "clf_10ppb",
  "algorithm_selected": "XGBoost",
  "selection_metric": "auc",
  "training_date": "2026-06-24T12:34:56Z",
  "class_distribution_train": {
    "negative (≤10 ppb)": 4230,
    "positive (>10 ppb)": 860
  },
  "RF": {
    "AUC": 0.957,
    "Brier": 0.042,
    "F1": 0.643,
    "Precision": 0.687,
    "Recall": 0.608
  },
  "XGB": {
    "AUC": 0.962,
    "Brier": 0.038,
    "F1": 0.668,
    "Precision": 0.721,
    "Recall": 0.624
  },
  "selection_value_rf": 0.957,
  "selection_value_xgb": 0.962,
  "difference": 0.005,
  "ensemble_margin": 0.01,
  "decision": "Ensemble (within margin)",
  "ensemble_weight_rf": 0.497,
  "ensemble_weight_xgb": 0.503,
  "smote_applied": false,
  "feature_importance": {
    "h2s_lag_3h": 0.158,
    "h2s_lag_1h": 0.121,
    "wind_speed_10m": 0.087,
    "hour_sin": 0.062,
    "h2s_rolling_6h": 0.058,
    "...": "..."
  }
}
```

---

## Troubleshooting

**"AUC is high (0.95) but recall is low at 0.5 threshold"**
- Model is ranking well but probability calibration is off.
- Try: (1) Check Brier score (good calibration should have low Brier), (2) Adjust threshold (lower it to 0.3 for higher recall), (3) Refit on more balanced data with SMOTE.

**"clf_30ppb always predicts <0.1 for SAN_YSIDRO"**
- Base rate extremely low (~0.3%); model learned "default to no".
- Expected behavior given training data. Monitor actual >30 ppb events separately.
- Solution: Trigger alerts at lower p30 threshold (e.g., 0.15 instead of 0.5) for that station.

**"Classifier and regression disagree: h2s_pred=25 but p30=0.1"**
- Not contradiction; they're independent models with different loss functions.
- Regression optimizes for magnitude fit (MAE/RMSE); classifier optimizes for threshold discrimination.
- Trust both: regression says "expect ~25 ppb" and classifier says "but we're not confident it will exceed 30 ppb".

**"SMOTE enabled but recall still low"**
- SMOTE was evaluated and degraded recall in backtest (unexpected).
- Possible reason: SMOTE introduces synthetic duplicates that aren't representative of true >30 ppb extremes.
- Current status: OFF by default; if needed, profile carefully with separate train fold.

---

## Future Enhancements

1. **Per-Station Thresholds:** Tune p5/p10/p30 thresholds independently per station (not global 0.5)
2. **Temporal Thresholds:** Different thresholds for day vs night (H2S behavior differs)
3. **High-Alert Specialist Model:** Separate model for >50 ppb events (extreme regime)
4. **Ensemble Voting:** Train 3 independent classifiers per task, vote on final alert
5. **Threshold Tuning:** Minimize false alarms while keeping recall >0.90 at operational level
