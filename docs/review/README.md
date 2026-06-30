# H2S Prediction Models — Complete Review

This directory contains comprehensive documentation of all models in the Tijuana H2S prediction system. Each model serves a distinct purpose within the operational forecasting pipeline.

## Quick Overview

The H2S prediction system uses three complementary model types working together:

1. **Regression Model** — Predicts actual H2S concentration (ppb) from meteorological and flow data
2. **Binary Classifiers (P>5, P>10, P>30)** — Estimate probability of exceeding alert thresholds
3. **Dispersion Model** — Determines source attribution and emission rates for Lagrangian/Gaussian forecasting

### Model Input/Output Flow

```
┌─────────────────────────────────────┐
│   Meteorological + Flow Data        │
│   (FORECAST for 24h leads)          │
└──────────────┬──────────────────────┘
               │
       ┌───────┴──────────┬────────────────┐
       │                  │                │
       ▼                  ▼                ▼
  ┌─────────┐      ┌──────────────┐  ┌──────────────┐
  │Regression│ ──→ │ Recursive    │  │ Classifiers  │
  │Model     │     │ H2S Features │  │ (P>5/10/30)  │
  │XGBoost   │     │ (lags + roll)│  │ XGBoost/RF   │
  └─────────┘     └──────────────┘  └──────────────┘
       │                  │                │
       └──────────────────┼────────────────┘
                          ▼
            ┌──────────────────────────┐
            │  Forecast Products       │
            │  (p5, p10, p30 probs)    │
            │  + h2s_pred (ppb)        │
            └──────────────────────────┘
                          │
            ┌─────────────┴──────────────┐
            │                            │
            ▼                            ▼
        ┌─────────┐           ┌──────────────────┐
        │  Alert  │           │  Validation/     │
        │ Cascade │           │  Performance     │
        │ Tier1-3 │           │  Reporting       │
        └─────────┘           └──────────────────┘
```

> **Latest revision (2026-06):** the operational primary variant is now **Lean**
> (not Evidence), `clf_30ppb` is **gated to NESTOR-BES only**, and a **two-head
> agreement** rubric (`classify_risk_agreement`) now sets the headline risk tier.
> See [risk_classification.md](./risk_classification.md).

## Document Structure

- **[model_architecture.md](./model_architecture.md)** — System-level architecture and how models connect
- **[regression_model.md](./regression_model.md)** — Regression model (H2S magnitude prediction)
- **[classifier_models.md](./classifier_models.md)** — Binary classifiers for alert thresholds
- **[risk_classification.md](./risk_classification.md)** — Two-head agreement, clf_30ppb station gating, underfitting findings
- **[dispersion_model.md](./dispersion_model.md)** — Source attribution and Lagrangian/Gaussian forecasting
- **[calibration.md](./calibration.md)** — Calibration loop and model selection strategy

## Key Concepts

### Operational Alert Thresholds

The system uses four hazard levels based on H2S concentration:

| Level | Range | Meaning | Status |
|-------|-------|---------|--------|
| **Green** | < 5 ppb | Safe, no action | Baseline |
| **Yellow-Low** | 5–10 ppb | Caution, elevated | Tier 1 Alert |
| **Yellow-High** | 10–30 ppb | Resident smell level | Tier 2 Alert |
| **Orange** | ≥ 30 ppb | Hazardous watch | Tier 3 Alert |

### Forecast Products

One recursive model pass produces three products, differing by how much predicted H2S is used as input:

- **Nowcast (Leads 1–3):** Mostly observed data; regression only recently uses its own predictions
- **Nearcast (Leads 4–6):** Mixed; 3h+ lags shift to predictions at lead 4
- **Forecast (Leads 7–24):** Fully recursive; all lags ≤6h are predictions

### Model Variants

The system trains two feature variants per station:

- **Lean (PRIMARY):** 19 features — the operational variant. As of 2026-06 every
  operational surface (daily forecast/dashboard, Tier 1–3 cascade trigger,
  heatmap board, digest, performance report, accuracy rollups) routes through
  Lean (`PRIMARY_VARIANT = "lean"` in `constants.py`).
- **Evidence:** 33 features — trained and deployed in parallel, shown alongside
  Lean (e.g. the cascade Slack report renders `E … / L …`) as the published
  "not-overdetermined" cross-check. Was the production default until 2026-06.

Both variants are deployed and scored independently; flipping `PRIMARY_VARIANT`
re-points every report and the alert trigger in one edit.

### Two-Head Agreement Classification (2026-06)

Every forecast row carries two independent hazard signals — the regression
**magnitude** (`h2s_pred` ppb) and the classifier **probability**
(`p5`/`p10`/`p30`). `classify_risk_agreement()` reports the tier **both** heads
confirm as the headline `risk`; a lone head surfaces as `risk_possible`
(*provisional*) and never escalates the headline. This cuts false ORANGE alarms
(a hard alert needs corroboration) while keeping a lone strong signal visible
(never a silent miss). Full detail in [risk_classification.md](./risk_classification.md).

### Per-Station ORANGE (≥30 ppb) Gating (2026-06)

`clf_30ppb` is trained and deployed for all three stations, but its P(>30)
probability is **emitted for NESTOR-BES only** (`CLF_30PPB_STATIONS`). NESTOR-BES
has ~344 training orange positives (dependable recall); IB Civic Center (~51) and
San Ysidro (~40) are too positive-starved for a trustworthy fixed-threshold
orange call, so their `p30` is suppressed (NaN) and they fall back to the ≥10 ppb
(yellow-high) tier as their top operational alert.

## Current Model Performance

**Hourly Pipeline (Single-station NESTOR legacy, retired):**
- Orange (≥30 ppb) detection: 61.3%
- False alarm rate: 5.4%

**Per-Station ORANGE (≥30 ppb) recall, Lean variant (in-sample 80/20 split):**

| Station | Baseline recall | Training orange positives |
|---------|-----------------|---------------------------|
| NESTOR-BES | 0.77 (up to ~0.86–0.95 at lower cutoff) | ~344 |
| IB Civic Ctr | 0.70 | ~51 |
| San Ysidro | 0.16–0.43 (gated off) | ~40 |

The 6/27/2026 Berry Elementary miss (103–219 ppb overnight, model predicted
green) drove an underfitting investigation — see
[risk_classification.md](./risk_classification.md) and
[classifier_models.md](./classifier_models.md) for per-station P(>30ppb) recall
and the pooled-model path to re-enabling the gated stations.

## Training Data & Features

### Feature Engineering Pipeline

The production 33-feature **Evidence** set (`CORE_FEATURES` in `constants.py`,
promoted from the 2026-06-10 "Berry" ablation, PR #27/#28) is built from:
- **Raw observations:** temperature, wind speed, wind gusts, humidity, pressure,
  cloud cover, precipitation, dewpoint
- **Engineered features:**
  - Time cyclicals (hour_sin/cos, month_sin/cos), `is_night`
  - Wind cyclicals (wind_direction_sin/cos)
  - Tide: tide_height, tidal_state_encoded
  - Lags: H2S (1/3/6 h), flow (flow_lag_6h)
  - Rolling windows: H2S (6/24 h), wind speed (2/3 h), wind gusts (2/3 h max),
    flow (flow_rolling_24h)
  - Derived: source_regime, stable_atm, wind_x_stable_atm,
    wind_temp_interaction, humidity_temp_interaction

The **Lean** 19-feature set drops the interactions, the 2/3 h wind rolling
aggregates, the lower-importance weather channels, and hour_sin/cos. The legacy
44-feature set (`MODEL_FEATURES_LEGACY`, including `flow_log`/`flow_low`/
`flow_high` and the SBIWTP effluent channels) is retained only for legacy-model
preprocessing — **not used for new training**. See
[regression_model.md](./regression_model.md) for the exact column lists.

### Training Strategy

**Multi-station approach (per-station partitioning):**
1. Feature extraction shared across all three stations (same training data snapshot)
2. Per-station models trained independently (one model per task × variant × station)
3. Immutable archive stores all models with training metadata
4. Production models deployed via explicit approval (no automatic promotion)

**SMOTE & Class Balancing:**
- Standard: Hazard multiplier (3× weight on yellow/orange classes)
- Optional: BorderlineSMOTE oversampling for sparse-positive tasks (e.g., clf_30ppb at low-base-rate stations)
- SMOTE evaluated for clf_30ppb — degraded recall, so kept OFF by default

## Next Steps

Read [model_architecture.md](./model_architecture.md) for the system-level overview, then dive into task-specific documentation.
