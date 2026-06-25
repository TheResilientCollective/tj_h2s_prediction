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

## Document Structure

- **[model_architecture.md](./model_architecture.md)** — System-level architecture and how models connect
- **[regression_model.md](./regression_model.md)** — Regression model (H2S magnitude prediction)
- **[classifier_models.md](./classifier_models.md)** — Binary classifiers for alert thresholds
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

- **Evidence:** 33 features (full meteorology, flow, interactions, time/wind cyclicals)
- **Lean:** 19 core features (wind, temperature, humidity, time only)

Both variants are deployed and scored independently.

## Current Model Performance

**Hourly Pipeline (Single-station NESTOR legacy):**
- Orange (≥30 ppb) detection: 61.3%
- False alarm rate: 5.4%

**Per-Station Performance (Walk-forward backtest):**
See [classifier_models.md](./classifier_models.md) for detailed per-station P(>30ppb) recall.

## Training Data & Features

### Feature Engineering Pipeline

Features are built from:
- **Raw observations:** Temperature, wind, humidity, pressure, cloud cover, precipitation
- **Engineered features:**
  - Time cyclicals (hour_sin/cos, month_sin/cos)
  - Wind cyclicals (direction_sin/cos)
  - Lags: H2S (1/3/6 h), flow (6 h)
  - Rolling windows: H2S (6/24 h), wind speed (2/3/4 h), wind gusts (2/3/4 h max)
  - Derived: Flow log, stability regime, tidal state, source regime

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
