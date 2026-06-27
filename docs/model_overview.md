# H2S Prediction System — Model Overview

The Tijuana River H2S prediction system combines machine learning forecasts
with atmospheric dispersion modeling to predict
hydrogen sulfide levels at three monitoring stations
(IB Civic Center, Nestor BES, San Ysidro).
The system operates both reactive forecasting pipelines and physics-based source attribution
to support air quality management in the Tijuana River Valley.

## Modeling Strategies

| Goal | Strategy | Description |
|------|----------|-------------|
| Forecast | Ensemble Multi-Site XGBoost | Predict H2S categories (green/yellow/orange) at three stations using averaged probabilities from XGBoost and Random Forest variants |
| Forecast | Multi-Horizon Forecast | 72-hour predictions at 0–6h, 6–24h, 24–48h, and 48–72h horizons using 36 station × task × horizon models |
| Source Determination | Backward Trajectory | Lagrangian particle model tracking 2000 particles backward from sensors to identify source locations, weighted by observed H2S |
| Source Emission | Backward Dispersion | Bayesian inversion of Lagrangian footprints to estimate per-zone emission rates (g/s) for east, west, and south source regions |
| Forecast | Forward Dispersion | Gaussian plume model (Pasquill-Gifford) producing 72-hour concentration forecasts using emission rates from backward dispersion |
| Calibration | Calibration Loop | Iterative refinement between backward and forward dispersion to converge on emission estimates *(planned — currently semi-static)* |
| Forecast | River Channel Dispersion | H2S transport model driven by effluent flow in the Tijuana River channel *(planned — not yet implemented)* |

## Pipeline Status

- **Operational:** Ensemble XGBoost (every 6h), Forward Dispersion (every 6h), Daily Multi-Station Forecasts (daily)
- **Operational (stopped):** Multi-Horizon Forecast, Backward Trajectory (weekly)
- **Planned:** Calibration Loop, River Channel Dispersion

## Per-Station Forecast Models (the models being run)

Each station trains a small model set — a regression (ppb magnitude) plus
exceedance classifiers `clf_5ppb` / `clf_10ppb` / `clf_30ppb` — in **two feature
variants**: **Evidence** (33 features) and **Lean** (19 features). Both variants
are trained and deployed in parallel for every station.

**Primary variant: Lean.** As of 2026-06 the operational surfaces — the daily
forecast/dashboard, the Tier 1–3 cascade alert trigger, the heatmap board, the
digest, the performance report, and the accuracy rollups — all route through the
**Lean** variant (`PRIMARY_VARIANT` in `constants.py`). Evidence is still
produced and shown alongside (e.g. the cascade Slack report shows `E … / L …`)
as the published "not-overdetermined" cross-check. Changing `PRIMARY_VARIANT`
re-points every report and the alert trigger in one edit.

**Per-station ≥30 ppb (ORANGE) coverage.** `clf_30ppb` is trained and deployed
for all three stations, but its **probability (P>30) is emitted for NESTOR-BES
only** (`CLF_30PPB_STATIONS` in `constants.py`). NESTOR-BES has ~344 orange
training positives (dependable recall); IB Civic Center (~51) and San Ysidro
(~40) are too positive-starved for a trustworthy fixed-threshold orange call, so
their `p30` is suppressed (NaN) and they fall back to the ≥10 ppb (yellow-high)
tier as their top operational alert. A pooled cross-station ≥30 model is the
path to re-enabling them. See `projects/h2s/experiments/underfitting_results/`.

**Two-head agreement classification.** Each forecast row carries two independent
hazard signals — the regression **magnitude** (`h2s_pred`) and the classifier
**probability** (`p5/p10/p30`). `classify_risk_agreement()` reports the level
**both** heads confirm as the headline tier; a lone head surfaces as a
`risk_possible` / *provisional* flag and never escalates the headline. This cuts
false orange alarms (a hard alert needs corroboration) while keeping a lone
strong signal visible (never a silent miss). It composes with the ≥30 gate:
where `p30` is NaN, magnitude-only orange is always *provisional*.
