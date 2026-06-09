# H₂S Prediction Model Status — Tijuana River Valley

*Audience: researchers familiar with probabilistic forecasting and ML.*

---

## Overview

Three distinct prediction approaches are deployed or under active development for the three monitoring stations (NESTOR-BES, IB CIVIC CTR, SAN YSIDRO): a threshold-based tiered alert system, a multi-horizon ensemble pipeline, and a per-station daily forecast pipeline. Each targets a different operational question. This document covers their current status, feature basis, and known failure modes.

---

## 1. Tiered Alert System (Tiers 1–3)

### What it does

A five-tier hierarchy partitions alerts by source (forecast vs. observation) and severity:

- **Tiers 1–3** (forecast-based, addressed here): issue pre-alerts hours in advance at 5, 10, and 30 ppb exceedance risk
- **Tiers 4–5** (observation-based, not discussed here): fire on live sensor reads at 30 and 100 ppb

Tiers 1–3 evaluate at four horizon windows anchored to the current time: nowcast (0–3h), near (3–6h), mid (6–12h), day-ahead (12–24h). Each window is scored independently.

### Architecture

The model is not a learned classifier in the conventional sense. It is a two-stage rule-based scorer:

1. **Hard gate** — a set of threshold conditions that must all pass before scoring begins. These are operationally interpretable (e.g., SBIWTP flow must be below baseline, atmospheric stability must exceed a fraction).
2. **Continuous score** — a weighted z-score sigmoid, where weights are derived from Cohen's *d* effect sizes computed on 273 fully-reported nights (Nov 2023 – Apr 2026) comparing nights where ≥2 stations exceeded threshold against quiet (0-active) nights.

A tier fires when `gate_passed AND score ≥ 0.5`. The score is clipped at 0.95 to prevent overconfidence during deep SBIWTP anomalies.

**Hard gate thresholds (Tier 3 example):**

| Condition | Value |
|---|---|
| SBIWTP flow | < 23.0 MGD (baseline − 0.5) |
| SBIWTP anomaly | < 0 |
| Wind speed (mean) | < 4.0 m/s |
| Min temperature | > 13.0°C |
| Dewpoint | > 11.0°C |
| Stable atm fraction | > 0.6 |

**Top scoring weights (Cohen's *d*, Tier 3):**

| Feature | *d* |
|---|---|
| sbiwtp_flow_mgd | −1.44 |
| temperature_2m | +0.70 |
| sbiwtp_anomaly | −0.93 |
| dewpoint_2m | +0.50 |
| sbiwtp_deficit | +0.64 |
| surface_pressure | −0.42 |

The sign convention is risk-directional: negative *d* means lower values are riskier.

Z-scores are computed against **frozen quiet-night statistics** (mean and standard deviation per feature) derived from the same 273-night calibration window.

### Calibration and post-April failure

The model was calibrated against data spanning Nov 2023 – Apr 2026, with the active analysis window concentrated in the anomalously active March–May 2026 period. This creates at least three compounding failure modes after April:

1. **Frozen z-score baseline.** Feature normalization uses statistics from the calibration window. As seasonal conditions shift into summer (higher baseline temperatures, altered SBIWTP patterns, drier stable-atm fraction), the z-scores drift away from the reference distribution. A feature that scored at the 80th percentile of the calibration distribution may be unremarkable in summer — or vice versa.

2. **Gate thresholds calibrated to an active season.** The hard-gate temperature floor (> 13°C) and dewpoint floor (> 11°C) were set to pass during the active Mar–May window. Summer conditions along the Tijuana valley corridor often exceed these floors continuously, meaning gates that were intended to *discriminate* active from quiet nights no longer do so — the gate passes whether or not an event is likely.

3. **Daytime horizon scoring.** Feature weights and quiet-night statistics were derived from nocturnal aggregates (hours 20–23, 0–5 local time). During summer months, the nowcast and near horizons evaluated during daytime are scored with night-calibrated weights and z-scores, which produces miscalibrated scores. A `daytime_horizon=True` flag is set but no weight correction is applied.

4. **NESTOR-BES dominance.** NB appears in 100% of two- and three-station active nights at the 30 ppb threshold. Any NB outage (sensor gap, maintenance) degrades the system substantially; a fallback to IB CIVIC CTR is implemented but flagged as `degraded=True`.

**Recommended remediation:** Re-derive quiet-night statistics on a rolling 90-day window with seasonal stratification. Fit separate gate thresholds and weight sets for summer (May–Oct) vs winter (Nov–Apr). Retrain tier weights with logistic regression on labeled nights rather than raw Cohen's *d*.

---

## 2. Multi-Horizon Forecast Pipeline (STOPPED)

### Architecture

Thirty-six sklearn/XGBoost models are trained: 4 horizon buckets × 3 stations × 3 core tasks (regression, clf_5ppb, clf_10ppb), with clf_30ppb as an optional fourth task. Two model types are evaluated at training time:

- **XGBoost:** 500 estimators, max_depth=6, lr=0.05, subsample=0.8, colsample_bytree=0.8
- **RandomForest:** 500 trees, max_depth=20, min_samples_leaf=5

An ensemble is automatically selected when the test-set AUC difference between the two is < 0.01 (classification) or R² difference < 0.02 (regression); ensemble weights are proportional to test-set metrics.

### Feature sets per horizon

Each horizon bucket receives a different feature vector, reflecting the shift from autocorrelation-dominated near-term prediction to exogenous-signal-dominated day-ahead prediction:

| Horizon | H₂S lags | Flow lags | Extra |
|---|---|---|---|
| 0–6h | 1h, 3h, 6h | 6h | — |
| 6–24h | 6h, 12h, 24h | 24h | yesterday max/mean/std, exceed_rate_7d, spill_active |
| 24–48h | 24h, 48h | 24h | 2-day max/mean/std, crisis_days |
| 48–72h | 48h, 72h | 48h | 2-day max/mean/std |

All horizons share a common base of 43 exogenous features: weather (temperature, humidity, wind speed/direction, surface pressure, cloud cover, dewpoint), wind rolling averages (2h, 3h, 4h), cyclical time encodings (hour, month, wind direction as sin/cos), streamflow derivatives (log-transformed, high/low regime flags), SBIWTP plant signal features, tidal state, and atmospheric stability flags.

Feature vectors are constructed to be **origin-anchored**: lags and rolling statistics are computed relative to the forecast issue time, not the target time, so the same feature can be used across all lead hours within a horizon bucket without information leakage.

### Current status and known issues

The pipeline is STOPPED pending re-evaluation of operational fit. The primary concern is the same seasonal calibration window issue as the tiered system: all 36 models were trained on Nov 2023 – Apr 2026 data, with no held-out seasonal validation set. Out-of-distribution generalization to summer months is unknown.

Additionally, the current operational system routes the `tiered_alert_features` asset through windowed slices of the standard hourly forecast rather than through multi-horizon outputs. If the MH pipeline is restarted, these two paths should be reconciled.

---

## 3. The clf_5ppb and clf_10ppb Classifiers

### Role in the pipeline

The per-station binary classifiers — clf_5ppb and clf_10ppb — are the most directly relevant ML components to the Tier 1 and Tier 2 alert logic, though they are not yet wired into the tiered alert scoring. They currently live in two places:

1. As **standalone per-station tasks** in the daily analysis pipeline (`h2s_multi_station_training.py`), where they serve as threshold exceedance predictors for the daily station forecast and dashboard.
2. As **multi-horizon tasks** within the MH pipeline (STOPPED), where they are fit separately per horizon bucket, allowing the model to use the appropriate lag structure at each lead time.

### What they predict

Each classifier outputs a probability P(H₂S > threshold) at the station level. The 5 ppb model identifies nights where any H₂S activity begins; the 10 ppb model targets the multi-station moderate-exceedance regime that is the analytical basis for Tier 2 gating.

### Relationship to the tiered alert gates

The tiered alert system currently ignores the clf_5ppb and clf_10ppb probability outputs entirely — the gates are deterministic (SBIWTP flow and anomaly thresholds), and the scores are derived from Cohen's *d* weights rather than classifier posteriors. A natural extension would be to **replace or augment** the hard gate in Tier 1 with `P(H₂S > 5ppb) > threshold_1` at NESTOR-BES, and similarly for Tier 2, which would give the gates genuine probabilistic calibration and allow them to generalize beyond the calibration window.

### Performance

Test-set performance is stored per-station in `tijuana/forecast/models/stations/{station_key}/training_report.json` (AUC, Brier score, F1). The classifiers show reasonable within-sample discrimination but, consistent with the MH pipeline, lack out-of-season validation.

---

## 4. Hourly Forecast Pipeline (Active)

The operational hourly pipeline uses a single XGBoost model trained on NESTOR-BES with a 43-feature set and three output classes (green < 5 ppb, yellow 5–30 ppb, orange ≥ 30 ppb). Reported performance on held-out data:

- Orange detection rate: **61.3%**
- False alarm rate: **5.4%**

This model is the current production system. It does not produce per-station outputs for IB CIVIC CTR or SAN YSIDRO, and it does not resolve at multiple lead times — all predictions use the same model regardless of forecast horizon.

---

## 5. Dispersion Models

Two physics-based models run alongside the ML pipeline and are not covered in depth here:

- **Gaussian forward plume** (every 6h): uses calibrated emission rates (east 87.3 g/s, west 29.9 g/s, south 49.8 g/s, total 167 g/s as of March 2026 calibration) with wind-dependent diffusion. Issues 30 ppb and 100 ppb threshold alerts based on 72h plume forecast.
- **Lagrangian backward inversion** (weekly, STOPPED by default): backward particle tracking to attribute source fractions to 16 candidate locations. Output emission rates feed the Gaussian forward model.

These models are independent of the ML classifiers but share the same meteorological inputs and alert routing infrastructure.

---

## Summary Table

| Model | Type | Stations | Status | Training data | Key failure risk |
|---|---|---|---|---|---|
| Tiered alert (T1–T3) | Rule-based / Cohen's *d* scoring | All 3 | Active (post-April degraded) | Nov 2023 – Apr 2026 | Seasonal distributional drift; frozen z-score baseline |
| Multi-horizon ensemble | XGBoost/RF, 36 models | All 3 | STOPPED | Nov 2023 – Apr 2026 | No out-of-season validation |
| clf_5ppb / clf_10ppb | Binary classifier (XGBoost/RF) | All 3 | Active (daily pipeline) | Nov 2023 – Apr 2026 | Same seasonal window; not wired into tier gates |
| Hourly XGBoost | 3-class classifier | NESTOR-BES only | Active | Ongoing retraining | Single-station; no lead-time resolution |
| Gaussian forward plume | Physics | N/A | Active | — | Emission rate staleness; wind field accuracy |
| Lagrangian inversion | Physics | N/A | STOPPED | — | — |
