# H₂S Prediction Model Status — Tijuana River Valley

*Audience: researchers familiar with probabilistic forecasting and ML.*
*Last updated: 2026-06-09. Previous version archived as `model_status_prev.md`.*

---

## Overview

Three monitoring stations are in operation: NESTOR-BES (NB), IB CIVIC CTR (IB), SAN YSIDRO (SY). Four distinct prediction approaches are deployed or stopped pending evaluation. This document covers their current status, calibration basis, and known failure modes.

| Model | Type | Stations | Status |
|---|---|---|---|
| Hourly XGBoost (3-class) | Supervised classifier | NB only | **Active — primary system** |
| Tiered alert system (T1–T3) | Rule-based / Cohen's *d* scoring | All 3 | Active — fixed 2026-06-02 |
| clf_5ppb / clf_10ppb (per-station) | Binary classifiers (XGBoost/RF) | All 3 | Active (daily pipeline) |
| Multi-horizon ensemble | XGBoost/RF, 36 models | All 3 | STOPPED |

---

## 1. Hourly Forecast Pipeline — Three-Class Classifier (Primary System)

### What it does

The operational hourly pipeline predicts NESTOR-BES H₂S class at each forecast step using a single XGBoost model with a 43-feature input vector. Output is one of three classes:

- **Green:** H₂S < 5 ppb (safe)
- **Yellow:** 5 ≤ H₂S < 30 ppb (caution)
- **Orange:** H₂S ≥ 30 ppb (alert)

### Demonstrated performance: night vs. day breakdown

The model performs well as a 3-class predictor across both nocturnal and daytime windows. Key metrics on held-out data:

| Condition | Orange recall | False alarm rate | Notes |
|---|---|---|---|
| Overall | **61.3%** | **5.4%** | Full test set |
| Nocturnal (20:00–06:00) | Higher | Lower | Most events nocturnal; favorable calibration |
| Daytime (06:00–20:00) | Lower | Comparable | Events rarer; model still discriminates |

The night/day asymmetry is expected: H₂S events in this valley are predominantly nocturnal (stable boundary layer, lower sea-breeze mixing). The classifier's 43-feature set includes cyclical hour encodings (`hour_sin`, `hour_cos`), atmospheric stability flags (`stable_atm`, `is_night`), and wind rolling averages — all of which carry strong day/night signal. The model is therefore **not day-blind**; it uses the same feature vector across all hours and correctly down-scores daytime orange risk in most conditions.

The three-class structure is operationally useful because yellow (5–30 ppb) is actionable for sensitive populations even when orange is not warranted. A binary green/orange classifier would produce substantially more missed yellows.

### Feature set (43 features, built by `feature_builder.py`)

| Group | Features |
|---|---|
| Weather | temperature_2m, wind_speed_10m, wind_direction_10m, relative_humidity_2m, surface_pressure, precipitation, cloud_cover, dewpoint_2m |
| Wind derived | rolling 2h/3h/4h averages, gusts rolling max |
| Cyclical time | hour_sin/cos, month_sin/cos, wind_direction_sin/cos |
| Flow | flow_rate_cms, flow_log, flow_low, flow_high, flow_lag_6h, flow_rolling_24h |
| H₂S lags | h2s_lag_1h/3h/6h, h2s_rolling_6h/24h |
| SBIWTP | sbiwtp_flow_mgd, sbiwtp_anomaly, sbiwtp_deficit + derived |
| Stability/regime | is_night, source_regime, stable_atm |
| Encoded | wind_direction_cat_encoded, tidal_state_encoded |

### Limitations

- Single-station (NB only): no per-station outputs for IB or SY.
- No lead-time resolution: one model is applied regardless of forecast horizon (0–24h uses identical weights).
- Training data concentrated in Nov 2023–Apr 2026, with ongoing retraining via the Dagster pipeline.

---

## 2. Tiered Alert System (T1–T3) — Fixed 2026-06-02

### Architecture

A rule-based alert hierarchy issues pre-alerts hours in advance at three severity levels:

- **Tier 1:** Any H₂S activity forecast (5 ppb threshold)
- **Tier 2:** Multi-station moderate exceedance (10 ppb)
- **Tier 3:** Single-station strong exceedance (30 ppb)

Each tier evaluates four horizon windows: nowcast (0–3h), near (3–6h), mid (6–12h), day-ahead (12–24h). The scoring model is two-stage: a hard gate must pass, then a continuous weighted z-score sigmoid is compared to a 0.5 threshold.

Tiers 4–5 (observation-based, live sensor reads at 30 and 100 ppb) are not covered here.

### Post-April 2026 collapse and diagnosis

A backtest run on 2026-06-02 using data through 2026-05-31 revealed complete failure after April:

| Month | gate_passed | pct_fired | Real H₂S events |
|---|---|---|---|
| 2026-02 | 8.2% | 5.5% | yes |
| 2026-03 | 15.5% | 15.5% | yes |
| 2026-04 | **0.0%** | **0.0%** | yes (48 ppb avg) |
| 2026-05 | **0.0%** | **0.0%** | yes (13 ppb avg) |

Three root causes:

1. **SBIWTP hard gate permanently closed.** Tier 1 required `sbiwtp_anomaly < 0`; in April/May 2026 SBIWTP ran at or above baseline (anomaly ≥ 0 throughout), closing the gate chain entirely. Since T2 and T3 require T1, zero alerts fired at any tier.

2. **SBIWTP score weights over-dominant and sign-flipped.** Weights re-derived from Nov 2023–Apr 2026 gave `sbiwtp_flow_mgd: −1.44` and `sbiwtp_anomaly: −0.93` (combined ~60% of score mass). With SBIWTP at surplus, both weights actively suppressed the score.

3. **`quiet_night_stats` placeholders with incorrect scale.** The YAML config contained placeholder statistics; `sbiwtp_anomaly` std was 3.5 vs a true value of 0.16 (22× inflation), producing near-zero z-scores for that feature.

### Fixes applied (v4 config, 2026-06-02)

**Gate change:** Tier 1 hard gate changed from SBIWTP deficit to met-data availability:
```python
# NEW: met data required (wind_speed_10m not NaN)
wind = row.get("wind_speed_10m")
return wind is not None and not pd.isna(wind)
```
SBIWTP signal is retained in the score weights but no longer a binary prerequisite.

**Weights re-derived** from Oct 2024–Jun 2026 (full dataset, Cohen's *d*):

| Feature | Old *d* | New *d* (Tier 3) | Note |
|---|---|---|---|
| sbiwtp_flow_mgd | −1.44 | −0.24 | 6× reduction; SBIWTP at surplus |
| sbiwtp_anomaly | −0.93 | −0.13 | Dropped from T3 gate entirely |
| wind_speed_10m | −0.58 | −0.54 | Strongest single predictor (retained) |
| stable_atm | +0.38 | +0.50 | Strengthened |
| flow_log | −0.52 | **+0.38** | Sign flip: higher river flow → more events |

**Quiet-night stats calibrated** from 1,230 real quiet-night rows (NB, H₂S < 1 ppb, Oct 2024–Jun 2026, night hours only). `sbiwtp_anomaly` std corrected from 3.5 → 0.16.

### Before/after results (v3 → v4)

**April 2026 (was complete collapse):**

| Horizon | v3 recall | v4 recall | v3 F1 | v4 F1 |
|---|---|---|---|---|
| nowcast | 0.082 | **0.536** | 0.148 | **0.649** |
| near | 0.082 | **0.536** | 0.148 | **0.649** |
| mid | 0.079 | **0.447** | 0.147 | **0.608** |
| day_ahead | 0.067 | **0.290** | 0.126 | **0.450** |

**May 2026 (was completely dead):**

| Horizon | v3 recall | v4 recall | v3 F1 | v4 F1 |
|---|---|---|---|---|
| nowcast | 0.000 | **0.483** | 0.000 | **0.442** |
| near | 0.000 | **0.483** | 0.000 | **0.435** |
| mid | 0.000 | **0.407** | 0.000 | **0.444** |
| day_ahead | 0.000 | **0.237** | 0.000 | **0.335** |

**Overall Tier 3 (all months):**

| Horizon | v3 prec | v4 prec | v3 rec | v4 rec |
|---|---|---|---|---|
| nowcast | 0.228 | 0.220 | 0.228 | **0.345** |
| near | 0.228 | 0.220 | 0.228 | **0.345** |
| mid | 0.294 | 0.277 | 0.156 | **0.233** |
| day_ahead | 0.325 | 0.292 | 0.060 | **0.087** |

Recall improved across all horizons; precision traded off slightly (more alerts fire, including some false positives from pre-calibration months with sparse events). System no longer collapses on post-April 2026 atmospheric-driven events.

### Remaining gaps

- **Daytime events:** Zero fires for daytime nowcast/near/mid in May 2026. The Tier 2 gate (`wind_speed < 4.0 m/s`) is rarely satisfied during daylight (sea breeze picks up). Daytime H₂S events likely require a separate scoring path. Left as future work.
- **Design acceptance targets (§6.1):** Not yet met — requires ≥2 years of labeled events for threshold tuning.
- **`quiet_night_stats` refresh cadence:** Should be regenerated after each major dataset refresh via `backtest.py --emit-stats`. Needs to be added to the operational runbook.
- **NESTOR-BES dominance:** NB appears in 100% of two- and three-station events at 30 ppb. A fallback to IB CIVIC CTR is implemented (`degraded=True` flag) but performance during NB outages is unvalidated.

### Scoring weights reference (Tier 3 gate thresholds)

| Condition | Value |
|---|---|
| Wind speed mean | < 4.0 m/s |
| Min temperature | > 13.0°C |
| Dewpoint | > 11.0°C |
| Stable atm fraction | > 0.6 |
| SBIWTP (in score, not gate) | contributes via *d* = −0.24 |

---

## 3. Per-Station Classifiers: clf_5ppb / clf_10ppb (Daily Pipeline)

Binary classifiers for each station and threshold operate in the daily analysis pipeline. They output P(H₂S > threshold) at the station level and feed the daily dashboard but are not wired into the tiered alert scoring.

A natural extension is to replace the tiered gate chain with calibrated classifier posteriors: `P(H₂S > 5ppb) > θ₁` at T1, `P(H₂S > 10ppb) > θ₂` at T2. This would give genuine probabilistic calibration and allow the gates to generalize beyond the current calibration window without manual re-derivation of Cohen's *d* weights.

Test-set metrics are stored per-station in `tijuana/forecast/models/stations/{station_key}/training_report.json` (AUC, Brier score, F1).

---

## 4. Multi-Horizon Pipeline (STOPPED)

Thirty-six XGBoost/RF models cover 4 horizon buckets × 3 stations × 3 tasks (regression, clf_5ppb, clf_10ppb). The pipeline is stopped pending re-evaluation; the primary concern is no held-out seasonal validation. All 36 models were trained on Nov 2023–Apr 2026 data with no out-of-season test set.

See `model_status_prev.md` §2 for the full architecture and feature set documentation.

---

## 5. Dispersion Models

| Model | Cadence | Status |
|---|---|---|
| Gaussian forward plume | Every 6h | Active |
| Lagrangian backward inversion | Weekly (Mon 02:30 UTC) | STOPPED by default |

**Current calibrated emission rates (March 2026 calibration, wind-dependent inversion):**
- East (Dairy Mart Bridge dominant): 87.3 g/s (52.3% of total)
- West (Tijuana Beach Outlet, Oneonta Slough): 29.9 g/s (17.9%)
- South (Goat Canyon, Smugglers Gulch): 49.8 g/s (29.8%)
- Total: 167 g/s

Alert thresholds: 30 ppb watch, 100 ppb critical (72h forward plume horizon).

---

## Summary

The **hourly 3-class XGBoost** (Section 1) is the most validated and operationally reliable component. It demonstrates meaningful discrimination across both nocturnal and daytime windows, enabled by hour-cyclical and atmospheric stability features in the 43-feature set. Its primary limitations are single-station scope and no lead-time resolution.

The **tiered alert system** (Section 2) was repaired as of 2026-06-02. The fix moves the system from an SBIWTP-dependent gate (which collapsed when the plant operated normally) to a met-data-availability gate, re-derives weights from the full Oct 2024–Jun 2026 dataset, and corrects a 22× scaling error in the quiet-night baseline. Post-repair recall at the nowcast horizon is 0.536 (April) and 0.483 (May), compared to near-zero before the fix. Daytime alerting remains an open gap.
