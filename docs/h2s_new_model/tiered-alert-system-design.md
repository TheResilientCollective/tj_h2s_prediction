# Tijuana Valley H₂S Tiered Alert System — Design Document

**Status:** Draft v2
**Date:** May 30, 2026
**Repo:** [`tj_h2s_prediction`](https://github.com/TheResilientCollective/tj_h2s_prediction) (master branch)
**Scope:** Refactors the existing two-tier (30 ppb WATCH / 100 ppb CRITICAL) logic in `projects/h2s/src/h2s/defs/h2s_alert_system.py` into a unified five-tier system that adds three forecast-based pre-alert tiers driven by SBIWTP operations and meteorology. The existing `apcd_sensor_watch.py` remains independent.

---

## 1. Motivation

The current `h2s_alert_system.py` fires only on observed H₂S exceedances at NESTOR. The `apcd_sensor_watch.py` does the same across all three APCD stations from the public bucket. Both are reactive. Neither uses the forecast pipeline's predictions or the upstream SBIWTP / met / dispersion features to provide lead time.

A multi-night analysis of 273 fully-reported nights (Nov 2023 – Apr 2026) demonstrated that the *features* discriminating multi-station detection events are different at different concentration thresholds, and that those differences map cleanly onto operational decision boundaries. This design exploits that mapping to add three predictive pre-alert tiers in front of the existing observation-based alerts.

The 23.5 MGD SBIWTP baseline already encoded in `constants.ALERT_SBIWTP_BASELINE_MGD` is the same baseline our analysis identifies as the regime boundary — independent validation that the operational team has already converged on the right number.

---

## 2. Analytical basis

Per-station nightly peak H₂S was computed for each "night" (the 12-hour window straddling midnight, using `is_night=1` hours only, attributed to the date on which the night begins). Three concentration thresholds were evaluated: 5, 10, and 30 ppb. For each, the number of stations whose peak exceeded the threshold was tallied per night, and feature distributions were compared between activity groups.

### 2.1 Threshold-dependent group sizes

| Threshold | 0 active | 1 active | 2 active | 3 active |
|---|---|---|---|---|
| 5 ppb | 34 (12%) | 38 (14%) | 85 (31%) | 116 (43%) |
| 10 ppb | 80 (29%) | 61 (22%) | 73 (27%) | 59 (22%) |
| 30 ppb | 154 (56%) | 68 (25%) | 35 (13%) | 16 (6%) |

### 2.2 NESTOR-BES as bellwether

NESTOR-BES is in 100% of 2- and 3-active nights at the 30 ppb threshold and in every 2-active station pair at 10 ppb. IB+SY without NB occurs only once across the entire record at 5 ppb. This justifies the existing `ALERT_SITE_NAME = "NESTOR"` choice in `constants.py` and the per-station fallback hierarchy (NB → IB → SY) used throughout the codebase.

### 2.3 Discriminating features by threshold

Effect sizes (Cohen's d) of multi-station (≥2 active) nights versus quiet (0 active) nights:

| Feature | 30 ppb d | 10 ppb d | 5 ppb d |
|---|---|---|---|
| SBIWTP flow ↓ | −0.79 | **−1.22** | **−1.44** |
| SBIWTP anomaly ↓ | — | −0.55 | **−0.93** |
| SBIWTP SLI ↑ | — | +0.55 | +0.63 |
| SBIWTP deficit ↑ | — | +0.54 | +0.64 |
| Precipitation ↓ | — | −0.33 | −0.64 |
| Border flow ↓ | −0.09 | −0.25 | −0.52 |
| Wind speed (vec) ↓ | **−0.61** | −0.45 | −0.58 |
| Wind speed (scalar) ↓ | −0.56 | −0.42 | −0.53 |
| Stable atm fraction ↑ | +0.54 | +0.38 | +0.31 |
| Min temp ↑ | **+0.70** | +0.17 | −0.14 |
| Dewpoint ↑ | +0.50 | +0.17 | ≈0 |
| Surface pressure ↓ | −0.42 | −0.11 | — |

### 2.4 Key structural finding: two-stage threshold logic

At 5 ppb, the SBIWTP signal saturates between 2-active and 3-active nights. The transition from "no stations" to "two stations" is governed by **SBIWTP operations and antecedent dryness**. The transition from "two stations" to "three stations" is governed by **boundary-layer meteorology** (low wind, stable atmosphere). At 30 ppb, both factors must align simultaneously.

This decoupling permits an operationally meaningful tiered system in which each forecast tier answers a distinct question:

- **5 ppb** — *Is the plant producing detectable upstream sewage tonight?* (SBIWTP-driven)
- **10 ppb** — *Is the plant signal reaching multiple receptors?* (SBIWTP + dispersion)
- **30 ppb** — *Is there an exceedance-grade exposure event ahead?* (full meteorological gate)

The 5 / 30 ppb thresholds align exactly with the existing **green / yellow / orange** classification used by the `h2s_alerts` asset in `h2s_pipeline.py`. The tiered system extends this classification with explicit feature gates rather than replacing the existing per-prediction logic.

---

## 3. System architecture

### 3.1 Tier definitions

The unified system uses five tiers. Tiers 1–3 are **forecast-based pre-alerts** (predictive, internal ops channel). Tiers 4–5 are **observation-based exceedance alerts** (existing WATCH/CRITICAL audiences retained).

| Tier | Name | Threshold | Source | Horizons | Audience |
|---|---|---|---|---|---|
| 1 | Plant Signal | ≥5 ppb forecast at ≥1 station | Forecast | All four | Ops (internal) |
| 2 | Multi-Site Risk | ≥10 ppb forecast at ≥2 stations | Forecast | All four | Ops (internal) |
| 3 | Exceedance Risk | ≥30 ppb forecast at ≥1 station | Forecast | All four | Ops (internal) |
| 4 | WATCH | ≥30 ppb observed | Live obs (current `h2s_alert_system.py`) | n/a | Monitoring staff |
| 5 | CRITICAL | ≥100 ppb observed | Live obs (current `h2s_alert_system.py`) | n/a | Agency decision-makers |

Each forecast tier is evaluated across four horizon windows (see §3.6). The tier audiences and labels are configured in the existing `ALERT_TIERS` dict in `constants.py`, extended to cover all five tiers. Tier 4 and Tier 5 keep their current onset/post-event-summary message logic.

### 3.2 Module structure

The new code lives under `projects/h2s/src/h2s/defs/` alongside the existing alerting modules. The dispatcher and Slack resource patterns are reused unchanged.

```
projects/h2s/src/h2s/defs/
├── h2s_alert_system.py            # EXISTING — refactored to host all 5 tiers
├── tiered_alerts/                 # NEW package
│   ├── __init__.py
│   ├── tiers.py                   # tier definitions, gates, scores
│   ├── features.py                # nightly aggregation of forecast features
│   ├── state.py                   # S3-backed state (extends existing pattern)
│   ├── messages.py                # tier-specific message templates
│   └── backtest.py                # historical replay against modeldata
├── apcd_sensor_watch.py           # UNCHANGED — independent per-station watch
└── ...

projects/h2s/src/h2s/
├── constants.py                   # ALERT_TIERS dict extended to 5 tiers
└── resources/
    └── slack.py                   # SlackAlertResource — reused
```

The existing `h2s_alert_dispatcher`, `h2s_alert_sensor`, and `h2s_alert_job` in `h2s_alert_system.py` are extended (not replaced) to handle the new tier outputs. Tier 4 and Tier 5 behavior is preserved bit-for-bit; Tiers 1–3 are added as new branches in the existing dispatch logic.

### 3.3 Asset graph (Dagster)

```
                ┌─────────────────────────────────┐
                │  h2s_predictions (existing)     │
                │  hourly forecast 0–48h          │
                └────────────────┬────────────────┘
                                 │
                                 ▼
            ┌─────────────────────────────────────┐
            │  tiered_alert_features (NEW)        │
            │  per-horizon aggregation per station│
            │  + SBIWTP, met, dispersion features │
            │  produces 4 horizon-windowed views  │
            └────────────────┬────────────────────┘
                             │
       ┌─────────────────────┼─────────────────────┐
       ▼                     ▼                     ▼
  ┌─────────┐           ┌─────────┐           ┌─────────┐
  │ tier_1  │           │ tier_2  │           │ tier_3  │
  │ scores  │           │ scores  │           │ scores  │
  │ × 4 hzns│           │ × 4 hzns│           │ × 4 hzns│
  └────┬────┘           └────┬────┘           └────┬────┘
       │                     │                     │
       └─────────┬───────────┴──────────┬──────────┘
                 ▼                      ▼
        ┌────────────────────┐    ┌─────────────────────────┐
        │ observation_check  │    │ tier_4 / tier_5         │
        │ (existing)         │    │ (existing WATCH/CRIT)   │
        └────────┬───────────┘    └──────────┬──────────────┘
                 │                           │
                 └─────────────┬─────────────┘
                               ▼
                  ┌─────────────────────────────┐
                  │ h2s_alert_dispatcher        │
                  │ (existing, extended)        │
                  │ consolidates per-tier       │
                  │ across horizons (1 msg/tier)│
                  └──────────────┬──────────────┘
                                 ▼
                  ┌─────────────────────────────┐
                  │ SlackAlertResource          │
                  │ (existing)                  │
                  └─────────────────────────────┘
```

Asset keys follow the existing namespace convention: `dg.AssetKey(["h2s", "tier_1_scores"])`, etc. Each `tier_N_scores` asset returns a list of `TierResult` objects, one per horizon × station evaluation cell.

### 3.4 Trigger logic per tier

Each forecast tier is evaluated independently across each of the four horizon windows defined in §3.6. The gate logic and feature weights below apply to each `(tier, horizon)` cell. Tier nesting (§6.2) is enforced **per horizon**: a Tier 3 fire in the nowcast window must coincide with Tier 2 and Tier 1 fires in the same nowcast window. It does *not* need to coincide with fires in other horizons.

Each tier combines a **hard threshold gate** (required) with a **continuous risk score** (shown for context and used to rank within and across horizons). The risk score is a sigmoid of a weighted sum of standardized feature values, with weights initialized from the Cohen's d values in §2.3 and intended to be refined via logistic regression in a follow-up calibration pass.

#### Tier 1 — Plant Signal (5 ppb forecast)

**Hard gate (all required):**
- SBIWTP forecast flow < `ALERT_SBIWTP_BASELINE_MGD - 0.5` (i.e., < 23 MGD)
- SBIWTP forecast anomaly < 0

**Score features:** `sbiwtp_flow_mgd`, `sbiwtp_anomaly`, `sbiwtp_sli`, `sbiwtp_deficit`, `precipitation`, `border_flow`

**Fire condition:** hard gate AND score ≥ 0.5

#### Tier 2 — Multi-Site Risk (10 ppb forecast)

**Hard gate (all required):**
- Tier 1 gate satisfied at ≥2 stations independently
- Forecast mean wind speed < 4 m/s for the night

**Score features:** Tier 1 features + `wind_speed_mean`, `wind_speed_min`, `stable_atm_fraction`

**Fire condition:** hard gate AND score ≥ 0.5

#### Tier 3 — Exceedance Risk (30 ppb forecast)

**Hard gate (all required):**
- Tier 2 gate satisfied
- Forecast min temp > 13 °C
- Forecast dewpoint > 11 °C
- Forecast stable_atm fraction > 0.6

**Score features:** Tier 2 features + `temp_min`, `dewpoint_mean`, `surface_pressure`

**Fire condition:** hard gate AND score ≥ 0.5

#### Tier 4 — WATCH (30 ppb observed)

Existing logic in `h2s_alert_system.py`. Retained unchanged. Threshold 30.0 ppb against NESTOR observations, with `ALERT_QUIET_HOURS` and `ALERT_CLOSE_WAIT_HOURS` controlling onset / post-event-summary timing.

#### Tier 5 — CRITICAL (100 ppb observed)

Existing logic in `h2s_alert_system.py`. Retained unchanged.

### 3.5 Threshold rationale

All hard threshold values are drawn from §2 — the inflection points in the feature distributions between quiet and multi-station groups. They are deliberately set near the medians of the active groups rather than at extremes, so the hard gate functions as a **necessary-condition filter** while the score handles ranking and confidence.

Thresholds and feature weights are stored in `projects/h2s/configs/tiered_alerts.yaml` (new file) and loaded via the existing config patterns. This keeps tuning out of code review and consistent with how the existing pipeline manages its hyperparameters.

### 3.6 Forecast horizons

Each forecast tier is evaluated across four horizon windows, each anchored to the current evaluation timestamp `t`:

| Horizon key | Window | Meaning |
|---|---|---|
| `nowcast` | `t+0h` → `t+3h` | What's happening right now and in the next three hours |
| `near` | `t+3h` → `t+6h` | Next-shift planning window |
| `mid` | `t+6h` → `t+12h` | Same-day operational planning |
| `day_ahead` | `t+12h` → `t+24h` | Tomorrow's overnight outlook |

For each horizon, features are aggregated by mean (or vector mean for wind direction; see §7.1) across the hourly forecast rows whose timestamps fall within the window. The `is_night` filter from the original analysis is applied to a horizon **only when ≥75% of the rows in that horizon fall in night hours** — for daytime horizons (typical for nowcast/near during the day) the gate and score are computed against the daytime aggregation directly. This is a calibration risk; see Open Question 6 in §8.

**Tier fire condition:** A `(tier, horizon)` cell fires when its hard gate passes and its score ≥ 0.5, exactly as in §3.4. Each cell has its own debounce state (§4).

**Per-tier message consolidation:** The dispatcher emits at most one Slack message per tier per evaluation cycle. That message lists all four horizon states for the tier and highlights which horizon(s) are firing. See §5 for the template.

**Tier nesting per horizon:** Tier 3 firing in horizon H requires Tier 2 and Tier 1 to also fire in horizon H. Cross-horizon nesting is not enforced — Tier 1 firing in the day-ahead horizon while Tier 3 is quiet across all horizons is a valid state (and a common one).

---

## 4. State management

State extends the existing `ALERT_STATE_S3_PATH` JSON file in MinIO (`tijuana/forecast/alerts/h2s_alert_state.json`). The JSON gains a `tiers` key that contains per-tier, per-horizon state:

```json
{
  "watch":    { ... existing structure preserved ... },
  "critical": { ... existing structure preserved ... },
  "tiers": {
    "tier_1": {
      "nowcast":   { "last_fired_at": "2026-04-15T22:00:00-07:00", "last_score": 0.78, "active": true,  "rolling_7d_fires": 4, "consecutive_clear_cycles": 0 },
      "near":      { "last_fired_at": null,                         "last_score": 0.42, "active": false, "rolling_7d_fires": 2, "consecutive_clear_cycles": 5 },
      "mid":       { ... },
      "day_ahead": { ... }
    },
    "tier_2": { "nowcast": {...}, "near": {...}, "mid": {...}, "day_ahead": {...} },
    "tier_3": { "nowcast": {...}, "near": {...}, "mid": {...}, "day_ahead": {...} }
  }
}
```

12 forecast-tier-horizon cells plus the two observation tiers. Read/written through the existing `S3Resource` using the same JSON helper functions the current alert system uses.

**Debounce rules (per `(tier, horizon)` cell):**

- A `(tier, horizon)` cell does not re-fire onset within `ALERT_QUIET_HOURS` (3 h) of its own previous onset.
- A cell clears when its score falls below 0.3 for 3 consecutive evaluation cycles. Clearing emits a post-event summary using the existing summary archive pattern (`ALERT_SUMMARY_ARCHIVE_PATH`).
- A higher-tier cell firing in horizon H while a lower-tier cell in the same horizon H is active suppresses the lower tier's onset Slack message (state still updates). Across horizons, no suppression.

**Per-tier message dedup (separate from debounce):**

The dispatcher consolidates all four horizons of a given tier into a single Slack message per cycle. The message is sent only if at least one horizon for that tier is firing for the first time since its last clear (i.e., a new onset in any horizon). If horizons are progressing through the time axis (a `day_ahead` fire becoming a `mid` fire becoming a `nowcast` fire as time advances), the dispatcher recognizes this as the same evolving event and rate-limits re-messaging to one per `ALERT_QUIET_HOURS` window per tier.

---

## 5. Alert routing

Routing is configured in `constants.ALERT_TIERS`, which is extended:

```python
ALERT_TIERS = {
    "tier_1":   {"label": "PLANT-SIGNAL",    "threshold": 5.0,   "audience": "Ops (internal)",         "channel_env": "SLACK_CHANNEL_OPS"},
    "tier_2":   {"label": "MULTI-SITE-RISK", "threshold": 10.0,  "audience": "Ops (internal)",         "channel_env": "SLACK_CHANNEL_OPS"},
    "tier_3":   {"label": "EXCEEDANCE-RISK", "threshold": 30.0,  "audience": "Ops (internal)",         "channel_env": "SLACK_CHANNEL_OPS"},
    "watch":    {"label": "WATCH",           "threshold": 30.0,  "audience": "Monitoring staff",       "channel_env": "SLACK_CHANNEL_MONITORING"},
    "critical": {"label": "CRITICAL",        "threshold": 100.0, "audience": "Agency decision-makers", "channel_env": "SLACK_CHANNEL_AGENCY"},
}
```

`SlackAlertResource` is instantiated once per tier with the appropriate channel resolved from env. The token (`SLACK_TOKEN`) is shared across all tiers — channels separate the audiences. Delivery uses `WebClient.chat_postMessage` (Slack SDK), consistent with the existing pattern in `slack.py`.

Forecast-tier messages are **consolidated per tier across horizons** — one message per tier per cycle. The message body lists all four horizon states and highlights which ones are firing. A representative Tier 2 message:

```
🟡 Tier 2 — Multi-Site Risk
Evaluated: Wed Apr 15, 17:00 PT

Horizon states:
  ⚠️ Nowcast (0–3h):    score 0.84  ← FIRING (gate satisfied at NB, IB)
  ⚠️ Near    (3–6h):    score 0.71  ← FIRING (gate satisfied at NB, IB)
     Mid     (6–12h):   score 0.42  (gate failed — wind speed mean 4.6 m/s)
     Day-ahead (12–24h): score 0.21  (gate failed — wind speed mean 5.8 m/s)

Top contributing factors (firing horizons):
  • SBIWTP forecast flow: 20.4 MGD  (3.1 MGD deficit below baseline)
  • Forecast wind speed:  2.8 m/s   (below 4 m/s threshold)
  • SBIWTP anomaly:       −0.15

Interpretation: Plant throughput drops into the multi-site detection regime within the next 6 hours, with light winds limiting dispersion. Mid and day-ahead winds recover above threshold. This is a pre-alert; no exceedance is yet observed.

Suggested response: Verify monitoring station status. Pre-position field response if NB peak exceeds 20 ppb within 6 hours.

Reference: docs/tiered-alert-system-design.md §3.4, §3.6
```

Tier 4 (WATCH) and Tier 5 (CRITICAL) message templates are preserved verbatim from `h2s_alert_system.py`.

---

## 6. Validation plan

### 6.1 Historical backtest

`projects/h2s/src/h2s/defs/tiered_alerts/backtest.py` (CLI: `uv run python -m h2s.defs.tiered_alerts.backtest`) replays `modeldata_h2s_nofill.parquet` (preferred) or `modeldata_h2s_nofill.csv` from Nov 2023 through the latest available date. For each evaluation timestamp the script computes all four horizon windows, all three forecast tiers per horizon, and records:

- per-cell: gate_passed, score, fire
- per-cell daytime_horizon flag
- per-night: actual max H₂S per station and number of stations exceeding each tier threshold
- per-cell: lead time (hours between cell fire and first observed exceedance during the cell's forecast window)

Acceptance criteria, per horizon:

| Horizon | Tier 3 precision | Tier 3 recall |
|---|---|---|
| nowcast | ≥ 0.65 | ≥ 0.80 |
| near | ≥ 0.60 | ≥ 0.75 |
| mid | ≥ 0.55 | ≥ 0.70 |
| day_ahead | ≥ 0.50 | ≥ 0.65 |

Looser criteria are accepted at longer horizons reflecting forecast uncertainty. The script exits non-zero if any horizon misses its targets — this gates merge.

### 6.2 Tier nesting invariant

A Tier 3 fire in horizon H must always co-occur with Tier 2 and Tier 1 fires in the same horizon H. This is enforced in `tiers.py` and asserted in `backtest.py`. A non-nested fire within a horizon is a hard failure. Cross-horizon non-nesting (e.g., Tier 1 firing in `day_ahead` while quiet in `nowcast`) is normal and expected.

### 6.3 Live shadow mode

Run the new tiers in shadow mode for two weeks alongside the existing WATCH/CRITICAL behavior. Compare lead times, false positive rates, and operator feedback before promoting Tiers 1–3 from shadow to active. The shadow flag is read from env (`TIERED_ALERTS_SHADOW=true` suppresses Slack dispatch but still writes state and logs).

---

## 7. Implementation notes

### 7.1 Forecast data source and horizon windowing

Forecast features are pulled from the existing hourly forecast pipeline output (`h2s_predictions` asset, hourly resolution to t+48h). The pipeline already produces per-station predictions at the partition cadence configured in `h2s_schedules.py`.

The new `tiered_alert_features` asset slices the hourly forecast into four overlapping horizon windows anchored to the evaluation timestamp `t`:

- `nowcast`:   rows where `time ∈ [t, t+3h)`
- `near`:      rows where `time ∈ [t+3h, t+6h)`
- `mid`:       rows where `time ∈ [t+6h, t+12h)`
- `day_ahead`: rows where `time ∈ [t+12h, t+24h)`

For each (horizon × station) cell, features are aggregated by mean. Wind direction is aggregated via vector mean:

```python
u = -wind_speed * np.sin(np.deg2rad(wind_direction))
v = -wind_speed * np.cos(np.deg2rad(wind_direction))
# After averaging u and v over the horizon window:
wind_dir_vec = np.rad2deg(np.arctan2(-u_mean, -v_mean)) % 360
```

`stable_atm_fraction` is the mean of the boolean `stable_atm` flag across the window. `wind_speed_min` is the minimum.

Timezone handling matches existing conventions: parquet inputs are timezone-aware; CSV fallback requires `pd.to_datetime(..., utc=True).dt.tz_convert("America/Los_Angeles")`.

**`is_night` handling per horizon:** the original analysis was night-aggregated, so the feature weights and quiet-night feature stats are calibrated against night-hour windows. For each horizon, the aggregation computes the fraction of in-window hours that are night hours; if that fraction is ≥ 0.75, the horizon is treated as a night window and weights apply as-is. Otherwise, a `daytime_horizon=True` flag is set on the result, the same weights and gates are applied (with the saturation clip from §7.4), and the message includes a note that confidence is reduced because the scoring was trained on nightly windows. The daytime-recalibration follow-up is Open Question 6 in §8.

The canonical met source is NESTOR-BES (`site_name == "NESTOR - BES"`). Per §8.4, if NB has no data for the cycle, fall back to IB Civic Center and mark `degraded=True` so dispatched messages can flag the degraded state.

### 7.2 Per-station vs aggregate forecasting

The forecast file is keyed on `site_name`. For Tier 1 (any station) the per-station forecast is used directly. For Tier 2 the gate requires the condition to be satisfied for at least two stations. NB+IB share Open-Meteo grid cells; the gate must be evaluated per station regardless of grid overlap (the saturated gate plus the score handles this naturally — both stations passing the gate independently is the signal that matters).

### 7.3 Score normalization

Feature standardization uses means and standard deviations from the quiet-night population in the training window (Nov 2023 – Apr 2026). These are committed to `projects/h2s/configs/tiered_alerts.yaml` and refreshed quarterly as part of the existing model retraining cadence.

### 7.4 Saturation behavior

The 5 ppb tier signal saturates at low SBIWTP flow. The hard gate captures the saturation point; the score is clipped at 0.95 to prevent overconfidence reporting under deep-SBIWTP-low conditions.

### 7.5 Relationship to `apcd_sensor_watch.py`

The APCD per-station sensor watch is independent of the new tiered system. It polls the APCD public bucket on a 5-minute cadence and is a complete loop of its own. The new tiered system reads from the internal forecast pipeline output, not the APCD bucket. The two systems share `constants.ALERT_TIERS` for label consistency and share `SlackAlertResource` for delivery, but their state, sensors, and dispatch are separate.

---

## 8. Open questions

1. **Tide signal at 5 ppb.** Min tide showed d = +0.45 between 3-active and 2-active nights at the 5 ppb threshold. May be a real boundary-layer proxy or a small-sample artifact. Re-evaluate once a full year of post-Feb-2026 data accumulates.
2. **Border flow vs SBIWTP overlap.** Border flow (d = −0.52 at 5 ppb) and SBIWTP anomaly (d = −0.93) are likely correlated. A collinearity check should inform whether to keep both in the score or combine into an "ingress signal" feature.
3. **Seasonal recalibration.** March–May 2026 was anomalously active. Threshold values may need a seasonal multiplier or a dry-season variant. Defer until the second full calendar year of data.
4. **NESTOR-BES degraded-mode fallback.** Because NB dominates multi-station nights, an NB outage substantially blinds the system. The existing pipeline already falls back NB → IB → SY in some assets; the tier evaluators should adopt the same pattern explicitly.
5. **Multihorizon pipeline as forecast source.** The `h2s_multihorizon_pipeline.py` module is currently STOPPED. If it returns to service, the tiered alerts should consume its dedicated multi-horizon outputs in place of the windowed slices of the standard 0–48 h hourly forecast.
6. **Nightly-calibration vs daytime horizons.** Feature weights and quiet-night feature stats were fit on nightly aggregates. Nowcast and near windows during the day are scored with the same weights and gates, flagged with `daytime_horizon=True` and a confidence caveat in the message. A proper daytime recalibration is a follow-up: re-fit weights on daytime windows and add a `weights_daytime.yaml` alongside the nightly weights. Until that's done, daytime horizon scores are advisory.
7. **Evaluation cadence vs horizon resolution.** The forecast pipeline currently materializes hourly. A 0–3h nowcast updated only once an hour is coarser than the horizon name implies. If operations want true 15-minute nowcast resolution, the upstream `h2s_predictions` materialization cadence would need to drop to 15 min — out of scope for this design but worth noting.

---

## 9. References

- Existing alert module being extended: `projects/h2s/src/h2s/defs/h2s_alert_system.py`
- Independent sister system: `projects/h2s/src/h2s/defs/apcd_sensor_watch.py`
- Forecast pipeline: `projects/h2s/src/h2s/defs/h2s_pipeline.py`
- Constants: `projects/h2s/src/h2s/constants.py`
- Slack resource: `projects/h2s/src/h2s/resources/slack.py`
- S3/MinIO resource: `projects/h2s/src/h2s/resources/minio.py`
- Object storage paths under `tijuana/forecast/...`
- Modeldata canonical source: `https://oss.resilientservice.mooo.com/resilentpublic/latest/tijuana/forecast_data/modeldata_h2s_nofill.parquet`
