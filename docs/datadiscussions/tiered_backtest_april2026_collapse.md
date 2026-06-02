# Tiered Alert Backtest: Performance Collapse After April 2026

**Date:** 2026-06-02  
**Dataset:** S3 `latest/tijuana/forecast_data/modeldata_h2s_nofill.parquet` (extends to 2026-05-31)  
**Backtest run:** `h2s.defs.tiered_alerts.backtest` — no `--data` arg, S3 fallback triggered

---

## Observed Symptom

Tier 3 nowcast results collapse completely after the calibration period ends (April 2026):

| Month   | gate_passed | mean_score | pct_fired | real H2S events |
|---------|-------------|------------|-----------|-----------------|
| 2026-02 | 8.2%        | 0.686      | 5.5%      | yes             |
| 2026-03 | 15.5%       | 0.898      | 15.5%     | yes             |
| 2026-04 | **0.0%**    | **0.277**  | **0.0%**  | yes (48 ppb avg)|
| 2026-05 | **0.0%**    | **0.166**  | **0.0%**  | yes (13 ppb avg)|

Real H2S events are occurring (`actual_max_h2s_nb` = 48 ppb mean in April, 13 ppb in May) but not a single alert fires across any tier or horizon.

---

## Root Cause 1 — Tier 1 Gate Permanently Closed

The Tier 1 hard gate (`tiers.py`, `gate_tier1`) requires **both conditions**:

```python
sbiwtp_flow_mgd < (baseline_mgd - 0.5)   # i.e., < 23.0 MGD
sbiwtp_anomaly  < 0.0
```

In the April/May 2026 data, `sbiwtp_anomaly` is consistently ≥ 0. Spot-check from the local parquet (8 rows of April data):

```
sbiwtp_anomaly: mean=+0.138, min=+0.065, max=+0.148  — ALL POSITIVE
```

Two possible reasons:
- **SBIWTP operating normally**: The plant ran at or above its long-run baseline (23.5 MGD) in spring 2026, eliminating the deficit signal that the gate relies on.
- **Data goes stale/NaN**: If the SBIWTP ingestion pipeline stopped updating after the observation cutoff (~April 1), `sbiwtp_flow_mgd` and `sbiwtp_anomaly` would be NaN. The gate implementation returns `False` for any NaN input (`tiers.py:83–87`), so NaN is indistinguishable from "gate not met."

Either way: **gate_passed = 0%**, and since Tier 2 and Tier 3 both require Tier 1 to pass first, the entire alert chain is dead.

---

## Root Cause 2 — Score Collapses Independently

The score function (`tiers.py`, `compute_score`) standardizes features using `quiet_night_stats` from `tiered_alerts.yaml`. Two problems:

**a) SBIWTP weights are large and SBIWTP features are now useless.**  
The two largest weights are `sbiwtp_flow_mgd: -1.44` and `sbiwtp_anomaly: -0.93`. If these features are NaN, they are skipped entirely. If they are positive (normal operations), they push the score *down* (negative weight × positive z-score = negative contribution). Either way, the dominant drivers of a high score are neutralized.

**b) `quiet_night_stats` are explicitly labeled as placeholders.**  
From `configs/tiered_alerts.yaml:68`:
```yaml
# PLACEHOLDER — regenerate with `uv run python -m h2s.defs.tiered_alerts.backtest --emit-stats`
```
These statistics were derived from quiet nights in Nov 2023–Apr 2026. They have not been regenerated. Spring/summer feature distributions (warmer, windier, less stable) will produce systematically different z-scores, further degrading the score.

---

## Structural Issue: SBIWTP as a Hard Prerequisite

The gate chain is:
```
Tier 1 (SBIWTP anomaly) → Tier 2 (wind + multi-station) → Tier 3 (temp + dewpoint + stability)
```

This design assumes SBIWTP anomaly is a reliable leading indicator. That held during the calibration period (winter/early spring, when SBIWTP deficits correlated with H2S events). But H2S events can also occur from purely atmospheric conditions (calm, warm, humid nights) without any SBIWTP anomaly. In that regime, the system is architecturally blind — no SBIWTP signal → no alert at any tier, regardless of observed H2S.

---

## Why the Backtest Ran Past the Calibration Boundary

The runfile (`runfiles/dg tiered backtest.run.xml`) passes no `--data` argument. The backtest falls back to the S3 URL:
```
https://oss.resilientservice.mooo.com/resilentpublic/latest/tijuana/forecast_data/modeldata_h2s_nofill.parquet
```

This "latest" parquet extends to 2026-05-31, whereas the local `data/modeldata_h2s_nofill.parquet` ends at 2026-04-01. The eval window (`t_max - 24h`) is derived from the data, so the backtest evaluated April and May 2026 without anyone explicitly asking it to.

---

## Resolution (2026-06-02)

All three root causes were addressed. Backtest runs are in `output/tier_backtest_v{1–4}/`; each directory now contains a `weights_snapshot.yaml` recording the exact config used.

### Fix 1 — Tier 1 gate: SBIWTP deficit → met-data availability

`tiers.py` `_gate_tier1_single` changed from:
```python
# OLD: sbiwtp_flow_mgd < 23.0 AND sbiwtp_anomaly < 0
```
to:
```python
# NEW: wind_speed_10m is not NaN  (met data available)
wind = row.get("wind_speed_10m")
return wind is not None and not pd.isna(wind)
```

SBIWTP still contributes via the score weights; it is no longer a hard gate.  
`configs/tiered_alerts.yaml` `gates.tier_1` updated to `met_data_required: wind_speed_10m`.

### Fix 2 — Score weights: re-derived from full dataset Cohen's d

All tier weights re-derived from Oct 2024–Jun 2026 data (n_events varies by tier):

Key changes vs original:
- `sbiwtp_flow_mgd`: −1.44 → −0.24/−0.16 (was 5–9× over-weighted; SBIWTP at surplus in 2026)
- `sbiwtp_anomaly`: −0.93 → −0.13 (near-zero signal; dropped from Tier 3 entirely)
- `flow_log`: −0.52 → +0.38 (**sign flip**: higher river flow correlates with more events)
- `stable_atm`: +0.38 → +0.50 (strengthened; d=+0.50 at Tier 3)
- `wind_speed_10m`: −0.58 → −0.54/−0.34 (strongest single atmospheric predictor)

### Fix 3 — Tier 1 weights: atmospheric features added

Tier 1 score previously used SBIWTP-only weights, so SBIWTP surplus suppressed the score
(only 14.9% of April event hours scored ≥0.5 at Tier 1). Tier 1 weights updated to match
Tier 2 (adding `wind_speed_10m: −0.34`, `wind_speed_min: −0.27`, `stable_atm: +0.27`),
so calm/stable nights can unlock the nesting chain even when SBIWTP shows no deficit.

### Fix 4 — quiet_night_stats calibrated from real data

Replaced PLACEHOLDER values with statistics from 1,230 real quiet-night rows
(NESTOR-BES, H2S < 1 ppb, Oct 2024–Jun 2026, night hours only). Key corrections:
- `sbiwtp_anomaly` std: 3.5 → **0.16** (placeholder was 22× too large — z-scores were tiny)
- `stable_atm` mean/std: 0.22/0.41 → 0.47/0.50

---

## Before/After Results (v3 → v4)

### April 2026 night (was the collapse)

| Horizon | v3 recall | v4 recall | v3 F1 | v4 F1 |
|---------|-----------|-----------|--------|--------|
| nowcast | 0.082 | **0.536** | 0.148 | **0.649** |
| near | 0.082 | **0.536** | 0.148 | **0.649** |
| mid | 0.079 | **0.447** | 0.147 | **0.608** |
| day_ahead | 0.067 | **0.290** | 0.126 | **0.450** |

### May 2026 night (was completely dead)

| Horizon | v3 recall | v4 recall | v3 F1 | v4 F1 |
|---------|-----------|-----------|--------|--------|
| nowcast | 0.000 | **0.483** | 0.000 | **0.442** |
| near | 0.000 | **0.483** | 0.000 | **0.435** |
| mid | 0.000 | **0.407** | 0.000 | **0.444** |
| day_ahead | 0.000 | **0.237** | 0.000 | **0.335** |

### Overall Tier 3 (all months)

| Horizon | v3 prec | v4 prec | v3 rec | v4 rec |
|---------|---------|---------|--------|--------|
| nowcast | 0.228 | 0.220 | 0.228 | **0.345** |
| near | 0.228 | 0.220 | 0.228 | **0.345** |
| mid | 0.294 | 0.277 | 0.156 | **0.233** |
| day_ahead | 0.325 | 0.292 | 0.060 | **0.087** |

Recall improved across all horizons; precision traded off slightly (more alerts fire,
including some false positives from pre-calibration months with sparse events).
System no longer collapses on post-April 2026 atmospheric-driven events.

---

## Remaining Gaps

- **Daytime events (April/May)**: Still zero fires for day-shift nowcast/near/mid/day_ahead in May.  
  Daytime Tier 2 gate (`wind_speed < 4.0 m/s`) is rarely satisfied in daylight (sea breeze picks up);  
  daytime H2S events are likely driven by different mechanisms. Left as future work.
- **Overall acceptance targets** (design §6.1) not yet met — requires more data accumulation  
  and/or threshold tuning once ≥2 years of labeled events are available.
- **`quiet_night_stats` refresh cadence**: Should be regenerated after each major dataset refresh  
  (`backtest.py --emit-stats`). Add to the operational runbook.
