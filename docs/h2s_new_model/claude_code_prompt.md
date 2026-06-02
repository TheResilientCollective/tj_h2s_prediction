# Claude Code Prompt: Tiered H₂S Alert System

Paste the block below into Claude Code at the root of `tj_h2s_prediction` (master branch). Commit `docs/h2s_new_model/tiered-alert-system-design.md` first so the prompt's references resolve.

---

## Task

Implement the tiered H₂S alert system specified in `docs/h2s_new_model/tiered-alert-system-design.md`. This **extends** (not replaces) the existing `projects/h2s/src/h2s/defs/h2s_alert_system.py` with three new forecast-based pre-alert tiers (Tiers 1–3, internal ops channel), each evaluated across four forecast horizons (nowcast 0–3h, near 3–6h, mid 6–12h, day-ahead 12–24h). The existing WATCH (Tier 4, 30 ppb) and CRITICAL (Tier 5, 100 ppb) observation-based logic stays intact. The unrelated `apcd_sensor_watch.py` is not touched.

Per-tier Slack messages are consolidated across all four horizons — one message per tier per cycle, with all four horizon states listed inside.

Read the design document first — every threshold, feature weight, audience, routing decision, and horizon window is grounded there. Read `CLAUDE.md` at the repo root for the broader codebase conventions (uv, Dagster, S3/MinIO, asset key namespace `["h2s", ...]`).

## Context you need before coding

1. **Read `docs/h2s_new_model/tiered-alert-system-design.md` end to end.** Hard-gate thresholds and score features come from a Cohen's d analysis of 273 nights at 5, 10, and 30 ppb. Do not change them without consulting the author.
2. **Read `CLAUDE.md` at the repo root.** Particularly the sections on asset development guidelines, S3 path conventions, and key design decisions.
3. **Read these existing files before writing new code:**
   - `projects/h2s/src/h2s/defs/h2s_alert_system.py` — the module you are extending. State management pattern, message templates, sensor cadence, dispatcher structure.
   - `projects/h2s/src/h2s/defs/apcd_sensor_watch.py` — the sister system. Mirrors a lot of the state/dispatch idiom, may have cleaner patterns to crib from.
   - `projects/h2s/src/h2s/constants.py` — `ALERT_TIERS`, `ALERT_SITE_NAME`, `ALERT_QUIET_HOURS`, `ALERT_CLOSE_WAIT_HOURS`, `ALERT_LOCAL_TZ`, `ALERT_SBIWTP_BASELINE_MGD`, `ALERT_STATE_S3_PATH`, `OBS_DATA_PATH`. Extend the `ALERT_TIERS` dict to include the three new tiers; don't fork a parallel constants dict.
   - `projects/h2s/src/h2s/resources/slack.py` — `SlackAlertResource`. Uses `slack_sdk.WebClient` and `chat_postMessage`, **not** webhooks. The token is shared; channels differ per tier.
   - `projects/h2s/src/h2s/resources/minio.py` — S3 resource for state and config reads.
   - `projects/h2s/src/h2s/defs/h2s_pipeline.py` — the existing `h2s_predictions`, `h2s_alerts`, and `slack_alerts` assets. Your `tiered_alert_features` asset takes `h2s_predictions` as input.
   - `projects/h2s/src/h2s/definitions.py` — where the new assets, sensors, and jobs must be registered.

4. **NESTOR-BES is the bellwether station.** Several pieces of logic depend on this (see design doc §2.2, §7.5). Don't treat the three stations as symmetric.

## Deliverables

All paths relative to repo root. Create:

```
projects/h2s/src/h2s/defs/tiered_alerts/
├── __init__.py
├── tiers.py                       # Tier definitions, hard gates, score functions
├── features.py                    # Nightly aggregation of forecast features
├── state.py                       # S3-backed state extensions (uses existing pattern)
├── messages.py                    # Slack message templates per tier
├── assets.py                      # Dagster assets: tiered_alert_features, tier_*_score
└── backtest.py                    # CLI replay against modeldata for validation

projects/h2s/configs/
└── tiered_alerts.yaml             # Hard-gate thresholds + score weights + feature stats

projects/h2s/tests/
├── test_tiered_alerts_tiers.py
├── test_tiered_alerts_state.py
└── test_tiered_alerts_backtest.py
```

Modify (do not rewrite from scratch):

- `projects/h2s/src/h2s/defs/h2s_alert_system.py` — extend `h2s_alert_dispatcher` to handle Tier 1–3 messages. Keep Tier 4/5 behavior bit-for-bit identical.
- `projects/h2s/src/h2s/constants.py` — extend `ALERT_TIERS` to include Tier 1–3 entries per design doc §5.
- `projects/h2s/src/h2s/definitions.py` — register the new assets and connect to the existing job.
- `projects/h2s/src/h2s/defs/h2s_schedules.py` — add tiered-alert evaluation schedule if Dagster cron differs from the existing forecast cadence (hourly is fine for now; reuse if convenient).
- `docs/h2s_new_model/tiered-alert-system-design.md` — already committed. Do not modify.
- `CHANGELOG.md` — add a `## [Unreleased]` entry describing the new tiers.

## Implementation specifics

### `tiers.py` — tier definitions, horizons, scoring

Horizon spec — define once and import everywhere:

```python
from enum import Enum
from dataclasses import dataclass

class Horizon(str, Enum):
    NOWCAST   = "nowcast"     # t+0h  to t+3h
    NEAR      = "near"        # t+3h  to t+6h
    MID       = "mid"         # t+6h  to t+12h
    DAY_AHEAD = "day_ahead"   # t+12h to t+24h

HORIZON_WINDOWS_H: dict[Horizon, tuple[int, int]] = {
    Horizon.NOWCAST:   (0,  3),
    Horizon.NEAR:      (3,  6),
    Horizon.MID:       (6, 12),
    Horizon.DAY_AHEAD: (12, 24),
}
```

Each `(tier, horizon)` evaluation produces:

```python
@dataclass(frozen=True)
class TierResult:
    tier: str                       # "tier_1" | "tier_2" | "tier_3"
    label: str                      # "PLANT-SIGNAL" | "MULTI-SITE-RISK" | "EXCEEDANCE-RISK"
    horizon: Horizon
    evaluated_at: pd.Timestamp
    window: tuple[pd.Timestamp, pd.Timestamp]   # (window_start, window_end) absolute times
    gate_passed: bool
    score: float                    # 0–0.95 (saturation clip)
    n_stations_passing_gate: int    # for Tier 2's ≥2 requirement
    contributing_features: dict[str, tuple[float, float, float]]   # name -> (value, z, weight)
    daytime_horizon: bool           # True if <75% of window is night hours (calibration caveat)
    degraded: bool                  # True if NB fallback to IB was used (§7.1 of design)
    fire: bool                      # gate_passed AND score >= 0.5
```

Score:

```python
z = (value - quiet_mean) / quiet_std
weighted_sum = sum(weight * z for each feature)
score = min(0.95, 1 / (1 + math.exp(-weighted_sum)))
```

Initial weights = Cohen's d values from design §2.3. Features where lower values indicate higher risk (SBIWTP flow, wind speed, precipitation) use negative weights so the sigmoid pushes toward 1 when the value is low.

Leave a `# TODO(weights): retrain logistic regression on labeled nights + daytime recalibration (see design §8.6)` comment near the weight-loading code.

### Hard gates

Per design §3.4 — match exactly. Evaluated **per `(tier, horizon, station)` cell**:

- **Tier 1:** `sbiwtp_flow_mgd < (ALERT_SBIWTP_BASELINE_MGD - 0.5) AND sbiwtp_anomaly < 0`. Evaluate per station.
- **Tier 2:** Tier 1 gate AND `wind_speed_mean < 4.0`. Require ≥2 stations to pass in the same horizon.
- **Tier 3:** Tier 2 gate AND `temp_min > 13.0 AND dewpoint_mean > 11.0 AND stable_atm_fraction > 0.6`.
- **Tier 4 / Tier 5:** existing logic. Do not touch. (No horizon dimension — observation-based.)

### Tier nesting invariant (per horizon)

A Tier 3 fire in horizon H must always co-occur with Tier 2 and Tier 1 fires in the same horizon H. Enforce in `tiers.py` and raise `TierNestingError` (define in `tiers.py`) if violated. `backtest.py` fails loudly on any non-nested fire within a horizon.

Cross-horizon non-nesting is normal: Tier 1 may fire in `day_ahead` while quiet in `nowcast`. Do not raise on this.

### `features.py` — horizon-windowed aggregation

For an evaluation timestamp `t`, slice the hourly `h2s_predictions` forecast into four overlapping windows defined in `HORIZON_WINDOWS_H`:

```python
def slice_horizon(df: pd.DataFrame, t: pd.Timestamp, horizon: Horizon) -> pd.DataFrame:
    start_h, end_h = HORIZON_WINDOWS_H[horizon]
    start = t + pd.Timedelta(hours=start_h)
    end   = t + pd.Timedelta(hours=end_h)
    return df[(df["time"] >= start) & (df["time"] < end)]
```

For each `(horizon, station)` cell, aggregate features by mean. Wind direction uses vector mean:

```python
u = -wind_speed * np.sin(np.deg2rad(wind_direction))
v = -wind_speed * np.cos(np.deg2rad(wind_direction))
# After averaging u and v over the window:
wind_dir_vec = np.rad2deg(np.arctan2(-u_mean, -v_mean)) % 360
```

`stable_atm_fraction` = mean of the boolean `stable_atm` flag across the window. `wind_speed_min` = minimum.

**`is_night` handling per horizon** (per design §7.1): compute the fraction of in-window hours that are night hours. If ≥ 0.75 → nightly aggregation, weights apply as-is. Otherwise → `daytime_horizon=True` is set on the result, the same weights and gates are applied, and the message format includes a confidence caveat. Do not skip the cell — produce it with the flag set.

Always normalize timestamps with `pd.to_datetime(..., utc=True).dt.tz_convert("America/Los_Angeles")` for CSV inputs; parquet inputs are timezone-aware already.

The canonical met source is NESTOR-BES (`site_name == "NESTOR - BES"`). Per design §7.1, if NB has no data for the cycle, fall back to IB Civic Center and mark `degraded=True` on every `TierResult` produced.

The asset returns a list of horizon-aggregated frames (or a single multi-indexed frame keyed on `(horizon, station)`) — pick whichever is cleaner for downstream tier-evaluation assets.

### `state.py` — S3-backed state with per-tier per-horizon cells

Extend the existing JSON at `ALERT_STATE_S3_PATH` by adding a `tiers` key per design §4. Each tier has four horizon sub-keys (`nowcast`, `near`, `mid`, `day_ahead`), each with its own debounce state:

```python
{
    "watch":    {...},   # existing — untouched
    "critical": {...},   # existing — untouched
    "tiers": {
        "tier_1": {
            "nowcast":   {"last_fired_at": "...", "last_score": 0.78, "active": True,
                          "rolling_7d_fires": 4, "consecutive_clear_cycles": 0},
            "near":      {...},
            "mid":       {...},
            "day_ahead": {...},
        },
        "tier_2": {...},
        "tier_3": {...},
    },
}
```

Read/write through the existing `S3Resource` — do not introduce a new state file. Preserve `watch` and `critical` keys exactly. **Round-trip safety:** a state file produced by the current `h2s_alert_system.py` (without a `tiers` key) must read successfully and be augmented in place; missing horizon sub-keys default to a fresh inactive cell.

Debounce rules per design §4:

- Per-cell onset suppression: no re-fire within `ALERT_QUIET_HOURS` (3 h) for the same `(tier, horizon)` cell.
- Per-cell clearing: score < 0.3 for 3 consecutive evaluation cycles → emit post-event summary for that cell.
- Within-horizon suppression: higher-tier active in horizon H suppresses lower-tier onset Slack message in the same horizon H (state still updates).
- **Per-tier message dedup** (design §4 final paragraph): The dispatcher consolidates all four horizons of a tier into one Slack message per cycle. The same tier is not re-messaged within `ALERT_QUIET_HOURS` even if new horizon cells fire — they're considered the same evolving event.

### `messages.py` — Slack templates

Use the existing `SlackAlertResource.get_client().chat_postMessage(channel=..., text=..., blocks=...)` pattern. Forecast-tier messages are **consolidated per tier across horizons** — one message per tier per cycle. Template:

```
🟡 *Tier 2 — Multi-Site Risk*
Evaluated: Wed Apr 15, 17:00 PT

*Horizon states:*
  ⚠️ Nowcast (0–3h):     score 0.84  ← FIRING (gate at NB, IB)
  ⚠️ Near    (3–6h):     score 0.71  ← FIRING (gate at NB, IB)
     Mid     (6–12h):    score 0.42  (gate failed — wind speed 4.6 m/s)
     Day-ahead (12–24h): score 0.21  (gate failed — wind speed 5.8 m/s)

*Top contributing factors (firing horizons):*
  • SBIWTP forecast flow: 20.4 MGD  (deficit: 3.1 MGD below baseline)
  • Forecast wind speed:  2.8 m/s   (below 4 m/s threshold)
  • SBIWTP anomaly:       −0.15

*Interpretation:* Plant throughput drops into the multi-site detection regime within the next 6 hours, with light winds limiting dispersion. Mid and day-ahead winds recover above threshold. This is a pre-alert; no exceedance is yet observed.

*Suggested response:* Verify monitoring station status. Pre-position field response if NB peak exceeds 20 ppb within 6 hours.

_Reference: docs/h2s_new_model/tiered-alert-system-design.md §3.4, §3.6_
```

If any contributing horizon has `daytime_horizon=True`, append a single one-line caveat at the bottom of the message:
> _Daytime-horizon scoring is advisory — weights are nightly-calibrated (see design §8.6)._

If `degraded=True` (NB → IB fallback), append:
> _NESTOR-BES unavailable; met inputs from IB Civic Center this cycle._

Use Slack Block Kit (`blocks=` argument) for layout. The `_deficit_label` and `_wind_flag` helpers in the existing `h2s_alert_system.py` are good — reuse them, don't reimplement.

**Do not change** the Tier 4 (WATCH) and Tier 5 (CRITICAL) message format. Those audiences have downstream consumers expecting the current shape. Onset / post-event-summary structure stays identical.

### `assets.py` — Dagster assets

Implement the asset graph from design §3.3. Each asset uses `@dg.asset(key_prefix="h2s", group_name="tiered_alerts", required_resource_keys={"s3", "slack"}, ...)` consistent with the existing pattern. Assets:

- `tiered_alert_features` — consumes `h2s_predictions` (existing). Returns horizon-windowed feature aggregates for all four horizons × three stations. Use a multi-indexed DataFrame keyed on `(horizon, site_name)` or a dict of frames — pick whichever downstream wiring is cleaner.
- `tier_1_scores`, `tier_2_scores`, `tier_3_scores` — each consumes `tiered_alert_features`, returns a `list[TierResult]` covering all four horizons (and per-station evaluations inside each).
- The existing `h2s_alert_dispatcher` is extended (in `h2s_alert_system.py`) to consume the three new score lists, consolidate per tier across horizons, apply the per-tier message dedup rule from §4, and dispatch to Slack.

Scheduling: reuse the existing forecast-pipeline schedule from `h2s_schedules.py`. The tier evaluators run at the same cadence as `h2s_predictions` (currently hourly). Do not introduce a new schedule unless the existing cadence is meaningfully wrong. Per design §8.7, true 15-min nowcast resolution would require dropping the upstream materialization cadence — out of scope.

### `backtest.py` — historical replay

CLI:

```bash
cd projects/h2s
uv run python -m h2s.defs.tiered_alerts.backtest \
    --data /mnt/data/modeldata_h2s_nofill.parquet \
    --output ./output/tier_backtest/
```

If the local path doesn't exist, fall back to the canonical URL:
`https://oss.resilientservice.mooo.com/resilentpublic/latest/tijuana/forecast_data/modeldata_h2s_nofill.parquet`

The script steps through historical evaluation timestamps (hourly), reconstructs the four horizon windows from the modeldata at each step, and evaluates every `(tier, horizon)` cell. Emit a parquet:

```
evaluated_at, tier, horizon, gate_passed, score, fired, daytime_horizon,
actual_max_h2s_nb, actual_max_h2s_ib, actual_max_h2s_sy,
n_stations_exceeding_tier_threshold_in_window, lead_time_hours
```

Print per-horizon precision/recall/F1 and per-horizon mean lead time. Acceptance per design §6.1:

| Horizon | Tier 3 precision | Tier 3 recall |
|---|---|---|
| nowcast | ≥ 0.65 | ≥ 0.80 |
| near | ≥ 0.60 | ≥ 0.75 |
| mid | ≥ 0.55 | ≥ 0.70 |
| day_ahead | ≥ 0.50 | ≥ 0.65 |

Exit non-zero if any horizon misses its targets. This gates the PR merge.

### Tests

Follow the conventions in `projects/h2s/tests/` (see `conftest.py`, `test_apcd_sensor_watch.py` for the closest analog). Minimum coverage:

- `test_tiered_alerts_tiers.py`:
  - Each tier's hard gate at boundary conditions (just-passing, just-failing) for each horizon.
  - Score function known-input → known-output.
  - Tier nesting invariant within a horizon (raises `TierNestingError` if violated).
  - Cross-horizon non-nesting allowed (no raise).
  - `daytime_horizon=True` propagates to `TierResult` when window is <75% night hours.
- `test_tiered_alerts_state.py`:
  - Per-cell onset / clearing transitions for each `(tier, horizon)`.
  - Debounce rejection within `ALERT_QUIET_HOURS` per cell.
  - Within-horizon higher-tier suppression of lower-tier onset.
  - Cross-horizon non-suppression (Tier 2 firing in nowcast does not suppress Tier 1 onset in day_ahead).
  - Per-tier message dedup: a tier with multiple horizons firing in one cycle produces only one message.
  - Round-trip read of a state file lacking the `tiers` key.
- `test_tiered_alerts_backtest.py`:
  - Regression test pinning per-horizon Tier 3 precision/recall above the per-horizon targets in design §6.1.
  - Use a fixed slice of `modeldata_h2s_nofill.parquet` committed as a test fixture or pulled in `conftest.py`.

Run with: `cd projects/h2s && uv run pytest tests/test_tiered_alerts_*.py -v`. Mark slow backtest tests with `@pytest.mark.slow`.

### Definitions registration

In `projects/h2s/src/h2s/definitions.py`, import and add the new assets:

```python
from h2s.defs.tiered_alerts.assets import (
    tiered_alert_features,
    tier_1_scores,    # note: plural — returns list[TierResult] across horizons
    tier_2_scores,
    tier_3_scores,
)
```

Add them to the existing assets list passed into `Definitions(...)`. Make sure they share the existing `s3_resource` and `slack_resource` resources — do not create new resource instances. Make sure they share the existing job (`h2s_alert_job`) — extend its asset selection rather than creating a parallel job.

## Acceptance criteria

You are done when:

1. `cd projects/h2s && uv run pytest` passes including the new backtest test.
2. Per-horizon Tier 3 backtest meets the targets in design §6.1:
   - nowcast: precision ≥ 0.65, recall ≥ 0.80
   - near:    precision ≥ 0.60, recall ≥ 0.75
   - mid:     precision ≥ 0.55, recall ≥ 0.70
   - day_ahead: precision ≥ 0.50, recall ≥ 0.65
3. `cd projects/h2s && uv run dg dev` starts cleanly and shows the new assets in the asset graph under the `tiered_alerts` group.
4. A manual job run (`uv run dg launch --job h2s_alert_job`) materializes the new tier assets, evaluates all four horizons per tier, and would dispatch correctly to a test channel when `TIERED_ALERTS_SHADOW=false`.
5. With `TIERED_ALERTS_SHADOW=true`, no Slack messages are sent for Tiers 1–3, but state and logs update normally.
6. The existing WATCH and CRITICAL alerts behave **identically** to before the change. Verify with a snapshot test against a known event window.
7. A tier with multiple firing horizons in one cycle produces exactly **one** consolidated Slack message listing all four horizon states.
8. No Tier 3 fire occurs without a corresponding Tier 2 and Tier 1 fire in the **same horizon** (per-horizon nesting invariant). Cross-horizon non-nesting is permitted.
9. `CHANGELOG.md` has an `## [Unreleased]` entry covering the change.

## Things to be careful about

- **Don't touch `apcd_sensor_watch.py`.** It is a sibling system with its own state and Slack delivery. Sharing `ALERT_TIERS` is fine; sharing modules is not.
- **Timezones.** The CSV is naive; the parquet is timezone-aware. Always normalize to `America/Los_Angeles` before computing horizon windows. Test that windows spanning DST transitions are handled.
- **Daytime horizon caveat.** Cohen's d weights and quiet-night feature stats were fit on nightly aggregates. Nowcast and near during the day are scored with the same weights and flagged `daytime_horizon=True`. The message must include the advisory caveat. A proper daytime recalibration is a follow-up — leave a `# TODO(daytime_calibration): see design §8.6` comment where the weights are loaded.
- **Per-tier message dedup, not per-cell.** Each `(tier, horizon)` cell has its own debounce state, but the Slack message is per-tier. Don't accidentally send four Slack messages when all four horizons of Tier 2 fire in the same cycle.
- **Slack delivery is SDK, not webhooks.** Use `SlackAlertResource.get_client().chat_postMessage(...)`. The token is shared across tiers; channels are not. Resolve channels via env vars per `ALERT_TIERS[tier]["channel_env"]`.
- **State backward compatibility.** A state file written by the current alert system (no `tiers` key) must read successfully. Add a migration helper in `state.py` that augments in place. Missing horizon sub-keys default to fresh inactive cells.
- **NESTOR-BES outage.** If NB has no data, fall back to IB. Mark `degraded=True` on every `TierResult` produced this cycle and include the caveat in the dispatched message.
- **No new top-level dependencies** without proposing them in the PR description. The existing stack covers what this task needs (pandas, numpy, slack_sdk, dagster, boto3).
- **Score weights are placeholders.** The Cohen's d initialization is reasonable but not optimal. Leave the TODO comments so the calibration follow-ups are easy to find.
- **Don't invent new thresholds.** Every numeric threshold comes from the analysis in §2. Flag any judgment-call values inline and ask in the PR description rather than quietly changing them.

## PR description

When ready, open a PR with the following sections:

1. **Summary** — the five-tier structure, what's new vs. preserved
2. **Backtest results** — precision, recall, F1, mean lead time per tier
3. **Configuration changes** — new env vars (`SLACK_CHANNEL_OPS`, `TIERED_ALERTS_SHADOW`), new config file (`projects/h2s/configs/tiered_alerts.yaml`)
4. **State migration** — note that the alert state JSON gains a `tiers` key, with backward-compatible read
5. **Open questions** — link to design doc §8
6. **Testing** — list of new tests and what they cover

Reference `docs/h2s_new_model/tiered-alert-system-design.md` by relative path so reviewers know where the analytical basis lives.
