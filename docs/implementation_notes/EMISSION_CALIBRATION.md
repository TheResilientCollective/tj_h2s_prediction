# Weekly Emission Calibration Pipeline

**Date:** 2026-04-17
**Scope:** `projects/h2s/src/h2s/defs/h2s_calibration_pipeline.py` and
`projects/h2s/src/h2s/dispersion/emission_inversion.py`
**Consumer:** `gaussian_forward_forecast_detailed` in
`h2s_dispersion_pipeline.py` — prefers `Q_field_latest.parquet` over
`EMISSION_RATES_PATH` when present.

## Problem

The earlier `emission_rate_inversion` asset fits three zone-level rates
(east / west / south) against a single static window (Feb 1 – Apr 1 2026)
and anchors them to a hard-coded total of 167 g/s from the March 13 2026
event. Two issues:

1. **No feedback loop.** Sensors see one thing, the forecast uses
   something else. Emissions drift with rainfall, wastewater loads, and
   season, but the forecast doesn't track any of that.
2. **Zone-level is too coarse.** Real H2S comes from continuous channel
   and wetland reaches, not three point-source-like zones. Attributing
   emissions to 104 river-channel segments captures the spatial
   structure that drives which sensor sees what.

## Approach — stacked-block NNLS over a 150 m channel grid

Three mathematical steps, reused from the standalone prototype that was
validated on the Apr 4 2026 events:

1. **LOCATE.** Project combined backward footprints onto a 150 m
   channel grid (104 segments along the Tijuana River main stem plus
   Goat Canyon, Smuggler's, estuary, Del Sol tributaries) via a 350 m
   Gaussian kernel.
2. **INVERT.** Build a Gaussian sensitivity matrix
   `A[sensor, segment]` from the same physics as `gaussian_forward`.
   Solve `Q = argmin ‖A·Q − C_obs‖² + λ₁‖Q‖²` subject to `Q ≥ 0` via
   `scipy.optimize.nnls` on the augmented system.
3. **ITERATE.** Re-weight footprints by positive residuals, re-project,
   add `ΔQ`. Converges in 1-3 iterations.

**Batch stacking** (this is what makes it work). A single event with 3
sensors is deeply underdetermined against 104 segments. We assemble
every qualifying event-timestep in the 7-day partition window and stack
their `A` matrices vertically:

```
A_stack ∈ ℝ^{N_events · N_sensors × N_segments}   # typically 100-200 rows
C_obs   ∈ ℝ^{N_events · N_sensors}
```

With a 5 ppb event threshold (see below) we typically get 100+ rows
against ~100 columns — overdetermined, and the NNLS floor (`Q ≥ 0`) plus
L1 regularization picks a sparse, physically plausible solution.

## Pipeline anatomy

One partition key = one Monday (UTC). Four assets, all partitioned:

| Asset | What it does | S3 outputs |
|---|---|---|
| `rolling_footprint_matrix` | Per-event Lagrangian residence-time footprints for each qualifying (sensor, timestep) in the 7-day window. | *(in-memory list)* |
| `channel_emission_inversion` | Stacked-block NNLS over the partition. | `weekly/{partition}/Q_field.parquet`, `Q_field.json` |
| `calibration_diagnostics` | Gate 1a (leave-one-sensor-out CV) + Gate 1b (leave-one-time-fold-out CV) + Σ Q budget check. | `weekly/{partition}/diagnostics.json` |
| `calibration_viz` | Four verification PNGs. | `weekly/{partition}/Q_field_map.png`, `loo_cv_scatter.png`, `loto_cv_scatter.png`, `budget_bar.png` |

### Skip-low-data gate

`CalibrationConfig.min_events_per_week = 3`. Weeks below this threshold
short-circuit in `rolling_footprint_matrix` — no particle simulations,
no NNLS, no diagnostics. Each asset emits `status =
skipped_insufficient_events` metadata. This makes the 2025-onward
historical backfill cheap: quiet weeks cost ~O(1), not O(N_particles ×
N_sensors × 168 h).

### Event threshold

`CalibrationConfig.h2s_threshold_ppb = 5.0` (matches
`H2S_THRESHOLD_LOW` — the community smell-detection threshold). 30 ppb
(the ORANGE alert) is too coarse a filter: residents complain well
before readings reach it, and a lower threshold gives the NNLS more
rows to fit against.

### `_latest` pointer protection

`Q_FIELD_LATEST_MAX_AGE_DAYS = 30`. A partition run only updates
`Q_field_latest.parquet` (and the `_latest` pointers for the JSON and
the three PNGs) when the partition's end is within 30 days of today.
Historical backfills of 2025 weeks can run freely without clobbering
the live dispersion forecast's Q field.

## S3 layout

```
tijuana/dispersion/calibration/
  weekly/
    2025-01-06/
      Q_field.parquet       # (segment_idx, lat, lon, Q_g_s, channel_name)
      Q_field.json          # GeoDemic-friendly sidecar (active segments only)
      diagnostics.json      # LOSO + LOTO CV + budget gate results
      Q_field_map.png       # channel segments colored by Q_g_s
      loo_cv_scatter.png    # Gate 1a: leave-one-sensor-out CV scatter
      loto_cv_scatter.png   # Gate 1b: leave-one-time-fold-out CV scatter
      budget_bar.png        # Σ Q vs allowed band
    2025-01-13/ ...
    2026-04-06/ ...
    index.json              # (reserved — not yet written)
  Q_field_latest.parquet    # canonical live Q field (recent-week only)
  Q_field_latest.json
  inversion_diagnostics_latest.json
  viz/
    Q_field_map_latest.png
    loo_cv_scatter_latest.png
    loto_cv_scatter_latest.png
    budget_bar_latest.png
  S_row_cache/{sensor}/{YYYYMMDDHH}.npy   # footprint row cache (planned)
```

## Diagnostic plots

### Gate 1a — Leave-one-sensor-out CV scatter (`loo_cv_scatter.png`)

**Purpose.** Cross-sensor consistency check: do the three sensors tell
the same story about the emission field?

**Construction**
(`h2s_calibration_pipeline.py` — `calibration_diagnostics` asset):

1. Pick one sensor (NESTOR, IB CIVIC, or SAN YSIDRO), hide all its rows.
2. Solve NNLS on rows from the other two sensors only → `Q_train`.
3. Predict what the held-out sensor should have seen:
   `c_pred = A_held_out @ Q_train`.
4. Repeat for each of the three sensors.

**Reading the plot**

- One colored dot per held-out event-timestep. `x` = observed ppb at
  that sensor; `y` = predicted ppb from Q fit to the *other* two
  sensors.
- Black dashed 1:1 line. Dots on the line mean the model can reproduce
  that sensor from information it never saw.
- Legend shows per-sensor RMSE and bias.

**Gate 1a — `leave_one_sensor_out_pass`.** For every sensor with test data:

```
rmse_over_std = RMSE(c_pred, c_obs) / std(c_obs) < 1.0
|bias|                                           < 10 ppb
```

Failing means one sensor disagrees with the story the other two tell —
most often a **geometric degeneracy** (because the inversion is
spatially underdetermined with 3 sensors against ~100 segments):

- Sensor too far from any channel segment that the other two can constrain.
- Wind direction puts the held-out sensor downwind of segments that the
  other two aren't sensitive to.
- Raw obs noise (bad calibration, intermittent drop-outs).

### Gate 1b — Leave-one-time-fold-out CV scatter (`loto_cv_scatter.png`)

**Purpose.** Temporal-stability check: does `Q` fit on some of the week's
events predict the rest? All 3 sensors stay in both train and test, so
spatial coverage is held constant and the test isolates whether the
inversion overfits to individual event clusters or captures a stable
underlying source distribution.

**Construction**
(`h2s_calibration_pipeline.py` — `calibration_diagnostics` asset,
`_time_fold_cv` helper):

1. Pre-compute per-event sensitivity rows (reused from Gate 1a).
2. Random k-fold shuffle with fixed seed (`np.random.default_rng(0)`) →
   reproducible fold assignments across reruns of the same partition.
3. `k = max(2, min(5, n_events // 3))` — adaptive: k=5 when events are
   plentiful, gracefully down to k=2 on quieter weeks.
4. For each fold `i`: train NNLS on rows from events in the other k-1
   folds; predict rows from events in fold `i`. Same `solve_nnls` call
   and regularization as the real inversion.

**Reading the plot**

- One colored dot per held-out event × sensor. `x` = observed ppb at
  that sensor; `y` = predicted ppb from Q fit to the *other* time folds.
- Black dashed 1:1 line. Dots on the line mean `Q` generalizes across
  time folds within the week.
- Legend shows per-sensor RMSE and bias (aggregated over all test folds).
- Title says `k=N` so you can spot weeks that fell back to fewer folds.

**Gate 1b — `leave_one_time_fold_out_pass`.**

```
overall.rmse_over_std = RMSE(c_pred, c_obs) / std(c_obs) < 1.0
|overall.bias_ppb|                                       < 10 ppb
```

Failing most often means **temporal non-stationarity of Q within the
week**:

- A source turned on or off mid-week (rainfall flush, discharge event,
  upstream repair) and the single `Q` can't fit both halves.
- An outlier event with meteorology the rest of the week doesn't cover —
  check the per-fold RMSEs in `diagnostics.json` for one fold with
  dramatically worse residuals.
- Raw obs noise concentrated in one fold.

**Why both gates?** They test different properties and fail from
different causes. LOSO fails on sensor-geometry problems that do not
imply the Q field is wrong; LOTO fails on temporal drift that LOSO can
miss entirely. A run passing both gives a much stronger pass signal
than either alone.

### Σ Q budget bar (`budget_bar.png`)

**Purpose.** One-line sanity check that the whole inversion gives a
total emission that's physically plausible.

**Construction.** Sums `Q_g_s` across all channel segments in the
partition's `Q_field.parquet` → `Σ Q`.

**Reading the plot**

- Green horizontal band at **30-500 g/s** — the allowed range.
- Gray dashed line at **167 g/s** — the March 13 2026 calibration anchor.
- Thick vertical bar at the current `Σ Q`. Green if inside the band,
  red if outside.

**Gate 2 — `budget_sanity_pass`.** `30 ≤ Σ Q ≤ 500 g/s`.

**Failure modes**

- **Σ Q below 30 g/s.** Too much L1 regularization, or the week had
  only weak events → inversion zeroed out real sources. Loosen
  `lambda_l1` or lower `h2s_threshold_ppb` (already at the 5 ppb
  smell-detection floor by default).
- **Σ Q above 500 g/s.** Numerical blowup — usually an event with a
  sensor reading the channel geometry can't explain (e.g. a wind shift
  puts the sensor outside any plausible plume). NNLS piles mass onto
  the nearest segment to force the fit. Look at the channel map —
  sharp spike on a single segment is the tell.

### Channel segment map (`Q_field_map.png`)

Gray dots = inactive segments (Q = 0). Colored dots = active segments,
sized and colored by `Q_g_s`. Blue triangles = the three sensors. This
one is descriptive rather than a gate — useful for visually confirming
that hot segments line up with known source regions (Smuggler's / Goat
Canyon / Saturn Blvd / Dairy Mart Bridge).

## Gate policy

Gates are **published as metadata but do not fail the asset.** The
dispersion forecast falls back to `EMISSION_RATES_PATH` only when
`Q_field_latest.parquet` is missing, not when a gate fails. That's
intentional:

- A backfill run of a 2025 week with `budget_sanity_pass = False` still
  writes its Q field for analysis (it just doesn't touch `_latest`
  anyway, because of the 30-day age filter).
- A recent-week run with `leave_one_sensor_out_pass = False` or
  `leave_one_time_fold_out_pass = False` still writes to `_latest` —
  the operator reviews the diagnostics and decides whether to disable
  the Q field source mode via ops config, rather than losing the whole
  forecast because one gate flagged.

## Tuning knobs (`CalibrationConfig`)

| Field | Default | Notes |
|---|---|---|
| `window_days` | 7 | Ignored for partitioned runs; partition key drives window. |
| `h2s_threshold_ppb` | 5.0 | Smell-detection threshold. 30 is alert; 10 is complaint-relevant. |
| `require_stable` | True | Only fit against nocturnal/calm events where Gaussian is valid. |
| `max_events` | 48 | Cap timesteps per partition (cost control). |
| `min_events_per_week` | 3 | Skip gate — see above. |
| `n_particles` | 1500 | Reduced from 2000 for batch speed. |
| `hours_back` | 2.0 | Valley-scale (1-7 km sources, 8-37 min travel). Do not raise without re-validating. |
| `segment_spacing_m` | 150.0 | Channel grid resolution. 150 m is a compromise between spatial resolution and NNLS conditioning. |
| `lambda_l1` | 0.3 | L1 sparsity. Raise to suppress ringing; lower if Σ Q collapses. **Only the value set on `channel_emission_inversion` matters** — diagnostics loads the `InversionConfig` from `Q_field.json` so CV runs under the same regularization that fit Q. |
| `background_ppb` | 1.0 | Subtracted from `C_obs`. Protects against fitting to pure noise. |
| `min_rows_for_inversion` | 9 | ≥ 3 events × 3 sensors before we trust NNLS. |

## Operator playbook

### Σ Q collapses well below the 167 g/s anchor on an event-heavy week

Symptom: gates all red, `Σ Q` in single-digit g/s, but the LOSO/LOTO
scatters show large observed-C dots (≥100 ppb) — real signal is
present and the inversion is not matching it. Both scatters pile
below the 1:1 line (predictions systematically low).

Typical cause: `lambda_l1 = 0.3` is too aggressive for this week — the
L1 penalty is zeroing out segments the big event actually needed.
Symptomatic tell: the channel map shows only a handful of active
segments (~20/143) clustered near one sensor.

Fix: drop `lambda_l1` to 0.05 – 0.10 and re-run the same partition.
Expect `Σ Q` to climb toward the anchor and the LOSO/LOTO scatters to
tighten against the 1:1 line.

`dg launch` only accepts a YAML *file* via `--config`, not inline JSON.
Write the override to a file (with `h2s__` op-name prefixes — required
because the calibration assets use `key_prefix="h2s"`):

```yaml
# /tmp/calibration_lambda_l1_008.yaml
ops:
  h2s__channel_emission_inversion:
    config:
      lambda_l1: 0.08
```

Only `channel_emission_inversion` needs the override — diagnostics reads
the `InversionConfig` from the `Q_field.json` sidecar so CV evaluates
under the same lambda Q was fit with. Then:

```bash
uv run dg launch --job emissions_calibration_job --partition 2026-03-09 \
  --config /tmp/calibration_lambda_l1_008.yaml
```

If `Σ Q` still undershoots, look at `per_fold` in
`diagnostics["leave_one_time_fold_out"]` — one fold with dramatically
worse RMSE means the week has a wind-regime outlier that a single `Q`
can't reconcile (e.g., one event has a wind direction no other event
in the partition shares). Not a tuning problem — that week genuinely
has two emission regimes and should be treated as such, or split.

### LOSO fails but LOTO passes (or vice-versa)

- **LOSO red, LOTO green** — geometric degeneracy between the three
  sensors, not a `Q` problem. The inversion found a physically
  plausible Q that generalizes across time; it just can't be
  reconstructed from only two sensors. Usually safe to accept.
- **LOSO green, LOTO red** — `Q` is non-stationary within the week.
  One or more sources turned on/off, or a rainfall flush happened.
  Don't tune your way around it — split the partition or accept that
  the weekly average Q masks sub-weekly dynamics.

## Running

### Single partition (smoke test)

```bash
cd projects/h2s
uv run dg launch --job emissions_calibration_job --partition 2026-04-06
```

### 2025-onward backfill

66 weekly partitions are available from 2025-01-06 through the most
recent completed Monday. Launch from the Dagster UI (Assets → `h2s /
rolling_footprint_matrix` → Backfill) or CLI:

```bash
# Backfill a range of weeks
uv run dg launch --job emissions_calibration_job \
  --partition-range 2025-01-06..2025-03-31
```

Expect most 2025 weeks to skip with
`status=skipped_insufficient_events` — that's fine and cheap. Weeks
with real source activity will write their full Q field + diagnostics.

### Schedule (weekly)

`emissions_calibration_schedule` fires Monday 03:30 UTC,
materializes the just-completed previous week. Default status is
`STOPPED` until the first backfill passes diagnostics. Enable via the
Dagster UI after reviewing weekly diagnostics JSONs.

## Verification — notebook

`notebooks/calibration_inspection.ipynb` pulls `Q_field_latest.parquet`
and `inversion_diagnostics_latest.json` directly from S3 and
reproduces the three plots locally. Useful for ad-hoc exploration
(e.g. comparing a specific backfilled week to the live `_latest`).

## Related notes

- `docs/implementation_notes/EMISSION_RATE_VALIDATION.md` — legacy
  3-zone `emission_rate_inversion` asset. Calibration pipeline's Q
  field supersedes it when present; EMISSION_RATES_PATH remains as
  fallback.
- `docs/implementation_notes/GRID_FORECAST_TUNING.md` — gridded Gaussian
  guards (`baseline_scale`, `ppb_clip`). Applies to both 16-source and
  Q-field forecast paths.
- `docs/implementation_notes/SOURCE_ATTRIBUTION.md` — Lagrangian
  backward-particle attribution used upstream by both the legacy
  inversion and the new calibration.
