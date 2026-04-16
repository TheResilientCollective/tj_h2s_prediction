# Gridded Gaussian Forecast — Sensitivity Tuning

**Date:** 2026-04-16
**Scope:** `run_forward_model_gridded_detailed()` in
`projects/h2s/src/h2s/dispersion/gaussian.py`
**Consumer:** GeoDemic map — `Gaussian Forecast` and `Gaussian Animation (12h)`
layers (`forward_grid_detailed_latest.json`,
`forward_grid_frames_detailed_latest.json`).

## Problem

On a typical day (sensors reading ~2 ppb at NESTOR-BES, IB CIVIC CTR,
SAN YSIDRO) the detailed gridded forecast was producing:

- `max = 8,646 ppb`
- 425 cells ≥ 100 ppb
- Near-source cells saturating to the top of the color ramp

Two root causes, independent:

1. **Event-calibrated emission rates applied as a constant baseline.**
   The inversion pipeline (`emission_rate_inversion`) fits rates against
   acute events — e.g. the March 13 2026 event where NESTOR-BES recorded
   394 ppb. Those same rates feed the *gridded* product every hour, so
   on calm baseline days the spatial field looks like a bad event even
   when sensors see nothing. Sensor-level timeseries forecasts match
   observed peaks because they're calibrated against those events; the
   grid has no matching dilution baseline.

2. **Near-source Gaussian singularity.** A grid cell whose center sits
   within 10 m of a source yielded σ_y ≈ 0.6 m, σ_z ≈ 0.3 m under
   stability F, driving the steady-state plume concentration to
   unphysical levels (∝ 1/(σ_y · σ_z)). Grid resolution is ~150 m, so
   cells cannot meaningfully resolve anything closer than ~75 m to a
   source centroid.

## Fix (2026-04-16)

Three defensive changes, all additive, no impact on the sensor-level
timeseries forecast (`run_forward_model_detailed`):

### 1. `pg_sigmas` near-field floor: 10 m → 100 m

```python
# gaussian.py — pg_sigmas()
x_km = max(x_km, 0.1)   # was 0.01
```

Every sensor receptor is > 300 m from the closest source, so the
sensor-level product is unaffected. The floor only bites on grid cells
co-located with or within a source cell.

### 2. `baseline_scale` parameter on `run_forward_model_gridded_detailed`

```python
GRID_BASELINE_SCALE: float = 0.1
```

Multiplies every per-source rate by this factor before the plume
evaluation. Default `0.1` produces a *baseline* spatial field (sensor
peak ~0.5–5 ppb, near-source peak ~50–100 ppb). To render at the full
event-calibrated level (same Q as the sensor-level forecast), pass
`baseline_scale=1.0`.

Only applies to the gridded product. The inversion output in
`emission_rates.json` and the sensor-level timeseries forecast are
unchanged.

### 3. `ppb_clip` output ceiling

```python
GRID_PPB_CLIP: float = 500.0
np.minimum(total_ppb, ppb_clip, out=total_ppb)
```

Hard clamp per cell. 500 ppb is ~5× the worst community-sensor
recording; anything above is numerical artefact from plume-core
singularities or stacked sources. Keeps the color ramp legible and
guards against future calibration drift.

## Metadata

Every frame now carries the applied parameters so consumers can
interpret the grid:

```json
"metadata": {
  "baseline_scale": 0.1,
  "ppb_clip": 500.0,
  "emission_rates_per_source_g_s": { ... }
}
```

## Validation

Back-of-envelope after the fix (baseline_scale=0.1, pg_sigmas floor=100m,
clip=500):

| Region | Before | After |
|---|---|---|
| Grid max | 8,646 ppb | ~50–500 ppb |
| Near-source core | 1,000–8,000 ppb | 50–200 ppb |
| Sensor proximity | 20–80 ppb | 2–8 ppb |
| Count of cells ≥ 100 ppb | 425 | ~5–20 |

These track reasonably with sensor observations on non-event days.

## Recalibration path (if needed)

If the baseline looks too cool after a field visit confirms real odors:

1. Raise `baseline_scale` toward `1.0` (e.g. `0.3`).
2. Re-run the dispersion pipeline and compare grid peaks to reported
   complaints for that hour.
3. If near-source cells look hot but downwind looks right, leave
   `baseline_scale` and raise `ppb_clip`.

Do **not** modify `emission_rates.json` or
`DISPERSION_DEFAULT_EMISSION_RATES_GS` to compensate — those drive the
sensor-level forecast too, which is independently calibrated against
observed peaks.

## Not touched

- The coarse 3-zone gridded model (`run_forward_model_gridded`) is not
  currently consumed by GeoDemic (the frontend is on the detailed
  path). If that changes, apply the same three fixes there.
- HYSPLIT forward-dispersion outputs are unaffected.