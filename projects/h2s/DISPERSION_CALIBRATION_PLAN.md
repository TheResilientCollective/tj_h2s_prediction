# Dispersion Calibration Loop — Geometry Refactor & SY Overprediction Fix

**Status:** Planning · **Owner:** dispersion pipeline · **Created:** 2026-06-19
**Branch:** `claude/charming-bell-klqfj9`

This is a *planning* document. Nothing here is implemented yet. It captures the
problem (San Ysidro overprediction), the proposed source-geometry change
(line sources + one point source + one estuary polygon), the redesigned
calibration loop, and a phased build plan. The **coordinate templates in
§4 are meant to be edited by hand** — fill in / move the real source
locations there, and the rest of the plan keys off that single config.

---

## 1. Problem statement

The Gaussian forward model (`dispersion/gaussian.py`) and the channel-snapped
inversion (`dispersion/emission_inversion.py`) currently represent every H2S
source as a **point**:

- `SOURCES` — 3 coarse zones (east/west/south), each a single lat/lon.
- `CANDIDATE_SOURCES` — 16 named points.
- `CHANNEL_WAYPOINTS` → `build_channel_grid()` — ~100 free point segments along
  the river centerline, each an independent unknown in the NNLS inversion.

**Symptom:** the dispersion model **overpredicts H2S at SAN YSIDRO**
(`32.552794, -117.047286`), the easternmost sensor.

### 1.1 Why points overpredict at SY (root cause)

San Ysidro sits ~250–600 m from the eastern channel points
(`tj_crossing_cdlp_e/w` at `-117.050 / -117.054`, `dairy_mart_bridge`,
`silva_drain`). Three compounding effects make a *point* source at that range
produce an unphysically sharp, high concentration:

1. **Near-field singularity.** `gaussian_plume_concentration()` floors the
   downwind distance via `pg_sigmas()` at `x_km = 0.1` (100 m). Inside a few
   hundred metres, σ_y·σ_z is tiny and `C ∝ Q/(π·u·σ_y·σ_z)` spikes. A point
   300 m upwind of SY in stable class **F** at night delivers a very large ppb
   per g/s.
2. **NNLS loads the spike.** `build_sensitivity_matrix()` gives the near-SY
   segments the largest `A[i, j]` (ppb per g/s) of any column. To fit *any*
   SY peak in the stacked system, `solve_nnls()` puts Q on exactly those
   high-sensitivity segments — which then overpredict whenever the wind aligns,
   because all the zone's emission is concentrated at one spot.
3. **No spatial regularization across the real source.** The true east source
   is a *distributed* line (the concrete river channel / Stewart's–Silva drain
   corridor), not a point. ~100 independent free segments give the inversion
   too many degrees of freedom and no incentive to spread emission.

**Key insight:** distributing a fixed total Q along a *line* (or over an
*area*) lowers the peak single-column sensitivity `max(A)` and smooths the
reconstructed field — which is exactly the lever that pulls the SY peak down
without hand-tuning a fudge factor. The geometry change *is* the calibration
fix; regularization and per-sensor weighting are second-order refinements.

---

## 2. Goals & non-goals

**Goals**
- Replace the point-only basis with a mixed geometry: **line sources** for the
  channel/drain corridors, **one point source** for Saturn Blvd Bridge, and
  **one area (polygon) source** for the Imperial Beach estuary.
- Make source geometry a **single source of truth** in one editable config,
  consumed by both the forward model and the inversion.
- Redesign the calibration loop to solve for **one Q per named source**
  (6–10 unknowns) instead of ~100 free segments, with per-sensor weighting and
  a leave-one-station-out (LOSO) diagnostic that explicitly tracks SY bias.
- Add standard dispersion-model skill metrics (bias, FB, NMSE, FAC2) so
  "overprediction at SY" becomes a number we drive to zero.

**Non-goals (this iteration)**
- No change to the Lagrangian backward footprint math (`lagrangian.py`) beyond
  reading the new geometry config for source labels.
- No HYSPLIT execution changes (CONTROL bundles regenerate from the new
  geometry automatically — §7).
- No new ML model; this is pure physical dispersion calibration.

---

## 3. Proposed source taxonomy

| Type | Source | Geometry | Rationale |
|------|--------|----------|-----------|
| **Line** | TJ River main stem (west) | polyline: Beach Outlet → Saturn → Hollister | concrete channel off-gasses along its length |
| **Line** | East drain corridor | polyline: Dairy Mart → Silva → TJ Crossing CDLP W/E | the source nearest SY — distributing it is the core fix |
| **Line** | Smuggler's Gulch / Goat Canyon | polyline: Smuggler's → Goat Canyon → Goat Canyon PS | south tributary canyons |
| **Point** | **Saturn Blvd Bridge** | single lat/lon | discrete bridge crossing, keep as point per request |
| **Polygon** | **Imperial Beach estuary** | closed polygon (marsh/mudflat) | broad area off-gassing, tide-modulated |

Notes:
- The **east drain corridor line** is the one that most directly fixes SY:
  today its emission collapses onto 1–2 points 300 m from SY; as a line it
  spreads over ~1.5 km.
- Hollister PS, Oneonta Slough, SD Bay outlets can stay as **points** or fold
  into a line — flagged as a decision in §9.
- The estuary polygon's Q should optionally scale with **tide state** (low tide
  → exposed mudflat → more off-gassing). Phase 4 note; default constant for v1.

---

## 4. Geometry config — single source of truth (EDIT THIS)

Create `projects/h2s/src/h2s/dispersion/source_geometry.yaml` (new). Both
`gaussian.py` and `emission_inversion.py` load it. Coordinates below are
pre-seeded from the existing `CHANNEL_WAYPOINTS` / `CANDIDATE_SOURCES` so the
model still runs before you relocate anything — **move/replace the lat/lon
values as needed** while you're in the field.

```yaml
# units: lat/lon in WGS84 degrees; q_prior in g/s (initial guess, refined by inversion)
# geometry: point | line | polygon
# discretize_spacing_m: sub-point spacing for line/polygon → forward & sensitivity kernels

sources:

  east_drain_corridor:           # <-- the SY-critical source
    geometry: line
    discretize_spacing_m: 120
    q_prior: 20.0
    zone: east
    vertices:
      - [32.548531, -117.064293]   # Dairy Mart Bridge
      - [32.539743, -117.064269]   # Silva Drain
      - [32.542103, -117.054117]   # TJ Crossing CDLP W
      - [32.542166, -117.050325]   # TJ Crossing CDLP E

  river_main_stem_west:
    geometry: line
    discretize_spacing_m: 150
    q_prior: 10.0
    zone: west
    vertices:
      - [32.556206, -117.126178]   # TJ River Beach Outlet
      - [32.559383, -117.092992]   # Saturn Blvd Bridge (also a discrete point below)
      - [32.554177, -117.084135]   # Hollister Bridge N
      - [32.551466, -117.084021]   # Hollister Bridge S

  smugglers_goat_canyon:
    geometry: line
    discretize_spacing_m: 150
    q_prior: 137.0
    zone: south
    vertices:
      - [32.538600, -117.086230]   # Smuggler's Gulch
      - [32.536900, -117.099160]   # Goat Canyon
      - [32.543476, -117.108026]   # Goat Canyon PS

  saturn_blvd_bridge:              # POINT per request
    geometry: point
    q_prior: 5.0
    zone: west
    location: [32.559383, -117.092992]

  imperial_beach_estuary:          # POLYGON per request — MOVE/EXTEND vertices
    geometry: polygon
    discretize_spacing_m: 150
    q_prior: 10.0
    zone: west
    tide_modulated: false          # set true in Phase 4 to scale Q by tidal_state
    vertices:                      # rough placeholder marsh boundary — replace
      - [32.573000, -117.130000]
      - [32.573000, -117.123000]
      - [32.566000, -117.123000]
      - [32.566000, -117.130000]

# Optional remaining points (decision §9: keep as points or fold into lines)
  oneonta_slough:   { geometry: point, q_prior: 0.0, zone: west,  location: [32.570082, -117.126724] }
  hollister_ps:     { geometry: point, q_prior: 0.0, zone: west,  location: [32.547600, -117.088374] }
```

A tiny loader (`load_source_geometry()`) parses this into:
- `point` → `{lat, lon}`
- `line` / `polygon` → list of discretized sub-points (reuse the metre-accurate
  interpolation in `build_channel_grid()`; for polygons, tessellate the
  interior on a `discretize_spacing_m` grid and keep points inside the ring).

Each sub-point carries its parent `source_id` so the inversion can **group
columns by source**.

---

## 5. Forward model changes (`gaussian.py`)

The point-source kernel `gaussian_plume_concentration()` stays unchanged. We add
a thin layer that turns geometry into sub-points and sums:

- **Line source → sum of sub-point plumes**, each sub-point emitting
  `Q_line · (segment_length_i / line_length_total)`. So a line's total Q is
  conserved while the peak per-cell sensitivity drops ∝ 1/N_subpoints.
- **Polygon source → sum over interior tessellation points**, each emitting
  `Q_poly / N_points`.
- **Point source → unchanged.**

New entry point (replaces the proliferation of `run_forward_model_*`
variants for this path):

```python
def run_forward_model_from_geometry(df, source_q_g_s: dict[str, float],
                                    geometry, start_time, hours=72, ...):
    # geometry: parsed source_geometry.yaml
    # source_q_g_s: {source_id -> Q g/s}
    # expands each source to sub-points once, then reuses gaussian_plume_concentration
```

Keep the existing `run_forward_model` (3-zone) and `run_forward_model_detailed`
(16-point) for backward compatibility / comparison, but route the operational
path through the geometry version. The gridded variants get the same
sub-point expansion.

**Near-field floor revisit:** with lines, no single sub-point should sit
< ~150 m from SY carrying the whole corridor's Q, so the 100 m floor stops
dominating. Still, raise the floor for sub-points *closer than the
discretization spacing* to avoid double-counting overlap. Validate against the
March 13 2026 calibration event (394 ppb @ NESTOR-BES) so we don't regress the
peak we calibrated to.

---

## 6. Calibration loop redesign

The existing `emission_inversion.py` loop (LOCATE → INVERT → ITERATE) is the
right skeleton. Changes:

### 6.1 Grouped sensitivity matrix
`build_sensitivity_matrix()` currently returns `A[sensor, segment]` over ~100
free segments. Replace with `A[sensor, source_id]` where each column is the
**sum of that source's sub-point sensitivities**:

```
A[i, s] = Σ_{p ∈ subpoints(s)}  ppb_at_sensor_i_from_unit_Q(p, met)
```

This collapses ~100 unknowns to ~6–10 named sources → a strongly
overdetermined, well-conditioned NNLS (200+ stacked rows). Fewer DOF = the
regularization that stops Q piling onto the one segment next to SY.

### 6.2 Per-sensor row weighting
In `solve_nnls()`, weight stacked rows so no single sensor dominates the fit and
SY's high-sensitivity rows don't force the solution to overpredict it. Weight
options (decision §9):
- inverse-variance by sensor (down-weight the noisiest), or
- equal-mass per sensor (each sensor contributes the same total row weight
  regardless of how many timesteps it has signal).

### 6.3 Optional per-sensor additive bias term
Jointly fit a small non-negative background/offset `b_s` per sensor to absorb
*local* micro-environment error (e.g., SY sitting in a pocket). Implement as
extra columns in `A` that are 1 for sensor `s`, 0 elsewhere. **Use cautiously**
— a bias term can mask genuine geometry error; gate it behind a config flag and
report it separately so we can see how much "unexplained SY offset" remains.

### 6.4 The loop, end to end
```
1. LOAD geometry (source_geometry.yaml) → sub-points per source.
2. For each event timestep in the rolling window:
     build grouped A[sensor, source] from met + sub-points.
3. STACK rows; solve NNLS for {Q_source} (+ optional b_s), with row weights
   and L2/L1 regularization (reuse lambda_l1; add cross-source smoothness if
   needed).
4. FORWARD-replay {Q_source} through run_forward_model_from_geometry.
5. SCORE per-sensor residuals + skill metrics (§6.5). Track SY bias explicitly.
6. ITERATE residual correction (existing logic) OR adjust geometry/weights and
   GOTO 2. Converged when max|residual| < tol AND |SY mean bias| < target.
```

### 6.5 Metrics — make "overprediction" a number
Add a scorer (extend `training/calibration_eval.py` or a new
`dispersion/calibration_metrics.py`) computing, per sensor and overall:

- **Mean bias** = mean(pred − obs). SY should move from **> 0** toward 0.
- **Fractional bias (FB)** and **NMSE** (Chang & Hanna 2004 dispersion-model
  eval standards).
- **FAC2** — fraction of predictions within a factor of 2 of obs.
- **RMSE / MAE** per sensor (already partially in `sensor_rmse_ppb`).
- **Threshold skill** — recall / false-alarm at 30 ppb (watch) and 100 ppb
  (critical), per sensor.

### 6.6 LOSO cross-validation (the SY-overprediction probe)
Fit Q on **NESTOR-BES + IB CIVIC CTR only**, predict **SY**. If LOSO-SY bias is
large and positive, the geometry/Q is overdriving SY independent of fit — the
direct test of the reported symptom. Repeat holding out each station; report a
3×(bias, FB, FAC2) table per run.

---

## 7. Downstream / HYSPLIT

- HYSPLIT CONTROL generation (`hysplit_controls.py`) already emits per-source
  emission points. Lines/polygons expand to their sub-points as multiple
  emission locations — wire the same `load_source_geometry()` expansion so
  backward/forward bundles stay consistent with the Gaussian model. No executor
  change.
- `emission_rates.json` schema grows from `{east, west, south}` to
  `{source_id: Q_g_s}` plus a `zone` rollup for back-compat with anything still
  reading the 3-zone keys (keep a derived `{east, west, south}` block).

---

## 8. Phased implementation plan

| Phase | Deliverable | Files |
|-------|-------------|-------|
| **0** | `source_geometry.yaml` + `load_source_geometry()` loader + sub-point/polygon tessellation helpers (unit-tested) | new `source_geometry.yaml`, `dispersion/geometry.py` |
| **1** | `run_forward_model_from_geometry()` (line/polygon/point), validate vs March 13 event (no peak regression) | `dispersion/gaussian.py`, tests |
| **2** | Grouped `A[sensor, source]` sensitivity + NNLS over named sources | `dispersion/emission_inversion.py` |
| **3** | Per-sensor weighting, optional bias term, metrics module, LOSO scorer | `emission_inversion.py`, new `calibration_metrics.py` |
| **4** | Wire into Dagster: extend `emission_rate_inversion` asset to write per-source `emission_rates.json`; tide-modulated estuary Q (optional) | `defs/h2s_dispersion_pipeline.py`, `constants.py` |
| **5** | HYSPLIT bundle expansion + docs + CHANGELOG | `hysplit_controls.py`, this doc, `CLAUDE.md` |

Each phase is independently shippable; Phases 0–3 are pure-library and testable
offline against stored events before any pipeline change.

---

## 9. Decisions to confirm (before/while you're relocating)

1. **Estuary polygon boundary** — the §4 vertices are a placeholder box. Replace
   with the real marsh/mudflat outline (the part that actually off-gasses).
2. **Which sources are lines vs points** — proposal: 3 lines + Saturn point +
   estuary polygon, with Oneonta/Hollister-PS as optional standalone points.
   Confirm whether to fold those into a line or keep discrete.
3. **Tide modulation of estuary Q** — implement now (Phase 4) or defer? Needs
   `tidal_state` joined into the met frame for the inversion window.
4. **Per-sensor weighting scheme** (§6.2) — inverse-variance vs equal-mass.
5. **Per-sensor bias term** (§6.3) — allow it, or force all SY error to be
   explained by geometry/Q? (Recommend: allow but report separately.)
6. **SY bias target** — what residual counts as "fixed"? (e.g., |mean bias|
   < 5 ppb and FAC2 > 0.5 on LOSO-SY.)

---

## 10. Quick reference — current code touchpoints

- `dispersion/gaussian.py` — point kernel `gaussian_plume_concentration()`,
  `SOURCES` (3 zones), `CANDIDATE_SOURCES` (16), forward-model variants, near-
  field floor in `pg_sigmas()`.
- `dispersion/emission_inversion.py` — `CHANNEL_WAYPOINTS`, `build_channel_grid()`,
  `build_sensitivity_matrix()`, `solve_nnls()`, `calibration_loop()`,
  `batch_inversion_stacked()`.
- `dispersion/lagrangian.py` — backward footprint, `GRID`, `SENSORS`,
  `CANDIDATE_SOURCES` (mirror).
- `constants.py` — `DISPERSION_DEFAULT_EMISSION_RATES_GS` (east=20/west=10/
  south=137), `EMISSION_RATES_PATH`.
- `defs/h2s_dispersion_pipeline.py` — `emission_rate_inversion`,
  `gaussian_forward_forecast`, `_ZONE_MAP`.
- `training/calibration_eval.py` — existing calibration-aligned evaluation
  harness to extend for §6.5 metrics.

**Sensors:** NESTOR-BES `32.567097, -117.090656` · IB CIVIC CTR
`32.576139, -117.115361` · **SAN YSIDRO `32.552794, -117.047286`** (overpredicted).
</content>
</invoke>
