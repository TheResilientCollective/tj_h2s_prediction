# Dispersion Calibration Loop — Geometry Refactor & SY Overprediction Fix

**Status:** Phases 0–4 complete · Phase 5 pending · **Branch:** `claude/charming-bell-klqfj9`

---

## 1. Problem statement

The Gaussian forward model overpredicts H2S at **SAN YSIDRO** (`32.552794, -117.047286`), the easternmost sensor. Root cause: every east-zone H2S source was represented as a **point**, and SY sits 250–600 m from the nearest point (`tj_crossing_cdlp_e`). In stable class F at night, the Gaussian σ_y·σ_z at 300 m is tiny, so `C ∝ Q/(π·u·σ_y·σ_z)` spikes. The NNLS loads all of the east corridor's Q onto the single point nearest SY, which then massively overpredicts whenever the wind aligns.

**Fix:** distribute the east drain corridor's Q along a *line source*. No single sub-point can carry the whole corridor's Q, so the peak sensitivity `A[SY, segment]` drops by ~1/N_subpoints.

---

## 2. What's implemented (Phases 0–4)

### Phase 0 — Source geometry config + loader
**Files:** `dispersion/source_geometry.toml`, `dispersion/geometry.py`

`source_geometry.toml` is the single source of truth for all source positions. The format uses TOML (stdlib `tomllib`, no new dependency). Sources:

| Source | Geometry | Zone | q_prior |
|--------|----------|------|---------|
| `east_drain_corridor` | line (Dairy Mart → Silva → CDLP W → CDLP E) | east | 20 g/s |
| `river_main_stem_west` | line (Beach Outlet → Saturn → Hollister N/S) | west | 10 g/s |
| `smugglers_goat_canyon` | line (Smuggler's → Goat Canyon → Goat Canyon PS) | south | 137 g/s |
| `saturn_blvd_bridge` | point | west | 5 g/s |
| `imperial_beach_estuary` | polygon | west | 10 g/s |
| `oneonta_slough` | point | west | 0 g/s |
| `hollister_ps` | point | west | 0 g/s |

`load_source_geometry()` → `dict[source_id, SourceSpec]` where each `SourceSpec` has `sub_points: list[SubPoint]` with `(lat, lon, weight)`. Weights sum to 1 per source; `Q_sub = Q_source × weight`.

**Note:** `imperial_beach_estuary` polygon vertices are placeholders. Replace with real marsh/mudflat boundary before calibration runs that include IB.

### Phase 1 — Geometry-aware forward model
**Files:** `dispersion/gaussian.py`

- `run_forward_model_from_geometry(df, specs, source_q_g_s, ...)` — expands each source to sub-points once, sums Gaussian plume contributions. No change to the point kernel.
- `run_forward_model_gridded_from_geometry(...)` — same for GeoDemic grid output.
- Validated against the March 13 2026 calibration event (394 ppb @ NESTOR-BES). No peak regression.

### Phase 2 — Grouped sensitivity matrix + geometry NNLS
**Files:** `dispersion/emission_inversion.py`

- `build_sensitivity_matrix_from_geometry(specs, met_row, sensors, cfg)` → `A[n_sensors, n_sources]`. Each column sums all sub-point sensitivities for that source. Reduces from ~100 free segments to 6–10 named unknowns.
- `batch_inversion_from_geometry(events, specs, cfg)` — stacks event rows, solves NNLS.
- `project_footprint_to_sources(...)` — Lagrangian footprint → per-source weights (for Lagrangian-seeded prior).
- `calibration_loop_from_geometry(...)` — single-event residual iteration.

### Phase 3 — Per-sensor weighting + bias term + metrics
**Files:** `dispersion/emission_inversion.py`, `dispersion/calibration_metrics.py`

- `InversionConfig.sensor_row_weight = "equal_mass"` — each active sensor contributes equal total NNLS row weight regardless of how many qualifying timesteps it has.
- `InversionConfig.fit_sensor_bias = False` — optional non-negative per-sensor additive offset (exempt from L1), reported separately as `sensor_bias_ppb`.
- `calibration_metrics.py` — Chang & Hanna 2004 metrics: `compute_dispersion_metrics(obs, pred)` → `{mean_bias, FB, NMSE, FAC2}`, plus `compute_threshold_skill` (POD/FAR/CSI at 30/100 ppb), `score_inversion_result`, `loso_cross_validate`.

**All 90 tests pass** (`tests/test_inversion_geometry.py`, `tests/test_calibration_metrics.py`, `tests/test_geometry.py`).

### Phase 4 — Dagster wiring
**Files:** `defs/h2s_dispersion_pipeline.py`

`EmissionInversionConfig` on `emission_rate_inversion` asset:

| Field | Default | Purpose |
|-------|---------|---------|
| `use_geometry_nnls` | `False` | Enable geometry NNLS (off = legacy Lagrangian path) |
| `date_start` / `date_end` | `"2026-02-01"` / `"2026-04-01"` | Obs window for event selection |
| `h2s_threshold_ppb` | `30.0` | Minimum H2S to qualify an event |
| `max_events` | `0` (all) | Cap on events |
| `gauss_meandering_deg` | `20.0` | Wind meandering σ for Gaussian sensitivity matrix |
| `lambda_l1` | `0.3` | L1 sparsity in NNLS |
| `background_ppb` | `1.0` | Subtracted before building obs vector |
| `fit_sensor_bias` | `False` | Non-negative per-sensor offset (absorbs IB local signal) |
| `require_anchor_sensor` | `""` | e.g. `"NESTOR - BES"` — drops events where this sensor is below threshold |
| `anchor_wd_gate` | `False` | For solo non-anchor events, require wind from [30, 270]° (FROM south/east) |

When `use_geometry_nnls=True`, `emission_rates.json` gains `emission_rates_by_geometry_g_s` (per source) and `geometry_inversion_meta` (n_events, RMSE, bias, filter_counts). The legacy `emission_rates_g_s` (3-zone) key is preserved.

`gaussian_forward_forecast` prefers geometry rates when `emission_rates_by_geometry_g_s` is present; falls back to 3-zone transparently.

---

## 3. Calibration runbook

### Step 1 — Baseline run (no filters)

```bash
cd projects/h2s
uv run dg launch --job dispersion_inversion_job \
  --config-json '{"ops":{"h2s__emission_rate_inversion":{"config":{
    "use_geometry_nnls": true,
    "date_start": "2026-02-01",
    "date_end": "2026-04-01",
    "h2s_threshold_ppb": 30.0
  }}}}'
```

Check `geometry_inversion_meta` in `emission_rates.json`:
- `n_events` — how many qualifying timesteps
- `sensor_rmse_ppb` — RMSE per sensor; SY should be lower than baseline
- `active_sources` — which sources got non-zero Q

### Step 2 — LOSO diagnostic (the SY-overprediction probe)

Run `loso_cross_validate` from `calibration_metrics.py` in a notebook or script:

```python
from h2s.dispersion import (
    load_source_geometry, batch_inversion_from_geometry,
    loso_cross_validate, InversionConfig,
)
# Load events from obs parquet (long format: time, site_name, H2S, wind_*)
# ...
specs = load_source_geometry()
cfg = InversionConfig(lambda_l1=0.3, background_ppb=1.0)
loso = loso_cross_validate(events, specs, cfg, source_ids=None,
                           thresholds=[30.0, 100.0],
                           sensor_names=["NESTOR - BES", "IB CIVIC CTR", "SAN YSIDRO"])
# loso["SAN YSIDRO"]["metrics"]["mean_bias"] should be close to 0
```

**Target:** `|mean_bias| < 5 ppb` and `FAC2 > 0.5` on LOSO-SY.

### Step 3 — IB calibration flags (if IB-without-NESTOR events are inflating west sources)

```bash
uv run dg launch --job dispersion_inversion_job \
  --config-json '{"ops":{"h2s__emission_rate_inversion":{"config":{
    "use_geometry_nnls": true,
    "fit_sensor_bias": true,
    "require_anchor_sensor": "NESTOR - BES",
    "anchor_wd_gate": true
  }}}}'
```

Interpretation:
- `sensor_bias_ppb["IB CIVIC CTR"]` large (> 10 ppb) → real local IB source geometry doesn't know about; bias term is doing real work. Consider adding a local IB point source to the TOML.
- `filter_counts.dropped_anchor_filter` large → many IB-only events were present; check whether removing them reduces or increases SY bias.
- LOSO-SY `mean_bias` improves → IB-only events were dragging Q_west up, causing SY overprediction via geometry correlation.

### Step 4 — Iterate on geometry

If LOSO-SY bias remains high after Step 3:

1. **Check `east_drain_corridor` vertices** — confirm CDLP E is in the right place (~117.050°W). If SY is still close to a sub-point, the 100 m near-field floor in `pg_sigmas()` may still be active for that sub-point.
2. **Add sub-point spacing** — reduce `spacing_m` on `east_drain_corridor` from 150 m to 75 m to spread Q more finely.
3. **Replace `imperial_beach_estuary` placeholder** — the current polygon is a rough box. Replace with real mudflat vertices; wrong polygon geometry drives IB predictions wrong and pulls the inversion off.
4. **Fold `oneonta_slough` into a line** — oneonta is the most likely driver of IB-without-NESTOR events. A short line (oneonta mouth → IB estuary edge) may attribute IB events more accurately than a single point.

### Step 5 — Forward forecast validation

```bash
uv run dg launch --job dispersion_forecast_job
```

The geometry forward model runs automatically once `emission_rates_by_geometry_g_s` is present. Check:
- `peak_ppb_SY` in asset metadata vs observed SY H2S at the same time
- Source map visualization (uploaded to S3) should show distributed corridor emission, not a single point

---

## 4. Open items (Phase 5 and field work)

| Item | Priority | Notes |
|------|----------|-------|
| Replace `imperial_beach_estuary` placeholder polygon | High | Current vertices are a rough box; wrong geometry distorts IB west-source attribution |
| Confirm `east_drain_corridor` CDLP E vertex | High | Verify in field: is `32.542166, -117.050325` the correct crossing? |
| HYSPLIT bundle: expand lines/polygons to sub-points | Medium | `hysplit_controls.py` currently uses single point per source; wire `load_source_geometry()` so sub-points generate multiple LOCATION records |
| Tide modulation of estuary Q | Low | Set `tide_modulated: true` in TOML + join `tidal_state` into the event met row |
| LOSO results table in S3 | Low | `score_inversion_result` is in the library; add it to `emission_rate_inversion` output |
| Fold `oneonta_slough` into a line | Low | Depends on Step 4 LOSO diagnosis |

---

## 5. Key code locations

| What | File | Symbol |
|------|------|--------|
| Geometry config | `dispersion/source_geometry.toml` | — |
| Geometry loader | `dispersion/geometry.py` | `load_source_geometry()` |
| Point kernel | `dispersion/gaussian.py` | `gaussian_plume_concentration()` |
| Geometry forward model | `dispersion/gaussian.py` | `run_forward_model_from_geometry()` |
| Sensitivity matrix | `dispersion/emission_inversion.py` | `build_sensitivity_matrix_from_geometry()` |
| NNLS solver | `dispersion/emission_inversion.py` | `solve_nnls()`, `batch_inversion_from_geometry()` |
| Metrics + LOSO | `dispersion/calibration_metrics.py` | `loso_cross_validate()`, `score_inversion_result()` |
| Dagster wiring | `defs/h2s_dispersion_pipeline.py` | `emission_rate_inversion`, `gaussian_forward_forecast` |
| Emission rates S3 path | `constants.py` | `EMISSION_RATES_PATH` |

**Sensors:** NESTOR-BES `32.567097, -117.090656` · IB CIVIC CTR `32.576139, -117.115361` · **SAN YSIDRO `32.552794, -117.047286`** (the overpredicted sensor)
