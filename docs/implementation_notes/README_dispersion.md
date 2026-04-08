# H2S Source Attribution & Dispersion Modeling
## Tijuana River Valley — San Diego APCD Monitoring Network

---

## Overview

This package implements three approaches to H2S source attribution and forward
dispersion forecasting for the Nestor/Imperial Beach/San Ysidro sensor network.

### Files

| File | Purpose |
|------|---------|
| `generate_hysplit_controls.py` | Generates HYSPLIT CONTROL files from modeldata |
| `lagrangian_backward.py` | Backward Lagrangian particle model (pure Python) |
| `gaussian_forward.py` | Gaussian plume forward model + Dagster asset |
| `dispersion_pipeline.py` | Full Dagster pipeline: inversion → forecast → alert |
| `hysplit_configs/CONTROL.traj_backward_template` | HYSPLIT backward trajectory template |
| `hysplit_configs/CONTROL.disp_backward_template` | HYSPLIT backward dispersion template |
| `hysplit_configs/CONTROL.disp_forward_template` | HYSPLIT forward dispersion template |
| `hysplit_configs/SETUP.CFG` | Shared HYSPLIT configuration |

---

## Recommended Inversion Window

Use **February 1 – March 31, 2026** for emission rate inversion.

Selection criteria:
- High event density: 158 stable nocturnal events > 30 ppb (vs ~60 in earlier months)
- SBIWTP deficit conditions most pronounced: strongest source signal
- Wind direction variability sufficient to separate East/West/South sources
- Post-February 25 weather data: validate against flat-fill artifact before use

Hold out **April 1–14, 2026** for forward model validation (POD, RMSE against observed).

---

## Emission Rate Scale

A critical finding from model calibration:

**H2S emission rates for open sewage channels in this system are in the range of
10–500 g/s per source zone**, not the initial order-of-magnitude placeholder values.

- The March 13 event (394 ppb at NESTOR-BES) required Q_south ≈ 137 g/s to reproduce.
- This is consistent with literature for open impoundments: 1–10 g/m²/hr over
  a 500–5000 m² surface area (Yegneswaran et al. 1999, Zhang et al. 2008).
- Do not use the default g/hr placeholder values in `ForwardConfig` for production runs.
- Always derive Q from backward dispersion inversion first.

---

## Workflow

### Step 1: Run HYSPLIT backward trajectories (source region identification)

Requires: HYSPLIT binary `hyts_std`, GDAS 0.5° met files

```bash
# Generate CONTROL files for all stable events in inversion window
python generate_hysplit_controls.py \
    --mode backward_traj \
    --data modeldata_h2s_nofill.csv \
    --met_dir /path/to/gdas/ \
    --output ./hysplit_runs/

# Run all trajectories (in your HYSPLIT container)
bash ./hysplit_runs/run_hysplit_backward_traj.sh
```

Output: `tdump_traj_*` files → use HYSPLIT trajectory cluster tool or
read with `hysplit_utils.py` (pysplit library recommended).

### Step 2: Run Lagrangian backward model (source footprints, no HYSPLIT needed)

```bash
python lagrangian_backward.py \
    --data modeldata_h2s_nofill.csv \
    --output ./lagrangian_output/ \
    --date_start 2026-02-01 \
    --date_end 2026-04-01 \
    --h2s_min 30 \
    --n_particles 2000 \
    --hours_back 6
```

Output: `footprint_ensemble.npy`, `source_attribution_ensemble.json`

### Step 3: HYSPLIT backward dispersion (emission rate inversion)

```bash
python generate_hysplit_controls.py \
    --mode backward_disp \
    --data modeldata_h2s_nofill.csv \
    --met_dir /path/to/gdas/ \
    --output ./hysplit_runs/

bash ./hysplit_runs/run_hysplit_backward_disp.sh
```

Post-process: for each event, divide observed concentration by cdump footprint value
at source location to estimate Q. Aggregate over all events using median (robust to
outliers). See `emission_inversion.py` (TODO — next step).

### Step 4: Run Gaussian forward model

```bash
# Standalone (emission rates in g/s from inversion)
python gaussian_forward.py \
    --data modeldata_h2s_nofill.csv \
    --emission_rates east=20,west=10,south=137 \
    --start "2026-04-01 06:00" \
    --hours 72

# Or via Dagster (reads emission_rates.json written by inversion step)
dagster asset materialize -f dispersion_pipeline.py \
    --select gaussian_forward_forecast
```

### Step 5: HYSPLIT forward dispersion

```bash
python generate_hysplit_controls.py \
    --mode forward \
    --met_dir /path/to/nam12/ \
    --forward_start "2026-04-01T06:00:00Z" \
    --emission_rates east=20,west=10,south=137 \
    --forward_hours 72 \
    --output ./hysplit_runs/

bash ./hysplit_runs/run_hysplit_forward_disp.sh
```

---

## Dagster Integration

Add to your existing `Definitions`:

```python
from dispersion_pipeline import dispersion_assets, dispersion_schedules

defs = Definitions(
    assets=[
        *existing_assets,
        *dispersion_assets,    # adds 4 assets
    ],
    schedules=[
        *existing_schedules,
        *dispersion_schedules, # weekly inversion, hourly forecast
    ],
)
```

Required env vars (add to `.env` alongside existing h2s_alerts vars):
```
H2S_DATA_PATH=/path/to/modeldata_h2s_nofill.parquet
H2S_ALERT_STATE_PATH=/path/to/alert_state/
SLACK_WEBHOOK_WATCH=https://hooks.slack.com/...
SLACK_WEBHOOK_CRITICAL=https://hooks.slack.com/...
```

---

## HYSPLIT Met Data

Download GDAS 0.5° archive from NOAA ARL:
```
https://www.ready.noaa.gov/gdas1.php
```
File naming: `gdas1.{mon}{yy}.w{N}` where N=week-of-month (1–4/5).

For real-time / 12h forecast: use NAM12 (12km CONUS) from:
```
https://www.ready.noaa.gov/READYnam.php
```

HYSPLIT binary must be in `$HYSPLIT_HOME/exec/`. Container setup:
```bash
export HYSPLIT_HOME=/opt/hysplit
export PATH=$HYSPLIT_HOME/exec:$PATH
```

---

## Model Validation Notes

Forward model calibration (March 13 2026, NESTOR-BES):
- Observed peak: 394.0 ppb
- Predicted (Q_south=137.5 g/s, meandering correction): 400.8 ppb  ✓
- Hour-2 observed: 96.8 ppb | predicted: 83.5 ppb  ✓
- Hour-3 observed: 57.6 ppb | predicted: 8.4 ppb  ✗ (wind shift not captured)

The Gaussian plume model captures the initial peak well but misses the
decay after wind shifts. The time-varying met update (hourly) helps but
Gaussian steady-state assumption breaks down during wind direction changes.
HYSPLIT forward dispersion will handle this better.

Lagrangian ensemble top sources (5-event test, March 13–15, stable >100 ppb):
  hollister_ps: 25%  |  hollister_bridge_s: 21%  |  hollister_bridge_n: 19%
  saturn_blvd_bridge: 13%  |  smugglers_gulch: 7%
→ Consistent with South/West source regime under southerly winds.

---

## Known Limitations

1. **Met data temporal resolution**: 1h met updates miss sub-hourly wind shifts.
   This is the dominant source of forward model error after the first 1–2 hours.

2. **Source geometry**: Zone centroids are approximations. Actual emission surfaces
   are linear (channels) not point sources. Multi-point or line-source extension
   recommended for production.

3. **Lagrangian met interpolation**: Currently uses sensor-collocated met only.
   Spatial interpolation between sensors would improve accuracy for long backward runs.

4. **Feb 25+ weather data**: Flat-fill artifact in some obs data versions. Validate
   wind columns before including those days in the inversion window.

5. **Emission rates**: Must come from inversion. Placeholder defaults (g/hr) in
   `ForwardConfig` will give non-physical (very low) concentrations.
