# Dispersion Model — Source Attribution & Gaussian Forecasting

## Overview

The dispersion model complements empirical H2S regression/classifiers with physics-based source attribution and forward plume forecasting.

**Two Complementary Systems:**

1. **Lagrangian Inversion** — Backward particle tracking to identify where H2S is coming from
2. **Gaussian Forward Forecast** — Predict plume from known sources using wind forecast

**Purpose:** Enable source-aware forecasting, identify emission hotspots, and validate empirical predictions against physics.

---

## System 1: Lagrangian Backward Inversion

### Run Schedule

**Job:** `dispersion_inversion_job`
- **Frequency:** Weekly, Monday 02:30 UTC (STOPPED by default in Phase 6)
- **Purpose:** Compute emission rates from 16 candidate sources based on wind patterns
- **Window:** 2-hour backward integration (tuned for Tijuana Valley scale)

### Physics: Backward Particle Tracking

For a given event with high H2S measurement at the sensor:

1. **Particle Release:** Release 100 air parcels at sensor location
2. **Backward Integration:** Trace each parcel backward 2 hours using observed wind fields
3. **Endpoint Distribution:** Particles spread across potential source regions
4. **Source Attribution:** % of particles landing near each source = contribution fraction

### Why 2-Hour Integration?

- **Tijuana Valley Scale:** Sources are 1–7 km upwind (travel time: 8–37 min @ 3 m/s)
- **6-Hour Integration (Outdated):** Particles travel 64 km — too far, misses local sources
- **2-Hour Integration (Current):** Particles travel ~21 km — appropriate for valley scale
- **Empirical Validation:** 6h gave East=0%, 2h gives East=46% (matches emission patterns)

### Candidate Sources (16 Total)

Grouped into 3 operational zones:

| Zone | Sources | Key Sites | Emission Rate |
|------|---------|-----------|-----------------|
| **East** | 6 sources | Dairy Mart Bridge (dominant), Stewart's Drain, Silva Drain | 87.3 g/s (52%) |
| **West** | 3 sources | Tijuana River Beach Outlet, Oneonta Slough Near IB, crossings | 29.9 g/s (18%) |
| **South** | 7 sources | Goat Canyon, Smugglers Gulch, Del Sol Canyon, Hollister bridges | 49.8 g/s (30%) |

**Total:** 167 g/s (calibrated from March 13, 2026 event)

### Wind-Dependent Diffusion (Implemented)

**Key Finding:** H2S strongly anti-correlated with wind speed (r = -0.246): calm winds → high H2S.

**Implementation:**
```
Diffusion parameter σ ~ U^0.5

Where U = wind speed (m/s)

Calm winds (U=1 m/s): σ ≈ 0.21 m/s    → sharp plume, high precision
Strong winds (U=5 m/s): σ ≈ 0.34 m/s  → broad plume, low precision
```

**Effect:** During calm events (high H2S), plume is sharper → sources correctly identified. During strong winds (low H2S), plume is diffuse → attribution uncertain but less critical operationally.

**Validation:**
- Fixed σ=0.3 → East=45.6%
- Wind-dependent σ ~ U^0.5 → East=52.3% (6.7 pp improvement)

### Output: Emission Rates

```json
{
  "timestamp": "2026-06-24T00:00:00Z",
  "window_start": "2026-02-01T00:00:00Z",
  "window_end": "2026-04-01T00:00:00Z",
  "integration_hours": 2,
  "total_g_per_s": 167.0,
  "zones": {
    "east": {
      "emission_rate_g_s": 87.3,
      "percent": 52.3,
      "sources": [
        {"name": "Dairy Mart Bridge", "percent": 14.2},
        {"name": "Stewart's Drain", "percent": 12.5},
        ...
      ]
    },
    "west": {
      "emission_rate_g_s": 29.9,
      "percent": 17.9,
      "sources": [...]
    },
    "south": {
      "emission_rate_g_s": 49.8,
      "percent": 29.8,
      "sources": [...]
    }
  },
  "methodology": "Lagrangian backward 2h integration, wind-dependent diffusion",
  "wind_mean": 4.2,
  "wind_std": 1.8
}
```

---

## System 2: Gaussian Forward Forecast

### Run Schedule

**Job:** `dispersion_forecast_job`
- **Frequency:** Every 6 hours (00, 06, 12, 18 UTC) — RUNNING in Phase 6
- **Purpose:** Forecast 72-hour plume based on latest emission rates and weather forecast
- **Meteorology:** Forecast winds (not observations) — so plume is purely predictive

### Physics: 3D Gaussian Plume Model

For each source with emission rate Q (g/s):

```
Downwind Concentration (ppb):

C(x, y, z) = (Q / (2π * u * σ_y * σ_z))
             × exp(-(y²/2σ_y²))
             × [exp(-((z-h)²/2σ_z²)) + exp(-((z+h)²/2σ_z²))]

Where:
  x, y, z = downwind distance, crosswind, height
  u = wind speed
  σ_y, σ_z = lateral and vertical diffusivity (Pasquill-Gifford classes)
  h = stack height
```

### Superposition

For multiple sources:
```
C_total(x,y,z) = Σ C_source_i(x,y,z)
```

### Forecast Steps

1. **Load Latest Emission Rates** (from inversion or calibrated defaults)
   - East: 87.3 g/s
   - West: 29.9 g/s
   - South: 49.8 g/s

2. **Fetch Forecast Meteorology** (NOAA forecast winds for next 72h)
   - U (eastward wind), V (northward wind) per 3-hour step
   - Derive wind speed U_total = √(U² + V²)

3. **For Each Forecast Hour (1–72):**
   - Get wind at that hour
   - Compute plume footprint over the domain
   - Interpolate concentration at sensor locations (3 stations)

4. **Threshold Check:**
   - Next 6 hours: Any sensor > 30 ppb (watch)? Post Slack alert
   - Any sensor > 100 ppb (critical)? Escalate alert

5. **Output:**
   - `forward_forecast_{run_tag}.json` (concentrations per hour/station)
   - `forward_bundle_{run_tag}.zip` (HYSPLIT CONTROL files + shell scripts)
   - Latest mirror: `forward_bundle_latest.zip`

### Example Forecast Output

```json
{
  "run_timestamp": "2026-06-24T18:00:00Z",
  "forecast_window": [1, 72],
  "hours_until_watch_threshold_30": 12,
  "hours_until_critical_threshold_100": null,
  "stations": {
    "NESTOR__BES": [
      {
        "lead_hour": 1,
        "time": "2026-06-25T19:00:00Z",
        "concentration_ppb": 2.3,
        "threshold_exceeded": null
      },
      {
        "lead_hour": 13,
        "time": "2026-06-26T07:00:00Z",
        "concentration_ppb": 31.4,
        "threshold_exceeded": "watch (30 ppb)"
      },
      ...
    ]
  },
  "alert_status": "watch",
  "alert_message": "Gaussian plume forecast predicts H2S >30 ppb at NESTOR__BES in 12 hours (lead 13)"
}
```

### HYSPLIT Bundle Generation

Instead of executing HYSPLIT directly in Dagster, the pipeline generates portable CONTROL bundles:

```bash
# Download from S3
aws s3 cp s3://resilientpublic/tijuana/dispersion/hysplit/forward_bundle_latest.zip .
unzip forward_bundle_latest.zip

# Option 1: Run in local HYSPLIT container
docker run -v $(pwd):/hysplit geodemdocker/hysplit-rundeck \
  /hysplit/exec/hysplit.csh CONTROL

# Option 2: Submit to NOAA READY server
curl -F controlfile=@CONTROL https://www.ready.noaa.gov/exec/htdocs/arl/hysplit4_std.html
```

**Rationale:** HYSPLIT needs ~20 GB GDAS meteorology files. Bundling keeps Dagster lightweight.

---

## Comparison: Empirical vs Physics-Based

| Aspect | Empirical H2S Model | Dispersion Model |
|--------|-------------------|-----------------|
| **Input** | Meteorology + H2S history | Emissions + forecast wind |
| **Output** | h2s_pred, p5/p10/p30 | Forward plume, source zones |
| **Time Horizon** | 24h (signal decays) | 72h (Gaussian stable) |
| **Physical Basis** | Machine-learned patterns | Gaussian plume physics |
| **Validation** | Measured vs predicted H2S | Observed concentrations vs plume |
| **Use Case** | Nowcast/alert cascade | Source ID + long-range forecast |
| **Failure Mode** | Misses anomalies, overfits | Fails if emissions misspecified |

**Integration:** Use both. Empirical model gives alert fast (familiar meteorology). Dispersion model provides physics validation and source tracking.

---

## Current Calibration & Tuning

### Emission Rate Calibration

Emission rates were derived from March 13, 2026 extreme event (observed >100 ppb at NESTOR__BES):

1. **Observed concentrations** at 3 stations during event
2. **Backward trajectory** from NESTOR to identify upwind sources
3. **Inverse Gaussian plume** fitting to derive source strengths
4. **Result:** 167 g/s total (East 52%, West 18%, South 30%)

**Stability:** These rates are held fixed in forward forecasts. Real-world emission rates vary (e.g., seasonal SBIWTP discharge, tide-driven seepage). Periodic recalibration (monthly-quarterly) recommended.

### Diffusivity Classes (Pasquill-Gifford)

Wind-dependent σ implementation currently uses empirical fit (σ ~ U^0.5). Could be refined with:
- Time-of-day (day/night stability classes)
- Cloud cover (mixing height proxy)
- Atmospheric pressure (stability indicator)

---

## Operational Use

### Use Case 1: Forecast Validation

```
"Empirical model predicts h2s_pred=22 ppb (Tier 2 alert)"
"Gaussian forecast shows East source at 40 ppb, West at 5 ppb"
→ Conclusion: Alert is plausible; East dominates
```

### Use Case 2: Source Attribution

```
"High H2S measured at SAN_YSIDRO but not NESTOR"
"Lagrangian backtrack shows South sources dominant"
→ Conclusion: Goat Canyon / Smugglers Gulch upwind of SAN_YSIDRO
→ Action: Notify environmental health of South zone sources
```

### Use Case 3: Extended Forecast

```
"Empirical forecast skill decays after lead 18 (skill ceiling)"
"Gaussian forecast extends to 72h; can rank risk 3+ days out"
→ Use Gaussian for long-range planning; empirical for < 24h operations
```

---

## Limitations & Caveats

1. **Emission Rates Fixed:** Real-world rates change with tide, season, SBIWTP discharge. Monthly recalibration needed.

2. **No Chemistry:** Gaussian plume assumes conservative transport (H2S doesn't degrade). Reality: H2S oxidizes over hours. Plume will be sharper than predicted.

3. **Meteorology Uncertainty:** Forecast winds have their own uncertainty. Wind error → plume position error.

4. **Point Source Assumption:** Real sources are spatially distributed (river miles, multiple channels). Modeling as point at "representative" location is approximation.

5. **No Vertical Mixing:** Mixing height not explicitly modeled. Tall plumes will affect ground concentrations less than predicted.

---

## Integration with Alert System

**Daily Workflow (current state in Phase 6):**

1. **Station Forecast Job (every 6h):** 
   - Empirical H2S regression + classifiers
   - Produces nowcast/nearcast/forecast (24h)
   - Powers Tier 1–3 cascade alerts

2. **Dispersion Forecast Job (every 6h):**
   - Gaussian forward forecast
   - Checks next 6h for >30 ppb
   - Posts independent Slack alert if threshold crossed
   - Uploads HYSPLIT bundles for user review

3. **Alert Performance Sensor (5-min poll):**
   - Observes measured H2S
   - Opens/closes alert events
   - Posts forecast-vs-measured close-out to Slack

**Flow:**
```
Empirical (fast, 24h) ──→ Primary Alert System
                         │
Dispersion (physics, 72h) → Validation & Source ID
                         │
Observation (real-time) → Performance Tracking
```

---

## Future Enhancements

1. **Time-Varying Emissions:** Calibrate hourly or daily emission rates from Lagrangian inversions
2. **Chemistry Module:** Add H2S oxidation kinetics (Pasquill-Gifford classes)
3. **ML-Optimized Diffusion:** Learn σ profile from measured plume spreads
4. **3D Topography:** Include terrain effects (valley funnel, elevation)
5. **Ensemble Gaussian:** Multiple meteorology ensemble members → plume confidence intervals
6. **Operational Integration:** Directly trigger alerts from Gaussian forecast at lower skill threshold (e.g., lead 36–72)

---

## References

- **Current Calibration:** March 13, 2026 extreme event analysis
- **Wind-Dependent Diffusion:** See `WIND_SPEED_DEPENDENCY.md`
- **Lagrangian Details:** `dispersion/lagrangian.py`
- **Gaussian Implementation:** `dispersion/gaussian.py`
- **HYSPLIT Controls:** `dispersion/hysplit_controls.py`
