# H2S Source Attribution — Implementation Notes

## Overview

The daily analysis pipeline (`daily_analysis_job`) attributes observed H2S elevations to likely
geographic source locations using two complementary methods:

1. **Per-hour source assignment** — for each observation, identify which source the wind is blowing from
2. **Source probability grid** — aggregate recent observations into a geographic probability surface

Both methods use the last 7 days of hourly observations loaded from
`latest/tijuana/forecast_data/modeldata_h2s_nofill.parquet`.

---

## Monitoring Stations

| Station | Key | Lat | Lon |
|---------|-----|-----|-----|
| SAN YSIDRO | SAN_YSIDRO | 32.5528 | -117.0473 |
| NESTOR - BES | NESTOR__BES | 32.5671 | -117.0907 |
| IB CIVIC CTR | IB_CIVIC_CTR | 32.5761 | -117.1154 |

## Known Sources

17 candidate source locations are defined in `constants.py`, including:
Stewart's Drain, Smuggler's Gulch, Hollister St PS, Goat Canyon, Goat Canyon PS,
Del Sol Canyon, Silva Drain, Saturn Blvd Bridge, Hollister St Bridge (N/S),
Dairy Mart Bridge, Oneonta Slough, Tijuana River Beach Outlet, Tijuana River
Crossing (E/W), San Diego Bay Ponds (Otay River Outlet, near Fruitdale).

---

## Method 1 — Per-Hour Source Assignment

### Day/night filter

Daytime hours (06:00–20:00 local) are unconditionally labeled `"Daytime (mixed)"` and excluded
from source attribution. Daytime thermally-driven sea breezes make wind direction an unreliable
indicator of source origin during these hours.

### Bearing calculation

For a given station and candidate source, the bearing is computed using a flat-earth approximation:

```
bearing = arctan2(Δlon, Δlat) mod 360
```

(`_bearing_from` in `h2s_daily_pipeline.py`)

### Wind alignment test

The signed angular difference between observed wind direction and each source bearing is:

```
diff = ((wind_dir - bearing + 180) mod 360) − 180
```

The source with the **smallest absolute difference** is selected, provided that difference is within
`ALIGNMENT_THRESHOLD_DEG = 30°`. If no source qualifies, the hour is labeled `"Unaligned"`.

### Output columns

Each processed hour produces:

| Column | Description |
|--------|-------------|
| `aligned_source` | Matched source name, `"Unaligned"`, or `"Daytime (mixed)"` |
| `emission_rate_gs` | Back-calculated emission rate in g/s (null if unmatched or H2S ≤ 2 ppb) |

---

## Method 2 — Gaussian Plume Emission Rate Estimate

For elevated hours (H2S > 2 ppb) with a matched source, a simplified Gaussian plume
back-calculation estimates the emission rate Q (g/s).

Assumes **Pasquill-Gifford stability class D** (neutral, typical of nocturnal conditions):

```
C [µg/m³] = H2S_ppb × 1.42
σ_y = 0.08 × d / sqrt(1 + 0.0001 × d)
σ_z = 0.06 × d / sqrt(1 + 0.0015 × d)
Q [µg/s] = C × π × σ_y × σ_z × u
Q [g/s]  = Q [µg/s] / 1×10⁶
```

Where `d` is the straight-line distance (meters) from station to source, and `u` is wind speed
(clamped to 0.5 m/s minimum). Distance is computed using the equirectangular approximation with
a cosine correction for longitude.

This is a simplified single-point receptor model. The plume centerline is assumed to pass
directly through the station (i.e., the station is treated as on-axis). The estimate is
approximate and intended for qualitative source strength comparison, not regulatory use.

---

## Method 3 — Source Probability Grid

### Purpose

Produces the filled contour map shown in the daily dashboard ("Source Probability — Last 7 Days").
Rather than assigning each hour to a single source, this method casts a continuous spatial vote
across the geographic domain for every elevated observation, then accumulates them.

### Domain and resolution

- Latitude: 32.525° – 32.595° N
- Longitude: 117.135° – 117.025° W
- Grid resolution: 0.0006° (~67 m)

### Weighting scheme

For each observation hour with H2S > 1 ppb, every grid cell receives a weight:

```
weight = dw × cw × distw
```

**Angular weight `dw`** — Gaussian centered on the upwind direction (σ = 15°):
```
dw = exp(−0.5 × (angular_diff(cell_bearing, wind_dir) / 15)²)
```
Grid cells directly upwind of the station score highest; off-axis cells decay rapidly.

**Concentration weight `cw`** — log-scaled H2S concentration:
```
cw = log1p(H2S_ppb)
```
Higher readings contribute more without letting extreme events dominate.

**Distance weight `distw`** — Gaussian centered on the expected upwind travel distance:
```
transport_dist = (wind_speed × 3600) / 111000   [degrees, ~1h travel]
distw = exp(−0.5 × ((dist − transport_dist × 0.3) / 0.015)²)
```
Rewards grid cells at approximately the right upwind distance given current wind speed.

All votes are summed across all observations and all stations. The final grid is normalized to
[0, 1] by dividing by its maximum value.

### Contour thresholds

The dashboard draws filled contours (0.15–1.0) and three labeled isolines:

| Level | Color | Linewidth | Interpretation |
|-------|-------|-----------|---------------|
| 0.50 | `#ff9800` (orange) | 0.6 | Moderate source signal |
| 0.70 | `#ff5722` (deep orange) | 1.0 | High source signal |
| 0.90 | `#d50000` (dark red) | 1.5 | Very high source signal |

These are relative probability values (normalized to the peak signal in the 7-day window),
not absolute probabilities.

---

## Source Attribution Bar Chart

Shown in the right panel of row 2 of the daily dashboard ("Source Attribution 7d").

Counts **nighttime-only, wind-aligned, elevated (H2S > 5 ppb) hours** per source over the
lookback window. Unaligned and daytime hours are excluded. Top 6 sources by hour count are shown.
No emission-rate weighting is applied — the chart reflects frequency of alignment, not intensity.

---

## Limitations

- Day/night split uses a simple fixed hour threshold (06:00–20:00), not astronomical twilight.
- The bearing test assumes a station-level receptor on the plume centerline; lateral dispersion and receptor offset are ignored.
- Wind direction is taken from the forecast data (`wind_direction_10m`), which may not match local surface flow near terrain features.
- The 30° alignment threshold is fixed; actual plume spread depends on atmospheric stability and travel distance.
- The probability grid accumulates all observations equally in time; there is no temporal decay within the 7-day window.

---

## Key Parameters (constants.py)

| Constant | Value | Description |
|----------|-------|-------------|
| `ALIGNMENT_THRESHOLD_DEG` | 30° | Max angular deviation for source match |
| `WIND_COL` | `wind_direction_10m` | Wind direction column name |
| `SPEED_COL` | `wind_speed_10m` | Wind speed column name |
| Lookback window | 7 days (configurable via `lookback_days` op config) | |
| PG class D σ_y coefficient | 0.08 | Gaussian plume lateral spread |
| PG class D σ_z coefficient | 0.06 | Gaussian plume vertical spread |
| Probability grid σ_bearing | 15° | Angular spread of spatial vote |
| Probability grid σ_dist | 0.015° | Distance spread of spatial vote (~1.7 km) |
