# Gaussian plume emission scale (`DISPERSION_EMISSION_SCALE`)

**Default: 0.15.** The Gaussian forward forecast (`gaussian_forward_forecast`,
`gaussian_forward_forecast_detailed`) multiplies whatever emission rates it loads
by this factor before evaluating the plume. The plume is linear in Q, so every
forecast ppb moves by the same factor. Override per deployment with the
`DISPERSION_EMISSION_SCALE` env var or per run with
`ForwardForecastConfig.emission_scale`; `1.0` restores the raw calibrated rates.
Every forecast payload records `emission_scale` and `emission_rates_raw_g_s`.

Re-fit with `scripts/calibrate_emission_scale.py` (read-only against S3).

## Why the forecast read high

All available rate sets are calibrated on acute events: the 3-zone defaults on
the March 13 2026 peak (394 ppb at NESTOR-BES) and the Lagrangian / NNLS
inversions on ≥30 ppb event windows, with the total held at 167 g/s. Under
ordinary meteorology the plume therefore reproduces event-level ppb.

## Evidence (fit of 2026-10-24)

779 stored coarse runs (2026-04-10 → 2026-10-23, all at 167 g/s), 15-min
forecast ppb averaged to hourly and joined to measured hourly H2S
(`modeldata_h2s_nofill.parquet`): 57,586 forecast-hours.

| | measured | raw forecast | ratio |
|---|---|---|---|
| mean ppb | 3.6 | 20.6 | 0.17 |
| p95 ppb | 10.4 | 78 | 0.13 |
| p99 ppb | 52.5 | 309 | 0.17 |
| hours ≥ 30 ppb | 1.7 % | 14.4 % | |

Median per-run predicted 24 h peak: 171 ppb (IB), 205 (NB), 344 (SY).
Median measured peak over the same windows: 2.2, 7.0, 7.6 ppb.

Uniform scale that makes the forecast exceed a threshold as often as the
measurements do:

| threshold | scale |
|---|---|
| 5 ppb | 0.11 |
| 10 ppb | 0.13 |
| 30 ppb (watch) | 0.14 |

**0.15** matches the ≥30 ppb exceedance frequency, the p95 ratio, and brings
overall mean bias to ≈ −0.5 ppb (SY +0.7, IB +2.0, NB −4.0).

## Per-zone scale (`DISPERSION_ZONE_SCALE`, default east 0.5 / west 1.0 / south 1.0)

A second multiplier composed on top of the global scale, per emission zone.
Applied to the zone-based rate sets (3-zone coarse rates, 16 per-source rates
via their zone, geometry sources via `SourceSpec.zone`); **not** applied to the
channel Q field, which is its own data-fitted calibration. Override with
`DISPERSION_ZONE_SCALE_{EAST,WEST,SOUTH}` or `ForwardForecastConfig.zone_scale`.
Every payload records `zone_scale`.

### Why east

SAN YSIDRO stayed high after the global 0.15 (latest run: SY peak 25 ppb, the
other two sensors under 3). Running the coarse model one zone at a time on
Apr–Oct 2026 measured met with the current inverted rates (east 49.5 / west
32.3 / south 85.2 g/s):

| share of forecast | SAN YSIDRO | NESTOR-BES | IB CIVIC CTR |
|---|---|---|---|
| east (Stewart's Drain / Dairy Mart, 1.6 km from SY) | 71 % of mean, 76 % of peak hours | 12 % | 3 % |
| west (Oneonta Slough) | 5 % | 33 % | 85 % |
| south (Goat Canyon) | 24 % | 55 % | 13 % |

The east emitter is essentially a SAN YSIDRO-only term, so scaling it is
surgical.

### Choosing 0.5 (with global 0.15; ratio = measured / forecast)

| east factor | SY mean ratio | SY p95 ratio | SY ≥10 ppb hours (fc / meas) | SY bias ppb | NB p95 ratio | IB p95 ratio |
|---|---|---|---|---|---|---|
| 1.0 | 0.45 | 0.59 | 0.094 / 0.052 | +2.9 | 1.93 | 0.18 |
| **0.5** | 0.71 | **0.99** | **0.050 / 0.052** | +1.0 | 2.14 | 0.18 |
| 0.3 | 0.91 | 1.35 | 0.044 / 0.052 | +0.2 | 2.32 | 0.18 |

0.5 makes SAN YSIDRO unbiased at p95 and matches its ≥10 ppb exceedance
frequency; 0.3 would zero the mean bias but under-predict the upper tail,
which is what the alert check keys on. NESTOR and IB move by < 3 %.

### What the same table says about the other zones

- IB CIVIC CTR is ~5× high and is 85 % **west** zone; NESTOR is ~2× low and
  is 55 % **south**. A joint non-negative fit of all three zone factors on
  each sensor's top-10 % measured hours gives roughly east 0.07, west 0.17,
  south 0.38 (on top of 0.15). Those are available through the same knob
  but were not applied here; the west/south adjustments trade IB false
  alarms against NESTOR misses and deserve a deliberate decision.

## Caveats the scale does not fix

- **Discrimination is poor regardless of scale.** Hourly Spearman is 0.21;
  at 30 ppb the false-alarm ratio stays ≥ 0.9 at every scale (hit rate 0.18 at
  1.0, 0.03 at 0.15). The scale corrects the bias, not the timing. Treat the
  dispersion forecast as a spatial-pattern product; event timing comes from the
  per-station ML products.
- **Per-sensor bias differs.** Measured/forecast mean ratio is 0.39 at NESTOR,
  0.12 at SAN YSIDRO, 0.07 at IB CIVIC CTR. A single scale is a compromise:
  0.15 is roughly unbiased at IB/SY and under-predicts NESTOR means. The spread
  points at zone split and geometry, not total Q (cf. the SY over-prediction
  diagnostic in `dispersion/calibration_metrics.py`).
- **Strong seasonality.** Measured/forecast ratio by month: 0.76 (Apr), 0.20
  (May), 0.25 (Jun), 0.09 (Jul), 0.04 (Aug), 0.05 (Sep), 0.11 (Oct). A fixed
  scale is too low in the wet season and too high in late summer. A
  flow- or temperature-dependent emission term (see
  `h2s_river_emissions_pipeline`) is the structural fix; until then, re-fit
  seasonally or raise the scale when river flow is high.
- **Grid product.** The gridded map already applies `GRID_BASELINE_SCALE`
  (0.1) for legibility; with the emission scale it now renders at 0.015 of
  raw. Set `GRID_BASELINE_SCALE = 1.0` if the map should match the sensor
  forecast.
