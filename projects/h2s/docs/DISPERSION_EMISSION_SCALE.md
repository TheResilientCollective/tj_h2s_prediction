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
