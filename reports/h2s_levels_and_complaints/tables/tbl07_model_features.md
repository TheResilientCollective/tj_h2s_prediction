| Family | Feature | In Lean | Source | Description |
|---|---|---|---|---|
| H2S history | `h2s_lag_1h` | yes | SD APCD monitor | H2S one hour earlier (ppb). The single most informative feature in every model; it is also why forecast skill decays once the recursion runs past the observed record. |
| H2S history | `h2s_lag_3h` | yes | SD APCD monitor | H2S three hours earlier (ppb). |
| H2S history | `h2s_lag_6h` | yes | SD APCD monitor | H2S six hours earlier (ppb). |
| H2S history | `h2s_rolling_24h` | yes | SD APCD monitor | 24 h rolling mean H2S (ppb) — the multi-night episode signal. |
| H2S history | `h2s_rolling_6h` | yes | SD APCD monitor | 6 h rolling mean H2S (ppb) — the event-in-progress signal, and the top feature for the ≥10 and ≥30 classifiers. |
| wind | `wind_direction_cos` | yes | derived | cos of wind direction, paired with the above. |
| wind | `wind_direction_sin` | yes | derived | sin of wind direction. Direction is split into sin/cos so the model sees it as a circle rather than a number that jumps at 360°. |
| wind | `wind_gusts_10m` | — | Open-Meteo | Wind gusts at 10 m (m/s). Evidence only. |
| wind | `wind_gusts_10m_max_2h` | — | derived | 2 h rolling maximum gust — a mixing-event marker. Evidence only. |
| wind | `wind_gusts_10m_max_3h` | — | derived | 3 h rolling maximum gust. Evidence only. |
| wind | `wind_speed_10m` | yes | Open-Meteo | Wind speed at 10 m (m/s). Calm nights ventilate the valley poorly; this is the dominant dispersion term. |
| wind | `wind_speed_10m_avg_2h` | — | derived | 2 h rolling mean wind speed — recent ventilation history. Evidence only. |
| wind | `wind_speed_10m_avg_3h` | — | derived | 3 h rolling mean wind speed. Evidence only. |
| regime | `humidity_temp_interaction` | — | derived | humidity × temperature. Evidence only, same argument. |
| regime | `is_night` | yes | derived | 1 between sunset and sunrise. Consistently a top-3 classifier feature, because the hazard is almost entirely nocturnal (Figure 4). |
| regime | `source_regime` | yes | derived | Coarse source regime: night flag crossed with the wind-direction quadrant, i.e. which part of the valley is upwind. |
| regime | `stable_atm` | yes | derived | Binary stable-atmosphere flag (calm and at night) — the nocturnal-inversion proxy that traps the plume near the ground. |
| regime | `wind_temp_interaction` | — | derived | wind_speed × temperature. Evidence only; dropped from Lean on the argument that a tree learns the product implicitly. |
| regime | `wind_x_stable_atm` | — | derived | wind_speed × stable_atm. Evidence only. |
| weather | `cloud_cover` | — | Open-Meteo | Total cloud cover (%), a proxy for overnight radiative cooling. Evidence only. |
| weather | `dewpoint_2m` | — | Open-Meteo | Dew point at 2 m (°C). Evidence only. |
| weather | `precipitation` | — | Open-Meteo | Hourly precipitation (mm). Evidence only. |
| weather | `relative_humidity_2m` | yes | Open-Meteo | Relative humidity at 2 m (%). |
| weather | `surface_pressure` | — | Open-Meteo | Surface pressure (hPa). Evidence only. |
| weather | `temperature_2m` | yes | Open-Meteo | Air temperature at 2 m (°C). The strongest single exogenous predictor, but see §4 — it acts as a marker for the season rather than a lever within one. |
| time | `hour_cos` | — | derived | cos of hour-of-day. Evidence only. |
| time | `hour_sin` | — | derived | sin of hour-of-day. Evidence only; Lean drops it as redundant with is_night. |
| time | `month_cos` | yes | derived | cos of month, paired with the above. |
| time | `month_sin` | yes | derived | sin of month — the seasonal cycle documented in §4 enters the model through this pair. |
| water | `flow_lag_6h` | yes | IBWC border gauge | Tijuana River flow 6 h earlier (m³/s) — travel time from the border reach to the monitor. **Stuck at a constant since 2026-01; see §3.4.** |
| water | `flow_rolling_24h` | yes | IBWC border gauge | 24 h rolling mean river flow (m³/s). **Same feed, same problem.** |
| water | `tidal_state_encoded` | yes | derived | Ordinal encoding of ebb / flood / high / low. |
| water | `tide_height` | yes | NOAA tides | Tide height (m) at the estuary mouth. |
