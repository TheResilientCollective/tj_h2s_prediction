# H2S Prediction System — Model Overview

The Tijuana River H2S prediction system combines machine learning forecasts
with atmospheric dispersion modeling to predict
hydrogen sulfide levels at three monitoring stations
(IB Civic Center, Nestor BES, San Ysidro).
The system operates both reactive forecasting pipelines and physics-based source attribution
to support air quality management in the Tijuana River Valley.

## Modeling Strategies

| Goal | Strategy | Description |
|------|----------|-------------|
| Forecast | Ensemble Multi-Site XGBoost | Predict H2S categories (green/yellow/orange) at three stations using averaged probabilities from XGBoost and Random Forest variants |
| Forecast | Multi-Horizon Forecast | 72-hour predictions at 0–6h, 6–24h, 24–48h, and 48–72h horizons using 36 station × task × horizon models |
| Source Determination | Backward Trajectory | Lagrangian particle model tracking 2000 particles backward from sensors to identify source locations, weighted by observed H2S |
| Source Emission | Backward Dispersion | Bayesian inversion of Lagrangian footprints to estimate per-zone emission rates (g/s) for east, west, and south source regions |
| Forecast | Forward Dispersion | Gaussian plume model (Pasquill-Gifford) producing 72-hour concentration forecasts using emission rates from backward dispersion |
| Calibration | Calibration Loop | Iterative refinement between backward and forward dispersion to converge on emission estimates *(planned — currently semi-static)* |
| Forecast | River Channel Dispersion | H2S transport model driven by effluent flow in the Tijuana River channel *(planned — not yet implemented)* |

## Pipeline Status

- **Operational:** Ensemble XGBoost (every 6h), Forward Dispersion (every 6h), Daily Multi-Station Forecasts (daily)
- **Operational (stopped):** Multi-Horizon Forecast, Backward Trajectory (weekly)
- **Planned:** Calibration Loop, River Channel Dispersion
