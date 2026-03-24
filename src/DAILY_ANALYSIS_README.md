# H₂S Daily Source Attribution & Forecast System

## Quick Start

```bash
python h2s_daily_analysis.py \
  --obs modeldata_h2s_nofill.parquet \
  --forecast model_forecast.parquet \
  --spills spills.csv \
  --models ./models \
  --output ./output
```

## What It Does

The script performs three analyses in a single run:

### 1. Source Attribution (Last 7 Days)
For every measured hour at each station, the system:
- Computes the bearing from the station to each known source location
- Compares the observed wind direction to each source bearing
- Classifies the hour as "aligned" with a source if the angular difference is < 30°
- Back-calculates an emission rate (g/s H₂S) using a Gaussian plume model when alignment is confirmed
- Generates a geographic source probability map using concentration-weighted back-projection

### 2. Forward Model (48-Hour Forecast)
Using the forecast weather/tide/flow data and Random Forest models:
- Predicts continuous H₂S concentration at each station
- Predicts exceedance probabilities: P(>5 ppb) and P(>10 ppb)
- Assigns hourly risk tiers (GREEN/YELLOW/ORANGE/RED)
- Tags each nighttime hour with the most likely wind-aligned source

### 3. Dashboard & JSON Output
- Combined 5-row dashboard PNG showing observations, source map, attribution, forecasts, and probabilities
- Machine-readable JSON summary for web dashboard integration
- Detailed CSVs for both attribution and forecast results

## Requirements

```
pip install pandas numpy scikit-learn xgboost matplotlib pyarrow
```

Note: If `xgboost` is not installed, the training script falls back to Random Forest only. The daily analysis script works with any model type (RF, XGB, or Ensemble) since all are pickled with the same interface.

## Input Files

| File | Description | Update Frequency |
|------|-------------|-----------------|
| `modeldata_h2s_nofill.parquet` | Observation data with H₂S, weather, tides, flow | Daily (append new hours) |
| `model_forecast.parquet` | Weather/tide/flow forecast (next 48-147h) | Daily (regenerate) |
| `spills.csv` | USIBWC spill event log | As events occur |
| `best_reg_*.pkl`, `best_clf_5ppb_*.pkl`, `best_clf_10ppb_*.pkl` | Auto-selected model files (9 total, XGB/RF/Ensemble per task) | Retrain periodically |

## Output Files

| File | Description |
|------|-------------|
| `dashboard.png` | Combined visual dashboard |
| `attribution.csv` | Hourly source attribution for lookback window |
| `forecast.csv` | Hourly forecast predictions for all stations |
| `summary.json` | Machine-readable summary for web dashboards |

## JSON Summary Structure

```json
{
  "generated_at": "2026-03-19T22:27:30Z",
  "stations": {
    "NB": {
      "name": "NESTOR - BES",
      "last_h2s": 340.9,
      "mean_24h": 41.5,
      "max_24h": 340.9,
      "pct_exceed_5": 72.0
    }
  },
  "forecast_48h": {
    "NB": {
      "max_h2s": 126.8,
      "max_prob_5": 82.6,
      "hours_red": 28,
      "hours_orange": 13
    }
  },
  "active_sources": {
    "Smuggler's Gulch": {
      "aligned_hours": 61,
      "mean_h2s": 80.9,
      "median_emission_gs": 12.21
    }
  }
}
```

## Known Source Locations

| Source | Latitude | Longitude |
|--------|----------|-----------|
| Stewart's Drain | 32.54064 | -117.05801 |
| Smuggler's Gulch | 32.53770 | -117.08623 |
| Hollister St Pump Station | 32.54760 | -117.08837 |
| Goat Canyon | 32.53690 | -117.09916 |
| Goat Canyon Pump Station | 32.54348 | -117.10803 |
| Del Sol Canyon Collector | 32.53930 | -117.06885 |
| Silva Drain | 32.53974 | -117.06427 |

## Risk Tiers

| Tier | Criteria | Suggested Action |
|------|----------|-----------------|
| GREEN | P(>5) < 25% and H₂S < 5 ppb | Normal operations |
| YELLOW | P(>5) 25–50% or H₂S 5–10 ppb | Monitor closely |
| ORANGE | P(>5) > 50% or H₂S 10–30 ppb | Consider community notification |
| RED | P(>10) > 50% or H₂S > 30 ppb | Issue alert |

## Automation (cron)

Run daily at 6 AM PST:
```cron
0 14 * * * cd /path/to/project && python h2s_daily_analysis.py --obs data/modeldata_h2s_nofill.parquet --forecast data/model_forecast.parquet --spills data/spills.csv --models models/ --output output/$(date +\%Y-\%m-\%d)/
```

## Emission Rate Back-Calculation

Uses a simplified Gaussian plume model (Pasquill-Gifford stability class D):
- σy = 0.08x / √(1 + 0.0001x)
- σz = 0.06x / √(1 + 0.0015x)
- C = Q / (π · σy · σz · u) for ground-level source/receptor
- 1 ppb H₂S ≈ 1.42 µg/m³ at 20°C

Estimates carry factor-of-2–5 uncertainty but are useful for relative comparison across events and for order-of-magnitude total emission estimates.

## Model Training (Retraining)

When new observation data accumulates (recommended: monthly, or after major spill events):

```bash
python train_models_auto.py \
  --data modeldata_h2s_nofill.parquet \
  --output ./models \
  --ensemble-margin 0.01
```

This trains both XGBoost and Random Forest for each station × task (regression, >5ppb classification, >10ppb classification), evaluates on a time-series holdout (last 20%), and automatically selects the best algorithm per combination. When two algorithms are within the margin threshold, it creates a weighted ensemble.

Current model selections (March 2026):

| Station | Regression | >5 ppb Clf | >10 ppb Clf |
|---------|-----------|------------|-------------|
| San Ysidro | Ensemble (RF+XGB) | XGBoost | XGBoost |
| NESTOR-BES | XGBoost | XGBoost | XGBoost |
| IB Civic Ctr | Random Forest | XGBoost | Ensemble (RF+XGB) |

Key insight: XGBoost dominates classification across all stations (AUC 0.94–0.98). Random Forest wins IB Civic Center regression (R²=0.33 vs XGB's -1.07). Ensembles are selected when the gap is < 0.01 AUC or < 0.02 R².
