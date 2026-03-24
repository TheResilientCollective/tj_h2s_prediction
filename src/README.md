# Standalone Scripts (`src/`)

These scripts run independently of the Dagster pipeline. They cover training, prediction, visualization, and daily source-attribution analysis for H2S at the Tijuana River Valley monitoring stations.

## Execution Order

```
1. train_models_auto.py        Train RF/XGB models per station
        |
        | outputs .pkl model files
        v
2a. predict_h2s.py             Single-file prediction (XGBoost 3-class)
      or batch_predict.py       Multi-file batch prediction (wraps predict_h2s)
2b. h2s_daily_analysis.py      Source attribution + 48h forecast (uses .pkl models)
        |
        v
3. generate_visualizations.py  Feature importance, confusion matrix, comparison plots
```

Steps 2a and 2b are independent -- run either or both depending on the use case. Step 3 is optional and can run at any point (only needs the model file; predictions/actuals are optional).

---

## Scripts

### 1. `train_models_auto.py` -- Train models

Trains XGBoost and Random Forest for each station x task (regression, >5 ppb classifier, >10 ppb classifier). Auto-selects the best algorithm per combination; creates a weighted ensemble when two algorithms are within the margin threshold.

```bash
python src/train_models_auto.py \
  --obs modeldata_h2s_nofill.parquet \
  --models ./models \
  --train-fraction 0.8 \
  --ensemble-margin 0.01
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--obs` | (required) | Path to `modeldata_h2s_nofill.parquet` (alias: `--data`) |
| `--models` | `./models` | Directory for output model `.pkl` files (alias: `--output`) |
| `--train-fraction` | `0.8` | Temporal train/test split ratio |
| `--ensemble-margin` | `0.01` | AUC/R2 margin below which an ensemble is created |

**Inputs:** `modeldata_h2s_nofill.parquet` with H2S, weather, tidal, and flow columns.

**Outputs (per station):**
- `best_reg_{STATION}.pkl` -- regression model
- `best_clf_5ppb_{STATION}.pkl` -- >5 ppb classifier
- `best_clf_10ppb_{STATION}.pkl` -- >10 ppb classifier
- `training_report.json` -- metrics, algorithm selections, feature importances

Station keys: `SAN_YSIDRO`, `NESTOR__BES`, `IB_CIVIC_CTR`.

---

### 2a. `predict_h2s.py` -- Single-file prediction

Generates H2S predictions using trained models from `train_models_auto.py`. Uses regression + binary classifiers to assign risk tiers (GREEN/YELLOW/ORANGE/RED), consistent with `h2s_daily_analysis.py`.

```bash
python src/predict_h2s.py \
  --input data.csv \
  --models ./models \
  --output ./results \
  --filter-alerts
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--input, -i` | (required) | Input CSV or parquet with environmental data |
| `--models` | (required) | Directory containing `best_*.pkl` files from `train_models_auto.py` |
| `--output, -o` | `.` | Output directory |
| `--prediction` | `prediction.csv` | Output filename |
| `--filter-alerts` | off | Only output ORANGE and RED predictions |
| `--site` | `NESTOR - BES` | Station (`NESTOR - BES`, `SAN YSIDRO`, `IB CIVIC CTR`) |

**Output columns added:** `h2s_predicted`, `prob_exceed_5ppb`, `prob_exceed_10ppb`, `risk` (GREEN/YELLOW/ORANGE/RED), `alert`.

**Risk tiers** (same as `h2s_daily_analysis.py`):

| Tier | Criteria |
|------|----------|
| GREEN | P(>5) < 25% and H2S < 5 ppb |
| YELLOW | P(>5) 25-50% or H2S 5-10 ppb |
| ORANGE | P(>5) > 50% or H2S 10-30 ppb |
| RED | P(>10) > 50% or H2S > 30 ppb |

---

### 2a. `batch_predict.py` -- All-station batch prediction

Runs predictions for all three stations in a single pass. Outputs one CSV per station plus a combined file.

```bash
python src/batch_predict.py \
  --obs data/model_forecast.csv \
  --models ./models \
  --output ./output
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--obs` | (required) | Input CSV or parquet with environmental data |
| `--models` | (required) | Directory containing `best_*.pkl` files |
| `--output, -o` | `./output` | Output directory |
| `--filter-alerts` | off | Only output ORANGE and RED predictions |

**Outputs:**
- `predictions_SAN_YSIDRO.csv`, `predictions_NESTOR__BES.csv`, `predictions_IB_CIVIC_CTR.csv` -- per-station
- `predictions_all_stations.csv` -- combined

**Dependency:** Imports from `predict_h2s.py`.

---

### 2b. `h2s_daily_analysis.py` -- Source attribution + 48h forecast

Performs three analyses in one run:
1. **Source attribution** (last 7 days) -- wind-alignment analysis with known source locations, Gaussian plume back-calculation of emission rates
2. **48-hour forecast** -- continuous H2S prediction, exceedance probabilities P(>5 ppb) and P(>10 ppb), risk tier assignment (GREEN/YELLOW/ORANGE/RED)
3. **Dashboard output** -- combined PNG, machine-readable JSON, detailed CSVs

```bash
python src/h2s_daily_analysis.py \
  --obs modeldata_h2s_nofill.parquet \
  --forecast model_forecast.parquet \
  --spills spills.csv \
  --models ./models \
  --output ./output \
  --lookback 7
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--obs` | (required) | Historical observation parquet |
| `--forecast` | (required) | Weather/tide/flow forecast parquet |
| `--spills` | (optional) | Spill event log CSV |
| `--models` | `.` | Directory with `.pkl` model files from `train_models_auto.py` |
| `--output` | `./output` | Output directory |
| `--lookback` | `7` | Days of lookback for source attribution |
| `--no-plot` | off | Skip PNG dashboard generation |

**Outputs:**
- `attribution.csv` -- hourly source attribution for lookback window
- `forecast.csv` -- 48h predictions with risk tiers
- `summary.json` -- machine-readable dashboard summary
- `dashboard.png` -- 5-panel visualization (unless `--no-plot`)

**Dependency:** Requires `.pkl` model files produced by `train_models_auto.py`.

**Risk tiers:**

| Tier | Criteria |
|------|----------|
| GREEN | P(>5) < 25% and H2S < 5 ppb |
| YELLOW | P(>5) 25-50% or H2S 5-10 ppb |
| ORANGE | P(>5) > 50% or H2S 10-30 ppb |
| RED | P(>10) > 50% or H2S > 30 ppb |

---

### 3. `generate_visualizations.py` -- Analysis plots

Generates feature importance and confusion matrix plots using trained `.pkl` models from `train_models_auto.py`. Runs all three stations by default.

```bash
# Feature importance for all stations
python src/generate_visualizations.py --models ./models --output ./reports

# With confusion matrix (requires predictions + actuals)
python src/generate_visualizations.py --models ./models \
  --predictions predictions.csv --actuals actuals.csv --output ./reports

# Single station
python src/generate_visualizations.py --models ./models --site "NESTOR - BES"
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--models` | (required) | Directory containing `best_*.pkl` files |
| `--predictions, -p` | (optional) | CSV with `time` and `risk` columns |
| `--actuals, -a` | (optional) | CSV with `time` and `H2S` (ppb) columns |
| `--output, -o` | `.` | Output directory for plots |
| `--site` | all stations | Specific station (`NESTOR - BES`, `SAN YSIDRO`, `IB CIVIC CTR`) |

**Outputs (per station):**
- `feature_importance_{STATION}.png` -- 3-panel plot (regression, >5ppb clf, >10ppb clf)
- `confusion_matrix_{STATION}.png` -- requires both predictions and actuals

**Dependency:** Imports from `predict_h2s.py`.

---

## Data Files

| File | Format | Description | Used by |
|------|--------|-------------|---------|
| `modeldata_h2s_nofill.parquet` | Parquet | Historical H2S + weather + tides + flow | `train_models_auto`, `h2s_daily_analysis` |
| `model_forecast.parquet` | Parquet | 48-147h weather/tide/flow forecast | `h2s_daily_analysis`, `predict_h2s` |
| `spills.csv` | CSV | USIBWC spill event log (`Start Date`, `End Date`) | `h2s_daily_analysis` (optional) |
| `best_reg_{STATION}.pkl` | Pickle | Regression model per station | `predict_h2s`, `batch_predict`, `h2s_daily_analysis`, `generate_visualizations` |
| `best_clf_5ppb_{STATION}.pkl` | Pickle | >5 ppb classifier per station | `predict_h2s`, `batch_predict`, `h2s_daily_analysis` |
| `best_clf_10ppb_{STATION}.pkl` | Pickle | >10 ppb classifier per station | `predict_h2s`, `batch_predict`, `h2s_daily_analysis` |
| `training_report.json` | JSON | Training metrics and algorithm selections | reference |

---

## Automation

Run the daily analysis at 6 AM PST:

```cron
0 14 * * * cd /path/to/project && python src/h2s_daily_analysis.py --obs data/modeldata_h2s_nofill.parquet --forecast data/model_forecast.parquet --spills data/spills.csv --models models/ --output output/$(date +\%Y-\%m-\%d)/
```

Retrain monthly (or after major spill events):

```cron
0 2 1 * * cd /path/to/project && python src/train_models_auto.py --obs data/modeldata_h2s_nofill.parquet --models models/
```
