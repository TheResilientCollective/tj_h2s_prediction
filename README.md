# H2S Prediction System — Tijuana River Valley

ML-based forecasting of H₂S levels at three monitoring stations in the Tijuana River
border region — **NESTOR - BES** (Berry Elementary), **IB Civic Ctr**, and
**San Ysidro** — orchestrated via Dagster with S3-backed model registry.

Built and validated alongside the sibling repo
[tijuana-dispersion-experiments](https://github.com/TheResilientCollective/tijuana-dispersion-experiments)
(physics-based dispersion modeling + calibration arc); see [tj_calibration findings
folded in](#calibration-findings-and-experiments) below.

---

## What this system does

| Pipeline | Frequency | Output |
|---|---|---|
| `forecast_prediction_job` | every 6 h | 3-class hourly H₂S forecast (green / yellow / orange) for NESTOR-BES |
| `daily_analysis_job` | daily | Per-station 48 h forecasts + source attribution + Slack dashboard |
| `dispersion_forecast_job` | every 6 h | 72 h Gaussian plume forecast + alert check |
| `dispersion_inversion_job` | weekly (stopped) | Lagrangian backward source attribution |
| `multi_station_training_job` | on demand | Retrain per-station XGB models (3 sites × 3 tasks) |

Categories follow SD County H₂S guidance:

- 🟢 **Green:** H₂S < 5 ppb (safe)
- 🟡 **Yellow Low:** 5 ≤ H₂S < 10 ppb (caution; **complaint-rate band**, with 8 ppb a known complaint-trigger)
- 🟡 **Yellow High:** 10 ≤ H₂S < 30 ppb (caution)
- 🟠 **Orange:** H₂S ≥ 30 ppb (watch alert)
- 🔴 **Critical:** H₂S ≥ 100 ppb (calibration-extreme threshold; agency notification)

The system runs a **5-tier alert ladder** (`ALERT_TIERS` in `constants.py`): the
lower three tiers (`PLANT-SIGNAL`, `MULTI-SITE-RISK`, `EXCEEDANCE-RISK` at 5 / 10 / 30 ppb)
are forecast-based pre-alerts routed to an internal ops channel; the upper two
(`WATCH`, `CRITICAL` at 30 / 100 ppb) are observation-based exceedance alerts routed
to monitoring staff and agency decision-makers respectively.

---

## Model Performance

### Hourly classifier (NESTOR-BES, deployed)

| Metric | Value |
|---|---|
| Orange detection rate | 61.3 % |
| False alarm rate | 5.4 % |
| Balanced accuracy | 63.1 % |
| Algorithm | XGBoost 3-class, weighted |
| Features | 44 (`CORE_FEATURES` + `SBIWTP_FEATURES`) |

### Per-station daily regression (Berry, held-out 2025-10 → 2026-04)

Evaluated on the calibration-aligned harness (Spearman + recall at the four categorical thresholds):

| Model | Features | Spearman | recall@5 | recall@10 | recall@30 | recall@100 |
|---|---|---|---|---|---|---|
| persistence floor (`h2s_lag_1h → ppb`) | — | 0.822 | 0.702 | 0.650 | 0.574 | 0.390 |
| **current XGB regression** (Evidence — production) | **33** | **0.817** | 0.856 | 0.861 | 0.794 | **0.675** |
| **Lean** (deployed in parallel) | **19** | 0.805 | 0.831 | 0.809 | 0.735 | 0.571 |
| legacy XGB regression | 44 | 0.782 | 0.861 | 0.869 | 0.803 | 0.636 |

The 33-feature Evidence model is the production default; the 19-feature Lean model is **deployed in parallel** (S3 path suffix `_lean`, see below) as a not-overdetermined argument that reviewers can verify by loading either model and reproducing the comparison. Lean drops below the acceptance gate at recall@30/100 — that's why Evidence ships — but Lean is operationally close enough that the system clearly isn't dependent on the 14 features Lean strips out. The 44-feature legacy set is retained in code as `MODEL_FEATURES_LEGACY` for backward compat with previously-deployed models. See [experiments/2026-06-10_feature_trim_berry/RESULTS.md](experiments/2026-06-10_feature_trim_berry/RESULTS.md).

### Threshold tuning (hourly classifier)

| Setting | Orange threshold | Orange recall | False positives |
|---|---|---|---|
| Conservative | 0.40 | ~55 % | ~3 % |
| **Default** | **0.33** | **61 %** | **5.4 %** |
| Sensitive | 0.25 | ~70 % | ~10 % |
| Very sensitive | 0.20 | ~75 % | ~15 % |

---

## Quick start

```bash
cd projects/h2s
uv sync
cp .env.example .env   # fill in S3 credentials

# 1. Seed S3 with starter models (hourly + per-station)
uv run dg launch --job seed_models_job

# 2. Hourly forecast (auto-runs every 6 h)
uv run dg launch --job forecast_prediction_job

# 3. Daily analysis (auto-runs daily)
uv run dg launch --job daily_analysis_job

# 4. Dispersion forecast (auto-runs every 6 h)
uv run dg launch --job dispersion_forecast_job

# Dagster UI
uv run dg dev   # http://localhost:3000
```

`seed_models_job` uploads starter models for every pipeline:

- `data/startmodels/` → `tijuana/forecast/models/` (hourly pipeline)
- `data/models_v2/` → `tijuana/forecast/models/stations/` (per-station daily pipeline)

For the full operational runbook (rebuilding, deploying, HYSPLIT worker setup), see [CLAUDE.md](CLAUDE.md).

---

## Calibration findings and experiments

Several recent experiments folded findings from the calibration arc (sibling repo) into the production model and evaluation discipline:

| Experiment | What it answers |
|---|---|
| [`experiments/2026-05-20_calm_night_feature_refresh/`](experiments/2026-05-20_calm_night_feature_refresh/) | Built the calibration-aligned eval harness (Spearman + recall at thresholds + regime stratification); fixed `train_and_select`'s R²-based selector that was hiding a 32 pp recall@100 gap on Berry. Shipped in [PR #26](https://github.com/TheResilientCollective/tj_h2s_prediction/pull/26). |
| [`experiments/2026-06-10_feature_trim_berry/`](experiments/2026-06-10_feature_trim_berry/) | Ablated four feature sets (44 / 33 / 19 / 11). Evidence-only at 33 features wins decisively (shipped in [PR #27](https://github.com/TheResilientCollective/tj_h2s_prediction/pull/27); promoted to `MODEL_FEATURES` default in PR #28). |

The calibration-aligned harness lives in [projects/h2s/src/h2s/training/calibration_eval.py](projects/h2s/src/h2s/training/calibration_eval.py) and is used for all forward-looking model evaluation. Headline metrics on this heavy-tailed series:

- **Spearman** (not Pearson) on the bulk
- **recall@{5, 10, 30, 100}** at the categorical boundaries — including the complaint-rate band (5–10 ppb) and the calibration headline (100 ppb)
- **Regime stratification** by `stable_atm` (calm nights carry 88 % of Berry's >100 ppb hours)
- **Persistence floor** (`h2s_lag_1h → ppb`) as the autoregressive ceiling any model must beat

See [tj_calibration's calibration_status.md](https://github.com/TheResilientCollective/tijuana-dispersion-experiments/blob/main/docs/calibration_status.md) for the full evidence trail.

---

## Schedules

| Schedule | Cron | Job | Default state |
|---|---|---|---|
| `forecast_prediction_schedule` | `0 */6 * * *` | `forecast_prediction_job` | RUNNING |
| `daily_analysis_schedule` | `0 8 * * *` | `daily_analysis_job` | RUNNING |
| `daily_validation_schedule` | `0 8 * * *` | `daily_validation_job` | RUNNING |
| `dispersion_forecast_schedule` | `0 */6 * * *` | `dispersion_forecast_job` | RUNNING |
| `dispersion_inversion_schedule` | Mon 02:30 UTC | `dispersion_inversion_job` | STOPPED |
| `monthly_model_training_schedule` | `0 4 1 * *` | `multi_station_training_job` | RUNNING |

---

## S3 path conventions

```
s3://test/
├── tijuana/forecast/
│   ├── models/
│   │   ├── nestor_xgboost_weighted_model.json     # hourly classifier
│   │   ├── nestor_preprocessing_info.json
│   │   ├── deployment_metadata.json
│   │   ├── xgboost_base/, xgboost_smote/, random_forest/   # variants
│   │   └── stations/{station_key}/
│   │       ├── {clf_5ppb,clf_10ppb,regression}_evidence.pkl  # Evidence (33 feat, production)
│   │       ├── {clf_5ppb,clf_10ppb,regression}_lean.pkl      # Lean (19 feat, parallel)
│   │       ├── features_evidence.json / features_lean.json   # per-variant schemas
│   │       ├── deployment_metadata.json                       # variants key lists both
│   │       └── training_report.json                           # metrics for both variants
│   ├── hourly/YYYY-MM-DD_HH/                      # hive-partitioned predictions
│   ├── daily_summary/                             # daily station summaries
│   ├── validation/YYYY-MM-DD/                     # metrics + viz
│   ├── visualizations/                            # plots
│   ├── alerts/h2s_alert_state.json                # NESTOR alert state
│   ├── sensor_events/{archive,index.json}         # APCD event reports
│   └── extreme_events/                            # extreme-event summaries
├── tijuana/dispersion/
│   ├── lagrangian/{ensemble.json,footprint_ensemble.parquet}
│   ├── emission_rates.json                        # per-zone Q (east/west/south, g/s)
│   ├── hysplit/{backward,forward}_bundle_{run_tag}.zip
│   ├── calibration/Q_field_*.parquet              # channel-snapped inversion
│   └── forward_forecast_{run_tag}.json            # Gaussian 72h forecast
└── latest/tijuana/
    ├── weather_forecast/latest.csv                # input (openmeteo)
    ├── tides/latest.csv
    ├── streamflow/latest.csv
    ├── dispersion/forward_forecast_latest.json
    └── forecast_data/
        ├── h2s_predictions.{csv,json}
        ├── modeldata_h2s_nofill.parquet           # historical observations
        ├── model_forecast.parquet                  # 15-min forecast input
        ├── daily_summary.json
        └── visualizations/
```

---

## Testing

All tests live in `projects/h2s/tests/`. Run from `projects/h2s/`:

```bash
cd projects/h2s
uv sync
```

### Run the calibration-aligned harness tests (fast, no S3)

```bash
uv run pytest tests/test_calibration_eval.py tests/test_feature_builder.py \
              tests/test_train_and_select.py tests/test_constants.py -v
```

These pin the eval harness, the feature_builder idempotency, the recall-aware selector default, and the candidate feature sets. They're the regression guard for any model-side change.

### Run all unit tests

```bash
uv run pytest -m "not s3" -v
```

### Run S3 integration tests (requires `.env`)

```bash
uv run pytest tests/test_s3_integration.py -v
```

### Test files

| File | Description | Requires S3 |
|---|---|---|
| `test_calibration_eval.py` | Calibration-aligned harness (Spearman, recall@threshold, persistence) | No |
| `test_feature_builder.py` | `ensure_base_features` idempotency + per-feature checks | No |
| `test_train_and_select.py` | Selector control flow + alert-recall metrics in `eval_regressor` | No |
| `test_constants.py` | Candidate feature-set definitions (counts, subsets, load-bearing preservation) | No |
| `test_h2s_pipeline.py` | Hourly pipeline asset logic | No |
| `test_predictor.py` | `H2SPredictor` class | No |
| `test_training_pipeline.py` | Training pipeline logic | No |
| `test_asset_materialization.py` | Dagster asset materialization with mocked resources | No |
| `test_apcd_sensor_watch.py` | APCD bucket polling + multi-station event reports | No |
| `test_s3_integration.py` | S3 upload/download, model loading from S3 | Yes |

### Pytest markers

```bash
uv run pytest -m "not slow"       # Skip slow tests
uv run pytest -m integration       # Integration tests only
uv run pytest -m s3                 # S3 tests only
uv run pytest -x                    # Stop on first failure
```

---

## Environment configuration

Create `projects/h2s/.env` (see `env.example`):

```bash
# S3 / MinIO
S3_BUCKET=test
S3_ADDRESS=oss.resilientservice.mooo.com
S3_PORT=443
S3_USE_SSL=true
S3_ACCESS_KEY=your_access_key
S3_SECRET_KEY=your_secret_key

# Slack alerting
SLACK_TOKEN=xoxb-...
SLACK_CHANNEL=#h2s-alerts
SLACK_CHANNEL_FAILURES=#h2s-failures
SLACK_CHANNEL_OPS=#h2s-ops

# Deployment context
DAGSTER_DEPLOYMENT=local
ENV_LABEL=DEV
HOST=local
```

Dagster reads these via `EnvVar`, not `os.getenv`, so they're loaded at runtime.

---

## Project structure

```
tj_h2s_prediction/
├── projects/h2s/                       # Dagster pipeline (primary)
│   ├── src/h2s/
│   │   ├── definitions.py              # Asset / job / schedule registration
│   │   ├── constants.py                # Station geo, thresholds, MODEL_FEATURES + candidate sets
│   │   ├── defs/
│   │   │   ├── h2s_pipeline.py                 # Hourly forecast (14 assets)
│   │   │   ├── h2s_daily_pipeline.py           # Daily per-station + source attribution
│   │   │   ├── h2s_dispersion_pipeline.py      # Gaussian + Lagrangian + HYSPLIT
│   │   │   ├── h2s_calibration_pipeline.py     # Channel-snapped emission inversion
│   │   │   ├── h2s_multi_station_training.py   # Per-station model training
│   │   │   ├── h2s_validation_pipeline.py      # Daily metrics + monthly dashboard
│   │   │   ├── h2s_alert_system.py             # NESTOR alert state machine
│   │   │   ├── apcd_sensor_watch.py            # APCD multi-station event reports
│   │   │   └── h2s_schedules.py                # All schedules and jobs
│   │   ├── predictor/                  # H2SPredictor + visualization helpers
│   │   ├── dispersion/                 # Gaussian, Lagrangian, HYSPLIT
│   │   ├── training/
│   │   │   ├── calibration_eval.py     # Spearman + recall@threshold harness
│   │   │   ├── feature_builder.py      # ensure_base_features (idempotent)
│   │   │   ├── multi_station_trainer.py  # per-station train + eval + selector
│   │   │   ├── model_trainer.py        # legacy single-model training
│   │   │   └── validation.py           # metrics + comparison
│   │   ├── reporting/                  # Weekly scorecard
│   │   └── resources/                  # S3, Slack
│   ├── scripts/                        # Local training helpers
│   └── tests/                          # Test suite (see above)
├── experiments/                        # Research arcs (one folder per question)
│   ├── 2026-05-20_calm_night_feature_refresh/   # PR #26
│   └── 2026-06-10_feature_trim_berry/           # PR #27
├── src/                                # Standalone CLI scripts (predict, batch, viz)
├── data/                               # Local data (parquets, models_v2, startmodels)
└── nestor_xgboost_weighted_model.json  # Deployed hourly classifier (root copy)
```

---

## Standalone scripts

For one-off training, prediction, and visualization outside the Dagster pipeline:

```bash
# Train per-station models locally (outputs to data/models_v2/YYYYMMDD/)
cd projects/h2s
uv run python scripts/train_station_models.py \
  --obs ../../data/modeldata_h2s_nofill.parquet \
  --models ../../data/models_v2/$(date +%Y%m%d)
# Then seed to S3: uv run dg launch --job seed_models_job

# Single-file prediction
python src/predict_h2s.py --input data.csv --output predictions.csv

# Batch prediction
python src/batch_predict.py --input-dir ./data --output-dir ./predictions

# Adjust sensitivity (lower threshold = more sensitive)
python src/predict_h2s.py --input data.csv --output predictions.csv --orange-threshold 0.25
```

For research-style ablations and feature experiments, see [experiments/](experiments/).

---

## Troubleshooting

| Problem | Solution |
|---|---|
| `ModuleNotFoundError: No module named 'h2s'` | Run `uv sync` from `projects/h2s/`, use `uv run` prefix |
| `Validation error for S3Resource` | Use `EnvVar('S3_BUCKET')` not `os.getenv()` in definitions |
| Assets not in Dagster UI | Check `uv run dg list defs --json`; verify registration in `definitions.py` |
| `daily_analysis_job` fails partway | Re-execute **all** in Dagster UI — re-running a single failed step reads stale IO cache |
| Too many false alarms | Increase threshold: `--orange-threshold 0.40` |
| Missing too many events | Decrease threshold: `--orange-threshold 0.25` |
| `AttributeError: 'bytes' object has no attribute 'read'` | `S3Resource.getFile()` returns raw bytes, not BytesIO — pass `model_bytes` directly |

---

## Related documentation

- [CLAUDE.md](CLAUDE.md) — comprehensive runbook + architecture reference (read this for operational detail)
- [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) — API and integration reference
- [NESTOR_BES_H2S_Forecasting_Report.md](NESTOR_BES_H2S_Forecasting_Report.md) — technical report on the hourly classifier
- [experiments/](experiments/) — research arcs; each folder has its own `README.md` and `RESULTS.md`
- [tj_calibration findings](https://github.com/TheResilientCollective/tijuana-dispersion-experiments/blob/main/docs/calibration_status.md) — the calibration evidence trail driving feature and evaluation decisions
