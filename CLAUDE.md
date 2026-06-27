# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Session Rules

- **Do all branch work in a git worktree.** Use `EnterWorktree` at the start of
  any task that creates commits or switches branches — never check out branches
  in the main working copy (`tj_h2s_prediction/`); it is the user's active
  checkout. The user also keeps manual worktrees as sibling directories
  (`tj_h2s_prediction-<name>`); when asked to work in one of those, `cd` to it
  directly.

## Repository Overview

This is an H2S (Hydrogen Sulfide) prediction system for the Tijuana River region, covering three monitoring stations: IB_CIVIC_CTR, NESTOR__BES, and SAN_YSIDRO. The system is the **Dagster orchestration pipeline** in `projects/h2s/` (production data pipeline with S3 integration). The original standalone `src/` scripts and the legacy single-NESTOR hourly/monthly pipelines were retired — the multi-station path (per-station training → products → validation) is the focus.

The system predicts H2S across all hazard levels. Reporting uses a 4-tier view
(the `h2s_category` helper in `constants.py`):
- **Green:** H2S < 5 ppb (safe)
- **Yellow:** 5 ≤ H2S < 10 ppb (caution)
- **Yellow-high:** 10 ≤ H2S < 30 ppb (resident-smell level — get it right)
- **Orange:** H2S ≥ 30 ppb (hazardous)

The trained per-station classifiers emit P(>5), P(>10) and P(>30) probabilities,
so the 4-tier view needs no retraining. The underlying 3-class models
(green / yellow 5–30 / orange) are unchanged.

**Model Performance (hourly pipeline):** 61.3% orange detection rate, 5.4% false alarm rate.

## Project Structure

```
tj_h2s_prediction/
├── data/
│   ├── models_v2/               # Trained per-station models (local)
│   └── startmodels/             # Seed models for initial S3 upload
├── projects/h2s/                 # Dagster orchestration project (the system)
│   ├── src/h2s/
│   │   ├── definitions.py       # Dagster definitions (asset + job registration)
│   │   ├── constants.py         # S3 paths, 4-tier h2s_category() + palette, thresholds
│   │   ├── defs/
│   │   │   ├── h2s_daily_pipeline.py    # Daily analysis: source attribution + station forecasts
│   │   │   ├── h2s_products_pipeline.py # nowcast/nearcast/forecast products (p5/p10/p30)
│   │   │   ├── h2s_forecast_validation_pipeline.py  # Validation store + skill curves
│   │   │   ├── forecast_digest.py       # Daily all-station sparkline digest → Slack
│   │   │   ├── forecast_heatmap.py      # All-station category + probability heatmap board → Slack
│   │   │   ├── forecast_performance.py  # Asymmetric-rubric performance report + AI narrative
│   │   │   ├── h2s_dispersion_pipeline.py  # Dispersion modeling: Lagrangian + Gaussian + HYSPLIT
│   │   │   ├── h2s_multi_station_training.py  # Per-station model training (partitioned)
│   │   │   └── h2s_schedules.py               # Schedules + dispersion/calibration/validation jobs
│   │   ├── forecasting/
│   │   │   ├── recursive.py            # Recursive nowcast/nearcast/forecast engine
│   │   │   ├── product_validation.py   # Join product rows to measured H2S; skill curves
│   │   │   ├── performance_report.py   # Asymmetric, temporally-tolerant verdict + cost rubric
│   │   │   └── narrative.py            # Plain-language narrative + ResilientLLM payload
│   │   ├── predictor/
│   │   │   ├── h2s_predictor.py  # H2SPredictor class with S3 loading
│   │   │   ├── visualizations.py # Plot generators returning BytesIO
│   │   │   ├── forecast_heatmap.py    # Category + probability heatmap grids (BytesIO)
│   │   │   └── performance_charts.py  # Scatter, skill-by-hour, verdict/confusion charts
│   │   ├── dispersion/          # lagrangian.py, gaussian.py, hysplit_controls.py
│   │   ├── training/            # feature_builder, model_trainer, multi_station_trainer, …
│   │   ├── resources/
│   │   │   ├── minio.py         # S3Resource
│   │   │   ├── slack.py         # SlackAlertResource
│   │   │   └── resilientllm.py  # ResilientLLMResource (n8n webhook → narrative)
│   │   └── utils/store_assets.py
│   ├── scripts/                 # Helper / backfill scripts
│   ├── tests/                   # Test suite
│   ├── pytest.ini
│   └── pyproject.toml
└── data/startmodels/            # Per-variant seed models for first S3 upload
```
### Asset Development Guidelines
When creating new assets:
1. Use consistent imports: `dagster`, `pandas`, `geopandas`, `requests`
2. Import shared utilities: `from utils import store_assets`
3. Require necessary resources: `s3`, `slack`
  * use s3.publicUrl to generate S3 URLs for visualization metadata and Slack alerts
  * avoid s3.get_presigned_url
6. Include proper error handling and logging with `get_dagster_logger()`
7. use utils/store_assets to Store data in both raw and processed formats.

8. Add automation conditions for scheduling (e.g., `AutomationCondition.eager()`)

## Operational Runbooks

### Quickstart — the core run loop

The shortest path from an empty store to working forecasts **and** accuracy
stats. Your `.env` sets the target bucket via `S3_BUCKET` (dev `test` /
prod `resilentpublic`); prefix any command with `S3_BUCKET=resilentpublic` to
operate on prod.

```bash
cd projects/h2s

# 1. Per-station models — trains all three stations against shared training data,
#    writes the promotable archive, AND deploys. Run in one batch to use a single
#    training snapshot across all stations.
uv run dg launch --job station_model_training_job --partition san_ysidro,nestor_bes,ib_civic_ctr
uv run dg launch --job station_model_deployment_job --partition san_ysidro,nestor_bes,ib_civic_ctr

# 2. Forecast products (nowcast / nearcast / forecast, all stations × variants)
uv run dg launch --job station_forecast_job

# 3. Accuracy / skill stats (rebuild from ALL stored product runs)
uv run dg launch --job station_forecast_validation_rebuild_job

# Optional: products + Tier 1–3 cascade Slack alerts in one job
uv run dg launch --job cascade_alerts_job
# Optional: daily source attribution + dashboard
uv run dg launch --job forecast_analysis_job
```

**Deploy vs. promote — you only need one.** `station_deployment_job` deploys the
models you just trained (running it *is* the approval) — use it for normal
operation. `promote_station_models_job` is only for promoting a *specific older
archived version* (the monthly review flow); skip it unless you are rolling back.
Both read the archive that step 1 writes, so promote fails with "No archived
versions found" until `station_model_training_job` has run at least once.

**On the stats (step 3):** the validation store joins forecasts against
*measured* H2S, so it only carries rows where a product run's target hours have
since been observed. It fills in over the days after you start running products;
re-run the rebuild whenever you want the latest numbers.

### Initial Installation

Run once when deploying to a new environment:

```bash
cd projects/h2s
uv sync
cp .env.example .env   # fill in S3 credentials

# 1. Train + deploy per-station daily models — the model-bootstrap path
#    (all three stations at once, sharing one training snapshot).
#    This also writes the immutable archive that promote_station_models_job reads.
uv run dg launch --job station_model_training_job --partition san_ysidro,nestor_bes,ib_civic_ctr
uv run dg launch --job station_model_deployment_job --partition san_ysidro,nestor_bes,ib_civic_ctr

# 2. Run the products pipeline (nowcast/nearcast/forecast, all stations × variants)
uv run dg launch --job station_forecast_job

# 3. Run daily analysis (source attribution + station forecasts + dashboard)
uv run dg launch --job station_forecast_analysis_job
```

The per-station training pipeline (`station_model_training_job` →
`station_deployment_job`) is the model-production path: it trains both feature
variants (evidence + lean), writes training reports and data snapshots, and
writes the immutable version archive that `promote_station_models_job` promotes
from. The previous one-off `seed_models_job` was removed — there is no separate
bootstrap; the training pipeline owns model production.

The legacy hourly NESTOR 3-class model
(`tijuana/forecast/models/nestor_xgboost_weighted_model.json`) and the legacy
monthly training pipeline that produced it were retired. The hindcast mode of
`scripts/backfill_validation.py` still reads that model from S3 if present, but
nothing trains or refreshes it. Model production is the per-station path above.

### Rebuilding Models (new training data available)

No approval gate is required — `station_deployment_job` acts as the explicit approval step.

```bash
cd projects/h2s

# 1. Train per-station models (all three stations at once, sharing one training data snapshot)
#    Runs multi_station_training_data (once) → per_station_trained_models (per station) →
#    station_training_report → station_model_archive
#    The archive asset writes an IMMUTABLE version to
#    models/archive/stations/{KEY}/{version}/ and posts a new-vs-production
#    comparison to Slack with a promote/review recommendation.
uv run dg launch --job station_model_training_job --partition san_ysidro,nestor_bes,ib_civic_ctr

# 2. Review training metrics in Dagster UI (station_training_report asset
#    metadata) and/or the Slack promotion report for each station.

# 3. Deploy to S3 (this IS the approval — running this job means you approve all three)
uv run dg launch --job station_model_deployment_job --partition san_ysidro,nestor_bes,ib_civic_ctr
# Default uploads to S3; pass approve_deployment=False in the run config
# for a dry run that validates without writing to S3:
#   uv run dg launch --job station_model_deployment_job --partition san_ysidro,nestor_bes,ib_civic_ctr \
#     --config-json '{"ops":{"h2s__station_model_deployment":{"config":{"approve_deployment":false}}}}'

# 4. Run daily analysis — it will re-load fresh models from S3
uv run dg launch --job station_forecast_analysis_job
```

**Important:** `station_model_training_job` stores models in Dagster's IO only
(plus the immutable S3 archive). `station_deployment_job` uploads them to the
production prefix where the daily pipeline reads from. Running
`forecast_analysis_job` after training but before deployment will use the
previously deployed models.

### Promoting an Archived Model Version (human-in-the-loop)

The monthly retraining flow archives every run WITHOUT touching production,
posts the comparison to Slack, and waits for a human:

```bash
# Promote the version named in the Slack report (running this IS the approval):
uv run dg launch --job promote_station_models_job --partition nestor_bes \
  --config-json '{"ops":{"h2s__station_model_promotion":{"config":{"version_tag":"20260612T214639Z-0f024ab"}}}}'

# Or promote the latest archived version for a station:
uv run dg launch --job promote_station_models_job --partition nestor_bes
```

Promotion copies all model pickles + feature schemas + training_report.json
from the archive to production and stamps `model_version` into
`deployment_metadata.json`. Every daily-pipeline forecast row carries that
`model_version`, so any analysis can be replayed against the exact archived
models that produced it (`models/archive/stations/{KEY}/{version}/`).

### Running the Forecast Pipelines

```bash
cd projects/h2s

# Products (nowcast/nearcast/forecast, all stations × variants; auto via cascade_alerts every 6h)
uv run dg launch --job station_forecast_job

# Daily source attribution + station forecasts + dashboard (auto-runs daily at 8am)
uv run dg launch --job forecast_analysis_job

# Dispersion modeling: 72h Gaussian forward forecast + alert check (auto-runs every 6h)
uv run dg launch --job dispersion_forecast_job

# Dispersion modeling: Weekly Lagrangian source attribution (Monday 02:30 UTC, STOPPED by default)
uv run dg launch --job dispersion_inversion_job

# Nowcast/nearcast/forecast products: 24 leads × 3 stations × 2 variants,
# stored to tijuana/forecast/products/run_ts=.../products.parquet (+ latest mirror).
uv run dg launch --job station_forecast_job

# Tier 1–3 forecast cascade: runs products, then alerts the ops Slack channel
# when a tier's product probability clears its cutoff (auto-runs every 6h).
uv run dg launch --job cascade_alerts_job

# Forecast validation store + per-lead-hour skill curves (rebuild from ALL runs):
uv run dg launch --job station_forecast_validation_rebuild_job
# (forecast_validation_job is the recent-window daily variant; schedule STOPPED)

# All-station heatmap board: category grid (station × hour, coloured by 4-tier
# hazard, labelled ppb) + probability grid (P>5/P>10/P>30 × hour) → Slack
# (auto-runs daily 08:45 UTC).
uv run dg launch --job forecast_heatmap_job

# Forecast performance report + AI narrative (asymmetric, temporally-tolerant
# rubric): rebuilds the validation store, scores predicted-vs-measured by the
# verdict rubric (dangerous-miss / smell-miss / early-warning-ok / …), renders
# scatter + skill-by-hour + verdict board, and posts a plain-language narrative
# (ResilientLLM webhook, local fallback). Schedule weekly Mon 14:00 UTC, STOPPED.
uv run dg launch --job forecast_performance_job
```

### The Forecast Heatmap Board + Performance Report

`forecast_heatmap_job` posts two grids built from the latest product rows: a
**category grid** (station × forecast hour, each cell coloured green / yellow /
yellow-high / orange and labelled with the predicted ppb) and a **probability
grid** (per station: P>5, P>10, P>30 across the forecast hours). The 4-tier
split comes from `h2s_category()` in `constants.py` (cut points 5/10/30) and the
existing p5/p10/p30 — no retraining.

`forecast_performance_job` scores the forecast with an **asymmetric, temporally-
tolerant rubric** (`forecasting/performance_report.py`): under-predicting a
hazard costs far more than over-predicting it, the ≥10 ppb resident-smell level
is called out, and an over-prediction the actual reaches within ±2 h is an
*early warning* rather than a false alarm. Headline metrics: tolerant-accuracy,
dangerous-miss rate, smell-miss rate, mean cost; plus Spearman/MAE by lead-hour
and by hour-of-day, and a predicted-vs-measured scatter. The narrative is
generated via the ResilientLLM n8n webhook (`resources/resilientllm.py`,
`execute_with_data`) when `RESILIENTLLM_*` env vars are set, otherwise a
deterministic local narrative. See `projects/h2s/docs/FORECAST_PERFORMANCE_RUBRIC.md`
for the full verdict definitions, cost weights, and tuning notes. These are
starting points — refine the rubric with health officials and the community.

### The Forecast Products (nowcast / nearcast / forecast)

`station_forecast_job` runs the recursive engine (`h2s/forecasting/recursive.py`)
for every station × variant:

One recursive pass produces all three products — they are window slices of
the same recursion, distinguished by how far it has drifted from observed data:

- **nowcast** (leads 1–3): recursion seeded at the last actual. Lead 1 is
  entirely observed; by lead 3 the short lags are predictions but the longer
  lags and rolling windows are still mostly actuals.
- **nearcast** (leads 4–6): the mid-window — lag_3h crosses into predictions
  at lead 4.
- **forecast** (leads 7–24): by lead 7 every lag ≤6h is a prediction.
  Honest scope: magnitude skill decays toward the exogenous ceiling at this
  horizon — treat as a risk ranking, not ppb truth.

Row schema (validation substrate): run_ts, product, station, lead_hour, time,
variant, model_version, h2s_pred, p5, p10, p30. Missing classifiers (e.g.
clf_30ppb before a station's first post-Phase-1 retrain) yield NaN
probabilities, never errors.

### Forecast Cascade + Alerting

`cascade_alerts_job` (every 6h, RUNNING) runs `station_forecast_job` then
evaluates the Tier 1–3 cascade at NESTOR-BES off the **Evidence** product
probabilities — Tier 1 P(>5) in nowcast, Tier 2 P(>10) in nearcast, Tier 3
P(>30) in forecast — and posts an escalating report to the ops Slack channel
when a tier's peak probability clears its cutoff (`CASCADE_TRIGGERS`, all 0.5).
Tiers fire independently (no nesting), and both variants (Evidence/Lean) are
shown in the report. A separate observed >10 ppb "Alert Performance" state
machine (`h2s_alert_performance_sensor`, 5-min poll) opens/closes events and
posts a forecast-vs-measured close-out. Both replaced the retired met-regime
gate + sigmoid-score `tiered_alerts` system. The observation tiers
(`watch` 30 / `critical` 100, in `h2s_alert_system`) are unchanged.

**Per-station clf_30ppb status (2026-06): NESTOR-BES only.** P(>30 ppb) / the
ORANGE call is **emitted for NESTOR-BES only** — gated by `CLF_30PPB_STATIONS`
in `constants.py`. clf_30ppb is still *trained and deployed* for all three
stations (the artifact set stays uniform), but the products
(`h2s_products_pipeline`) and daily (`h2s_daily_pipeline`) engines pass
`clf_30ppb=None` for the other stations, so their `p30` is **NaN** — the same
path the schema already takes for a missing classifier. IB_CIVIC_CTR and
SAN_YSIDRO therefore fall back to the ≥10 ppb (yellow-high) tier as their top
operational alert; `classify_risk` already reverts to the `prob_10`-driven
orange decision when `prob_30` is absent. The cascade (`cascade_alerts`) was
already NESTOR-only (`NB_SITE`), so it is unaffected.

*Why:* per-station orange recall is only dependable where there are enough ≥30
training positives. NESTOR-BES has ~344 (holdout recall ~0.77, walk-forward
~0.95 @0.25); IB_CIVIC_CTR (~51) and SAN_YSIDRO (~40) collapse at a fixed
operating point (SAN_YSIDRO ~0.16 holdout / ~0.43 walk-forward — below the 0.50
target). A **pooled cross-station ≥30 model** (435 combined positives) recovers
their recall in experiments and is the path to re-enabling them: add the
station's `site_name` to `CLF_30PPB_STATIONS` once that model ships. See
`projects/h2s/experiments/underfitting_results/` (FINDINGS + pooled results) and
`projects/h2s/docs/RETRAIN_STATION_CLF30PPB_BRIEF.md` §11. BorderlineSMOTE
oversampling was evaluated and *degraded* recall (AUC was already 0.96–0.98), so
it is OFF by default (opt-in via `enable_smote_clf_30ppb`).

### Forecast Validation Store + Skill Curves

`station_forecast_validation_rebuild_job` joins every stored product row to the H2S
actually measured at its target hour and writes:

- a consolidated, **rebuildable** parquet at
  `tijuana/forecast/validation_store/validation.parquet` (+ dated snapshot) —
  recomputed from the immutable product runs each time, so it is idempotent;
- per-(product, variant, lead-hour) **skill curves** (`skill_curves.parquet` +
  `skill_report.json` + a `latest` mirror): n, Spearman, MAE, and recall@{5,10,
  30,100} in two flavours — *magnitude* (cut h2s_pred at k, calibration harness)
  and *probability-call* (did the classifier flag it: p_k > 0.5 vs actual ≥ k),
  Evidence vs Lean.

`forecast_validation_job` is the daily recent-window variant (`max_age_days`
config, schedule STOPPED — enable in Phase 6); the rebuild job passes
`max_age_days=None` to recompute the entire history. Stats only populate where a
forecast's target hours have since been observed, so the store fills in over the
days after products start running.

### Re-executing a Failed forecast_analysis_job

If `forecast_analysis_job` fails partway through, use **"Re-execute all"** in the Dagster UI — not "Re-execute failed steps". Re-executing only the failed step reads `multi_station_model_artifacts` from a stale IO cache and will fail again. Running all steps re-loads models fresh from S3.

### Dispersion Pipeline Operations

**Forward forecast** (`dispersion_forecast_job`, runs every 6h):
- Loads latest emission rates from S3 (or uses calibrated defaults: east=20, west=10, south=137 g/s)
- Runs 72h Gaussian plume model using FORECAST meteorology
- Checks next 6h for threshold crossings (30 ppb watch, 100 ppb critical)
- Sends Slack alert if thresholds exceeded
- Uploads HYSPLIT forward CONTROL bundle to S3 (no execution)

**Source attribution** (`dispersion_inversion_job`, weekly Monday 02:30 UTC, STOPPED):
- Runs Lagrangian backward particle tracking over inversion window (default: Feb 1 - Apr 1 2026)
- Computes ensemble source fractions from 16 candidate sources
- Groups sources into east/west/south zones, derives emission rates (g/s)
- Uploads emission_rates.json to S3 for use by forward forecasts
- Generates HYSPLIT backward CONTROL bundle (no execution)

**HYSPLIT bundles**: Download from S3 (`tijuana/dispersion/hysplit/{backward|forward}_bundle_latest.zip`), unzip, and run `bash run_hysplit_*.sh` in a HYSPLIT container or submit to NOAA READY.

### HYSPLIT Celery Worker (`dispersion_hysplit_execution_job`)

A dedicated HYSPLIT worker container executes bundles automatically via
`dagster-celery` queue routing. The worker reads work from a Redis broker
populated by the Dagster code-server and writes outputs to
`s3://.../tijuana/forecasts/dispersion/hysplit/runs/{run_tag}/`.

**Architecture:**
- `dagster-code-h2s` enqueues the `hysplit_run_results` asset (tagged
  `dagster-celery/queue: hysplit`) on Redis.
- `hysplit-worker` consumes the queue, runs `hyts_std` / `hycs_std`
  against GDAS meteorology, uploads tdump/cdump + `summary.json` to S3.
- Other dispersion jobs (`dispersion_forecast_job`, `dispersion_inversion_job`)
  are unchanged and still run in-process on the code-server.

**One-time setup — place the HYSPLIT tarball:**
```bash
# From a machine that has access to the GeoDemic HYSPLIT archive:
cp /path/to/GeoDemic/backend/hysplit.v5.4.2_x86_64.tar.gz \
   tj_h2s_prediction/build/
# (License-gated; excluded from git via .gitignore → build/hysplit*.tar.gz)
```

**Build the worker image:**
```bash
cd tj_h2s_prediction
docker build --platform linux/amd64 \
    -f build/Dockerfile_hysplit_worker \
    -t docker.io/resilientucsd/dagster-resilient-h2s-hysplit-worker:latest .
```

**Mount GDAS meteorology (host bind mount):**
- Put GDAS files under `/data/gdas` on the host (or set
  `HYSPLIT_MET_HOST_DIR` in `.env` to an alternate path).
- The compose file mounts this directory read-only at
  `/data/hysplit/meteo` inside the worker.

**Launch a queue-driven run:**
```bash
# Ensure redis and hysplit-worker are up
docker compose -f deployments/compose.yaml up -d redis hysplit-worker

# Dispatch the HYSPLIT execution job via Dagster
cd projects/h2s
uv run dg launch --job dispersion_hysplit_execution_job
```

**Expected outputs:**
- `tijuana/forecasts/dispersion/hysplit/runs/{run_tag}/` — tdump/cdump files
- `tijuana/forecasts/dispersion/hysplit/runs/latest/` — mirror of latest run
- `tijuana/forecasts/dispersion/hysplit/runs/{run_tag}/summary.json` —
  per-control exit codes and MESSAGE diagnostics

**Troubleshooting the worker:**
- `docker compose logs hysplit-worker` — look for `hyts_std` stderr or
  MESSAGE-file diagnostics echoed by `HysplitRunner`.
- A missing met file prints a HYSPLIT `metset.f: Bad value` message and
  the op fails cleanly — check that `HYSPLIT_MET_HOST_DIR` covers the
  run window's GDAS week file (`gdas1.{mon}{yy}.w{1..5}`).
- `redis-cli -h redis-${PROJECT} monitor` — watch broker traffic during
  a run to confirm enqueue → consume flow.

## Common Commands

### Dagster Development

```bash
# Navigate to Dagster project
cd projects/h2s

# Install dependencies
uv sync

# Check definitions (validate assets load correctly)
uv run dg check defs

# List all assets and resources
uv run dg list defs

# Start Dagster UI (default: http://localhost:3000)
uv run dg dev

# Materialize a specific asset
uv run dg launch --assets h2s/h2s_predictions
```

### Training Scripts

```bash
cd projects/h2s

# Train per-station models locally for inspection (outputs to data/models_v2/YYYYMMDD/)
# NOTE: local outputs are for analysis only — deploy through
# station_model_training_job → station_deployment_job.
uv run python scripts/train_station_models.py \
  --obs ../../data/modeldata_h2s_nofill.parquet \
  --models ../../data/models_v2/$(date +%Y%m%d)
```

### Standalone Scripts

```bash
# Single prediction
python src/predict_h2s.py --input data.csv --output predictions.csv

# Batch processing
python src/batch_predict.py --input-dir ./data --output-dir ./predictions

# Alerts only (filter out green predictions)
python src/predict_h2s.py --input data.csv --output alerts.csv --filter-alerts

# Adjust sensitivity (lower threshold = more sensitive)
python src/predict_h2s.py --input data.csv --output predictions.csv --orange-threshold 0.25
```

### Testing

```bash
cd projects/h2s

# Install test dependencies
uv sync

# Run all tests
uv run pytest

# Run specific test file
uv run pytest tests/test_h2s_pipeline.py -v

# Run tests with coverage
uv run pytest --cov=h2s --cov-report=html

# Skip S3 integration tests (if credentials not available)
uv run pytest -m "not s3"

# Run only fast tests
uv run pytest -m "not slow"

# Stop on first failure
uv run pytest -x
```

### Testing S3 Integration

```bash
cd projects/h2s
uv run python -c "
from h2s.predictor.h2s_predictor import H2SPredictor
from h2s.resources.minio import S3Resource
import os

s3 = S3Resource(
    S3_BUCKET=os.getenv('S3_BUCKET'),
    S3_ADDRESS=os.getenv('S3_ADDRESS'),
    S3_PORT=os.getenv('S3_PORT'),
    S3_USE_SSL=os.getenv('S3_USE_SSL', 'true').lower() == 'true',
    S3_ACCESS_KEY=os.getenv('S3_ACCESS_KEY'),
    S3_SECRET_KEY=os.getenv('S3_SECRET_KEY'),
)

predictor = H2SPredictor.from_s3(
    s3,
    'tijuana/forecast/models/nestor_xgboost_weighted_model.json',
    'tijuana/forecast/models/nestor_preprocessing_info.json'
)
print(f'Model loaded: {len(predictor.feature_cols)} features')
"
```

## Architecture

### Active Pipelines

> **Retired (multi-station replaces single-NESTOR):** the legacy
> `forecast_prediction_job` (hourly NESTOR forecast), the monthly single-model
> training jobs (`monthly_data_extraction_job` / `monthly_model_training_job` /
> `deploy_approved_model_job` / `approve_and_deploy_job`), and the legacy hourly
> validation (`station_forecast_validation_job`) were removed along with
> `defs/h2s_pipeline.py` and `defs/h2s_training_pipeline.py`. Model production is
> the per-station path (`station_model_training_job` → `station_model_deployment_job`)
> feeding the products pipeline (`station_forecast_job`).

**`station_forecast_job`** (the products pipeline) — nowcast/nearcast/forecast
rows (p5/p10/p30 per station × variant), the substrate for everything below.

**`forecast_heatmap_job`** (daily 08:45 UTC, RUNNING) — all-station heatmap board
```
products_model_artifacts → h2s_products → forecast_heatmap_board
   → category grid (station × hour, coloured by 4-tier hazard, labelled ppb)
   → probability grid (station × {P>5,P>10,P>30} × hour) → S3 + Slack
```

**`forecast_performance_job`** (weekly Mon 14:00 UTC, STOPPED) — asymmetric rubric report + AI narrative
```
forecast_validation_store → forecast_performance_report → forecast_performance_narrative
   report: predicted-vs-measured scatter + skill-by-hour correlation + verdict/
   confusion board (tolerant-accuracy, dangerous-miss, smell-miss); narrative via
   ResilientLLM webhook with a deterministic local fallback.
```

**`forecast_analysis_job`** (every 6h) — multi-station source attribution + 48h forecasts
```
multi_station_model_artifacts → source_attribution → daily_station_forecasts → daily_dashboard_viz
                                                                              → daily_summary_json
```

**`dispersion_inversion_job`** (weekly Monday 02:30 UTC, STOPPED by default) — backward source attribution
```
lagrangian_source_attribution → emission_rate_inversion → hysplit_controls_generation (backward CONTROL bundle)
```

**`dispersion_forecast_job`** (every 6h, RUNNING) — forward Gaussian plume forecast
```
emission_rate_inversion → gaussian_forward_forecast → dispersion_alert_check
                                                    → hysplit_controls_generation (forward CONTROL bundle)
```

### S3 Path Conventions

```
s3://test/
├── tijuana/forecast/
│   ├── models/
│   │   ├── nestor_xgboost_weighted_model.json  # hourly pipeline model
│   │   ├── nestor_preprocessing_info.json       # 33-feature preprocessing metadata (PR #28)
│   │   ├── deployment_metadata.json
│   │   ├── xgboost_base/model.json              # variants
│   │   ├── xgboost_smote/model.json
│   │   ├── random_forest/model.joblib
│   │   ├── stations/{station_key}/              # IB_CIVIC_CTR, NESTOR__BES, SAN_YSIDRO
│   │   │   ├── clf_5ppb_evidence.pkl            # Evidence variant — 33 feat, production default
│   │   │   ├── clf_10ppb_evidence.pkl
│   │   │   ├── clf_30ppb_evidence.pkl           # P(>30ppb) for the Tier-3 cascade trigger
│   │   │   ├── regression_evidence.pkl
│   │   │   ├── clf_5ppb_lean.pkl                # Lean variant — 19 feat, deployed in parallel
│   │   │   ├── clf_10ppb_lean.pkl
│   │   │   ├── clf_30ppb_lean.pkl
│   │   │   ├── regression_lean.pkl
│   │   │   ├── features_evidence.json           # Evidence column schema
│   │   │   ├── features_lean.json               # Lean column schema
│   │   │   ├── deployment_metadata.json         # `variants` + `model_version`
│   │   │   └── training_report.json             # metrics for both variants
│   │   └── archive/stations/{station_key}/{version_tag}/  # immutable version history
│   │       └── (same artifact set + archive_metadata.json; version_tag =
│   │            YYYYMMDDTHHMMSSZ-{gitsha} for production,
│   │            backfill_{month_key}_{timestamp}-{sha} for backfill; never overwritten)
│   └── output/YYYY-MM-DD_HH/                   # Timestamped hourly predictions
├── tijuana/dispersion/
│   ├── lagrangian/
│   │   ├── ensemble.json                        # Source attribution ensemble (16 candidate sources)
│   │   └── footprint_ensemble.parquet           # Ensemble footprint heatmap (lat × lon)
│   ├── emission_rates.json                      # Per-zone Q (east/west/south in g/s)
│   ├── hysplit/
│   │   ├── backward_bundle_{run_tag}.zip        # HYSPLIT backward CONTROL bundle
│   │   ├── backward_bundle_latest.zip
│   │   ├── forward_bundle_{run_tag}.zip         # HYSPLIT forward CONTROL bundle
│   │   └── forward_bundle_latest.zip
│   └── forward_forecast_{run_tag}.json          # Gaussian 72h plume forecast
└── latest/tijuana/
    ├── weather_forecast/latest.csv              # Input (from openmeteo.py)
    ├── tides/latest.csv
    ├── streamflow/latest.csv
    ├── dispersion/forward_forecast_latest.json  # Latest Gaussian forecast
    └── forecast_data/
        ├── h2s_predictions.{csv,json}
        ├── daily_summary.json
        ├── modeldata_h2s.csv                    # Historical H2S measurements
        └── visualizations/
```

### Key Design Decisions

**`algorithm_choices` field in `archive_metadata.json`**

Every archived model version (production and backfill) carries an `algorithm_choices` field
documenting which algorithm `train_and_select()` auto-selected per task/variant. See
`docs/ALGORITHM_CHOICES.md` for the full schema, selection logic, and code examples.

**Backfill archive layout** (same root as production, `backfill_` prefix in version tag):
```
STATION_MODELS_ARCHIVE_BASE/{station_key}/backfill_{month_key}_{timestamp}-{sha}/
  {task}_{variant}.pkl       — model pickle (8 files: 4 tasks × 2 variants)
  training_report.json       — in-sample val metrics, feature lists per variant, is_backfill: true
  archive_metadata.json      — version tag, algorithm_choices, artifacts list, backfill metadata
```
`features_{variant}.json` is NOT written separately — the feature lists are inside `training_report.json`
under `"features": {"evidence": [...], "lean": [...]}`.

**Why JSON instead of pickle for preprocessing?**
- S3-friendly (human-readable, portable, secure)
- Eliminates sklearn version warnings
- Uses dict lookups instead of LabelEncoder objects

**Why copy code from resilient_workflows_public?**
- Avoids sys.path manipulation and import issues
- Simplified `store_assets.py` without heavy dependencies
- Self-contained S3Resource in `h2s/resources/minio.py`

**Why tempfile for XGBoost model loading?**
- XGBoost requires file path (not BytesIO)
- `S3Resource.getFile()` returns raw bytes — write to tempfile, load, delete

**Why FORECAST_DATA_PATH for Gaussian forward forecast?**
- `gaussian_forward_forecast` uses forecast meteorology (model_forecast.parquet), not observations
- This is the operational forecast use case — predicting future H2S based on weather forecasts
- Lagrangian inversion uses OBS_DATA_PATH for backward attribution on historical events

**Why 2-hour backward integration time for Lagrangian inversion?**
- Valley-scale sources: 1-7 km from sensor (travel time: 8-37 min @ 3 m/s wind)
- 6-hour integration was 10× too long (particles travel 64 km, miss local sources)
- 2-hour integration (21 km reach) is appropriate for Tijuana River Valley scale
- Critical fix: 6h gave east=0%, 2h gives east=46% (east sources now correctly detected)

**Current emission rates (wind-dependent Lagrangian inversion, Feb-Apr 2026):**
- East: 87.3 g/s (Dairy Mart Bridge dominant: 14.2%; 52.3% of total)
- West: 29.9 g/s (Tijuana Beach Outlet, Oneonta Slough: 17.9% of total)
- South: 49.8 g/s (Goat Canyon, Smugglers Gulch: 29.8% of total)
- Total: 167 g/s (conserved from March 13 2026 calibration event)

**Wind-dependent diffusion (implemented):**
- H2S strongly anti-correlated with wind speed (r = -0.246): low wind → high H2S
- Lagrangian model now uses σ ~ U^0.5 (calm winds: σ ~ 0.21 m/s; strong winds: σ ~ 0.34 m/s)
- Sharper attribution during calm events → east sources properly identified
- Comparison: Fixed σ=0.3 gave east=45.6%; wind-dependent gave east=52.3%
- See WIND_SPEED_DEPENDENCY.md for implementation details

**Why upload HYSPLIT bundles but not execute?**
- HYSPLIT requires ~20 GB GDAS meteorology files and specialized container environment
- Bundles are generated as CONTROL files + shell scripts, uploaded to S3
- User downloads and executes in local HYSPLIT container or submits to NOAA READY server
- Keeps Dagster pipeline lightweight and portable

## Environment Configuration

Create `.env` file (see `env.example`):

```bash
# S3/MinIO Configuration (required for Dagster pipeline)
S3_BUCKET=test
S3_ADDRESS=oss.resilientservice.mooo.com
S3_PORT=443
S3_USE_SSL=true
S3_ACCESS_KEY=your_access_key
S3_SECRET_KEY=your_secret_key

# Optional: Latest path configuration
PUBLIC_BUCKET=test
LATEST_BASEPATH=latest/

# Slack alerting
SLACK_TOKEN=xoxb-...
SLACK_CHANNEL=#h2s-alerts
SLACK_CHANNEL_FAILURES=#h2s-failures

# Deployment context
DAGSTER_DEPLOYMENT=local     # or production
ENV_LABEL=DEV                # shown in dashboard titles and Slack alerts
SCHED_HOSTNAME=sched         # for Dagster UI URL in failure alerts
HOST=local
```

**Dagster uses `EnvVar` for S3 and Slack credentials** - environment variables are loaded at runtime, not at definitions.py import time.

## Model Files

**Location:** Root directory and S3

- `nestor_xgboost_weighted_model.json` - 4.2 MB trained XGBoost classifier (hourly pipeline)
- `nestor_preprocessing_info.json` - metadata (feature names, class mappings)

**Features (43 total) — built by `feature_builder.py`:**
- Weather: temperature_2m, wind_speed_10m, wind_direction_10m, relative_humidity_2m, surface_pressure, precipitation, cloud_cover, dewpoint_2m
- Wind rolling averages (2h, 3h, 4h) + gusts rolling max
- Cyclical encodings: hour_sin/cos, month_sin/cos, wind_direction_sin/cos
- Flow: flow_rate_cms, flow_log, flow_low, flow_high, flow_lag_6h, flow_rolling_24h
- H2S lags: h2s_lag_1h/3h/6h, h2s_rolling_6h/24h
- SBIWTP: sbiwtp_flow_mgd, sbiwtp_anomaly, sbiwtp_deficit, etc.
- Stability/regime: is_night, source_regime, stable_atm
- Encoded: wind_direction_cat_encoded, tidal_state_encoded

**Classes:** ['green', 'orange', 'yellow']

## Daily Partitions and Validation Metrics

> **Mostly retired.** This section described the legacy hourly
> `forecast_prediction_job` + `daily_validation_*` partition workflow, which was
> removed with the single-NESTOR pipeline. Forecast accuracy now comes from the
> products → `forecast_validation_store` → `forecast_skill_report` path and the
> asymmetric `forecast_performance_job`. The partition mechanics below are kept
> for historical context only.

### Partition System

**Forecast and validation jobs use daily partitions** (start_date=2026-01-01, timezone=UTC):

```bash
# Run forecast for specific date
uv run dg launch --job forecast_prediction_job --partition 2026-04-02

# Run validation for specific date
uv run dg launch --job daily_validation_metrics_job --partition 2026-04-02

# Run full validation with monthly dashboard (requires >0 days of metrics)
uv run dg launch --job daily_validation_job --partition 2026-04-02
```

**Jobs:**
- `forecast_prediction_job` — Generates predictions for a date (uses forecast data from that date)
- `daily_validation_metrics_job` — Creates metrics.json only (for backfilling)
- `daily_validation_job` — Creates metrics.json + monthly dashboard (fails if zero metrics days available)

### Validation Metrics Accumulation

**Natural accumulation workflow** (recommended):

1. **Day 1**: Forecast runs → predictions stored to S3
2. **Day 2**: Validation runs → compares Day 1 predictions vs Day 1 actuals → creates metrics.json
3. **Days 3-7**: Repeat daily
4. **Day 8+**: Monthly dashboard generates successfully (uses last 30 days of metrics)

**Daily schedules:**
- `forecast_prediction_schedule`: Every 6 hours (00, 06, 12, 18 UTC) → materializes TODAY's partition
- `daily_validation_schedule`: Daily at 8 AM UTC → materializes YESTERDAY's partition

**Important:** Validation requires predictions and observations to have matching timestamps. The current system:
- ✅ Works for daily production runs (forecast uses today's data, validation uses yesterday's data)
- ❌ Cannot backfill historical validations (forecast data not partitioned by date)

### Historical Backfills (Future Enhancement)

**Current limitation:** `preprocessed_features` loads from `latest/tijuana/forecast_data/model_forecast.parquet` which always contains the most recent forecast, not historical forecasts. Backfilling partition `2026-03-26` loads today's forecast, generates predictions with today's timestamps, then validation finds zero matches with March 26 observations.

**Solution:** Partition forecast data by generation date. See `projects/h2s/FORECAST_DATA_PARTITIONING.md` for detailed implementation guide to enable true historical backfills.

### Metrics Storage

```
s3://test/tijuana/forecast/validation/
  2026-04-01/
    metrics.json          # Daily metrics (balanced accuracy, confusion matrix, FAR)
    confusion_matrix.png  # Visualization
    model_comparison.png
  2026-04-02/
    metrics.json
    ...
```

**metrics.json structure:**
```json
{
  "date": "2026-04-01",
  "site": "NESTOR__BES",
  "n_predictions": 462,
  "n_matched": 450,
  "match_rate": 0.974,
  "balanced_accuracy": 0.856,
  "false_alarm_rate": 0.034,
  "class_metrics": {
    "green": {"precision": 0.92, "recall": 0.95, "f1": 0.93},
    "yellow": {"precision": 0.78, "recall": 0.71, "f1": 0.74},
    "orange": {"precision": 0.88, "recall": 0.81, "f1": 0.84}
  },
  "confusion_matrix": [[240, 12, 3], [15, 145, 8], [2, 5, 20]]
}
```

## Troubleshooting

**"ModuleNotFoundError: No module named 'h2s'"**
- Ensure you're in `projects/h2s/` and run commands with `uv run`
- Check `uv sync` completed successfully

**"Validation error for S3Resource: Input should be a valid string"**
- S3 config must use `EnvVar('S3_BUCKET')` not `os.getenv('S3_BUCKET')`
- Dagster definitions.py already uses EnvVar correctly

**"AttributeError: 'bytes' object has no attribute 'read'"**
- `S3Resource.getFile()` returns raw bytes, not BytesIO
- Use `model_bytes` directly, not `model_bytes.read()`

**"Assets not appearing in Dagster UI"**
- Assets must be explicitly registered in `definitions.py`
- Check `uv run dg list defs --json` to see if assets are loaded
- Verify `from h2s.defs.h2s_pipeline import ...` in definitions.py

**"Too many false alarms / missing events"**
- Adjust thresholds in standalone scripts: `--orange-threshold 0.25` (more sensitive) or `0.40` (less sensitive)
- Default: 0.33 (61% detection, 5.4% false positives)

## Input Data Requirements

CSV must include these columns:
- `time` - Timestamp
- `temperature_2m`, `wind_speed_10m`, `wind_direction_10m`, `relative_humidity_2m`
- `surface_pressure`, `precipitation`, `cloud_cover`
- `wind_direction_categorical` - Cardinal direction (N, NE, E, etc.)
- `flow_rate_cms`, `tide_height_m`, `tidal_state` - Tidal data

See README.md for complete column list.

## Related Documentation

- `README.md` - Quick start, usage examples, model details
- `DEPLOYMENT_GUIDE.md` - Complete API reference, integration examples
- `NESTOR_BES_H2S_Forecasting_Report.md` - Technical report
- `Complete_Model_Testing_Summary.md` - Model evaluation
- `projects/h2s/VALIDATION_AND_ACCURACY_REPORTING.md` - Validation pipeline guide, backfill scripts, accuracy reporting (see this for detailed validation workflows)
- `projects/h2s/docs/FORECAST_PERFORMANCE_RUBRIC.md` - Forecast performance rubric: verdict definitions, cost weights, and tuning notes for health official refinement
- `experiments/` - Research-style retrain experiments; each subfolder has its own `README.md` / `RESULTS.md`. Calibration-aligned evaluation harness lives in `projects/h2s/src/h2s/training/calibration_eval.py`.
