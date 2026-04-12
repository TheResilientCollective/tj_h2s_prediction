# Forecast Data Partitioning for Historical Backfills

## Problem Statement

The current H2S prediction pipeline has daily partitions enabled, but **cannot backfill historical validations** because the forecast data source (`latest/tijuana/forecast_data/model_forecast.parquet`) always contains the most recent forecast, not historical forecasts.

When backfilling partition `2026-03-26`, the pipeline loads today's forecast data (2026-04-02), generates predictions with today's timestamps, then tries to validate against March 26 observations — finding zero time matches.

## Current Architecture

```
Forecast Service → latest/tijuana/forecast_data/model_forecast.parquet (LATEST ONLY)
                                ↓
                   H2S Pipeline preprocessed_features asset
                                ↓
                   Predictions stored with partition: tijuana/forecast/hourly/model=.../year={Y}/month={M}/day={D}/hour={H}/
                                ↓
                   Validation fails: predictions have today's timestamps, actuals have historical timestamps
```

## Solution: Partition Forecast Data by Date

### Required Changes in Forecast Service

The forecast service (openmeteo.py or equivalent) needs to:

1. **Store forecast data with date partitions** instead of always overwriting `latest/`:

```
# CURRENT (overwrites daily):
latest/tijuana/forecast_data/model_forecast.parquet

# NEW (append with partitions):
tijuana/forecast_data/date={YYYY-MM-DD}/model_forecast.parquet
tijuana/forecast_data/date=2026-03-26/model_forecast.parquet
tijuana/forecast_data/date=2026-03-27/model_forecast.parquet
tijuana/forecast_data/date=2026-04-02/model_forecast.parquet

# ALSO keep latest path for current day:
latest/tijuana/forecast_data/model_forecast.parquet  (copy of today's partition)
```

2. **Include forecast generation timestamp** in the parquet metadata or as a column:
   - Column: `forecast_generated_at` (UTC timestamp when forecast was generated)
   - This allows distinguishing between "forecast generated on 2026-03-26 for next 10 days" vs "forecast generated on 2026-04-02 retroactively"

3. **Retention policy**: Keep partitioned forecast data for at least 30 days to support validation dashboard (requires 30-day lookback).

### File Format

**Schema (no changes, just add partition):**
- All existing columns (43 features: temperature_2m, wind_speed_10m, etc.)
- `time` column: Future timestamps (forecast valid times)
- **NEW**: Optional `forecast_generated_at` column (when this forecast was created)

**Storage pattern:**
```python
from datetime import datetime
import pandas as pd

# When generating forecast
forecast_date = datetime.now().strftime("%Y-%m-%d")
partition_path = f"tijuana/forecast_data/date={forecast_date}/model_forecast.parquet"

df['forecast_generated_at'] = datetime.now()  # Optional but recommended
df.to_parquet(partition_path)

# Also update latest path
df.to_parquet("latest/tijuana/forecast_data/model_forecast.parquet")
```

## Required Changes in H2S Pipeline

Once forecast service implements partitioning, update `preprocessed_features` asset:

**File:** `projects/h2s/src/h2s/defs/h2s_pipeline.py`

**Current code (line 120-128):**
```python
def preprocessed_features(context, h2s_model_artifacts):
    s3 = context.resources.s3
    context.log.info(f"Loading forecast data from S3: {FORECAST_DATA_PATH}")

    try:
        forecast_url = s3.get_presigned_url(path=FORECAST_DATA_PATH, bucket=s3.S3_BUCKET)
        df = pd.read_parquet(forecast_url)
```

**NEW code (with partition support):**
```python
def preprocessed_features(context, h2s_model_artifacts):
    s3 = context.resources.s3

    # Use partition key to load correct forecast data
    partition_date = context.partition_key  # e.g., "2026-03-26"
    partitioned_path = f"tijuana/forecast_data/date={partition_date}/model_forecast.parquet"

    # Try partitioned path first (for backfills), fall back to latest (for current runs)
    try:
        forecast_url = s3.get_presigned_url(path=partitioned_path, bucket=s3.S3_BUCKET)
        df = pd.read_parquet(forecast_url)
        context.log.info(f"✓ Loaded forecast data from partitioned path: {partitioned_path}")
    except Exception as e:
        context.log.warning(f"Partitioned forecast not found ({e}), falling back to latest")
        forecast_url = s3.get_presigned_url(path=FORECAST_DATA_PATH, bucket=s3.S3_BUCKET)
        df = pd.read_parquet(forecast_url)
```

**Update constants.py:**
```python
# Add new constant
FORECAST_DATA_PARTITIONED_PATH = "tijuana/forecast_data"  # Base path for partitions
```

## Benefits After Implementation

✅ **Historical backfills work**: Can re-run any date's predictions using that date's forecast
✅ **Reproducibility**: Exact forecast used for each prediction is preserved
✅ **Validation accuracy**: Predictions use correct timestamps matching when forecast was generated
✅ **Debugging**: Can investigate "why did model predict X on date Y" using exact forecast inputs
✅ **Data lineage**: Clear provenance from forecast → prediction → validation

## Migration Path

### Phase 1: Forecast Service Changes (No Breaking Changes)
1. Update forecast service to write partitioned data
2. Keep writing to `latest/` path (existing pipeline continues working)
3. Accumulate 7-30 days of partitioned data

### Phase 2: H2S Pipeline Changes (After Phase 1 Complete)
1. Update `preprocessed_features` to try partitioned path first
2. Test backfill on 7-day range
3. Verify validations succeed with historical data

### Phase 3: Production Rollout
1. Enable daily partition backfills
2. Monitor metrics accumulation
3. Monthly dashboard works after 7 days

## Testing Checklist

After forecast service implements partitioning:

```bash
# 1. Verify partitioned data exists
aws s3 ls s3://test/tijuana/forecast_data/date=2026-04-02/

# 2. Test single partition
uv run dg launch --job forecast_prediction_job --partition 2026-04-02

# 3. Test backfill (3 days)
for date in 2026-03-30 2026-03-31 2026-04-01; do
  uv run dg launch --job forecast_prediction_job --partition $date
done

# 4. Validate backfilled predictions
for date in 2026-03-30 2026-03-31 2026-04-01; do
  uv run dg launch --job daily_validation_metrics_job --partition $date
done

# 5. Check metrics files created
aws s3 ls s3://test/tijuana/forecast/validation/2026-03-30/metrics.json
aws s3 ls s3://test/tijuana/forecast/validation/2026-03-31/metrics.json
aws s3 ls s3://test/tijuana/forecast/validation/2026-04-01/metrics.json

# 6. Generate monthly dashboard (after 7 days of metrics)
uv run dg launch --job daily_validation_job --partition 2026-04-07
```

## Example: Forecast Service Implementation

```python
# openmeteo.py or forecast generation script
import pandas as pd
from datetime import datetime
from minio_client import s3_client  # Your S3 client

def generate_and_store_forecast():
    # Generate forecast (your existing logic)
    forecast_df = fetch_weather_forecast()

    # Add generation timestamp
    forecast_date = datetime.now().strftime("%Y-%m-%d")
    forecast_df['forecast_generated_at'] = datetime.now()

    # Store partitioned copy
    partition_path = f"tijuana/forecast_data/date={forecast_date}/model_forecast.parquet"
    s3_client.put_object(
        bucket="test",
        key=partition_path,
        data=forecast_df.to_parquet()
    )
    print(f"✓ Stored partitioned forecast: {partition_path}")

    # Also update latest path (for current runs)
    latest_path = "latest/tijuana/forecast_data/model_forecast.parquet"
    s3_client.put_object(
        bucket="test",
        key=latest_path,
        data=forecast_df.to_parquet()
    )
    print(f"✓ Updated latest forecast: {latest_path}")

if __name__ == "__main__":
    generate_and_store_forecast()
```

## Contact

For questions about this implementation:
- H2S Pipeline: This service (tj_h2s_prediction)
- Forecast Data: Other service (openmeteo.py or equivalent)
