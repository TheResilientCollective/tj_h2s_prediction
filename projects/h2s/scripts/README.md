# H2S Pipeline Materialization Scripts

Helper scripts for materializing Dagster assets with different configurations.

## Scripts

### Production Mode (S3 Data)

**`materialize_data.sh`** - Run full pipeline with S3 environmental data
```bash
bash scripts/materialize_data.sh
```
- **Model:** Loaded from S3 (`tijuana/forecast/models/`)
- **Environmental Data:** Loaded from S3 (`latest/tijuana/weather_forecast/latest.csv`)
- **H2S Actuals:** Merged from local `data/modeldata_h2s.csv` if not in S3 data
- **Behavior:** FAILS if S3 environmental data is not available (no fallback)

### Test Mode (Local Data)

**`materialize_local_test.sh`** - Run pipeline with local test data
```bash
bash scripts/materialize_local_test.sh
```
- **Model:** Loaded from S3 (`tijuana/forecast/models/`)
- **Environmental Data:** Loaded from LOCAL (`data/latest.csv`)
- **H2S Actuals:** Already included in `data/latest.csv`
- **Use Case:** Testing when S3 environmental data is not available

### Model Only

**`materialize_artifacts.sh`** - Load model artifacts only
```bash
bash scripts/materialize_artifacts.sh
```
- Loads just the model from S3
- Useful for verifying S3 model access

## Configuration Details

### Production Mode Config
By default, `raw_environmental_data` asset uses:
```yaml
use_local_data: false  # Load from S3
```
This will **FAIL** if S3 data is not available (no silent fallbacks).

### Test Mode Config
`materialize_local_test.sh` passes this config:
```yaml
ops:
  raw_environmental_data:
    config:
      use_local_data: true
      local_data_path: "/path/to/data/latest.csv"
```

## Data Requirements

### Production Mode
- S3 must have: `latest/tijuana/weather_forecast/latest.csv`
- Optional: H2S measurements (will merge from local if missing)

### Test Mode
- Local file: `data/latest.csv` (350KB, includes environmental + H2S data)
- Backup: `data/modeldata_h2s.csv` (68KB, H2S measurements only)

## Error Handling

**Production mode will FAIL with clear error if:**
- S3 environmental data path does not exist
- S3 connection fails
- Data is corrupted/unreadable

**Test mode will FAIL with clear error if:**
- Local test data file does not exist
- File path is incorrect
- Data is corrupted/unreadable

No silent fallbacks or mock data - all failures are explicit.
