# Changelog

All notable changes to the H2S Forecasting System are documented here.

## [Unreleased]

## [2026-01-27] - Monthly Training Partitions
### Added
- **MonthlyPartitionsDefinition** for h2s_training_data asset group
  - Partition start date: 2025-09-01
  - Enables backfilling historical training runs
  - Cumulative data filtering: each partition includes all data from start → end of that month
  
- **Partitioned Assets** (5 total in h2s_training_data group):
  - `monthly_training_data` - Loads historical data with cumulative filtering by partition month
  - `relabeled_training_data` - Applies new H2S thresholds (Yellow: 5-30 ppb, Orange: ≥30 ppb)
  - `data_quality_report` - Validates data completeness, missing values, class balance
  - `training_data` - Time-based 80/20 train split from relabeled data
  - `validation_data` - Time-based 20% validation split

- **S3 Path Constants** (`h2s/constants.py`):
  - `MODEL_PATH = 'tijuana/forecast/models'`
  - `TRAINING_PATH = 'tijuana/forecast/models/training'`
  - `ARCHIVE_PATH = 'tijuana/forecast/models/archive'`
  - `PREDICTIONS_PATH = 'tijuana/forecast/predictions'`
  - `LATEST_BASEPATH = 'latest/tijuana'`

### Changed
- Updated H2S category thresholds (Epic 0):
  - Yellow: 5-30 ppb (previously 5-15 ppb)
  - Orange: ≥30 ppb (previously ≥15 ppb)
  - Green: <5 ppb (unchanged)

- Monthly training data filtering now uses cumulative approach:
  - Before: Filtered to only data within the partition month
  - After: Includes all data from beginning up to end of partition month
  - Ensures realistic model retraining with progressively growing datasets

### Fixed
- Timezone handling in partition date filtering (UTC-aware timestamps)
- S3 path constants centralized (eliminates hardcoded path strings)

### Technical Details
**Partition Usage:**
```bash
# Materialize specific partition
uv run dg launch --assets monthly_training_data --partition 2025-09-01

# Backfill historical partitions
uv run dg launch --assets training_data --partition-range 2025-09-01...2025-12-01
```

**Implementation Files:**
- `projects/h2s/src/h2s/defs/h2s_training_pipeline.py` (lines 58-61, 72, 191, 250, 328, 374)
- `projects/h2s/src/h2s/constants.py` (new file)
- `system_plan.md` (updated with partition documentation)

## [2026-01] - Training Pipeline Foundation
### Added
- Complete monthly model retraining pipeline (13 assets across 4 phases)
- Phase 1: Data extraction and preparation
- Phase 2: Model training with cross-validation
- Phase 3: Validation and comparison (new vs current model)
- Phase 4: Manual approval gate and deployment

### Documentation
- System plan updated with partition backfilling examples
- Model retraining process documentation with quality gates

---

*Format: [YYYY-MM-DD] - Description*
