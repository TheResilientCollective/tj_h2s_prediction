# Changelog

All notable changes to the H2S Forecasting System are documented here.

## [Unreleased]

### Added — Tiered H₂S Pre-Alert System (Tiers 1–3)

Three forecast-based pre-alert tiers complement the existing observation-based
WATCH (Tier 4) and CRITICAL (Tier 5) alerts. All three new tiers post to the
`SLACK_CHANNEL_OPS` channel (internal ops), support shadow mode
(`TIERED_ALERTS_SHADOW=true`), and are evaluated across four forecast horizons
(nowcast 0–3 h, near 3–6 h, mid 6–12 h, day-ahead 12–24 h).

| Tier | Label | Signal |
|------|-------|--------|
| 1 | PLANT-SIGNAL | SBIWTP flow below baseline + negative anomaly |
| 2 | MULTI-SITE-RISK | Tier 1 at ≥ 2 stations + wind speed < 4 m/s |
| 3 | EXCEEDANCE-RISK | Tier 2 + high temp, dewpoint, atmospheric stability |

**New files:**
- `projects/h2s/src/h2s/defs/tiered_alerts/__init__.py`
- `projects/h2s/src/h2s/defs/tiered_alerts/tiers.py` — Horizon enum, gate functions, score function, nesting invariant
- `projects/h2s/src/h2s/defs/tiered_alerts/features.py` — Horizon feature slicing, NB→IB fallback, `stable_atm_fraction`
- `projects/h2s/src/h2s/defs/tiered_alerts/state.py` — Backward-compatible S3 state extension with per-cell debounce
- `projects/h2s/src/h2s/defs/tiered_alerts/messages.py` — Slack Block Kit tier message builder
- `projects/h2s/src/h2s/defs/tiered_alerts/assets.py` — Dagster assets: `tiered_alert_features`, `tier_1/2/3_scores`, `tier_alert_dispatcher`
- `projects/h2s/src/h2s/defs/tiered_alerts/schedules.py` — `tiered_alerts_job`, `tiered_alerts_schedule` (6-hourly, auto-started)
- `projects/h2s/src/h2s/defs/tiered_alerts/backtest.py` — CLI replay against historical parquet
- `projects/h2s/configs/tiered_alerts.yaml` — Hard-gate thresholds, Cohen's d score weights, quiet-night feature stats
- `projects/h2s/tests/test_tiered_alerts_tiers.py`
- `projects/h2s/tests/test_tiered_alerts_state.py`
- `projects/h2s/tests/test_tiered_alerts_backtest.py`

**Modified files:**
- `projects/h2s/src/h2s/constants.py` — Extended `ALERT_TIERS` with tier_1/2/3 entries
- `projects/h2s/src/h2s/defs/h2s_alert_system.py` — Scoped obs-tier loops to `_OBS_TIER_KEYS = ("watch", "critical")`
- `projects/h2s/src/h2s/defs/apcd_sensor_watch.py` — Same `_OBS_TIER_KEYS` scoping
- `projects/h2s/src/h2s/definitions.py` — Registered new assets, job, and schedule

**New environment variables:**
- `SLACK_CHANNEL_OPS` — Slack channel for Tier 1–3 internal ops alerts (required in production)
- `TIERED_ALERTS_SHADOW` — Set `true` to suppress Slack dispatch while writing state (recommended for initial deploy)

**State migration:** The existing S3 state JSON at `ALERT_STATE_S3_PATH` gains a `tiers` key on first read. Existing `watch`/`critical` state is preserved unchanged.

**Preserved:** Existing WATCH/CRITICAL alert behavior (Tiers 4–5) is unchanged. The `h2s_alert_dispatcher` asset and `h2s_alert_sensor` sensor are unmodified.

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
