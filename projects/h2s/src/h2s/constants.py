"""Shared S3 path constants for H2S prediction pipeline.

Following resilient_workflows_public conventions:
- Timestamped paths: tijuana/forecast/{category}/{timestamp}/
- Latest paths: latest/tijuana/{category}/
"""

# S3 Path Constants
MODEL_PATH = 'tijuana/forecast/models'
TRAINING_PATH = 'tijuana/forecast/models/training'
ARCHIVE_PATH = 'tijuana/forecast/models/archive'
PREDICTIONS_PATH = 'tijuana/forecast/predictions'  # Legacy - use HOURLY_PREDICTIONS_PATH
HOURLY_PREDICTIONS_PATH = 'tijuana/forecast/hourly'  # Hive-partitioned base for pyarrow
OUTPUT_PATH = 'tijuana/forecast/output'  # Legacy - use DAILY_SUMMARY_PATH
DAILY_SUMMARY_PATH = 'tijuana/forecast/daily_summary'  # Daily multi-station output
VISUALIZATIONS_PATH = 'tijuana/forecast/visualizations'

# Latest path base
LATEST_BASEPATH = 'latest/tijuana'
LATEST_FORECAST = 'tijuana/forecast'
VALIDATION_PATH = 'tijuana/forecast/validation'
