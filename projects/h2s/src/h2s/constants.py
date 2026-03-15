"""Shared S3 path constants for H2S prediction pipeline.

Following resilient_workflows_public conventions:
- Timestamped paths: tijuana/forecast/{category}/{timestamp}/
- Latest paths: latest/tijuana/{category}/
"""

# S3 Path Constants
MODEL_PATH = 'tijuana/forecast/models'
TRAINING_PATH = 'tijuana/forecast/models/training'
ARCHIVE_PATH = 'tijuana/forecast/models/archive'
PREDICTIONS_PATH = 'tijuana/forecast/predictions'
OUTPUT_PATH = 'tijuana/forecast/output'  # Legacy - prefer PREDICTIONS_PATH

# Latest path base
LATEST_BASEPATH = 'latest/tijuana'
LATEST_FORECAST_DATA = 'tijuana/forecast_data'
VALIDATION_PATH = 'tijuana/forecast/validation'