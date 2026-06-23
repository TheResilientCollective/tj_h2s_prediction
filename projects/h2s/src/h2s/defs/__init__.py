"""H2S Dagster Definitions - convenience re-exports.

The legacy single-NESTOR hourly pipeline (h2s_pipeline) was retired; the
multi-station training/products/validation pipelines are registered explicitly
in ``h2s.definitions``.
"""

from h2s.defs.h2s_validation_pipeline import (
    daily_station_validation_report,
)

from h2s.defs.accuracy_reporting_pipeline import (
    daily_accuracy_scorecard,
    rolling_accuracy_scorecards,
    monthly_accuracy_scorecard,
    alert_performance,
    monthly_accuracy_report_html,
    weekly_scorecard_post,
    accuracy_reporting_job,
    monthly_accuracy_job,
    weekly_scorecard_job,
    daily_accuracy_schedule,
    monthly_accuracy_schedule,
    weekly_scorecard_schedule,
)

__all__ = [
    # Validation
    "daily_station_validation_report",
    # Accuracy reporting
    "daily_accuracy_scorecard",
    "rolling_accuracy_scorecards",
    "monthly_accuracy_scorecard",
    "alert_performance",
    "monthly_accuracy_report_html",
    "weekly_scorecard_post",
    "accuracy_reporting_job",
    "monthly_accuracy_job",
    "weekly_scorecard_job",
    "daily_accuracy_schedule",
    "monthly_accuracy_schedule",
    "weekly_scorecard_schedule",
]
