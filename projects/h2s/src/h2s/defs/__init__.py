"""H2S Dagster Definitions - Export all assets."""

from h2s.defs.h2s_pipeline import (
    # Model Management
    h2s_model_artifacts,

    # Prediction Pipeline
    preprocessed_features,
    h2s_predictions,
    h2s_alerts,
    # Visualization & Export
    feature_importance_viz,
    confusion_matrix_viz,
    model_comparison_viz,
    prediction_timeline_viz,
    predictions_export,
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
    "h2s_model_artifacts",
    "raw_environmental_data",
    "actual_h2s_data",
    "preprocessed_features",
    "h2s_predictions",
    "h2s_alerts",
    "feature_importance_viz",
    "confusion_matrix_viz",
    "model_comparison_viz",
    "prediction_timeline_viz",
    "predictions_export",
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
