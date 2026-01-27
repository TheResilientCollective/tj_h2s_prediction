"""H2S Model Retraining Schedules and Jobs.

This module defines schedules and jobs for monthly model retraining:

1. monthly_retraining_schedule:
   - Runs on 1st of month at 2 AM
   - Executes: data extraction → training → validation
   - Stops at validation_report (does NOT auto-deploy)
   - Allows human review before deployment

2. deploy_approved_model_job:
   - Manual job (no schedule)
   - Must be triggered manually after reviewing validation_report
   - Executes: deployment_approval → archive → production deployment
   - Requires approve_deployment=True config

Workflow:
1. Schedule auto-runs monthly retraining (assets 1-10)
2. Human reviews validation_report.json in S3
3. If acceptable, manually trigger deploy_approved_model_job with approval
4. Job archives old model and deploys new model to production
"""

import dagster as dg

from h2s.defs.h2s_training_pipeline import (
    # Phase 1: Data Extraction
    monthly_training_data,
    relabeled_training_data,
    data_quality_report,
    training_data,
    validation_data,
    # Phase 2: Training
    trained_model_cv,
    model_training_metrics,
    feature_importance_analysis,
    # Phase 3: Validation
    validation_predictions,
    validation_report,
    model_comparison_report,
    # Phase 4: Deployment
    deployment_approval,
    archived_previous_model,
    production_model_deployment,
)


# ============================================================================
# JOB 1: Monthly Retraining (Automated)
# ============================================================================
# Runs data extraction, training, and validation
# STOPS at validation_report for human review

monthly_retraining_job = dg.define_asset_job(
    name="monthly_retraining_job",
    description="Monthly model retraining job (training + validation only, no deployment)",
    selection=dg.AssetSelection.assets(
        # Phase 1: Data Extraction
        monthly_training_data,
        relabeled_training_data,
        data_quality_report,
        training_data,
        validation_data,
        # Phase 2: Training
        trained_model_cv,
        model_training_metrics,
        feature_importance_analysis,
        # Phase 3: Validation
        validation_predictions,
        validation_report,
        model_comparison_report,
    ),
    tags={"environment": "production", "pipeline": "h2s_retraining"},
)

# ============================================================================
# JOB 2: Deploy Approved Model (Manual Trigger Only)
# ============================================================================
# Executes deployment after human approval

deploy_approved_model_job = dg.define_asset_job(
    name="deploy_approved_model_job",
    description="Deploy approved model to production (manual trigger only)",
    selection=dg.AssetSelection.assets(
        # Phase 4: Deployment
        deployment_approval,
        archived_previous_model,
        production_model_deployment,
    ),
    tags={"environment": "production", "pipeline": "h2s_deployment"},
    # Note: approve_deployment config must be set at launch time:
    # uv run dg launch --job deploy_approved_model_job \
    #   --config '{"ops": {"deployment_approval": {"config": {"approve_deployment": true}}}}'
)


# ============================================================================
# SCHEDULE: Monthly Retraining (1st of month, 2 AM)
# ============================================================================


@dg.schedule(
    job=monthly_retraining_job,
    cron_schedule="0 2 1 * *",  # Minute Hour Day Month DayOfWeek
    description="Monthly model retraining (1st of month at 2 AM UTC)",
    default_status=dg.DefaultScheduleStatus.RUNNING,
    tags={"environment": "production", "schedule_type": "monthly"},
)
def monthly_retraining_schedule(context: dg.ScheduleEvaluationContext):
    """Execute monthly model retraining on 1st of each month at 2 AM UTC.

    Cron: 0 2 1 * * (2 AM on day 1 of every month)

    This schedule automatically:
    1. Loads training data from previous month
    2. Relabels with new H2S thresholds
    3. Trains XGBoost with 5-fold CV
    4. Validates new model vs current production model
    5. Generates validation report in S3

    Does NOT automatically deploy - stops at validation_report to allow
    human review before deployment.

    Review Process:
    1. Check S3 for validation_report.json:
       s3://test/tijuana/forecast/models/training/{YYYY_MM}/validation_report.json

    2. Review metrics:
       - Balanced accuracy delta
       - Orange recall delta (critical for safety)
       - Quality gates passed/failed

    3. If acceptable, trigger deploy_approved_model_job manually:
       uv run dg launch --job deploy_approved_model_job

    4. If not acceptable, investigate issues and retrain with adjustments
    """
    run_config = {
        "resources": {
            "s3": {
                "config": {
                    # S3 resources configured via EnvVar in definitions.py
                }
            }
        }
    }

    return dg.RunRequest(
        run_key=f"monthly_retraining_{context.scheduled_execution_time.strftime('%Y_%m')}",
        run_config=run_config,
        tags={
            "scheduled_execution_time": context.scheduled_execution_time.isoformat(),
            "retraining_period": context.scheduled_execution_time.strftime('%Y_%m'),
        }
    )


# ============================================================================
# SENSORS (Optional Future Enhancement)
# ============================================================================
# Could add sensors to:
# - Notify on training completion
# - Alert if quality gates fail
# - Monitor for deployment approval timeout (e.g., >7 days without action)
