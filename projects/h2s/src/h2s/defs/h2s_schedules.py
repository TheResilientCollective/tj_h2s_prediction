"""H2S Model Retraining Schedules and Jobs.

Jobs are split by partition type:
  - Data assets use monthly_training_partitions (single dimension)
  - Model assets use model_run_partitions (month × variant, multi-dimensional)

Workflow:
1. monthly_data_schedule (2 AM 1st of month) → materializes monthly data assets
2. monthly_model_training_schedule (4 AM 1st of month) → trains both variants in parallel
3. Human reviews validation reports in S3 per variant
4. Manual trigger of deploy_approved_model_job for the chosen variant
"""

import dagster as dg

from h2s.defs.h2s_training_pipeline import (
    # Phase 1: Data Extraction (monthly partitioned)
    monthly_training_data,
    relabeled_training_data,
    data_quality_report,
    training_data,
    validation_data,
    # Phase 2-3: Training + Validation (model_run partitioned)
    trained_model_cv,
    model_training_metrics,
    feature_importance_analysis,
    validation_predictions,
    validation_report,
    model_comparison_report,
    # Phase 4: Deployment (model_run partitioned)
    deployment_approval,
    archived_previous_model,
    production_model_deployment,
    # Partition definitions
    monthly_training_partitions,
    model_run_partitions,
)


# ============================================================================
# JOB 1: Monthly Data Extraction (monthly partitioned)
# ============================================================================

monthly_data_extraction_job = dg.define_asset_job(
    name="monthly_data_extraction_job",
    description="Extract and prepare training/validation data for a given month",
    selection=dg.AssetSelection.assets(
        monthly_training_data,
        relabeled_training_data,
        data_quality_report,
        training_data,
        validation_data,
    ),
    partitions_def=monthly_training_partitions,
    tags={"environment": "production", "pipeline": "h2s_data_extraction"},
)


# ============================================================================
# JOB 2: Monthly Model Training + Validation (model_run partitioned)
# ============================================================================

monthly_model_training_job = dg.define_asset_job(
    name="monthly_model_training_job",
    description="Train and validate a model variant for a given month",
    selection=dg.AssetSelection.assets(
        trained_model_cv,
        model_training_metrics,
        feature_importance_analysis,
        validation_predictions,
        validation_report,
        model_comparison_report,
    ),
    partitions_def=model_run_partitions,
    tags={"environment": "production", "pipeline": "h2s_model_training"},
)


# ============================================================================
# JOB 3: Deploy Approved Model (model_run partitioned, manual trigger)
# ============================================================================

deploy_approved_model_job = dg.define_asset_job(
    name="deploy_approved_model_job",
    description="Deploy an approved model variant to production (manual trigger only)",
    selection=dg.AssetSelection.assets(
        deployment_approval,
        archived_previous_model,
        production_model_deployment,
    ),
    partitions_def=model_run_partitions,
    tags={"environment": "production", "pipeline": "h2s_deployment"},
)


# ============================================================================
# SCHEDULE 1: Monthly Data Extraction (2 AM on 1st of month)
# ============================================================================

@dg.schedule(
    job=monthly_data_extraction_job,
    cron_schedule="0 2 1 * *",
    description="Monthly data extraction (1st of month at 2 AM UTC)",
    default_status=dg.DefaultScheduleStatus.RUNNING,
    tags={"environment": "production", "schedule_type": "monthly_data"},
)
def monthly_data_schedule(context: dg.ScheduleEvaluationContext):
    """Materialize monthly training/validation data on the 1st of each month."""
    month_key = context.scheduled_execution_time.strftime('%Y-%m-01')
    return dg.RunRequest(
        partition_key=month_key,
        run_key=f"data_extraction_{month_key}",
        tags={"retraining_period": month_key},
    )


# ============================================================================
# SCHEDULE 2: Monthly Model Training (4 AM on 1st of month, both variants)
# ============================================================================

@dg.schedule(
    job=monthly_model_training_job,
    cron_schedule="0 4 1 * *",
    description="Monthly model training for all variants (1st of month at 4 AM UTC — 2h after data extraction)",
    default_status=dg.DefaultScheduleStatus.RUNNING,
    tags={"environment": "production", "schedule_type": "monthly_training"},
)
def monthly_model_training_schedule(context: dg.ScheduleEvaluationContext):
    """Train all model variants for the current month.

    Runs 2 hours after data extraction to allow data assets to materialize.
    Emits one RunRequest per variant so they train in parallel.
    """
    month_key = context.scheduled_execution_time.strftime('%Y-%m-01')
    return [
        dg.RunRequest(
            partition_key=dg.MultiPartitionKey({"month": month_key, "variant": variant}),
            run_key=f"model_training_{month_key}_{variant}",
            tags={"retraining_period": month_key, "variant": variant},
        )
        for variant in model_run_partitions.get_partitions_def_for_dimension("variant").get_partition_keys()
    ]