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

from datetime import timedelta

import dagster as dg

from h2s.defs.h2s_pipeline import (
    forecast_daily_partitions,
    h2s_model_artifacts,
    preprocessed_features,
    h2s_predictions,
    h2s_alerts,
    h2s_variant_predictions,
    h2s_ensemble_predictions,
    predictions_export,
    feature_importance_viz,
    confusion_matrix_viz,
    model_comparison_viz,
    prediction_timeline_viz,
    daily_validation_report,
    monthly_performance_viz,
)

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
    production_model_deployment,
    # Partition definitions
    monthly_training_partitions,
    model_run_partitions,
)

from h2s.defs.h2s_multi_station_training import (
    multi_station_training_data,
    per_station_trained_models,
    station_training_report,
    multi_station_training_job,
    STATION_PARTITIONS,
)

from h2s.defs.h2s_daily_pipeline import (
    daily_analysis_job,
)

from h2s.defs.h2s_multihorizon_training import (
    mh_training_job,
    STATION_PARTITIONS as MH_STATION_PARTITIONS,
)

from h2s.defs.h2s_multihorizon_pipeline import (
    mh_forecast_job,
)
from h2s.constants import SCHEDULE_6HR
from h2s.defs.h2s_dispersion_pipeline import (
    lagrangian_source_attribution,
    emission_rate_inversion,
    hysplit_controls_generation,
    gaussian_forward_forecast,
    gaussian_forward_forecast_detailed,
    dispersion_alert_check,
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
        model_comparison_report,
        deployment_approval,
        production_model_deployment,
    ),
    partitions_def=model_run_partitions,
    tags={"environment": "production", "pipeline": "h2s_deployment"},
)


# ============================================================================
# JOB 3b: Approve and Deploy (pre-approved, no config editing needed)
# ============================================================================

approve_and_deploy_job = dg.define_asset_job(
    name="approve_and_deploy_job",
    description=(
        "Approve and deploy a model variant to production. "
        "Select the partition (month | variant) and launch — no config editing required."
    ),
    selection=dg.AssetSelection.assets(
        model_comparison_report,
        deployment_approval,
        production_model_deployment,
    ),
    partitions_def=model_run_partitions,
    config={
        "ops": {
            "h2s__deployment_approval": {
                "config": {"approve_deployment": True}
            }
        }
    },
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
    """Materialize monthly training/validation data on the 1st of each month.

    Uses the previous month as the partition key — the schedule fires on the 1st
    of the current month, but we want to process the just-completed previous month,
    which is the latest available partition under end_offset=0.
    """
    prev_month = (context.scheduled_execution_time.replace(day=1) - timedelta(days=1)).replace(day=1)
    month_key = prev_month.strftime('%Y-%m-%d')
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
    """Train all model variants for the previous (completed) month.

    Runs 2 hours after data extraction to allow data assets to materialize.
    Emits one RunRequest per variant so they train in parallel.
    Uses the previous month partition key to match end_offset=0.
    """
    prev_month = (context.scheduled_execution_time.replace(day=1) - timedelta(days=1)).replace(day=1)
    month_key = prev_month.strftime('%Y-%m-%d')
    return [
        dg.RunRequest(
            partition_key=dg.MultiPartitionKey({"month": month_key, "variant": variant}),
            run_key=f"model_training_{month_key}_{variant}",
            tags={"retraining_period": month_key, "variant": variant},
        )
        for variant in model_run_partitions.get_partitions_def_for_dimension("variant").get_partition_keys()
    ]


# ============================================================================
# JOB 4: 6-Hourly Forecast Prediction (unpartitioned)
# ============================================================================

forecast_prediction_job = dg.define_asset_job(
    name="forecast_prediction_job",
    description="Run full H2S prediction pipeline and export results to S3",
    selection=dg.AssetSelection.assets(
        h2s_model_artifacts,
        preprocessed_features,
        h2s_predictions,
        h2s_alerts,
        h2s_variant_predictions,
        h2s_ensemble_predictions,
        predictions_export,
        feature_importance_viz,
        confusion_matrix_viz,
        model_comparison_viz,
        prediction_timeline_viz,
    ),
    partitions_def=forecast_daily_partitions,
    tags={"environment": "production", "pipeline": "h2s_forecast"},
)


# ============================================================================
# JOB 5: Daily Validation Report (unpartitioned)
# ============================================================================

daily_validation_metrics_job = dg.define_asset_job(
    name="daily_validation_metrics_job",
    description="Generate daily metrics only (useful for backfilling)",
    selection=dg.AssetSelection.assets(
        daily_validation_report,
    ),
    partitions_def=forecast_daily_partitions,
    tags={"environment": "production", "pipeline": "h2s_validation"},
)

daily_validation_job = dg.define_asset_job(
    name="daily_validation_job",
    description="Compare previous day's H2S predictions against actual measurements and generate performance dashboard",
    selection=dg.AssetSelection.assets(
        daily_validation_report,
        monthly_performance_viz,
    ),
    partitions_def=forecast_daily_partitions,
    tags={"environment": "production", "pipeline": "h2s_validation"},
)


# ============================================================================
# SCHEDULE 3: 6-Hourly Forecast (00:00, 06:00, 12:00, 18:00 UTC)
# ============================================================================

@dg.schedule(
    job=forecast_prediction_job,
    cron_schedule=SCHEDULE_6HR,
    description="Run H2S forecast every 6 hours (00:00, 06:00, 12:00, 18:00 UTC)",
    default_status=dg.DefaultScheduleStatus.STOPPED,
    tags={"environment": "production", "schedule_type": "forecast"},
)
def forecast_prediction_schedule(context: dg.ScheduleEvaluationContext):
    """Trigger the full H2S prediction pipeline every 6 hours.

    Each run materializes TODAY's partition (all 4 daily runs update same partition).
    S3 hour paths differentiate individual runs.
    """
    today_utc = context.scheduled_execution_time.date().strftime("%Y-%m-%d")

    return dg.RunRequest(
        partition_key=today_utc,
        run_key=f"forecast_{context.scheduled_execution_time.strftime('%Y-%m-%d_%H')}",
        tags={
            "dagster/schedule_execution_time": context.scheduled_execution_time.isoformat(),
        },
    )


# ============================================================================
# SCHEDULE 4: Daily Validation (8 AM UTC)
# ============================================================================

@dg.schedule(
    job=daily_validation_job,
    cron_schedule="0 8 * * *",
    description="Daily validation report at 8 AM UTC (after actuals data is expected)",
    default_status=dg.DefaultScheduleStatus.RUNNING,
    tags={"environment": "production", "schedule_type": "validation"},
)
def daily_validation_schedule(context: dg.ScheduleEvaluationContext):
    """Trigger daily validation report comparing predictions vs actuals.

    Partition key is YESTERDAY's date (the day being validated).
    """
    yesterday_utc = (context.scheduled_execution_time - timedelta(days=1)).date().strftime("%Y-%m-%d")

    return dg.RunRequest(
        partition_key=yesterday_utc,
        run_key=f"validation_{yesterday_utc}",
    )


# ============================================================================
# SCHEDULE 5: Multi-Station Model Training (2 AM on 1st of month)
# ============================================================================

@dg.schedule(
    job=multi_station_training_job,
    cron_schedule="0 2 1 * *",
    description="Monthly multi-station training — all 3 stations on 1st of month at 2 AM UTC",
    default_status=dg.DefaultScheduleStatus.RUNNING,
    tags={"environment": "production", "schedule_type": "multi_station_training"},
)
def multi_station_training_schedule(context: dg.ScheduleEvaluationContext):
    """Train per-station models for all partitions (one RunRequest per station)."""
    return [
        dg.RunRequest(
            partition_key=partition_key,
            run_key=f"multi_station_training_{context.scheduled_execution_time.strftime('%Y-%m')}_{partition_key}",
            tags={"training_month": context.scheduled_execution_time.strftime('%Y-%m')},
        )
        for partition_key in STATION_PARTITIONS.get_partition_keys()
    ]


# ============================================================================
# SCHEDULE 6: Daily Analysis (14:00 UTC = 6 AM PST)
# ============================================================================

@dg.schedule(
    job=daily_analysis_job,
    cron_schedule=SCHEDULE_6HR,
    description="Daily H2S source attribution + 48h forecast + dashboard (14:00 UTC / 6 AM PST)",
    default_status=dg.DefaultScheduleStatus.RUNNING,
    tags={"environment": "production", "schedule_type": "daily_analysis"},
)
def daily_analysis_schedule(context: dg.ScheduleEvaluationContext):
    """Trigger daily H2S analysis pipeline."""
    return dg.RunRequest(
        run_key=f"daily_analysis_{context.scheduled_execution_time.strftime('%Y-%m-%d')}",
    )


# ============================================================================
# SCHEDULE 7: Multi-Horizon Training (3 AM on 1st of month)
# ============================================================================

@dg.schedule(
    job=mh_training_job,
    cron_schedule="0 3 1 * *",
    description="Monthly multi-horizon training — all 3 stations on 1st of month at 3 AM UTC",
    default_status=dg.DefaultScheduleStatus.STOPPED,
    tags={"environment": "production", "schedule_type": "mh_training"},
)
def mh_training_schedule(context: dg.ScheduleEvaluationContext):
    """Train multi-horizon models for all stations (one RunRequest per station)."""
    return [
        dg.RunRequest(
            partition_key=partition_key,
            run_key=f"mh_training_{context.scheduled_execution_time.strftime('%Y-%m')}_{partition_key}",
            tags={"training_month": context.scheduled_execution_time.strftime('%Y-%m')},
        )
        for partition_key in MH_STATION_PARTITIONS.get_partition_keys()
    ]


# ============================================================================
# SCHEDULE 8: Multi-Horizon Forecast (14:00 UTC daily)
# ============================================================================

@dg.schedule(
    job=mh_forecast_job,
    cron_schedule=SCHEDULE_6HR,
    description="Daily multi-horizon H2S forecast (14:00 UTC / 6 AM PST)",
    default_status=dg.DefaultScheduleStatus.STOPPED,
    tags={"environment": "production", "schedule_type": "mh_forecast"},
)
def mh_forecast_schedule(context: dg.ScheduleEvaluationContext):
    """Trigger daily multi-horizon forecast."""
    return dg.RunRequest(
        run_key=f"mh_forecast_{context.scheduled_execution_time.strftime('%Y-%m-%d')}",
    )


# ============================================================================
# JOB 9: Weekly Dispersion Inversion (Lagrangian + emission rates + HYSPLIT backward bundle)
# ============================================================================

dispersion_inversion_job = dg.define_asset_job(
    name="dispersion_inversion_job",
    description=(
        "Weekly source attribution: Lagrangian backward model → emission rate inversion "
        "→ HYSPLIT backward CONTROL bundle upload. No HYSPLIT execution."
    ),
    selection=dg.AssetSelection.assets(
        lagrangian_source_attribution,
        emission_rate_inversion,
        hysplit_controls_generation,
    ),
    config={
        "ops": {
            "h2s__hysplit_controls_generation": {
                "config": {"mode": "backward_traj"}
            }
        }
    },
    tags={"environment": "production", "pipeline": "h2s_dispersion"},
)


# ============================================================================
# JOB 10: 6-hourly Dispersion Forecast (Gaussian forward + alert check + HYSPLIT forward bundle)
# ============================================================================

dispersion_forecast_job = dg.define_asset_job(
    name="dispersion_forecast_job",
    description=(
        "6-hourly Gaussian plume forward forecast using forecast meteorology, "
        "dispersion alert check, and HYSPLIT forward CONTROL bundle upload. "
        "Runs both 3-source coarse and 16-source detailed models in parallel."
    ),
    selection=dg.AssetSelection.assets(
        emission_rate_inversion,
        gaussian_forward_forecast,
        gaussian_forward_forecast_detailed,
        dispersion_alert_check,
        hysplit_controls_generation,
    ),
    config={
        "ops": {
            "h2s__hysplit_controls_generation": {
                "config": {"mode": "forward_disp"}
            }
        }
    },
    tags={"environment": "production", "pipeline": "h2s_dispersion"},
)


# ============================================================================
# SCHEDULE 9: Weekly Dispersion Inversion (Monday 02:30 UTC)
# Offset 30min from monthly_data_schedule (02:00) to avoid collision on 1st of month.
# Starts STOPPED — enable after reviewing first emission rate inversion results.
# ============================================================================

@dg.schedule(
    job=dispersion_inversion_job,
    cron_schedule="30 2 * * 1",
    description="Weekly Lagrangian inversion + HYSPLIT backward bundle (Monday 02:30 UTC)",
    default_status=dg.DefaultScheduleStatus.STOPPED,
    tags={"environment": "production", "schedule_type": "dispersion_inversion"},
)
def dispersion_inversion_schedule(context: dg.ScheduleEvaluationContext):
    """Re-run source attribution inversion weekly to capture new high-H2S events."""
    return dg.RunRequest(
        run_key=f"dispersion_inversion_{context.scheduled_execution_time.strftime('%Y-%m-%d')}",
    )


# ============================================================================
# SCHEDULE 10: 6-hourly Dispersion Forecast (tied to SCHEDULE_6HR)
# Starts RUNNING — uses calibrated default emission rates until first inversion completes.
# ============================================================================

@dg.schedule(
    job=dispersion_forecast_job,
    cron_schedule=SCHEDULE_6HR,
    description="6-hourly Gaussian forward forecast + alert check + HYSPLIT forward bundle",
    default_status=dg.DefaultScheduleStatus.RUNNING,
    tags={"environment": "production", "schedule_type": "dispersion_forecast"},
)
def dispersion_forecast_schedule(context: dg.ScheduleEvaluationContext):
    """Trigger 6-hourly dispersion forward forecast using current forecast meteorology."""
    return dg.RunRequest(
        run_key=f"dispersion_forecast_{context.scheduled_execution_time.strftime('%Y-%m-%d_%H')}",
    )
