"""H2S Schedules and Jobs (multi-station pipeline).

Holds the per-station training schedule, the station forecast-analysis schedule,
the dispersion/calibration jobs + schedules, and the daily-station validation
job. The legacy single-NESTOR hourly forecast and monthly single-model training
pipelines were retired — model production is the per-station training path
(station_model_training_job → station_model_deployment_job) and the products
pipeline (station_forecast_job).
"""

import os
from datetime import timedelta

import dagster as dg

from h2s.defs.h2s_multi_station_training import (
    multi_station_training_data,
    per_station_trained_models,
    station_training_report,
    station_model_training_job,
    STATION_PARTITIONS,
)

from h2s.defs.h2s_daily_pipeline import (
    station_forecast_analysis_job,
)

from h2s.constants import SCHEDULE_6HR,SCHEDULE_1HR
from h2s.defs.h2s_dispersion_pipeline import (
    lagrangian_source_attribution,
    emission_rate_inversion,
    hysplit_controls_generation,
    hysplit_run_results,
    gaussian_forward_forecast,
    gaussian_forward_forecast_detailed,
    dispersion_alert_check,
)
from h2s.defs.h2s_river_emissions_pipeline import river_emission_grid

from h2s.defs.h2s_calibration_pipeline import (
    CALIBRATION_WEEKLY_PARTITIONS,
    rolling_footprint_matrix,
    channel_emission_inversion,
    calibration_diagnostics,
    calibration_viz,
)

from h2s.defs.h2s_validation_pipeline import (
    daily_station_validation_report,
    forecast_daily_partitions as validation_daily_partitions,
)

# ============================================================================
# SCHEDULE 5: Station Model Training (2 AM on 1st of month)
# ============================================================================

@dg.schedule(
    job=station_model_training_job,
    cron_schedule="0 2 1 * *",
    execution_timezone="America/Los_Angeles",
    description="Monthly station model training — all 3 stations on 1st of month at 2 AM ",
    default_status=dg.DefaultScheduleStatus.RUNNING,
    tags={"environment": "production", "schedule_type": "station_model_training"},
)
def station_model_training_schedule(context: dg.ScheduleEvaluationContext):
    """Train per-station models for all partitions (one RunRequest per station)."""
    return [
        dg.RunRequest(
            partition_key=partition_key,
            run_key=f"station_model_training_{context.scheduled_execution_time.strftime('%Y-%m')}_{partition_key}",
            tags={"training_month": context.scheduled_execution_time.strftime('%Y-%m')},
        )
        for partition_key in STATION_PARTITIONS.get_partition_keys()
    ]


# ============================================================================
# SCHEDULE 6: Station Forecast Analysis (every 6h)
# ============================================================================

@dg.schedule(
    job=station_forecast_analysis_job,
    cron_schedule=SCHEDULE_1HR,
    execution_timezone="America/Los_Angeles",
    description="Station forecast analysis: H2S source attribution + 24h forecast + dashboard (every 6 hours: 00, 06, 12, 18 )",
    default_status=dg.DefaultScheduleStatus.RUNNING,
    tags={"environment": "production", "schedule_type": "station_forecast_analysis"},
)
def station_forecast_analysis_schedule(context: dg.ScheduleEvaluationContext):
    """Trigger station forecast analysis pipeline."""
    return dg.RunRequest(
        run_key=f"station_forecast_analysis_{context.scheduled_execution_time.strftime('%Y-%m-%d_%H')}",
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
        river_emission_grid,
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
# JOB 10b: HYSPLIT Queue Execution (bundle generation + queue-driven run)
#
# This job routes `hysplit_run_results` to a dedicated HYSPLIT worker via
# dagster-celery. The worker container ships HYSPLIT binaries, mounts the
# GDAS meteorology directory, and consumes work from a Redis queue tagged
# with `dagster-celery/queue: hysplit` (set on the asset's op_tags).
#
# The existing dispersion_forecast_job / dispersion_inversion_job keep the
# default in-process executor — minimal blast radius, existing schedules
# unchanged.
# ============================================================================

try:
    from dagster_celery import celery_executor

    _celery_broker_url = os.environ.get(
        "DAGSTER_CELERY_BROKER", "redis://redis:6379/0"
    )
    _celery_backend_url = os.environ.get(
        "DAGSTER_CELERY_BACKEND", "redis://redis:6379/0"
    )
    _dispersion_hysplit_executor = celery_executor.configured(
        {
            "broker": _celery_broker_url,
            "backend": _celery_backend_url,
        }
    )
except ImportError:
    # dagster-celery not installed (e.g. dev environment without worker) —
    # fall back to in-process so `dg check defs` still works.
    _dispersion_hysplit_executor = None


_dispersion_hysplit_job_kwargs = dict(
    name="dispersion_hysplit_execution_job",
    description=(
        "Generate HYSPLIT forward bundle and execute it on the dedicated "
        "HYSPLIT worker via dagster-celery queue routing. Uploads tdump/cdump "
        "outputs to tijuana/forecasts/dispersion/hysplit/runs/."
    ),
    selection=dg.AssetSelection.assets(
        hysplit_controls_generation,
        hysplit_run_results,
    ),
    config={
        "ops": {
            "h2s__hysplit_controls_generation": {
                "config": {"mode": "forward_disp"}
            },
            "h2s__hysplit_run_results": {
                "config": {"mode": "forward_disp"}
            },
        }
    },
    tags={"environment": "production", "pipeline": "h2s_dispersion_hysplit_execution"},
)
if _dispersion_hysplit_executor is not None:
    _dispersion_hysplit_job_kwargs["executor_def"] = _dispersion_hysplit_executor

dispersion_hysplit_execution_job = dg.define_asset_job(
    **_dispersion_hysplit_job_kwargs,
)

# ============================================================================
# JOB 10b: HYSPLIT Queue Execution (bundle generation + queue-driven run)
#
# This job routes `hysplit_run_results` to a dedicated HYSPLIT worker via
# dagster-celery. The worker container ships HYSPLIT binaries, mounts the
# GDAS meteorology directory, and consumes work from a Redis queue tagged
# with `dagster-celery/queue: hysplit` (set on the asset's op_tags).
#
# The existing dispersion_forecast_job / dispersion_inversion_job keep the
# default in-process executor — minimal blast radius, existing schedules
# unchanged.
# ============================================================================

try:
    from dagster_celery import celery_executor

    _celery_broker_url = os.environ.get(
        "DAGSTER_CELERY_BROKER", "redis://redis:6379/0"
    )
    _celery_backend_url = os.environ.get(
        "DAGSTER_CELERY_BACKEND", "redis://redis:6379/0"
    )
    _dispersion_hysplit_executor = celery_executor.configured(
        {
            "broker": _celery_broker_url,
            "backend": _celery_backend_url,
        }
    )
except ImportError:
    # dagster-celery not installed (e.g. dev environment without worker) —
    # fall back to in-process so `dg check defs` still works.
    _dispersion_hysplit_executor = None


_dispersion_hysplit_job_kwargs = dict(
    name="dispersion_hysplit_execution_job",
    description=(
        "Generate HYSPLIT forward bundle and execute it on the dedicated "
        "HYSPLIT worker via dagster-celery queue routing. Uploads tdump/cdump "
        "outputs to tijuana/forecasts/dispersion/hysplit/runs/."
    ),
    selection=dg.AssetSelection.assets(
        hysplit_controls_generation,
        hysplit_run_results,
    ),
    config={
        "ops": {
            "h2s__hysplit_controls_generation": {
                "config": {"mode": "forward_disp"}
            },
            "h2s__hysplit_run_results": {
                "config": {"mode": "forward_disp"}
            },
        }
    },
    tags={"environment": "production", "pipeline": "h2s_dispersion_hysplit_execution"},
)
if _dispersion_hysplit_executor is not None:
    _dispersion_hysplit_job_kwargs["executor_def"] = _dispersion_hysplit_executor

dispersion_hysplit_execution_job = dg.define_asset_job(
    **_dispersion_hysplit_job_kwargs,
)


# ============================================================================
# SCHEDULE 9: Weekly Dispersion Inversion (Monday 02:30 UTC)
# Offset 30min from the 02:00 station-model-training schedule to avoid collision.
# Starts STOPPED — enable after reviewing first emission rate inversion results.
# ============================================================================

@dg.schedule(
    job=dispersion_inversion_job,
    cron_schedule="30 2 * * 1",
    execution_timezone="America/Los_Angeles",
    description="Weekly Lagrangian inversion + HYSPLIT backward bundle (Monday 02:30 )",
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
    execution_timezone="America/Los_Angeles",
    description="6-hourly Gaussian forward forecast + alert check + HYSPLIT forward bundle",
    default_status=dg.DefaultScheduleStatus.RUNNING,
    tags={"environment": "production", "schedule_type": "dispersion_forecast"},
)
def dispersion_forecast_schedule(context: dg.ScheduleEvaluationContext):
    """Trigger 6-hourly dispersion forward forecast using current forecast meteorology."""
    return dg.RunRequest(
        run_key=f"dispersion_forecast_{context.scheduled_execution_time.strftime('%Y-%m-%d_%H')}",
    )


# ============================================================================
# JOB 11: Weekly Rolling Emissions Calibration (partitioned)
# ============================================================================
#
# Runs channel-snapped NNLS over one weekly partition (Monday-start). Produces
# a per-partition Q field parquet under weekly/{partition}/Q_field.parquet and
# updates Q_field_latest.parquet only when the partition is recent. Skips
# weeks with fewer than `min_events_per_week` qualifying events (default 3).
#
# Schedule fires Monday mornings to materialize the just-completed previous
# week. Supports 2025-onward historical backfills via Dagster partition UI.
#
# Starts STOPPED. Enable after first backfill passes diagnostics.
# ============================================================================

emissions_calibration_job = dg.define_asset_job(
    name="emissions_calibration_job",
    description=(
        "Weekly-partitioned channel-snapped emission calibration. Each "
        "partition covers [Monday, Monday+7d). Produces per-week Q field "
        "parquets (100-segment g/s field along the Tijuana River main stem "
        "+ tributaries) plus LOO CV and budget diagnostics. Recent-week runs "
        "update Q_field_latest.parquet for the live dispersion forecast."
    ),
    selection=dg.AssetSelection.assets(
        rolling_footprint_matrix,
        channel_emission_inversion,
        calibration_diagnostics,
        calibration_viz,
    ),
    partitions_def=CALIBRATION_WEEKLY_PARTITIONS,
    tags={"environment": "production", "pipeline": "h2s_calibration"},
)


# ============================================================================
# SCHEDULE 11: Weekly emissions calibration (Monday 03:30 )
# ============================================================================
# Materializes the previous week's partition (the just-completed Monday-start
# week). Offset 30 min from dispersion_inversion_schedule (Monday 02:30 ).

@dg.schedule(
    job=emissions_calibration_job,
    cron_schedule="30 3 * * 1",
    execution_timezone="America/Los_Angeles",
    description="Weekly rolling emissions calibration (Monday 03:30 )",
    default_status=dg.DefaultScheduleStatus.STOPPED,
    tags={"environment": "production", "schedule_type": "emissions_calibration"},
)
def emissions_calibration_schedule(context: dg.ScheduleEvaluationContext):
    """Materialize the just-completed previous week's Q field."""
    scheduled = context.scheduled_execution_time
    # Back up to the Monday that begins the just-completed week.
    days_since_monday = scheduled.weekday()            # Monday = 0
    this_monday = (scheduled - timedelta(days=days_since_monday)).date()
    previous_monday = this_monday - timedelta(days=7)  # the partition being materialized
    partition_key = previous_monday.strftime("%Y-%m-%d")
    return dg.RunRequest(
        partition_key=partition_key,
        run_key=f"emissions_calibration_{partition_key}",
        tags={"calibration_week": partition_key},
    )


# ============================================================================
# JOB 12: Daily Station Validation (partitioned)
# ============================================================================

daily_station_validation_job = dg.define_asset_job(
    name="daily_station_validation_job",
    description="Validate daily per-station forecasts against observations",
    selection=dg.AssetSelection.assets(daily_station_validation_report),
    partitions_def=validation_daily_partitions,
    tags={"environment": "production", "pipeline": "h2s_validation"},
)


# ============================================================================
# SCHEDULE 12: Daily Station Validation (9 AM  — 1h after hourly validation)
# ============================================================================

@dg.schedule(
    job=daily_station_validation_job,
    cron_schedule="0 7 * * *",
    execution_timezone="America/Los_Angeles",
    description="Daily station forecast validation at 7 AM ",
    default_status=dg.DefaultScheduleStatus.RUNNING,
    tags={"environment": "production", "schedule_type": "validation"},
)
def daily_station_validation_schedule(context: dg.ScheduleEvaluationContext):
    """Validate yesterday's daily station forecasts against observations."""
    yesterday_utc = (context.scheduled_execution_time - timedelta(days=1)).date().strftime("%Y-%m-%d")
    return dg.RunRequest(
        partition_key=yesterday_utc,
        run_key=f"daily_station_validation_{yesterday_utc}",
    )


# ============================================================================
# SCHEDULE 13: Monthly backfill training + backtest (2 AM on 2nd of month)
# Offset from station_model_training_schedule (1st of month at 2 AM) to avoid
# overlap. Starts STOPPED — run station_backfill_training_job --partition
# YYYY-MM-DD to backfill historical months, then enable.
# ============================================================================

from h2s.defs.h2s_backfill_pipeline import (  # noqa: E402
    BACKFILL_MONTHLY_PARTITIONS,
    station_backfill_training_job,
    station_backtest_index_job,
)


@dg.schedule(
    job=station_backfill_training_job,
    cron_schedule="0 2 2 * *",
    execution_timezone="America/Los_Angeles",
    description=(
        "Monthly walk-forward backfill: train on pre-cutoff data, evaluate OOS "
        "(2nd of month at 2 AM  — day after station_model_training_schedule)"
    ),
    default_status=dg.DefaultScheduleStatus.STOPPED,
    tags={"environment": "production", "schedule_type": "backfill_training"},
)
def station_backfill_schedule(context: dg.ScheduleEvaluationContext):
    """Train the previous month's backfill models and evaluate OOS."""
    prev_month = (
        context.scheduled_execution_time.replace(day=1) - timedelta(days=1)
    ).replace(day=1)
    partition_key = prev_month.strftime("%Y-%m-%d")
    return dg.RunRequest(
        partition_key=partition_key,
        run_key=f"backfill_{partition_key}",
        tags={"backfill_month": partition_key},
    )
