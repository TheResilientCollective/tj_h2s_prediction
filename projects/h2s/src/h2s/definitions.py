import os
from pathlib import Path

from dagster import (
    Definitions, EnvVar, definitions, load_from_defs_folder,
   RunFailureSensorContext
)
from dagster_slack import make_slack_on_run_failure_sensor

from h2s.resources.minio import S3Resource
from h2s.resources.slack import SlackAlertResource

# Configure S3 resource (using EnvVar for Dagster config)
s3_resource = S3Resource(
    S3_BUCKET=EnvVar('S3_BUCKET'),
    S3_ADDRESS=EnvVar('S3_ADDRESS'),
    S3_PORT=EnvVar('S3_PORT'),
    S3_USE_SSL=os.environ.get('S3_USE_SSL', 'true').lower() == 'true',
    S3_ACCESS_KEY=EnvVar('S3_ACCESS_KEY'),
    S3_SECRET_KEY=EnvVar('S3_SECRET_KEY'),
)

# Configure Slack resource for alert notifications
slack_resource = SlackAlertResource(
    token=EnvVar('SLACK_TOKEN'),
    channel=os.environ.get('SLACK_CHANNEL', '#test'),
)
def slack_message_fn(context: RunFailureSensorContext) -> str:
    return (
        f"Job *[{context.dagster_run.job_name}]* failed! "
        f"Error: {context.failure_event.message}"
    )
slack_on_run_failure = make_slack_on_run_failure_sensor(
     os.environ.get("SLACK_CHANNEL_FAILURES", "test_failure"),
    os.getenv("SLACK_TOKEN"),
    webserver_base_url=f'https://{os.environ.get("SCHED_HOSTNAME", "sched")}.{os.environ.get("HOST", "local")}/',
    text_fn=slack_message_fn
)

resources = {
    "local": {"s3": s3_resource, "slack": slack_resource},
    "production": {"s3": s3_resource, "slack": slack_resource},
}

deployment_name = os.environ.get("DAGSTER_DEPLOYMENT", "local")


@definitions
def defs():
    # Import prediction pipeline assets
    from h2s.defs.h2s_pipeline import (
        h2s_model_artifacts,
        preprocessed_features,
        h2s_predictions,
        h2s_alerts,
        slack_alerts,
        h2s_variant_predictions,
        h2s_ensemble_predictions,
        feature_importance_viz,
        confusion_matrix_viz,
        model_comparison_viz,
        prediction_timeline_viz,
        cross_correlation_viz,
        predictions_export,
        daily_validation_report,
        monthly_performance_viz,
    )

    # Import two-tier alert system (Tiers 4–5: observation-based)
    from h2s.defs.h2s_alert_system import (
        h2s_alert_dispatcher,
        h2s_alert_sensor,
        h2s_alert_job,
    )

    # Import forecast cascade pre-alert system (Tiers 1–3: product-probability-driven)
    from h2s.defs.cascade_alerts.assets import (
        cascade_alert_dispatcher,
    )
    from h2s.defs.cascade_alerts.schedules import (
        cascade_alerts_job,
        cascade_alerts_schedule,
    )

    # Import observed >10 ppb "Alert Performance" machine (yellow-tier state machine)
    from h2s.defs.alert_performance.assets import (
        h2s_alert_performance_dispatcher,
        h2s_alert_performance_sensor,
        h2s_alert_performance_job,
    )

    # Import APCD multi-station sensor watch
    from h2s.defs.apcd_sensor_watch import (
        apcd_sensor_alert_dispatcher,
        apcd_sensor_watch_sensor,
        apcd_sensor_watch_job,
    )

    # Import multi-station training pipeline assets
    from h2s.defs.h2s_multi_station_training import (
        multi_station_training_data,
        per_station_trained_models,
        station_training_report,
        station_model_archive,
        station_model_deployment,
        station_model_promotion,
        station_model_training_job,
        station_model_deployment_job,
        station_model_promotion_job,
    )

    # Import nowcast/nearcast/forecast products pipeline
    from h2s.defs.h2s_products_pipeline import (
        products_model_artifacts,
        h2s_products,
        station_forecast_job,
    )

    # Import forecast validation store + accuracy reporting (Phase 5)
    from h2s.defs.h2s_forecast_validation_pipeline import (
        forecast_validation_store,
        forecast_skill_report,
        forecast_validation_job,
        station_forecast_validation_rebuild_job,
        forecast_validation_schedule,
    )

    # Import station forecast analysis pipeline assets
    from h2s.defs.h2s_daily_pipeline import (
        multi_station_model_artifacts,
        source_attribution,
        station_forecasts,
        station_forecast_dashboard,
        station_forecast_summary,
        station_forecast_analysis_job,
    )

    # Import dispersion modeling pipeline assets
    from h2s.defs.h2s_dispersion_pipeline import (
        lagrangian_source_attribution,
        emission_rate_inversion,
        hysplit_controls_generation,
        hysplit_run_results,
        gaussian_forward_forecast,
        gaussian_forward_forecast_detailed,
        dispersion_alert_check,
    )

    # Import physics-based river emission grid asset
    from h2s.defs.h2s_river_emissions_pipeline import (
        river_emission_grid,
    )

    # Import rolling emissions calibration pipeline assets
    from h2s.defs.h2s_calibration_pipeline import (
        rolling_footprint_matrix,
        channel_emission_inversion,
        calibration_diagnostics,
        calibration_viz,
    )

    # Import validation pipeline assets
    from h2s.defs.h2s_validation_pipeline import (
        daily_station_validation_report,
    )

    # Import backfill pipeline assets + jobs
    from h2s.defs.h2s_backfill_pipeline import (
        backfill_training_data,
        backfill_station_models,
        station_backtest_results,
        backtest_comparison_index,
        station_backfill_training_job,
        station_backtest_index_job,
    )

    # Import schedules and jobs
    from h2s.defs.h2s_schedules import (
        monthly_data_extraction_job,
        monthly_model_training_job,
        deploy_approved_model_job,
        approve_and_deploy_job,
        monthly_data_schedule,
        monthly_model_training_schedule,
        forecast_prediction_job,
        forecast_prediction_schedule,
        station_forecast_validation_job,
        station_forecast_validation_metrics_job,
        station_forecast_validation_schedule,
        station_model_training_schedule,
        station_forecast_analysis_schedule,
        station_backfill_schedule,
        dispersion_inversion_job,
        dispersion_forecast_job,
        dispersion_hysplit_execution_job,
        dispersion_inversion_schedule,
        dispersion_forecast_schedule,
        emissions_calibration_job,
        emissions_calibration_schedule,
        daily_station_validation_job,
        daily_station_validation_schedule,
    )

    # Create definitions with assets, jobs, schedules, and resources
    all_defs = Definitions(
        assets=[
            # Prediction Pipeline Assets
            h2s_model_artifacts,
            preprocessed_features,
            h2s_predictions,
            h2s_alerts,
            slack_alerts,
            h2s_alert_dispatcher,
            apcd_sensor_alert_dispatcher,
            h2s_variant_predictions,
            h2s_ensemble_predictions,
            feature_importance_viz,
            confusion_matrix_viz,
            model_comparison_viz,
            prediction_timeline_viz,
            cross_correlation_viz,
            predictions_export,
            daily_validation_report,
            monthly_performance_viz,
            # Multi-Station Training Pipeline Assets
            multi_station_training_data,
            per_station_trained_models,
            station_training_report,
            station_model_archive,
            station_model_deployment,
            station_model_promotion,
            # Products pipeline (nowcast / nearcast / forecast)
            products_model_artifacts,
            h2s_products,
            # Station Forecast Analysis Pipeline Assets
            multi_station_model_artifacts,
            source_attribution,
            station_forecasts,
            station_forecast_dashboard,
            station_forecast_summary,
            # Dispersion Pipeline Assets
            lagrangian_source_attribution,
            emission_rate_inversion,
            hysplit_controls_generation,
            hysplit_run_results,
            gaussian_forward_forecast,
            gaussian_forward_forecast_detailed,
            dispersion_alert_check,
            # Physics-based river emission grid
            river_emission_grid,
            # Rolling Emissions Calibration
            rolling_footprint_matrix,
            channel_emission_inversion,
            calibration_diagnostics,
            calibration_viz,
            # Validation Pipeline Assets
            daily_station_validation_report,
            # Forecast Validation Store + Accuracy Reporting (Phase 5)
            forecast_validation_store,
            forecast_skill_report,
            # Forecast Cascade Pre-Alert (Tiers 1–3, product-probability-driven)
            cascade_alert_dispatcher,
            # Observed >10 ppb Alert-Performance machine (yellow tier)
            h2s_alert_performance_dispatcher,
            # Walk-forward backfill pipeline
            backfill_training_data,
            backfill_station_models,
            station_backtest_results,
            backtest_comparison_index,
        ],
        jobs=[
            # Prediction jobs
            forecast_prediction_job,
            station_forecast_validation_job,
            station_forecast_validation_metrics_job,
            # Training jobs (old single-model pipeline — kept for reference)
            monthly_data_extraction_job,
            monthly_model_training_job,
            deploy_approved_model_job,
            approve_and_deploy_job,
            # New multi-station training jobs
            station_model_training_job,
            station_model_deployment_job,
            station_model_promotion_job,
            # Products job
            station_forecast_job,
            # Forecast validation store + rebuild (Phase 5)
            station_forecast_validation_rebuild_job,
            # Station forecast analysis job
            station_forecast_analysis_job,
            # Dispersion jobs
            dispersion_inversion_job,
            dispersion_forecast_job,
            dispersion_hysplit_execution_job,
            # Rolling emissions calibration job
            emissions_calibration_job,
            # Two-tier observation alert job (Tiers 4–5)
            h2s_alert_job,
            # APCD multi-station sensor watch job
            apcd_sensor_watch_job,
            # Forecast cascade pre-alert job (Tiers 1–3)
            cascade_alerts_job,
            # Observed >10 ppb Alert-Performance job (yellow tier)
            h2s_alert_performance_job,
            # Validation jobs
            daily_station_validation_job,
            # Walk-forward backfill jobs
            station_backfill_training_job,
            station_backtest_index_job,
        ],
        schedules=[
            forecast_prediction_schedule,
            station_forecast_validation_schedule,
            monthly_data_schedule,
            monthly_model_training_schedule,
            station_model_training_schedule,
            station_forecast_analysis_schedule,
            # Dispersion schedules
            dispersion_inversion_schedule,
            dispersion_forecast_schedule,
            # Calibration schedule
            emissions_calibration_schedule,
            # Validation schedules
            daily_station_validation_schedule,
            # Forecast cascade pre-alert schedule
            cascade_alerts_schedule,
            # Walk-forward backfill schedule
            station_backfill_schedule,
        ],
        sensors=[slack_on_run_failure, h2s_alert_sensor, apcd_sensor_watch_sensor,
                 h2s_alert_performance_sensor],
        resources=resources[deployment_name]
    )

    # Load any additional component definitions from defs/ folder
    component_defs = load_from_defs_folder(path_within_project=Path(__file__).parent / "defs")

    return Definitions.merge(all_defs, component_defs)
