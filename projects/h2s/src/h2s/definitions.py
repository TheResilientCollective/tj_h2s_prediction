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

    # Import two-tier alert system
    from h2s.defs.h2s_alert_system import (
        h2s_alert_dispatcher,
        h2s_alert_sensor,
        h2s_alert_job,
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
        station_model_deployment,
        multi_station_training_job,
        station_deployment_job,
    )

    # Import seed models job
    from h2s.defs.h2s_seed_models import (
        seed_models,
        seed_models_job,
    )

    # Import daily analysis pipeline assets
    from h2s.defs.h2s_daily_pipeline import (
        multi_station_model_artifacts,
        source_attribution,
        daily_station_forecasts,
        daily_dashboard_viz,
        daily_summary_json,
        daily_analysis_job,
    )

    # Import multi-horizon training pipeline assets
    from h2s.defs.h2s_multihorizon_training import (
        mh_training_data,
        mh_trained_models,
        mh_training_report,
        mh_model_deployment,
        mh_training_job,
        mh_deployment_job,
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
    )

    # Import multi-horizon forecast pipeline assets
    from h2s.defs.h2s_multihorizon_pipeline import (
        mh_model_artifacts,
        mh_observation_state,
        mh_forecasts,
        mh_dashboard_viz,
        mh_summary_export,
        mh_slack_alerts,
        mh_forecast_job,
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
        daily_validation_job,
        daily_validation_metrics_job,
        daily_validation_schedule,
        multi_station_training_schedule,
        daily_analysis_schedule,
        mh_training_schedule,
        mh_forecast_schedule,
        dispersion_inversion_job,
        dispersion_forecast_job,
        dispersion_hysplit_execution_job,
        dispersion_inversion_schedule,
        dispersion_forecast_schedule,
        emissions_calibration_job,
        emissions_calibration_schedule,
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
            station_model_deployment,
            # Daily Analysis Pipeline Assets
            multi_station_model_artifacts,
            source_attribution,
            daily_station_forecasts,
            daily_dashboard_viz,
            daily_summary_json,
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
            # Seed Models
            seed_models,
            # Multi-Horizon Training Pipeline Assets
            mh_training_data,
            mh_trained_models,
            mh_training_report,
            mh_model_deployment,
            # Multi-Horizon Forecast Pipeline Assets
            mh_model_artifacts,
            mh_observation_state,
            mh_forecasts,
            mh_dashboard_viz,
            mh_summary_export,
            mh_slack_alerts,
        ],
        jobs=[
            # Prediction jobs
            forecast_prediction_job,
            daily_validation_job,
            daily_validation_metrics_job,
            # Training jobs (old single-model pipeline — kept for reference)
            monthly_data_extraction_job,
            monthly_model_training_job,
            deploy_approved_model_job,
            approve_and_deploy_job,
            # New multi-station training jobs
            multi_station_training_job,
            station_deployment_job,
            # Daily analysis job
            daily_analysis_job,
            # Seed models job
            seed_models_job,
            # Multi-horizon jobs
            mh_training_job,
            mh_deployment_job,
            mh_forecast_job,
            # Dispersion jobs
            dispersion_inversion_job,
            dispersion_forecast_job,
            dispersion_hysplit_execution_job,
            # Rolling emissions calibration job
            emissions_calibration_job,
            # Two-tier alert job
            h2s_alert_job,
            # APCD multi-station sensor watch job
            apcd_sensor_watch_job,
        ],
        schedules=[
            forecast_prediction_schedule,
            daily_validation_schedule,
            monthly_data_schedule,
            monthly_model_training_schedule,
            multi_station_training_schedule,
            daily_analysis_schedule,
            mh_training_schedule,
            mh_forecast_schedule,
            # Dispersion schedules
            dispersion_inversion_schedule,
            dispersion_forecast_schedule,
            # Calibration schedule
            emissions_calibration_schedule,
        ],
        sensors=[slack_on_run_failure, h2s_alert_sensor, apcd_sensor_watch_sensor],
        resources=resources[deployment_name]
    )

    # Load any additional component definitions from defs/ folder
    component_defs = load_from_defs_folder(path_within_project=Path(__file__).parent / "defs")

    return Definitions.merge(all_defs, component_defs)
