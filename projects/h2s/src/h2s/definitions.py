import os
from pathlib import Path

from dagster import Definitions, EnvVar, definitions, load_from_defs_folder

from h2s.resources.minio import S3Resource

# Configure S3 resource (using EnvVar for Dagster config)
s3_resource = S3Resource(
    S3_BUCKET=EnvVar('S3_BUCKET'),
    S3_ADDRESS=EnvVar('S3_ADDRESS'),
    S3_PORT=EnvVar('S3_PORT'),
    S3_USE_SSL=os.environ.get('S3_USE_SSL', 'true').lower() == 'true',
    S3_ACCESS_KEY=EnvVar('S3_ACCESS_KEY'),
    S3_SECRET_KEY=EnvVar('S3_SECRET_KEY'),
)

resources = {
    "local": {"s3": s3_resource},
    "production": {"s3": s3_resource},
}

deployment_name = os.environ.get("DAGSTER_DEPLOYMENT", "local")


@definitions
def defs():
    # Import prediction pipeline assets
    from h2s.defs.h2s_pipeline import (
        h2s_model_artifacts,
        raw_environmental_data,
        actual_h2s_data,
        preprocessed_features,
        h2s_predictions,
        h2s_alerts,
        feature_importance_viz,
        confusion_matrix_viz,
        model_comparison_viz,
        prediction_timeline_viz,
        predictions_export,
        daily_validation_report,
    )

    # Import training pipeline assets
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

    # Import schedules and jobs
    from h2s.defs.h2s_schedules import (
        monthly_data_extraction_job,
        monthly_model_training_job,
        deploy_approved_model_job,
        monthly_data_schedule,
        monthly_model_training_schedule,
        forecast_prediction_job,
        forecast_prediction_schedule,
        daily_validation_job,
        daily_validation_schedule,
    )

    # Create definitions with assets, jobs, schedules, and resources
    all_defs = Definitions(
        assets=[
            # Prediction Pipeline Assets (12 assets)
            h2s_model_artifacts,
            raw_environmental_data,
            actual_h2s_data,
            preprocessed_features,
            h2s_predictions,
            h2s_alerts,
            feature_importance_viz,
            confusion_matrix_viz,
            model_comparison_viz,
            prediction_timeline_viz,
            predictions_export,
            daily_validation_report,
            # Training Pipeline Assets (14 assets)
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
        ],
        jobs=[
            monthly_data_extraction_job,
            monthly_model_training_job,
            deploy_approved_model_job,
            forecast_prediction_job,
            daily_validation_job,
        ],
        schedules=[
            monthly_data_schedule,
            monthly_model_training_schedule,
            forecast_prediction_schedule,
            daily_validation_schedule,
        ],
        resources=resources[deployment_name]
    )

    # Load any additional component definitions from defs/ folder
    component_defs = load_from_defs_folder(path_within_project=Path(__file__).parent / "defs")

    return Definitions.merge(all_defs, component_defs)
