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
    # Import assets from h2s_pipeline
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
    )

    # Create definitions with assets and resources
    asset_defs = Definitions(
        assets=[
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
        ],
        resources=resources[deployment_name]
    )

    # Load any additional component definitions
    component_defs = load_from_defs_folder(path_within_project=Path(__file__).parent / "defs")

    return Definitions.merge(asset_defs, component_defs)
