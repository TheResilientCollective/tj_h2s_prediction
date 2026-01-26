"""Tests for S3 integration and resource connections.

These tests require S3 credentials in environment variables.
Use pytest markers to skip if credentials not available.
"""

import os
import pytest
from io import BytesIO
import pandas as pd
import numpy as np

from h2s.resources.minio import S3Resource
from h2s.predictor.h2s_predictor import H2SPredictor
from h2s.utils.store_assets import store_dataframe_to_s3, objectMetadata


# Skip all tests in this file if S3 credentials not available
pytestmark = pytest.mark.skipif(
    not all([
        os.getenv('S3_BUCKET'),
        os.getenv('S3_ADDRESS'),
        os.getenv('S3_ACCESS_KEY'),
        os.getenv('S3_SECRET_KEY')
    ]),
    reason="S3 credentials not available in environment"
)


@pytest.fixture
def s3_resource():
    """Create S3Resource with environment credentials."""
    return S3Resource(
        S3_BUCKET=os.getenv('S3_BUCKET', 'test'),
        S3_ADDRESS=os.getenv('S3_ADDRESS'),
        S3_PORT=os.getenv('S3_PORT', '443'),
        S3_USE_SSL=os.getenv('S3_USE_SSL', 'true').lower() == 'true',
        S3_ACCESS_KEY=os.getenv('S3_ACCESS_KEY'),
        S3_SECRET_KEY=os.getenv('S3_SECRET_KEY'),
    )


class TestS3ResourceConnection:
    """Test S3Resource connection and basic operations."""

    def test_can_create_s3_resource(self, s3_resource):
        """Test that S3Resource can be instantiated."""
        assert s3_resource is not None
        assert s3_resource.S3_BUCKET is not None
        assert s3_resource.S3_ADDRESS is not None

    def test_can_get_minio_client(self, s3_resource):
        """Test that MinIO client can be obtained."""
        client = s3_resource.getClient()
        assert client is not None

    def test_base_url_format(self, s3_resource):
        """Test that base URL is correctly formatted."""
        url = s3_resource.baseUrl()

        if s3_resource.S3_USE_SSL:
            assert url.startswith('https://'), "SSL enabled should use https"
        else:
            assert url.startswith('http://'), "SSL disabled should use http"

        assert s3_resource.S3_ADDRESS in url, "URL should contain address"

    def test_public_url_format(self, s3_resource):
        """Test that public URL is correctly formatted."""
        url = s3_resource.publicUrl(path='test/file.csv', bucket='test')

        assert 'test/file.csv' in url, "URL should contain path"
        assert 'test' in url, "URL should contain bucket"


class TestS3FileOperations:
    """Test S3 file upload and download operations."""

    def test_can_upload_and_download_text(self, s3_resource):
        """Test uploading and downloading text data."""
        test_data = "Test H2S data\n123,456,789"
        test_path = "test/h2s_test_upload.txt"

        # Upload
        object_name = s3_resource.putFile_text(
            data=test_data,
            path=test_path,
            bucket=s3_resource.S3_BUCKET
        )

        assert object_name == test_path

        # Download
        downloaded = s3_resource.getFile(
            path=test_path,
            bucket=s3_resource.S3_BUCKET
        )

        assert downloaded.decode('utf-8') == test_data

    def test_can_upload_and_download_binary(self, s3_resource):
        """Test uploading and downloading binary data."""
        test_data = b"Binary test data\x00\x01\x02"
        test_path = "test/h2s_test_binary.bin"

        # Upload
        object_name = s3_resource.putFile(
            data=test_data,
            path=test_path,
            bucket=s3_resource.S3_BUCKET
        )

        assert object_name == test_path

        # Download
        downloaded = s3_resource.getFile(
            path=test_path,
            bucket=s3_resource.S3_BUCKET
        )

        assert downloaded == test_data

    def test_can_stream_file(self, s3_resource):
        """Test getting file as stream."""
        test_data = "Stream test data"
        test_path = "test/h2s_test_stream.txt"

        # Upload first
        s3_resource.putFile_text(
            data=test_data,
            path=test_path,
            bucket=s3_resource.S3_BUCKET
        )

        # Get as stream
        stream = s3_resource.get_stream(
            path=test_path,
            bucket=s3_resource.S3_BUCKET
        )

        # Stream should have data attribute
        assert hasattr(stream, 'data')
        downloaded = stream.data.decode('utf-8')
        assert downloaded == test_data


class TestModelLoading:
    """Test loading H2S model from S3."""

    def test_model_files_exist_on_s3(self, s3_resource):
        """Test that model files exist on S3."""
        model_path = 'tijuana/forecast/models/nestor_xgboost_weighted_model.json'
        prep_path = 'tijuana/forecast/models/nestor_preprocessing_info.json'

        # Try to get file info
        try:
            model_bytes = s3_resource.getFile(path=model_path, bucket=s3_resource.S3_BUCKET)
            assert len(model_bytes) > 0, "Model file should not be empty"

            prep_bytes = s3_resource.getFile(path=prep_path, bucket=s3_resource.S3_BUCKET)
            assert len(prep_bytes) > 0, "Preprocessing file should not be empty"

        except Exception as e:
            pytest.fail(f"Model files not found on S3: {e}")

    def test_can_load_predictor_from_s3(self, s3_resource):
        """Test that H2SPredictor can load from S3."""
        model_path = 'tijuana/forecast/models/nestor_xgboost_weighted_model.json'
        prep_path = 'tijuana/forecast/models/nestor_preprocessing_info.json'

        predictor = H2SPredictor.from_s3(
            s3_resource,
            model_path,
            prep_path
        )

        assert predictor is not None, "Predictor should be created"
        assert predictor.model is not None, "Model should be loaded"
        assert predictor.prep_info is not None, "Preprocessing info should be loaded"

    def test_loaded_predictor_has_correct_attributes(self, s3_resource):
        """Test that loaded predictor has expected attributes."""
        model_path = 'tijuana/forecast/models/nestor_xgboost_weighted_model.json'
        prep_path = 'tijuana/forecast/models/nestor_preprocessing_info.json'

        predictor = H2SPredictor.from_s3(
            s3_resource,
            model_path,
            prep_path
        )

        # Check feature count (should be 20)
        assert len(predictor.feature_cols) == 20, \
            f"Expected 20 features, got {len(predictor.feature_cols)}"

        # Check class names
        expected_classes = {'green', 'orange', 'yellow'}
        assert set(predictor.class_names) == expected_classes, \
            f"Expected classes {expected_classes}, got {set(predictor.class_names)}"

        # Check site name
        assert predictor.site_name == 'NESTOR - BES', \
            f"Expected site 'NESTOR - BES', got '{predictor.site_name}'"

        # Check mappings exist
        assert predictor.wind_cat_mapping is not None, "Wind category mapping missing"
        assert predictor.tidal_mapping is not None, "Tidal mapping missing"


class TestStoreAssets:
    """Test store_assets utility functions."""

    @pytest.fixture
    def test_dataframe(self):
        """Create test dataframe for export."""
        return pd.DataFrame({
            'time': pd.date_range('2024-01-01', periods=5, freq='h'),
            'predicted_category': ['green', 'yellow', 'orange', 'green', 'yellow'],
            'probability_green': [0.8, 0.3, 0.1, 0.9, 0.4],
            'probability_orange': [0.1, 0.3, 0.7, 0.05, 0.2],
            'probability_yellow': [0.1, 0.4, 0.2, 0.05, 0.4],
            'confidence': [0.8, 0.4, 0.7, 0.9, 0.4],
            'alert': [False, True, True, False, True],
        })

    def test_can_create_metadata(self):
        """Test that metadata object can be created."""
        metadata = objectMetadata(
            name="Test H2S Predictions",
            description="Test predictions for validation",
            variableMeasured=["H2S Category", "Probability Scores"]
        )

        assert metadata.name == "Test H2S Predictions"
        assert metadata.description == "Test predictions for validation"
        assert len(metadata.variableMeasured) == 2

    def test_can_export_dataframe_to_s3(self, s3_resource, test_dataframe):
        """Test exporting dataframe to S3 with metadata."""
        metadata = objectMetadata(
            name="Test H2S Export",
            description="Test export functionality",
            variableMeasured=["H2S Category", "Probabilities"]
        )

        test_path = "test/h2s_export_test"
        test_identifier = "test_predictions"

        # Export
        store_dataframe_to_s3(
            df=test_dataframe,
            path=test_path,
            dataset_identifier=test_identifier,
            s3_resource=s3_resource,
            metadata=metadata,
            enable_latest_path=False,
            formats=['csv']
        )

        # Verify CSV was created
        csv_path = f"{test_path}/{test_identifier}.csv"
        csv_data = s3_resource.getFile(path=csv_path, bucket=s3_resource.S3_BUCKET)
        assert len(csv_data) > 0, "CSV file should not be empty"

        # Verify metadata was created
        metadata_path = f"{test_path}/{test_identifier}.metadata.json"
        metadata_data = s3_resource.getFile(path=metadata_path, bucket=s3_resource.S3_BUCKET)
        assert len(metadata_data) > 0, "Metadata file should not be empty"

    def test_can_export_with_latest_path(self, s3_resource, test_dataframe):
        """Test exporting with latest path enabled."""
        metadata = objectMetadata(
            name="Test H2S Latest",
            description="Test latest path functionality",
            variableMeasured=["H2S Category"]
        )

        test_path = "test/h2s_latest_test"
        test_identifier = "test_latest"
        latest_path = "test/latest_data"

        # Export with latest path
        store_dataframe_to_s3(
            df=test_dataframe,
            path=test_path,
            dataset_identifier=test_identifier,
            s3_resource=s3_resource,
            metadata=metadata,
            latestdatasetpath=latest_path,
            enable_latest_path=True,
            formats=['csv']
        )

        # Verify both paths exist
        timestamped_path = f"{test_path}/{test_identifier}.csv"
        latest_full_path = f"latest/{latest_path}/{test_identifier}.csv"

        timestamped_data = s3_resource.getFile(path=timestamped_path, bucket=s3_resource.S3_BUCKET)
        latest_data = s3_resource.getFile(path=latest_full_path, bucket=s3_resource.S3_BUCKET)

        assert len(timestamped_data) > 0, "Timestamped file should exist"
        assert len(latest_data) > 0, "Latest file should exist"

    def test_can_export_multiple_formats(self, s3_resource, test_dataframe):
        """Test exporting in multiple formats."""
        metadata = objectMetadata(
            name="Test Multi-format Export",
            description="Test CSV and JSON export",
            variableMeasured=["H2S Category"]
        )

        test_path = "test/h2s_multiformat_test"
        test_identifier = "test_multiformat"

        # Export in both formats
        store_dataframe_to_s3(
            df=test_dataframe,
            path=test_path,
            dataset_identifier=test_identifier,
            s3_resource=s3_resource,
            metadata=metadata,
            enable_latest_path=False,
            formats=['csv', 'json']
        )

        # Verify both formats exist
        csv_path = f"{test_path}/{test_identifier}.csv"
        json_path = f"{test_path}/{test_identifier}.json"

        csv_data = s3_resource.getFile(path=csv_path, bucket=s3_resource.S3_BUCKET)
        json_data = s3_resource.getFile(path=json_path, bucket=s3_resource.S3_BUCKET)

        assert len(csv_data) > 0, "CSV file should exist"
        assert len(json_data) > 0, "JSON file should exist"


class TestVisualizationUpload:
    """Test visualization generation and upload."""

    def test_can_generate_and_upload_visualization(self, s3_resource):
        """Test that visualizations can be generated as BytesIO and uploaded."""
        from h2s.predictor.visualizations import generate_feature_importance
        from unittest.mock import Mock

        # Create mock model and prep_info
        mock_model = Mock()
        mock_booster = Mock()
        mock_booster.get_score.return_value = {
            'f0': 100, 'f1': 90, 'f2': 80, 'f3': 70, 'f4': 60
        }
        mock_model.get_booster.return_value = mock_booster

        mock_prep_info = {
            'feature_cols': ['temp', 'wind_speed', 'humidity', 'pressure', 'cloud_cover']
        }

        # Generate visualization
        plot_bytes = generate_feature_importance(mock_model, mock_prep_info, top_n=5)

        assert isinstance(plot_bytes, BytesIO), "Should return BytesIO"
        assert plot_bytes.tell() > 0, "BytesIO should contain data"

        # Upload to S3
        plot_bytes.seek(0)
        test_path = "test/h2s_viz_test/feature_importance.png"
        s3_resource.putFile(
            data=plot_bytes.read(),
            path=test_path,
            bucket=s3_resource.S3_BUCKET,
            content_type='image/png'
        )

        # Verify upload
        downloaded = s3_resource.getFile(path=test_path, bucket=s3_resource.S3_BUCKET)
        assert len(downloaded) > 0, "Uploaded visualization should exist"
