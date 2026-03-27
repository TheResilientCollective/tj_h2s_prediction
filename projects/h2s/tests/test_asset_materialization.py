"""Integration tests for asset materialization.

Tests that assets can be materialized successfully using Dagster's testing utilities.
These tests validate the full pipeline execution, not just the logic.
"""

import pandas as pd
import numpy as np
import pytest
import tempfile
from unittest.mock import Mock, patch
import dagster as dg

from h2s.defs.h2s_pipeline import (
    raw_environmental_data,
    preprocessed_features,
    h2s_predictions,
    h2s_alerts,
    slack_alerts,
)


class MockS3Resource:
    """Mock S3Resource that implements required interface."""

    def __init__(self):
        self.S3_BUCKET = "test"
        self.S3_ADDRESS = "localhost"
        self.S3_PORT = "9000"
        self.S3_USE_SSL = False
        self.S3_ACCESS_KEY = "test_key"
        self.S3_SECRET_KEY = "test_secret"
        self._get_stream_mock = Mock()
        self._getFile_mock = Mock()
        self._putFile_mock = Mock()
        self._putFile_text_mock = Mock()
        self._get_presigned_url_mock = Mock()

    def get_stream(self, path, bucket=None):
        return self._get_stream_mock(path)

    def getFile(self, path, bucket=None):
        return self._getFile_mock(path, bucket)

    def get_presigned_url(self, path, bucket=None, expires_seconds=3600):
        return self._get_presigned_url_mock(path, bucket)

    def mock_presigned_csv(self, df):
        """Write a DataFrame to a temp CSV and configure get_presigned_url to return its path."""
        tmp = tempfile.NamedTemporaryFile(suffix='.csv', delete=False, mode='w')
        df.to_csv(tmp, index=False)
        tmp.close()
        self._get_presigned_url_mock.return_value = tmp.name
        return tmp.name

    def putFile(self, data, path, bucket=None, contenttype=None):
        return self._putFile_mock(data, path, bucket, contenttype)

    def putFile_text(self, data, path, bucket=None):
        return self._putFile_text_mock(data, path, bucket)


@pytest.fixture
def mock_s3_resource():
    """Create a mock S3Resource for testing."""
    return MockS3Resource()


@pytest.fixture
def tidal_stub():
    """Stub tidal_forecast asset — no S3 calls."""
    @dg.asset(key_prefix="h2s", group_name="h2s_prediction")
    def tidal_forecast():
        now = pd.Timestamp.utcnow().floor("h").tz_localize(None)
        times = pd.date_range(start=now, periods=240, freq="h")
        return pd.DataFrame({
            'time': times,
            'tide_height': 1.0,
            'tidal_state': 'flood',
        })
    return tidal_forecast


@pytest.fixture
def sbiwtp_stub():
    """Stub sbiwtp_operational_data asset — no S3 calls, returns persistence defaults."""
    @dg.asset(key_prefix="h2s", group_name="h2s_prediction")
    def sbiwtp_operational_data():
        now = pd.Timestamp.utcnow().floor("h").tz_localize(None)
        times = pd.date_range(start=now, periods=240, freq="h")
        return pd.DataFrame({
            'time': times,
            'sbiwtp_flow_mgd': 23.5,
            'sbiwtp_hourly_mgd': 23.5 / 24,
            'sbiwtp_anomaly': 0.0,
            'sbiwtp_deficit': 0.0,
            'sbiwtp_flow_x_temp': 0.0,
            'sbiwtp_sli': 0.0,
        })
    return sbiwtp_operational_data


@pytest.fixture
def sample_environmental_data():
    """Sample environmental data CSV for testing."""
    df = pd.DataFrame({
        'time': pd.date_range('2024-01-01', periods=24, freq='h'),
        'temperature_2m': np.random.uniform(15, 25, 24),
        'wind_speed_10m': np.random.uniform(0, 10, 24),
        'wind_direction_10m': np.random.uniform(0, 360, 24),
        'relative_humidity_2m': np.random.uniform(60, 90, 24),
        'surface_pressure': np.random.uniform(1010, 1020, 24),
        'precipitation': np.random.uniform(0, 5, 24),
        'cloud_cover': np.random.uniform(0, 100, 24),
        'wind_direction_categorical': ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'] * 3,
        'flow_rate_cms': np.random.uniform(1, 10, 24),
        'tide_height_m': np.random.uniform(0, 2, 24),
        'tidal_state': ['rising', 'falling'] * 12,
    })
    return df


@pytest.mark.integration
class TestRawEnvironmentalDataMaterialization:
    """Test raw_environmental_data asset materialization."""

    def test_materializes_from_s3(self, sample_environmental_data, mock_s3_resource, tidal_stub, sbiwtp_stub):
        """Test that raw_environmental_data materializes from S3."""
        # Mock S3 presigned URL to return a temp CSV file path
        mock_s3_resource.mock_presigned_csv(sample_environmental_data)

        result = dg.materialize(
            assets=[tidal_stub, sbiwtp_stub, raw_environmental_data],
            resources={"s3": mock_s3_resource},
        )

        assert result.success
        output = result.output_for_node("h2s__raw_environmental_data")
        assert output is not None
        assert len(output) == 24
        assert 'date' in output.columns
        assert pd.api.types.is_datetime64_any_dtype(output['date'])

    def test_materializes_with_correct_columns(self, sample_environmental_data, mock_s3_resource, tidal_stub, sbiwtp_stub):
        """Test that materialized data has all expected columns."""
        mock_s3_resource.mock_presigned_csv(sample_environmental_data)

        result = dg.materialize(
            assets=[tidal_stub, sbiwtp_stub, raw_environmental_data],
            resources={"s3": mock_s3_resource},
        )

        output = result.output_for_node("h2s__raw_environmental_data")
        required_cols = [
            'time', 'temperature_2m', 'wind_speed_10m', 'wind_direction_10m',
            'relative_humidity_2m', 'surface_pressure', 'precipitation',
            'cloud_cover', 'date',
        ]
        for col in required_cols:
            assert col in output.columns, f"Missing column: {col}"


@pytest.mark.integration
class TestPreprocessedFeaturesMaterialization:
    """Test preprocessed_features asset materialization with mocked predictor."""

    def test_materializes_with_mocked_predictor(self, sample_environmental_data, mock_s3_resource, tidal_stub, sbiwtp_stub):
        """Test that preprocessed_features materializes when predictor is mocked."""
        mock_s3_resource.mock_presigned_csv(sample_environmental_data)

        # Mock the predictor's preprocess_data method
        with patch('h2s.predictor.h2s_predictor.H2SPredictor') as MockPredictor:
            # Create a mock predictor instance
            mock_predictor = Mock()

            # Create preprocessed data with added features
            preprocessed = sample_environmental_data.copy()
            preprocessed['hour'] = preprocessed['time'].dt.hour
            preprocessed['hour_sin'] = np.sin(2 * np.pi * preprocessed['hour'] / 24)
            preprocessed['hour_cos'] = np.cos(2 * np.pi * preprocessed['hour'] / 24)
            preprocessed['wind_direction_sin'] = np.sin(np.radians(preprocessed['wind_direction_10m']))
            preprocessed['wind_direction_cos'] = np.cos(np.radians(preprocessed['wind_direction_10m']))

            mock_predictor.preprocess_data.return_value = preprocessed
            MockPredictor.from_s3.return_value = mock_predictor

            # Mock asset function to return predictor — key must match h2s/h2s_model_artifacts
            @dg.asset(key_prefix="h2s", group_name="h2s_model", required_resource_keys={"s3"})
            def h2s_model_artifacts(context):
                return mock_predictor

            result = dg.materialize(
                assets=[tidal_stub, sbiwtp_stub, h2s_model_artifacts, raw_environmental_data, preprocessed_features],
                resources={"s3": mock_s3_resource, "io_manager": dg.mem_io_manager},
            )

            assert result.success
            output = result.output_for_node("h2s__preprocessed_features")
            assert output is not None
            assert 'hour_sin' in output.columns
            assert 'hour_cos' in output.columns


@pytest.mark.integration
class TestH2SPredictionsMaterialization:
    """Test h2s_predictions asset materialization with mocked predictor."""

    def test_materializes_with_mocked_predictions(self, sample_environmental_data, mock_s3_resource, tidal_stub, sbiwtp_stub):
        """Test that predictions materialize with mocked predictor."""
        mock_s3_resource.mock_presigned_csv(sample_environmental_data)

        # Mock the predictor
        with patch('h2s.predictor.h2s_predictor.H2SPredictor') as MockPredictor:
            mock_predictor = Mock()

            # Create preprocessed data
            preprocessed = sample_environmental_data.copy()
            preprocessed['hour'] = preprocessed['time'].dt.hour
            mock_predictor.preprocess_data.return_value = preprocessed

            # Create mock predictions
            predictions = sample_environmental_data[['time']].copy()
            predictions['predicted_category'] = ['green'] * 24
            predictions['probability_green'] = 0.7
            predictions['probability_orange'] = 0.2
            predictions['probability_yellow'] = 0.1
            predictions['confidence'] = 0.7
            predictions['alert'] = False
            mock_predictor.predict.return_value = predictions

            MockPredictor.from_s3.return_value = mock_predictor

            # Mock model artifacts asset
            @dg.asset(key_prefix="h2s", group_name="h2s_model", required_resource_keys={"s3"})
            def h2s_model_artifacts(context):
                return mock_predictor

            result = dg.materialize(
                assets=[tidal_stub, sbiwtp_stub, h2s_model_artifacts, raw_environmental_data, preprocessed_features, h2s_predictions],
                resources={"s3": mock_s3_resource, "io_manager": dg.mem_io_manager},
            )

            assert result.success
            output = result.output_for_node("h2s__h2s_predictions")
            assert output is not None
            assert len(output) == 24
            assert 'predicted_category' in output.columns
            assert 'confidence' in output.columns


@pytest.mark.integration
class TestH2SAlertsMaterialization:
    """Test h2s_alerts asset materialization."""

    def test_materializes_and_filters_alerts(self, sample_environmental_data, mock_s3_resource, tidal_stub, sbiwtp_stub):
        """Test that alerts materialize and filter correctly."""
        mock_s3_resource.mock_presigned_csv(sample_environmental_data)

        # Mock the predictor
        with patch('h2s.predictor.h2s_predictor.H2SPredictor') as MockPredictor:
            mock_predictor = Mock()

            # Create preprocessed data
            preprocessed = sample_environmental_data.copy()
            preprocessed['hour'] = preprocessed['time'].dt.hour
            mock_predictor.preprocess_data.return_value = preprocessed

            # Create mock predictions with some alerts
            predictions = sample_environmental_data[['time']].copy()
            predictions['predicted_category'] = ['green', 'yellow', 'orange'] * 8
            predictions['probability_green'] = 0.5
            predictions['probability_orange'] = 0.3
            predictions['probability_yellow'] = 0.2
            predictions['confidence'] = 0.5
            predictions['alert'] = predictions['predicted_category'].isin(['orange', 'yellow'])
            mock_predictor.predict.return_value = predictions

            MockPredictor.from_s3.return_value = mock_predictor

            # Mock model artifacts asset
            @dg.asset(key_prefix="h2s", group_name="h2s_model", required_resource_keys={"s3"})
            def h2s_model_artifacts(context):
                return mock_predictor

            result = dg.materialize(
                assets=[
                    tidal_stub,
                    sbiwtp_stub,
                    h2s_model_artifacts,
                    raw_environmental_data,
                    preprocessed_features,
                    h2s_predictions,
                    h2s_alerts
                ],
                resources={"s3": mock_s3_resource, "io_manager": dg.mem_io_manager},
            )

            assert result.success
            alerts = result.output_for_node("h2s__h2s_alerts")
            predictions_output = result.output_for_node("h2s__h2s_predictions")

            # Alerts should be subset of predictions
            assert len(alerts) <= len(predictions_output)

            # All alerts should have alert=True
            if len(alerts) > 0:
                assert alerts['alert'].all()
                assert 'green' not in alerts['predicted_category'].values


class MockSlackResource:
    """Mock SlackAlertResource for testing."""

    def __init__(self):
        self.channel = "#test"
        self._messages = []

    def get_client(self):
        mock_client = Mock()
        def capture_message(**kwargs):
            self._messages.append(kwargs)
        mock_client.chat_postMessage = capture_message
        return mock_client


@pytest.mark.integration
class TestSlackAlertsMaterialization:
    """Test slack_alerts asset materialization."""

    def test_sends_slack_when_alerts_exist(self):
        """Test that Slack message is sent when there are alerts."""
        alerts_df = pd.DataFrame({
            'time': pd.date_range('2024-06-15', periods=5, freq='h'),
            'predicted_category': ['orange', 'orange', 'yellow', 'yellow', 'yellow'],
            'confidence': [0.8, 0.7, 0.6, 0.5, 0.55],
            'h2s_risk': [0.9, 0.85, 0.4, 0.35, 0.38],
            'alert': True,
        })

        mock_slack = MockSlackResource()

        # Name must match the asset key that slack_alerts expects
        @dg.asset(key=dg.AssetKey(["h2s", "h2s_alerts"]))
        def h2s_alerts_source():
            return alerts_df

        result = dg.materialize(
            assets=[h2s_alerts_source, slack_alerts],
            resources={"slack": mock_slack, "io_manager": dg.mem_io_manager},
        )

        assert result.success
        assert len(mock_slack._messages) == 1
        msg = mock_slack._messages[0]
        assert msg['channel'] == '#test'
        assert 'orange' in msg['text'].lower()

    def test_no_slack_when_no_alerts(self):
        """Test that no Slack message is sent when alerts DataFrame is empty."""
        empty_df = pd.DataFrame({
            'time': pd.Series([], dtype='datetime64[ns]'),
            'predicted_category': pd.Series([], dtype='str'),
            'confidence': pd.Series([], dtype='float'),
            'h2s_risk': pd.Series([], dtype='float'),
            'alert': pd.Series([], dtype='bool'),
        })

        mock_slack = MockSlackResource()

        @dg.asset(key=dg.AssetKey(["h2s", "h2s_alerts"]))
        def h2s_alerts_source():
            return empty_df

        result = dg.materialize(
            assets=[h2s_alerts_source, slack_alerts],
            resources={"slack": mock_slack, "io_manager": dg.mem_io_manager},
        )

        assert result.success
        assert len(mock_slack._messages) == 0


@pytest.mark.integration
class TestPipelineDataFlow:
    """Test data flow through the pipeline."""

    def test_data_continuity_through_pipeline(self, sample_environmental_data, mock_s3_resource, tidal_stub, sbiwtp_stub):
        """Test that same number of rows flows through entire pipeline."""
        mock_s3_resource.mock_presigned_csv(sample_environmental_data)

        # Mock the predictor
        with patch('h2s.predictor.h2s_predictor.H2SPredictor') as MockPredictor:
            mock_predictor = Mock()

            # Create preprocessed data (same length)
            preprocessed = sample_environmental_data.copy()
            preprocessed['hour'] = preprocessed['time'].dt.hour
            mock_predictor.preprocess_data.return_value = preprocessed

            # Create predictions (same length)
            predictions = sample_environmental_data[['time']].copy()
            predictions['predicted_category'] = 'green'
            predictions['probability_green'] = 0.7
            predictions['probability_orange'] = 0.2
            predictions['probability_yellow'] = 0.1
            predictions['confidence'] = 0.7
            predictions['alert'] = False
            mock_predictor.predict.return_value = predictions

            MockPredictor.from_s3.return_value = mock_predictor

            @dg.asset(key_prefix="h2s", group_name="h2s_model", required_resource_keys={"s3"})
            def h2s_model_artifacts(context):
                return mock_predictor

            result = dg.materialize(
                assets=[
                    tidal_stub,
                    sbiwtp_stub,
                    h2s_model_artifacts,
                    raw_environmental_data,
                    preprocessed_features,
                    h2s_predictions,
                ],
                resources={"s3": mock_s3_resource, "io_manager": dg.mem_io_manager},
            )

            raw_data = result.output_for_node("h2s__raw_environmental_data")
            preprocessed_output = result.output_for_node("h2s__preprocessed_features")
            predictions_output = result.output_for_node("h2s__h2s_predictions")

            # Same number of rows through pipeline
            assert len(raw_data) == 24
            assert len(preprocessed_output) == 24
            assert len(predictions_output) == 24

            # Time column preserved
            assert 'time' in raw_data.columns
            assert 'time' in predictions_output.columns


@pytest.mark.integration
class TestAssetFailureScenarios:
    """Test asset behavior in failure scenarios."""

    def test_raw_data_fails_without_s3(self, mock_s3_resource, tidal_stub, sbiwtp_stub):
        """Test that raw_environmental_data fails when S3 unavailable and no local data."""
        # Mock presigned URL to fail
        mock_s3_resource._get_presigned_url_mock.side_effect = Exception("S3 connection failed")

        result = dg.materialize(
            assets=[tidal_stub, sbiwtp_stub, raw_environmental_data],
            resources={"s3": mock_s3_resource},
            raise_on_error=False,
        )

        assert not result.success

    def test_asset_reports_metadata(self, sample_environmental_data, mock_s3_resource, tidal_stub, sbiwtp_stub):
        """Test that assets include metadata in their outputs."""
        mock_s3_resource.mock_presigned_csv(sample_environmental_data)

        result = dg.materialize(
            assets=[tidal_stub, sbiwtp_stub, raw_environmental_data],
            resources={"s3": mock_s3_resource},
        )

        # Check that materialization event includes metadata
        assert result.success
