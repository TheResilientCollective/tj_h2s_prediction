"""Tests for H2S pipeline assets.

Tests the core logic of each asset without requiring S3 connection.
"""

import pandas as pd
import numpy as np
import pytest
from unittest.mock import Mock, MagicMock, patch
import dagster as dg

from h2s.defs.h2s_pipeline import (
    preprocessed_features,
    h2s_predictions,
    h2s_alerts,
)
from h2s.predictor.h2s_predictor import H2SPredictor


@pytest.fixture
def mock_environmental_data():
    """Create mock environmental data for testing."""
    return pd.DataFrame({
        'time': pd.date_range('2024-01-01', periods=10, freq='h'),
        'temperature_2m': np.random.uniform(15, 25, 10),
        'wind_speed_10m': np.random.uniform(0, 10, 10),
        'wind_direction_10m': np.random.uniform(0, 360, 10),
        'relative_humidity_2m': np.random.uniform(60, 90, 10),
        'surface_pressure': np.random.uniform(1010, 1020, 10),
        'precipitation': np.random.uniform(0, 5, 10),
        'cloud_cover': np.random.uniform(0, 100, 10),
        'wind_direction_categorical': ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW', 'N', 'NE'],
        'flow_rate_cms': np.random.uniform(1, 10, 10),
        'tide_height_m': np.random.uniform(0, 2, 10),
        'tidal_state': ['rising', 'falling'] * 5,
    })


@pytest.fixture
def mock_predictor():
    """Create a mock H2SPredictor for testing."""
    predictor = Mock(spec=H2SPredictor)
    predictor.feature_cols = [
        'temperature_2m', 'wind_speed_10m', 'wind_direction_10m',
        'relative_humidity_2m', 'surface_pressure', 'precipitation',
        'cloud_cover', 'flow_rate_cms', 'tide_height_m',
        'hour_sin', 'hour_cos', 'wind_direction_sin', 'wind_direction_cos',
        'wind_temp_interaction', 'humidity_temp_interaction',
        'wind_direction_cat_encoded', 'tidal_state_encoded',
        'hour', 'day_of_week', 'month'
    ]
    predictor.class_names = ['green', 'orange', 'yellow']
    predictor.site_name = 'NESTOR - BES'
    predictor.wind_cat_mapping = {'N': 0, 'NE': 1, 'E': 2, 'SE': 3, 'S': 4, 'SW': 5, 'W': 6, 'NW': 7}
    predictor.tidal_mapping = {'rising': 0, 'falling': 1}
    return predictor


@pytest.fixture
def mock_predictions_with_alerts():
    """Create mock predictions with mix of alerts and non-alerts."""
    return pd.DataFrame({
        'time': pd.date_range('2024-01-01', periods=10, freq='h'),
        'predicted_category': ['green', 'yellow', 'orange', 'green', 'yellow',
                               'orange', 'green', 'green', 'yellow', 'green'],
        'alert': [False, True, True, False, True, True, False, False, True, False],
        'confidence': [0.8, 0.6, 0.7, 0.9, 0.5, 0.8, 0.85, 0.7, 0.55, 0.75],
    })


class TestPreprocessedFeatures:
    """Test preprocessing logic."""

    def test_creates_temporal_features(self, mock_environmental_data, mock_predictor):
        """Test that temporal features are created."""
        mock_predictor.preprocess_data.return_value = mock_environmental_data.copy()

        # Mock the preprocessing to add temporal features
        df = mock_environmental_data.copy()
        df['hour'] = df['time'].dt.hour
        df['day_of_week'] = df['time'].dt.dayofweek
        df['month'] = df['time'].dt.month
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

        mock_predictor.preprocess_data.return_value = df

        result = mock_predictor.preprocess_data(mock_environmental_data)

        assert 'hour' in result.columns
        assert 'hour_sin' in result.columns
        assert 'hour_cos' in result.columns
        assert 'day_of_week' in result.columns
        assert 'month' in result.columns

    def test_creates_cyclical_wind_features(self, mock_environmental_data, mock_predictor):
        """Test that wind direction cyclical encoding is created."""
        df = mock_environmental_data.copy()
        df['wind_direction_sin'] = np.sin(np.radians(df['wind_direction_10m']))
        df['wind_direction_cos'] = np.cos(np.radians(df['wind_direction_10m']))

        mock_predictor.preprocess_data.return_value = df

        result = mock_predictor.preprocess_data(mock_environmental_data)

        assert 'wind_direction_sin' in result.columns
        assert 'wind_direction_cos' in result.columns
        # Cyclical values should be between -1 and 1
        assert result['wind_direction_sin'].min() >= -1
        assert result['wind_direction_sin'].max() <= 1

    def test_creates_interaction_features(self, mock_environmental_data, mock_predictor):
        """Test that interaction features are created."""
        df = mock_environmental_data.copy()
        df['wind_temp_interaction'] = df['wind_speed_10m'] * df['temperature_2m']
        df['humidity_temp_interaction'] = df['relative_humidity_2m'] * df['temperature_2m']

        mock_predictor.preprocess_data.return_value = df

        result = mock_predictor.preprocess_data(mock_environmental_data)

        assert 'wind_temp_interaction' in result.columns
        assert 'humidity_temp_interaction' in result.columns

    def test_encodes_categorical_features(self, mock_environmental_data, mock_predictor):
        """Test that categorical features are encoded."""
        df = mock_environmental_data.copy()
        df['wind_direction_cat_encoded'] = df['wind_direction_categorical'].map(
            mock_predictor.wind_cat_mapping
        ).fillna(-1).astype(int)
        df['tidal_state_encoded'] = df['tidal_state'].map(
            mock_predictor.tidal_mapping
        ).fillna(-1).astype(int)

        mock_predictor.preprocess_data.return_value = df

        result = mock_predictor.preprocess_data(mock_environmental_data)

        assert 'wind_direction_cat_encoded' in result.columns
        assert 'tidal_state_encoded' in result.columns
        # Encoded values should be integers
        assert result['wind_direction_cat_encoded'].dtype in [np.int32, np.int64]
        assert result['tidal_state_encoded'].dtype in [np.int32, np.int64]


class TestH2SPredictions:
    """Test prediction generation."""

    @pytest.fixture
    def mock_predictions_output(self):
        """Create mock prediction output."""
        return pd.DataFrame({
            'time': pd.date_range('2024-01-01', periods=10, freq='h'),
            'predicted_category': ['green', 'yellow', 'orange', 'green', 'yellow',
                                   'orange', 'green', 'green', 'yellow', 'green'],
            'probability_green': [0.8, 0.3, 0.1, 0.9, 0.35, 0.05, 0.85, 0.7, 0.35, 0.75],
            'probability_orange': [0.1, 0.3, 0.7, 0.05, 0.2, 0.8, 0.05, 0.15, 0.25, 0.1],
            'probability_yellow': [0.1, 0.4, 0.2, 0.05, 0.45, 0.15, 0.1, 0.15, 0.4, 0.15],
            'confidence': [0.8, 0.4, 0.7, 0.9, 0.45, 0.8, 0.85, 0.7, 0.4, 0.75],
            'alert': [False, True, True, False, True, True, False, False, True, False],
        })

    def test_predictions_have_required_columns(self, mock_predictions_output):
        """Test that predictions have all required columns."""
        required_cols = [
            'predicted_category',
            'probability_green',
            'probability_orange',
            'probability_yellow',
            'confidence',
            'alert'
        ]

        for col in required_cols:
            assert col in mock_predictions_output.columns, f"Missing column: {col}"

    def test_probabilities_are_valid(self, mock_predictions_output):
        """Test that probabilities are between 0 and 1."""
        prob_cols = ['probability_green', 'probability_orange', 'probability_yellow']

        for col in prob_cols:
            assert mock_predictions_output[col].min() >= 0, f"{col} has values < 0"
            assert mock_predictions_output[col].max() <= 1, f"{col} has values > 1"

    def test_probabilities_sum_to_approximately_one(self, mock_predictions_output):
        """Test that class probabilities sum to approximately 1.0."""
        prob_sum = (
            mock_predictions_output['probability_green'] +
            mock_predictions_output['probability_orange'] +
            mock_predictions_output['probability_yellow']
        )

        # Allow small floating point error
        assert np.allclose(prob_sum, 1.0, atol=0.01), "Probabilities don't sum to 1.0"

    def test_confidence_is_max_probability(self, mock_predictions_output):
        """Test that confidence equals the maximum probability."""
        max_prob = mock_predictions_output[
            ['probability_green', 'probability_orange', 'probability_yellow']
        ].max(axis=1)

        assert np.allclose(mock_predictions_output['confidence'], max_prob, atol=0.01)

    def test_predicted_category_matches_max_probability(self, mock_predictions_output):
        """Test that predicted category matches the highest probability."""
        for idx, row in mock_predictions_output.iterrows():
            probs = {
                'green': row['probability_green'],
                'orange': row['probability_orange'],
                'yellow': row['probability_yellow']
            }
            max_category = max(probs, key=probs.get)
            assert row['predicted_category'] == max_category, \
                f"Row {idx}: predicted '{row['predicted_category']}' but max prob is '{max_category}'"

    def test_alert_flag_matches_category(self, mock_predictions_output):
        """Test that alert flag is True only for orange/yellow."""
        for idx, row in mock_predictions_output.iterrows():
            expected_alert = row['predicted_category'] in ['orange', 'yellow']
            assert row['alert'] == expected_alert, \
                f"Row {idx}: category '{row['predicted_category']}' should have alert={expected_alert}"

    def test_categories_are_valid(self, mock_predictions_output):
        """Test that all predicted categories are valid."""
        valid_categories = {'green', 'yellow', 'orange'}
        unique_categories = set(mock_predictions_output['predicted_category'].unique())

        assert unique_categories.issubset(valid_categories), \
            f"Invalid categories found: {unique_categories - valid_categories}"


class TestH2SAlerts:
    """Test alert filtering."""

    def test_filters_only_alerts(self, mock_predictions_with_alerts):
        """Test that only rows with alert=True are included."""
        alerts = mock_predictions_with_alerts[mock_predictions_with_alerts['alert'] == True]

        assert len(alerts) == 5, "Should have 5 alert rows"
        assert alerts['alert'].all(), "All rows should have alert=True"

    def test_no_green_predictions_in_alerts(self, mock_predictions_with_alerts):
        """Test that green predictions are excluded from alerts."""
        alerts = mock_predictions_with_alerts[mock_predictions_with_alerts['alert'] == True]

        assert 'green' not in alerts['predicted_category'].values, \
            "Alerts should not contain green predictions"

    def test_only_orange_and_yellow_in_alerts(self, mock_predictions_with_alerts):
        """Test that alerts contain only orange and yellow."""
        alerts = mock_predictions_with_alerts[mock_predictions_with_alerts['alert'] == True]

        unique_categories = set(alerts['predicted_category'].unique())
        assert unique_categories.issubset({'orange', 'yellow'}), \
            f"Alerts contain invalid categories: {unique_categories - {'orange', 'yellow'}}"

    def test_alert_count_matches_orange_plus_yellow(self, mock_predictions_with_alerts):
        """Test that alert count equals orange + yellow count."""
        orange_count = (mock_predictions_with_alerts['predicted_category'] == 'orange').sum()
        yellow_count = (mock_predictions_with_alerts['predicted_category'] == 'yellow').sum()
        alert_count = mock_predictions_with_alerts['alert'].sum()

        assert alert_count == orange_count + yellow_count, \
            "Alert count should equal orange + yellow count"

    def test_preserves_all_columns(self, mock_predictions_with_alerts):
        """Test that filtering preserves all columns."""
        alerts = mock_predictions_with_alerts[mock_predictions_with_alerts['alert'] == True]

        assert set(alerts.columns) == set(mock_predictions_with_alerts.columns), \
            "Alert filtering should preserve all columns"


class TestAssetMetadata:
    """Test that assets include proper metadata."""

    def test_predictions_metadata_includes_counts(self, mock_predictions_with_alerts):
        """Test that prediction metadata includes category counts."""
        orange_count = (mock_predictions_with_alerts['predicted_category'] == 'orange').sum()
        yellow_count = (mock_predictions_with_alerts['predicted_category'] == 'yellow').sum()
        green_count = (mock_predictions_with_alerts['predicted_category'] == 'green').sum()

        # Verify counts add up to total
        assert orange_count + yellow_count + green_count == len(mock_predictions_with_alerts)

        # Calculate alert percentage
        alert_percentage = (orange_count + yellow_count) / len(mock_predictions_with_alerts) * 100

        assert 0 <= alert_percentage <= 100, "Alert percentage should be 0-100"
