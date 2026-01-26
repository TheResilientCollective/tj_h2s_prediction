"""Tests for H2SPredictor class functionality.

Tests the core prediction logic without requiring S3 connection.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import json
import os
from unittest.mock import Mock

from h2s.predictor.h2s_predictor import H2SPredictor


@pytest.fixture
def sample_preprocessing_info():
    """Create sample preprocessing info for testing."""
    return {
        'feature_cols': [
            'temperature_2m', 'wind_speed_10m', 'wind_direction_10m',
            'relative_humidity_2m', 'surface_pressure', 'precipitation',
            'cloud_cover', 'flow_rate_cms', 'tide_height_m',
            'hour_sin', 'hour_cos', 'wind_direction_sin', 'wind_direction_cos',
            'wind_temp_interaction', 'humidity_temp_interaction',
            'wind_direction_cat_encoded', 'tidal_state_encoded',
            'hour', 'day_of_week', 'month'
        ],
        'class_names': ['green', 'orange', 'yellow'],
        'site_name': 'NESTOR - BES',
        'wind_cat_mapping': {'N': 0, 'NE': 1, 'E': 2, 'SE': 3, 'S': 4, 'SW': 5, 'W': 6, 'NW': 7},
        'tidal_mapping': {'rising': 0, 'falling': 1, 'high': 2, 'low': 3}
    }


@pytest.fixture
def sample_raw_data():
    """Create sample raw environmental data."""
    return pd.DataFrame({
        'time': pd.date_range('2024-01-01 12:00:00', periods=5, freq='h'),
        'temperature_2m': [20.5, 21.0, 21.5, 22.0, 22.5],
        'wind_speed_10m': [5.2, 5.5, 6.0, 6.5, 7.0],
        'wind_direction_10m': [180, 190, 200, 210, 220],
        'relative_humidity_2m': [75, 76, 77, 78, 79],
        'surface_pressure': [1013, 1013.5, 1014, 1014.5, 1015],
        'precipitation': [0, 0, 0.1, 0.2, 0],
        'cloud_cover': [50, 55, 60, 65, 70],
        'wind_direction_categorical': ['S', 'S', 'SW', 'SW', 'SW'],
        'flow_rate_cms': [5.0, 5.2, 5.4, 5.6, 5.8],
        'tide_height_m': [1.0, 1.2, 1.4, 1.6, 1.8],
        'tidal_state': ['rising', 'rising', 'high', 'falling', 'falling'],
    })


class TestH2SPredictorInitialization:
    """Test H2SPredictor initialization."""

    def test_can_initialize_with_mock_model(self, sample_preprocessing_info):
        """Test that predictor can be initialized."""
        mock_model = Mock()
        predictor = H2SPredictor(mock_model, sample_preprocessing_info)

        assert predictor.model == mock_model
        assert predictor.prep_info == sample_preprocessing_info
        assert predictor.feature_cols == sample_preprocessing_info['feature_cols']
        assert predictor.class_names == sample_preprocessing_info['class_names']
        assert predictor.site_name == sample_preprocessing_info['site_name']

    def test_loads_encoders_as_dicts(self, sample_preprocessing_info):
        """Test that encoders are loaded as dictionaries."""
        mock_model = Mock()
        predictor = H2SPredictor(mock_model, sample_preprocessing_info)

        assert isinstance(predictor.wind_cat_mapping, dict)
        assert isinstance(predictor.tidal_mapping, dict)

        # Check mappings have expected keys
        assert 'N' in predictor.wind_cat_mapping
        assert 'rising' in predictor.tidal_mapping


class TestPreprocessing:
    """Test data preprocessing functionality."""

    def test_creates_temporal_features(self, sample_raw_data, sample_preprocessing_info):
        """Test that temporal features are created from time column."""
        mock_model = Mock()
        predictor = H2SPredictor(mock_model, sample_preprocessing_info)

        processed = predictor.preprocess_data(sample_raw_data)

        # Check temporal features exist
        assert 'hour' in processed.columns
        assert 'day_of_week' in processed.columns
        assert 'month' in processed.columns
        assert 'hour_sin' in processed.columns
        assert 'hour_cos' in processed.columns

        # Check hour is correct (should be 12, 13, 14, 15, 16)
        expected_hours = [12, 13, 14, 15, 16]
        assert list(processed['hour']) == expected_hours

    def test_cyclical_hour_encoding(self, sample_raw_data, sample_preprocessing_info):
        """Test cyclical encoding of hour."""
        mock_model = Mock()
        predictor = H2SPredictor(mock_model, sample_preprocessing_info)

        processed = predictor.preprocess_data(sample_raw_data)

        # For hour 12 (noon), sin should be ~0, cos should be ~-1
        hour_12_sin = np.sin(2 * np.pi * 12 / 24)
        hour_12_cos = np.cos(2 * np.pi * 12 / 24)

        assert np.isclose(processed['hour_sin'].iloc[0], hour_12_sin, atol=0.01)
        assert np.isclose(processed['hour_cos'].iloc[0], hour_12_cos, atol=0.01)

        # Check values are in valid range [-1, 1]
        assert processed['hour_sin'].min() >= -1
        assert processed['hour_sin'].max() <= 1
        assert processed['hour_cos'].min() >= -1
        assert processed['hour_cos'].max() <= 1

    def test_cyclical_wind_direction_encoding(self, sample_raw_data, sample_preprocessing_info):
        """Test cyclical encoding of wind direction."""
        mock_model = Mock()
        predictor = H2SPredictor(mock_model, sample_preprocessing_info)

        processed = predictor.preprocess_data(sample_raw_data)

        assert 'wind_direction_sin' in processed.columns
        assert 'wind_direction_cos' in processed.columns

        # Check values are in valid range [-1, 1]
        assert processed['wind_direction_sin'].min() >= -1
        assert processed['wind_direction_sin'].max() <= 1
        assert processed['wind_direction_cos'].min() >= -1
        assert processed['wind_direction_cos'].max() <= 1

    def test_interaction_features(self, sample_raw_data, sample_preprocessing_info):
        """Test creation of interaction features."""
        mock_model = Mock()
        predictor = H2SPredictor(mock_model, sample_preprocessing_info)

        processed = predictor.preprocess_data(sample_raw_data)

        assert 'wind_temp_interaction' in processed.columns
        assert 'humidity_temp_interaction' in processed.columns

        # Verify calculation is correct for first row
        expected_wind_temp = sample_raw_data['wind_speed_10m'].iloc[0] * sample_raw_data['temperature_2m'].iloc[0]
        expected_humidity_temp = sample_raw_data['relative_humidity_2m'].iloc[0] * sample_raw_data['temperature_2m'].iloc[0]

        assert np.isclose(processed['wind_temp_interaction'].iloc[0], expected_wind_temp, atol=0.01)
        assert np.isclose(processed['humidity_temp_interaction'].iloc[0], expected_humidity_temp, atol=0.01)

    def test_categorical_encoding(self, sample_raw_data, sample_preprocessing_info):
        """Test encoding of categorical variables."""
        mock_model = Mock()
        predictor = H2SPredictor(mock_model, sample_preprocessing_info)

        processed = predictor.preprocess_data(sample_raw_data)

        assert 'wind_direction_cat_encoded' in processed.columns
        assert 'tidal_state_encoded' in processed.columns

        # Check encoding is correct
        # 'S' should map to 4, 'SW' should map to 5
        assert processed['wind_direction_cat_encoded'].iloc[0] == 4  # 'S'
        assert processed['wind_direction_cat_encoded'].iloc[2] == 5  # 'SW'

        # 'rising' should map to 0, 'high' should map to 2, 'falling' should map to 1
        assert processed['tidal_state_encoded'].iloc[0] == 0  # 'rising'
        assert processed['tidal_state_encoded'].iloc[2] == 2  # 'high'
        assert processed['tidal_state_encoded'].iloc[3] == 1  # 'falling'

    def test_handles_missing_categorical_values(self, sample_raw_data, sample_preprocessing_info):
        """Test that unknown categorical values are handled gracefully."""
        mock_model = Mock()
        predictor = H2SPredictor(mock_model, sample_preprocessing_info)

        # Add unknown category
        data_with_unknown = sample_raw_data.copy()
        data_with_unknown.loc[0, 'wind_direction_categorical'] = 'UNKNOWN'

        processed = predictor.preprocess_data(data_with_unknown)

        # Unknown categories should be encoded as -1
        assert processed['wind_direction_cat_encoded'].iloc[0] == -1

    def test_preserves_original_columns(self, sample_raw_data, sample_preprocessing_info):
        """Test that original columns are preserved."""
        mock_model = Mock()
        predictor = H2SPredictor(mock_model, sample_preprocessing_info)

        processed = predictor.preprocess_data(sample_raw_data)

        # All original columns should still exist
        for col in sample_raw_data.columns:
            assert col in processed.columns, f"Original column '{col}' should be preserved"


class TestPredictionLogic:
    """Test prediction generation logic."""

    def test_predict_returns_correct_columns(self, sample_preprocessing_info):
        """Test that predict returns all required columns."""
        # Create mock model that returns predictions
        mock_model = Mock()
        mock_model.predict.return_value = np.array([0, 1, 2, 0, 1])  # green, orange, yellow, green, orange
        mock_model.predict_proba.return_value = np.array([
            [0.8, 0.1, 0.1],  # green
            [0.2, 0.6, 0.2],  # orange
            [0.1, 0.2, 0.7],  # yellow
            [0.9, 0.05, 0.05],  # green
            [0.3, 0.5, 0.2],  # orange
        ])

        predictor = H2SPredictor(mock_model, sample_preprocessing_info)

        # Create sample preprocessed data
        preprocessed = pd.DataFrame({
            col: [1.0] * 5 for col in sample_preprocessing_info['feature_cols']
        })

        predictions = predictor.predict(preprocessed)

        # Check required columns exist
        required_cols = [
            'predicted_category',
            'probability_green',
            'probability_orange',
            'probability_yellow',
            'confidence',
            'alert'
        ]

        for col in required_cols:
            assert col in predictions.columns, f"Missing column: {col}"

    def test_predicted_categories_are_correct(self, sample_preprocessing_info):
        """Test that predicted categories match class names."""
        mock_model = Mock()
        mock_model.predict.return_value = np.array([0, 1, 2])
        mock_model.predict_proba.return_value = np.array([
            [0.8, 0.1, 0.1],
            [0.2, 0.6, 0.2],
            [0.1, 0.2, 0.7],
        ])

        predictor = H2SPredictor(mock_model, sample_preprocessing_info)

        preprocessed = pd.DataFrame({
            col: [1.0] * 3 for col in sample_preprocessing_info['feature_cols']
        })

        predictions = predictor.predict(preprocessed)

        assert predictions['predicted_category'].iloc[0] == 'green'
        assert predictions['predicted_category'].iloc[1] == 'orange'
        assert predictions['predicted_category'].iloc[2] == 'yellow'

    def test_probabilities_are_assigned_correctly(self, sample_preprocessing_info):
        """Test that probability columns match predict_proba output."""
        mock_model = Mock()
        mock_model.predict.return_value = np.array([0, 1, 2])
        mock_model.predict_proba.return_value = np.array([
            [0.8, 0.1, 0.1],
            [0.2, 0.6, 0.2],
            [0.1, 0.2, 0.7],
        ])

        predictor = H2SPredictor(mock_model, sample_preprocessing_info)

        preprocessed = pd.DataFrame({
            col: [1.0] * 3 for col in sample_preprocessing_info['feature_cols']
        })

        predictions = predictor.predict(preprocessed)

        # Check probabilities match
        assert np.isclose(predictions['probability_green'].iloc[0], 0.8)
        assert np.isclose(predictions['probability_orange'].iloc[1], 0.6)
        assert np.isclose(predictions['probability_yellow'].iloc[2], 0.7)

    def test_confidence_is_max_probability(self, sample_preprocessing_info):
        """Test that confidence equals maximum probability."""
        mock_model = Mock()
        mock_model.predict.return_value = np.array([0, 1, 2])
        mock_model.predict_proba.return_value = np.array([
            [0.8, 0.1, 0.1],
            [0.2, 0.6, 0.2],
            [0.1, 0.2, 0.7],
        ])

        predictor = H2SPredictor(mock_model, sample_preprocessing_info)

        preprocessed = pd.DataFrame({
            col: [1.0] * 3 for col in sample_preprocessing_info['feature_cols']
        })

        predictions = predictor.predict(preprocessed)

        assert np.isclose(predictions['confidence'].iloc[0], 0.8)
        assert np.isclose(predictions['confidence'].iloc[1], 0.6)
        assert np.isclose(predictions['confidence'].iloc[2], 0.7)

    def test_alert_flag_is_correct(self, sample_preprocessing_info):
        """Test that alert flag is True for orange/yellow, False for green."""
        mock_model = Mock()
        mock_model.predict.return_value = np.array([0, 1, 2, 0])  # green, orange, yellow, green
        mock_model.predict_proba.return_value = np.array([
            [0.8, 0.1, 0.1],
            [0.2, 0.6, 0.2],
            [0.1, 0.2, 0.7],
            [0.9, 0.05, 0.05],
        ])

        predictor = H2SPredictor(mock_model, sample_preprocessing_info)

        preprocessed = pd.DataFrame({
            col: [1.0] * 4 for col in sample_preprocessing_info['feature_cols']
        })

        predictions = predictor.predict(preprocessed)

        assert predictions['alert'].iloc[0] == False  # green
        assert predictions['alert'].iloc[1] == True   # orange
        assert predictions['alert'].iloc[2] == True   # yellow
        assert predictions['alert'].iloc[3] == False  # green

    def test_handles_missing_values(self, sample_preprocessing_info):
        """Test that missing values are handled."""
        mock_model = Mock()
        mock_model.predict.return_value = np.array([0, 1])
        mock_model.predict_proba.return_value = np.array([
            [0.8, 0.1, 0.1],
            [0.2, 0.6, 0.2],
        ])

        predictor = H2SPredictor(mock_model, sample_preprocessing_info)

        # Create data with missing values
        preprocessed = pd.DataFrame({
            col: [1.0, np.nan] for col in sample_preprocessing_info['feature_cols']
        })

        # Should not raise an error
        predictions = predictor.predict(preprocessed)

        assert len(predictions) == 2


class TestPredictWithAlerts:
    """Test predict_with_alerts method."""

    def test_filters_to_alerts_only(self, sample_preprocessing_info):
        """Test that predict_with_alerts returns only orange/yellow."""
        mock_model = Mock()
        mock_model.predict.return_value = np.array([0, 1, 2, 0, 1])
        mock_model.predict_proba.return_value = np.array([
            [0.8, 0.1, 0.1],
            [0.2, 0.6, 0.2],
            [0.1, 0.2, 0.7],
            [0.9, 0.05, 0.05],
            [0.3, 0.5, 0.2],
        ])

        predictor = H2SPredictor(mock_model, sample_preprocessing_info)

        preprocessed = pd.DataFrame({
            col: [1.0] * 5 for col in sample_preprocessing_info['feature_cols']
        })

        alerts = predictor.predict_with_alerts(preprocessed)

        # Should only have 3 alerts (orange, yellow, orange)
        assert len(alerts) == 3
        assert 'green' not in alerts['predicted_category'].values
        assert all(alerts['alert'])
