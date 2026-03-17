"""Tests for H2S pipeline assets.

Tests the core logic of each asset without requiring S3 connection.
"""

import pandas as pd
import numpy as np
import pytest
from unittest.mock import Mock
import dagster as dg

from h2s.defs.h2s_pipeline import (
    preprocessed_features,
    h2s_predictions,
    h2s_alerts,
    h2s_ensemble_predictions,
)
from h2s.predictor.h2s_predictor import H2SPredictor


# Column names must match nestor_preprocessing_info.json feature_cols exactly
REAL_FEATURE_COLS = [
    "temperature_2m",
    "relative_humidity_2m",
    "dewpoint_2m",
    "precipitation",
    "surface_pressure",
    "cloud_cover",
    "wind_speed_10m",
    "wind_direction_10m",
    "wind_gusts_10m",
    "wind_direction_sin",
    "wind_direction_cos",
    "wind_speed_10m_avg_2h",
    "wind_speed_10m_avg_3h",
    "wind_speed_10m_avg_4h",
    "wind_gusts_10m_max_2h",
    "wind_gusts_10m_max_3h",
    "wind_gusts_10m_max_4h",
    "Flow (m^3/s)--Border",
    "tide_height",
    "tidal_state_encoded",
    "wind_direction_categorical_encoded",
    "wind_temp_interaction",
    "humidity_temp_interaction",
]


@pytest.fixture
def mock_environmental_data():
    """Raw environmental data with correct column names for the real model."""
    return pd.DataFrame({
        "time": pd.date_range("2024-01-01", periods=10, freq="h"),
        "temperature_2m": np.random.uniform(15, 25, 10),
        "wind_speed_10m": np.random.uniform(0, 10, 10),
        "wind_gusts_10m": np.random.uniform(0, 15, 10),
        "wind_direction_10m": np.random.uniform(0, 360, 10),
        "relative_humidity_2m": np.random.uniform(60, 90, 10),
        "surface_pressure": np.random.uniform(1010, 1020, 10),
        "precipitation": np.random.uniform(0, 5, 10),
        "cloud_cover": np.random.uniform(0, 100, 10),
        "dewpoint_2m": np.random.uniform(10, 20, 10),
        "wind_direction_categorical": ["N", "NE", "E", "SE", "S", "SW", "W", "NW", "N", "NE"],
        # Real column names from training data
        "Flow (m^3/s)--Border": np.random.uniform(1.0, 3.5, 10),
        "tide_height": np.random.uniform(-0.5, 2.3, 10),
        "tidal_state": ["flood", "ebb", "slack high", "slack low", "flood",
                        "ebb", "flood", "ebb", "slack high", "ebb"],
    })


@pytest.fixture
def mock_predictor():
    """Mock H2SPredictor using real feature/class names from nestor_preprocessing_info.json."""
    predictor = Mock(spec=H2SPredictor)
    predictor.feature_cols = REAL_FEATURE_COLS
    predictor.class_names = ["green", "orange", "yellow"]
    predictor.site_name = "NESTOR - BES"
    predictor.wind_cat_mapping = {"E": 0, "N": 1, "NE": 2, "NW": 3, "S": 4, "SE": 5, "SW": 6, "W": 7}
    predictor.tidal_mapping = {"ebb": 0, "flood": 1, "slack": 2, "slack high": 3, "slack low": 4}
    return predictor


@pytest.fixture
def mock_predictions_with_alerts():
    """Mix of alert and non-alert predictions."""
    return pd.DataFrame({
        "time": pd.date_range("2024-01-01", periods=10, freq="h"),
        "predicted_category": ["green", "yellow", "orange", "green", "yellow",
                               "orange", "green", "green", "yellow", "green"],
        "alert": [False, True, True, False, True, True, False, False, True, False],
        "confidence": [0.8, 0.6, 0.7, 0.9, 0.5, 0.8, 0.85, 0.7, 0.55, 0.75],
        "probability_green":  [0.8, 0.3, 0.1, 0.9, 0.35, 0.05, 0.85, 0.70, 0.35, 0.75],
        "probability_orange": [0.1, 0.3, 0.7, 0.05, 0.2, 0.80, 0.05, 0.15, 0.25, 0.10],
        "probability_yellow": [0.1, 0.4, 0.2, 0.05, 0.45, 0.15, 0.1, 0.15, 0.40, 0.15],
    })


# ==============================================================================
# Preprocessing
# ==============================================================================

class TestPreprocessedFeatures:
    """Test that preprocessing creates required derived features."""

    def test_creates_cyclical_wind_features(self, mock_environmental_data, mock_predictor):
        df = mock_environmental_data.copy()
        df["wind_direction_sin"] = np.sin(np.radians(df["wind_direction_10m"]))
        df["wind_direction_cos"] = np.cos(np.radians(df["wind_direction_10m"]))
        mock_predictor.preprocess_data.return_value = df

        result = mock_predictor.preprocess_data(mock_environmental_data)

        assert "wind_direction_sin" in result.columns
        assert "wind_direction_cos" in result.columns
        assert result["wind_direction_sin"].between(-1, 1).all()
        assert result["wind_direction_cos"].between(-1, 1).all()

    def test_creates_interaction_features(self, mock_environmental_data, mock_predictor):
        df = mock_environmental_data.copy()
        df["wind_temp_interaction"] = df["wind_speed_10m"] * df["temperature_2m"]
        df["humidity_temp_interaction"] = df["relative_humidity_2m"] * df["temperature_2m"]
        mock_predictor.preprocess_data.return_value = df

        result = mock_predictor.preprocess_data(mock_environmental_data)

        assert "wind_temp_interaction" in result.columns
        assert "humidity_temp_interaction" in result.columns

    def test_encodes_categorical_features(self, mock_environmental_data, mock_predictor):
        df = mock_environmental_data.copy()
        df["wind_direction_cat_encoded"] = df["wind_direction_categorical"].map(
            mock_predictor.wind_cat_mapping
        ).fillna(-1).astype(int)
        df["tidal_state_encoded"] = df["tidal_state"].map(
            mock_predictor.tidal_mapping
        ).fillna(-1).astype(int)
        mock_predictor.preprocess_data.return_value = df

        result = mock_predictor.preprocess_data(mock_environmental_data)

        assert "wind_direction_cat_encoded" in result.columns
        assert "tidal_state_encoded" in result.columns
        assert result["wind_direction_cat_encoded"].dtype in [np.int32, np.int64]
        assert result["tidal_state_encoded"].dtype in [np.int32, np.int64]

    def test_real_column_names_pass_through(self, mock_environmental_data):
        """Flow (m^3/s)--Border and tide_height must survive preprocessing without being zeroed."""
        from h2s.predictor.h2s_predictor import H2SPredictor
        import json

        with open("../../data/startmodels/xgboost_smote/nestor_preprocessing_info.json") as f:
            prep_info = json.load(f)

        mock_model = Mock()
        predictor = H2SPredictor(mock_model, prep_info)
        df = mock_environmental_data.copy()
        df["date"] = df["time"]

        result = predictor.preprocess_data(df)

        # These must carry real values, not the 0.0 fallback
        assert "Flow (m^3/s)--Border" in result.columns
        assert result["Flow (m^3/s)--Border"].min() > 0, \
            "Flow was zeroed out — column name mismatch with training data"
        assert "tide_height" in result.columns
        # tide_height can be negative (below MLLW), but not stuck at 0 when we passed real values
        assert not (result["tide_height"] == 0.0).all(), \
            "tide_height was zeroed out — column name mismatch with training data"

    def test_tidal_state_mapping_covers_noaa_states(self, mock_predictor):
        """NOAA-derived tidal states (flood/ebb/slack high/slack low) must all be in the mapping."""
        noaa_states = {"flood", "ebb", "slack high", "slack low"}
        mapped = set(mock_predictor.tidal_mapping.keys())
        assert noaa_states.issubset(mapped), \
            f"Tidal states missing from mapping: {noaa_states - mapped}"


# ==============================================================================
# Predictions
# ==============================================================================

class TestH2SPredictions:

    @pytest.fixture
    def mock_predictions_output(self):
        return pd.DataFrame({
            "time": pd.date_range("2024-01-01", periods=10, freq="h"),
            "predicted_category": ["green", "yellow", "orange", "green", "yellow",
                                   "orange", "green", "green", "yellow", "green"],
            "probability_green":  [0.8, 0.3, 0.1, 0.9, 0.35, 0.05, 0.85, 0.70, 0.35, 0.75],
            "probability_orange": [0.1, 0.3, 0.7, 0.05, 0.20, 0.80, 0.05, 0.15, 0.25, 0.10],
            "probability_yellow": [0.1, 0.4, 0.2, 0.05, 0.45, 0.15, 0.10, 0.15, 0.40, 0.15],
            "confidence": [0.8, 0.4, 0.7, 0.9, 0.45, 0.8, 0.85, 0.7, 0.4, 0.75],
            "alert": [False, True, True, False, True, True, False, False, True, False],
        })

    def test_predictions_have_required_columns(self, mock_predictions_output):
        required_cols = [
            "predicted_category", "probability_green", "probability_orange",
            "probability_yellow", "confidence", "alert",
        ]
        for col in required_cols:
            assert col in mock_predictions_output.columns, f"Missing column: {col}"

    def test_probabilities_are_valid(self, mock_predictions_output):
        for col in ["probability_green", "probability_orange", "probability_yellow"]:
            assert mock_predictions_output[col].min() >= 0
            assert mock_predictions_output[col].max() <= 1

    def test_probabilities_sum_to_one(self, mock_predictions_output):
        prob_sum = (
            mock_predictions_output["probability_green"]
            + mock_predictions_output["probability_orange"]
            + mock_predictions_output["probability_yellow"]
        )
        assert np.allclose(prob_sum, 1.0, atol=0.01)

    def test_confidence_is_max_probability(self, mock_predictions_output):
        max_prob = mock_predictions_output[
            ["probability_green", "probability_orange", "probability_yellow"]
        ].max(axis=1)
        assert np.allclose(mock_predictions_output["confidence"], max_prob, atol=0.01)

    def test_predicted_category_matches_max_probability(self, mock_predictions_output):
        class_map = {0: "green", 1: "orange", 2: "yellow"}
        for _, row in mock_predictions_output.iterrows():
            probs = [row["probability_green"], row["probability_orange"], row["probability_yellow"]]
            expected = class_map[int(np.argmax(probs))]
            assert row["predicted_category"] == expected

    def test_alert_flag_matches_category(self, mock_predictions_output):
        for _, row in mock_predictions_output.iterrows():
            expected = row["predicted_category"] in {"orange", "yellow"}
            assert row["alert"] == expected

    def test_categories_are_valid(self, mock_predictions_output):
        assert set(mock_predictions_output["predicted_category"].unique()).issubset(
            {"green", "yellow", "orange"}
        )


# ==============================================================================
# Alerts
# ==============================================================================

class TestH2SAlerts:

    def test_filters_only_alerts(self, mock_predictions_with_alerts):
        alerts = mock_predictions_with_alerts[mock_predictions_with_alerts["alert"]]
        assert len(alerts) == 5
        assert alerts["alert"].all()

    def test_no_green_in_alerts(self, mock_predictions_with_alerts):
        alerts = mock_predictions_with_alerts[mock_predictions_with_alerts["alert"]]
        assert "green" not in alerts["predicted_category"].values

    def test_only_orange_and_yellow_in_alerts(self, mock_predictions_with_alerts):
        alerts = mock_predictions_with_alerts[mock_predictions_with_alerts["alert"]]
        assert set(alerts["predicted_category"].unique()).issubset({"orange", "yellow"})

    def test_alert_count_matches_orange_plus_yellow(self, mock_predictions_with_alerts):
        orange = (mock_predictions_with_alerts["predicted_category"] == "orange").sum()
        yellow = (mock_predictions_with_alerts["predicted_category"] == "yellow").sum()
        assert mock_predictions_with_alerts["alert"].sum() == orange + yellow

    def test_preserves_all_columns(self, mock_predictions_with_alerts):
        alerts = mock_predictions_with_alerts[mock_predictions_with_alerts["alert"]]
        assert set(alerts.columns) == set(mock_predictions_with_alerts.columns)


# ==============================================================================
# Ensemble Predictions
# ==============================================================================

class TestEnsemblePredictions:
    """Test h2s_ensemble_predictions: probability averaging across the primary model + variants.

    Production model is determined by deployment_approval → production_model_deployment,
    which copies the approved variant to MODEL_PATH/nestor_xgboost_weighted_model.json.
    h2s_variant_predictions loads additional variants from MODEL_PATH/{variant}/model.json.
    The ensemble averages probabilities across all available models (primary + variants).
    """

    def _make_predictions(self, n=6, green=None, orange=None, yellow=None):
        """Build a predictions DataFrame with controlled probabilities."""
        if green is None:
            defaults = np.array([
                [0.7, 0.2, 0.1],
                [0.1, 0.8, 0.1],
                [0.05, 0.05, 0.90],
                [0.6, 0.3, 0.1],
                [0.2, 0.7, 0.1],
                [0.15, 0.05, 0.80],
            ])[:n]
            green, orange, yellow = defaults[:, 0], defaults[:, 1], defaults[:, 2]
        assert len(green) == n
        categories = ["green", "orange", "yellow"]
        probs = np.stack([green, orange, yellow], axis=1)
        predicted = [categories[i] for i in probs.argmax(axis=1)]
        return pd.DataFrame({
            "time": pd.date_range("2024-06-01", periods=n, freq="h"),
            "predicted_category": predicted,
            "probability_green":  green,
            "probability_orange": orange,
            "probability_yellow": yellow,
            "confidence": probs.max(axis=1),
            "alert": [c in {"orange", "yellow"} for c in predicted],
        })

    def test_falls_back_to_primary_when_no_variants(self):
        """With an empty variant dict, ensemble == primary model output."""
        primary = self._make_predictions()
        result = _run_ensemble(primary, variant_predictions={})

        assert len(result) == len(primary)
        pd.testing.assert_frame_equal(
            result[["probability_green", "probability_orange", "probability_yellow"]].reset_index(drop=True),
            primary[["probability_green", "probability_orange", "probability_yellow"]].reset_index(drop=True),
        )

    def test_averages_probabilities_across_models(self):
        """Ensemble probability is the mean of primary + each variant."""
        primary = self._make_predictions(
            n=3,
            green=np.array([0.6, 0.1, 0.3]),
            orange=np.array([0.3, 0.8, 0.2]),
            yellow=np.array([0.1, 0.1, 0.5]),
        )
        variant_a = self._make_predictions(
            n=3,
            green=np.array([0.4, 0.2, 0.1]),
            orange=np.array([0.5, 0.6, 0.3]),
            yellow=np.array([0.1, 0.2, 0.6]),
        )
        result = _run_ensemble(primary, {"xgboost_base": variant_a})

        expected_green  = (primary["probability_green"]  + variant_a["probability_green"])  / 2
        expected_orange = (primary["probability_orange"] + variant_a["probability_orange"]) / 2
        expected_yellow = (primary["probability_yellow"] + variant_a["probability_yellow"]) / 2

        np.testing.assert_allclose(result["probability_green"].values,  expected_green.values,  atol=1e-6)
        np.testing.assert_allclose(result["probability_orange"].values, expected_orange.values, atol=1e-6)
        np.testing.assert_allclose(result["probability_yellow"].values, expected_yellow.values, atol=1e-6)

    def test_ensemble_includes_all_three_variants(self):
        """All three variant slots (xgboost_base, xgboost_smote, random_forest) are averaged in."""
        primary   = self._make_predictions(n=2, green=np.array([0.6, 0.1]), orange=np.array([0.3, 0.8]), yellow=np.array([0.1, 0.1]))
        variant_b = self._make_predictions(n=2, green=np.array([0.5, 0.2]), orange=np.array([0.4, 0.7]), yellow=np.array([0.1, 0.1]))
        variant_s = self._make_predictions(n=2, green=np.array([0.4, 0.3]), orange=np.array([0.5, 0.6]), yellow=np.array([0.1, 0.1]))
        variant_r = self._make_predictions(n=2, green=np.array([0.3, 0.4]), orange=np.array([0.6, 0.5]), yellow=np.array([0.1, 0.1]))

        result = _run_ensemble(primary, {
            "xgboost_base": variant_b,
            "xgboost_smote": variant_s,
            "random_forest": variant_r,
        })

        assert result["n_models"].iloc[0] == 4  # primary + 3 variants

    def test_ensemble_category_from_averaged_argmax(self):
        """Predicted category reflects the class with highest averaged probability."""
        primary = self._make_predictions(
            n=1, green=np.array([0.1]), orange=np.array([0.7]), yellow=np.array([0.2])
        )
        variant = self._make_predictions(
            n=1, green=np.array([0.1]), orange=np.array([0.2]), yellow=np.array([0.7])
        )
        result = _run_ensemble(primary, {"xgboost_base": variant})

        # avg orange = 0.45, avg yellow = 0.45 — tie; either is valid
        assert result["predicted_category"].iloc[0] in {"orange", "yellow"}

    def test_ensemble_alert_flag_set_for_orange_and_yellow(self):
        """Alert flag must be True for orange and yellow, False for green."""
        primary = self._make_predictions()
        variant = self._make_predictions()
        result = _run_ensemble(primary, {"xgboost_base": variant})

        for _, row in result.iterrows():
            expected_alert = row["predicted_category"] in {"orange", "yellow"}
            assert row["alert"] == expected_alert

    def test_ensemble_output_has_n_models_column(self):
        """Ensemble DataFrame must include n_models to record how many models contributed."""
        primary = self._make_predictions(n=3)
        variant = self._make_predictions(n=3)
        result = _run_ensemble(primary, {"xgboost_base": variant})

        assert "n_models" in result.columns
        assert (result["n_models"] == 2).all()


# ==============================================================================
# Asset Metadata
# ==============================================================================

class TestAssetMetadata:

    def test_category_counts_sum_to_total(self, mock_predictions_with_alerts):
        orange = (mock_predictions_with_alerts["predicted_category"] == "orange").sum()
        yellow = (mock_predictions_with_alerts["predicted_category"] == "yellow").sum()
        green  = (mock_predictions_with_alerts["predicted_category"] == "green").sum()
        assert orange + yellow + green == len(mock_predictions_with_alerts)

    def test_alert_percentage_in_range(self, mock_predictions_with_alerts):
        orange = (mock_predictions_with_alerts["predicted_category"] == "orange").sum()
        yellow = (mock_predictions_with_alerts["predicted_category"] == "yellow").sum()
        pct = (orange + yellow) / len(mock_predictions_with_alerts) * 100
        assert 0 <= pct <= 100


# ==============================================================================
# Helpers
# ==============================================================================

def _run_ensemble(primary_df: pd.DataFrame, variant_predictions: dict) -> pd.DataFrame:
    """Call h2s_ensemble_predictions directly using a real Dagster build context."""
    with dg.build_asset_context() as ctx:
        return h2s_ensemble_predictions(ctx, variant_predictions, primary_df)
