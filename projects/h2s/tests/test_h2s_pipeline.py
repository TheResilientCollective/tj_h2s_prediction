"""Tests for H2S pipeline assets and predictor logic.

Tests core logic without requiring S3 connection.
"""

import json
import numpy as np
import pandas as pd
import pytest
from unittest.mock import Mock
import dagster as dg

from h2s.defs.h2s_pipeline import h2s_ensemble_predictions
from h2s.predictor.h2s_predictor import H2SPredictor


# ==============================================================================
# Shared fixtures
# ==============================================================================

PREP_INFO = {
    "feature_cols": [
        "temperature_2m", "relative_humidity_2m", "dewpoint_2m", "precipitation",
        "surface_pressure", "cloud_cover", "wind_speed_10m", "wind_direction_10m",
        "wind_gusts_10m", "wind_direction_sin", "wind_direction_cos",
        "wind_speed_10m_avg_2h", "wind_speed_10m_avg_3h", "wind_speed_10m_avg_4h",
        "wind_gusts_10m_max_2h", "wind_gusts_10m_max_3h", "wind_gusts_10m_max_4h",
        "Flow (m^3/s)--Border", "tide_height", "tidal_state_encoded",
        "wind_direction_categorical_encoded", "wind_temp_interaction",
        "humidity_temp_interaction",
    ],
    "class_names": ["green", "orange", "yellow"],
    "site_name": "NESTOR - BES",
    "wind_cat_mapping": {"E": 0, "N": 1, "NE": 2, "NW": 3, "S": 4, "SE": 5, "SW": 6, "W": 7},
    "tidal_mapping": {"ebb": 0, "flood": 1, "slack": 2, "slack high": 3, "slack low": 4},
}


@pytest.fixture
def mock_environmental_data():
    """Raw environmental data matching real column names from training data."""
    n = 24
    return pd.DataFrame({
        "time": pd.date_range("2024-06-15 00:00", periods=n, freq="h"),
        "temperature_2m": np.random.uniform(15, 25, n),
        "wind_speed_10m": np.random.uniform(0, 10, n),
        "wind_gusts_10m": np.random.uniform(0, 15, n),
        "wind_direction_10m": np.random.uniform(0, 360, n),
        "relative_humidity_2m": np.random.uniform(60, 90, n),
        "surface_pressure": np.random.uniform(1010, 1020, n),
        "precipitation": np.random.uniform(0, 5, n),
        "cloud_cover": np.random.uniform(0, 100, n),
        "dewpoint_2m": np.random.uniform(10, 20, n),
        "wind_direction_categorical": (["N", "NE", "E", "SE", "S", "SW", "W", "NW"] * 3)[:n],
        "Flow (m^3/s)--Border": np.random.uniform(1.0, 3.5, n),
        "tide_height": np.random.uniform(-0.5, 2.3, n),
        "tidal_state": (["flood", "ebb", "slack high", "slack low"] * 6)[:n],
    })


@pytest.fixture
def predictor():
    """Real H2SPredictor with mock model and PREP_INFO."""
    mock_model = Mock()
    return H2SPredictor(mock_model, PREP_INFO)


@pytest.fixture
def mock_predictions_with_alerts():
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
# H2SPredictor.preprocess_data — real implementation tests
# ==============================================================================

class TestH2SPredictorPreprocess:
    """Test preprocess_data() against the real implementation."""

    def test_time_cyclicals_created(self, mock_environmental_data, predictor):
        result = predictor.preprocess_data(mock_environmental_data)
        for col in ("hour_sin", "hour_cos", "month_sin", "month_cos"):
            assert col in result.columns, f"Missing: {col}"
        assert result["hour_sin"].between(-1, 1).all()
        assert result["hour_cos"].between(-1, 1).all()

    def test_is_night_flag(self, predictor):
        df = pd.DataFrame({
            "time": pd.to_datetime(["2024-01-01 03:00",  # night (< 6)
                                    "2024-01-01 12:00",  # day
                                    "2024-01-01 21:00"]), # night (>= 20)
        })
        result = predictor.preprocess_data(df)
        assert result["is_night"].tolist() == [1, 0, 1]

    def test_source_regime_zero_during_day(self, predictor):
        df = pd.DataFrame({
            "time": pd.to_datetime(["2024-01-01 12:00"]),
            "wind_direction_10m": [90.0],
        })
        result = predictor.preprocess_data(df)
        assert result["source_regime"].iloc[0] == 0

    def test_source_regime_nonzero_at_night_with_east_wind(self, predictor):
        df = pd.DataFrame({
            "time": pd.to_datetime(["2024-01-01 02:00"]),  # night
            "wind_direction_10m": [90.0],  # East — should be regime 1 (22.5–135)
        })
        result = predictor.preprocess_data(df)
        assert result["source_regime"].iloc[0] == 1

    def test_wind_direction_cyclicals(self, mock_environmental_data, predictor):
        result = predictor.preprocess_data(mock_environmental_data)
        assert "wind_direction_sin" in result.columns
        assert "wind_direction_cos" in result.columns
        assert result["wind_direction_sin"].between(-1, 1).all()

    def test_rolling_wind_features_created(self, mock_environmental_data, predictor):
        result = predictor.preprocess_data(mock_environmental_data)
        for h in (2, 3, 4):
            assert f"wind_speed_10m_avg_{h}h" in result.columns
            assert f"wind_gusts_10m_max_{h}h" in result.columns

    def test_interaction_features(self, mock_environmental_data, predictor):
        result = predictor.preprocess_data(mock_environmental_data)
        assert "wind_temp_interaction" in result.columns
        assert "humidity_temp_interaction" in result.columns
        # Values should equal the product
        expected = mock_environmental_data["wind_speed_10m"] * mock_environmental_data["temperature_2m"]
        np.testing.assert_allclose(result["wind_temp_interaction"].values, expected.values, rtol=1e-5)

    def test_stable_atm_flag(self, predictor):
        df = pd.DataFrame({
            "time": pd.to_datetime(["2024-01-01 02:00",  # night, calm
                                    "2024-01-01 12:00",  # day, calm
                                    "2024-01-01 02:00"]),# night, windy
            "wind_speed_10m": [2.0, 2.0, 8.0],
        })
        result = predictor.preprocess_data(df)
        assert result["stable_atm"].tolist() == [1, 0, 0]

    def test_h2s_lags_zero_in_forecast_mode(self, mock_environmental_data, predictor):
        """Without H2S measurements, lag features should default to 0."""
        assert "H2S" not in mock_environmental_data.columns
        result = predictor.preprocess_data(mock_environmental_data)
        for col in ("h2s_lag_1h", "h2s_lag_3h", "h2s_lag_6h",
                    "h2s_rolling_6h", "h2s_rolling_24h"):
            assert col in result.columns
            assert (result[col] == 0.0).all(), f"{col} should be zero in forecast mode"

    def test_sbiwtp_defaults_filled_when_absent(self, mock_environmental_data, predictor):
        """SBIWTP columns should be filled with defaults when not in input."""
        assert "sbiwtp_flow_mgd" not in mock_environmental_data.columns
        result = predictor.preprocess_data(mock_environmental_data)
        assert "sbiwtp_flow_mgd" in result.columns
        assert (result["sbiwtp_flow_mgd"] == 23.5).all()
        assert (result["sbiwtp_anomaly"] == 0.0).all()
        assert (result["sbiwtp_deficit"] == 0.0).all()

    def test_sbiwtp_passthrough_when_present(self, mock_environmental_data, predictor):
        """SBIWTP columns already in input should not be overwritten."""
        df = mock_environmental_data.copy()
        df["sbiwtp_flow_mgd"] = 30.0
        result = predictor.preprocess_data(df)
        assert (result["sbiwtp_flow_mgd"] == 30.0).all()

    def test_flow_derivatives_created(self, mock_environmental_data, predictor):
        result = predictor.preprocess_data(mock_environmental_data)
        assert "flow_log" in result.columns
        assert "flow_low" in result.columns
        assert "flow_high" in result.columns
        # flow_log should equal log1p(flow)
        expected = np.log1p(mock_environmental_data["Flow (m^3/s)--Border"].values)
        np.testing.assert_allclose(result["flow_log"].values, expected, rtol=1e-5)

    def test_real_column_names_pass_through(self, mock_environmental_data):
        """Flow (m^3/s)--Border and tide_height must survive without being zeroed."""
        with open("../../data/startmodels/xgboost_base/nestor_preprocessing_info.json") as f:
            prep_info = json.load(f)
        real_predictor = H2SPredictor(Mock(), prep_info)
        result = real_predictor.preprocess_data(mock_environmental_data)

        assert "Flow (m^3/s)--Border" in result.columns
        assert result["Flow (m^3/s)--Border"].min() > 0, \
            "Flow was zeroed — column name mismatch with training data"
        assert "tide_height" in result.columns
        assert not (result["tide_height"] == 0.0).all(), \
            "tide_height was zeroed — column name mismatch with training data"

    def test_tidal_mapping_covers_noaa_states(self, predictor):
        noaa_states = {"flood", "ebb", "slack high", "slack low"}
        assert noaa_states.issubset(set(predictor.tidal_mapping.keys()))


# ==============================================================================
# H2SPredictor.predict output schema
# ==============================================================================

class TestH2SPredictorOutput:

    @pytest.fixture
    def predictions_df(self, mock_environmental_data, predictor):
        n = len(mock_environmental_data)
        mock_model = predictor.model
        mock_model.predict.return_value = np.zeros(n, dtype=int)  # all green
        mock_model.predict_proba.return_value = np.column_stack([
            np.full(n, 0.7),   # green
            np.full(n, 0.2),   # orange
            np.full(n, 0.1),   # yellow
        ])
        preprocessed = predictor.preprocess_data(mock_environmental_data)
        return predictor.predict(preprocessed)

    def test_required_columns_present(self, predictions_df):
        required = {"predicted_category", "probability_green", "probability_orange",
                    "probability_yellow", "confidence", "alert", "h2s_risk"}
        assert required.issubset(set(predictions_df.columns))

    def test_h2s_risk_in_unit_interval(self, predictions_df):
        assert predictions_df["h2s_risk"].between(0.0, 1.0).all()

    def test_probabilities_valid(self, predictions_df):
        for col in ("probability_green", "probability_orange", "probability_yellow"):
            assert predictions_df[col].between(0, 1).all()

    def test_confidence_is_max_probability(self, predictions_df):
        max_prob = predictions_df[
            ["probability_green", "probability_orange", "probability_yellow"]
        ].max(axis=1)
        np.testing.assert_allclose(predictions_df["confidence"].values, max_prob.values, atol=1e-6)

    def test_alert_matches_category(self, predictions_df):
        expected = predictions_df["predicted_category"].isin({"orange", "yellow"})
        pd.testing.assert_series_equal(predictions_df["alert"], expected, check_names=False)


# ==============================================================================
# H2S Alerts
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
# Multi-station trainer
# ==============================================================================

class TestMultiStationTrainer:

    @pytest.fixture
    def raw_parquet_df(self):
        """Minimal multi-site DataFrame matching the real parquet schema."""
        n_per_site = 200
        sites = ["SAN YSIDRO", "NESTOR - BES", "IB CIVIC CTR"]
        dfs = []
        for site in sites:
            times = pd.date_range("2023-01-01", periods=n_per_site, freq="h", tz="UTC")
            dfs.append(pd.DataFrame({
                "time": times,
                "site_name": site,
                "H2S": np.random.uniform(0, 50, n_per_site),
                "h2s_measured": True,
                "temperature_2m": np.random.uniform(15, 25, n_per_site),
                "wind_speed_10m": np.random.uniform(0, 10, n_per_site),
                "wind_gusts_10m": np.random.uniform(0, 15, n_per_site),
                "wind_direction_10m": np.random.uniform(0, 360, n_per_site),
                "relative_humidity_2m": np.random.uniform(60, 90, n_per_site),
                "surface_pressure": np.random.uniform(1010, 1020, n_per_site),
                "precipitation": np.random.uniform(0, 2, n_per_site),
                "cloud_cover": np.random.uniform(0, 100, n_per_site),
                "dewpoint_2m": np.random.uniform(10, 20, n_per_site),
                "Flow (m^3/s)--Border": np.random.uniform(1, 5, n_per_site),
                "tide_height": np.random.uniform(-0.5, 2.0, n_per_site),
                "tidal_state": np.random.choice(["flood", "ebb", "slack high", "slack low"], n_per_site),
                # SBIWTP all null — simulates real parquet before feed is connected
                "sbiwtp_flow_mgd": np.nan,
                "sbiwtp_hourly_mgd": np.nan,
                "sbiwtp_anomaly": np.nan,
                "sbiwtp_deficit": np.nan,
                "sbiwtp_flow_x_temp": np.nan,
                "sbiwtp_sli": np.nan,
            }))
        return pd.concat(dfs, ignore_index=True)

    def test_sbiwtp_nulls_filled_with_defaults(self, raw_parquet_df):
        from h2s.training.multi_station_trainer import prepare_multi_station_features
        result = prepare_multi_station_features(raw_parquet_df)
        assert (result["sbiwtp_flow_mgd"] == 23.5).all()
        assert (result["sbiwtp_anomaly"] == 0.0).all()
        assert (result["sbiwtp_deficit"] == 0.0).all()

    def test_all_stations_have_rows(self, raw_parquet_df):
        from h2s.training.multi_station_trainer import prepare_multi_station_features
        result = prepare_multi_station_features(raw_parquet_df)
        for site in ["SAN YSIDRO", "NESTOR - BES", "IB CIVIC CTR"]:
            count = (result["site_name"] == site).sum()
            assert count > 0, f"No rows for {site} after feature engineering"

    def test_target_columns_created(self, raw_parquet_df):
        from h2s.training.multi_station_trainer import prepare_multi_station_features
        result = prepare_multi_station_features(raw_parquet_df)
        assert "exceed_5" in result.columns
        assert "exceed_10" in result.columns
        assert result["exceed_5"].isin([0, 1]).all()
        assert result["exceed_10"].isin([0, 1]).all()

    def test_model_features_all_present(self, raw_parquet_df):
        from h2s.training.multi_station_trainer import prepare_multi_station_features, MODEL_FEATURES
        result = prepare_multi_station_features(raw_parquet_df)
        missing = [f for f in MODEL_FEATURES if f not in result.columns]
        assert not missing, f"Missing MODEL_FEATURES: {missing}"

    def test_no_nulls_in_model_features(self, raw_parquet_df):
        from h2s.training.multi_station_trainer import prepare_multi_station_features, MODEL_FEATURES
        result = prepare_multi_station_features(raw_parquet_df)
        null_cols = [f for f in MODEL_FEATURES if result[f].isna().any()]
        assert not null_cols, f"NaN values remain in: {null_cols}"

    def test_rolling_features_do_not_cross_stations(self, raw_parquet_df):
        """Rolling H2S stats must be computed per-station."""
        from h2s.training.multi_station_trainer import prepare_multi_station_features
        # Set SAN YSIDRO H2S to high, others to zero — lag of SY should not bleed into others
        raw_parquet_df.loc[raw_parquet_df["site_name"] == "SAN YSIDRO", "H2S"] = 100.0
        raw_parquet_df.loc[raw_parquet_df["site_name"] != "SAN YSIDRO", "H2S"] = 0.0
        result = prepare_multi_station_features(raw_parquet_df)
        other = result[result["site_name"] != "SAN YSIDRO"]
        # h2s_rolling_24h for non-SY stations should be low (0 or close)
        assert other["h2s_rolling_24h"].max() <= 1.0, \
            "H2S rolling features bled across station boundaries"

    def test_station_filter_parameter(self, raw_parquet_df):
        from h2s.training.multi_station_trainer import prepare_multi_station_features
        result = prepare_multi_station_features(raw_parquet_df, station="NESTOR - BES")
        assert set(result["site_name"].unique()) == {"NESTOR - BES"}


# ==============================================================================
# Ensemble predictions
# ==============================================================================

class TestEnsemblePredictions:
    """Test h2s_ensemble_predictions: probability averaging across primary + variants."""

    def _make_predictions(self, n=6, green=None, orange=None, yellow=None):
        if green is None:
            probs = np.array([
                [0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.05, 0.05, 0.90],
                [0.6, 0.3, 0.1], [0.2, 0.7, 0.1], [0.15, 0.05, 0.80],
            ])[:n]
            green, orange, yellow = probs[:, 0], probs[:, 1], probs[:, 2]
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
        primary = self._make_predictions()
        result = _run_ensemble(primary, variant_predictions={})
        pd.testing.assert_frame_equal(
            result[["probability_green", "probability_orange", "probability_yellow"]].reset_index(drop=True),
            primary[["probability_green", "probability_orange", "probability_yellow"]].reset_index(drop=True),
        )

    def test_averages_probabilities_across_models(self):
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
        np.testing.assert_allclose(
            result["probability_green"].values,
            ((primary["probability_green"] + variant_a["probability_green"]) / 2).values,
            atol=1e-6,
        )

    def test_n_models_column_present(self):
        primary = self._make_predictions(n=3)
        variant = self._make_predictions(n=3)
        result = _run_ensemble(primary, {"xgboost_base": variant})
        assert "n_models" in result.columns
        assert (result["n_models"] == 2).all()

    def test_three_variants_counted(self):
        primary = self._make_predictions(n=2, green=np.array([0.6, 0.1]),
                                          orange=np.array([0.3, 0.8]), yellow=np.array([0.1, 0.1]))
        vb = self._make_predictions(n=2, green=np.array([0.5, 0.2]),
                                     orange=np.array([0.4, 0.7]), yellow=np.array([0.1, 0.1]))
        vs = self._make_predictions(n=2, green=np.array([0.4, 0.3]),
                                     orange=np.array([0.5, 0.6]), yellow=np.array([0.1, 0.1]))
        vr = self._make_predictions(n=2, green=np.array([0.3, 0.4]),
                                     orange=np.array([0.6, 0.5]), yellow=np.array([0.1, 0.1]))
        result = _run_ensemble(primary, {"xgboost_base": vb, "xgboost_smote": vs, "random_forest": vr})
        assert result["n_models"].iloc[0] == 4

    def test_alert_flag_correct_after_ensemble(self):
        primary = self._make_predictions()
        variant = self._make_predictions()
        result = _run_ensemble(primary, {"xgboost_base": variant})
        for _, row in result.iterrows():
            assert row["alert"] == (row["predicted_category"] in {"orange", "yellow"})


# ==============================================================================
# Asset metadata helpers
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
# Helper
# ==============================================================================

def _run_ensemble(primary_df: pd.DataFrame, variant_predictions: dict) -> pd.DataFrame:
    with dg.build_asset_context() as ctx:
        return h2s_ensemble_predictions(ctx, variant_predictions, primary_df)
