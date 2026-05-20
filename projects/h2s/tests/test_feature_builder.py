"""Tests for h2s.training.feature_builder.

Scope: the wind × regime interaction added per calibration finding #3,
plus idempotency / no-overwrite guarantees the rest of the codebase
already depends on.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from h2s.constants import CORE_FEATURES, MODEL_FEATURES
from h2s.training.feature_builder import ensure_base_features


def _minimal_input(n: int = 6) -> pd.DataFrame:
    """Smallest input that exercises wind_x_stable_atm + co-features."""
    return pd.DataFrame(
        {
            "time": pd.date_range("2026-03-01", periods=n, freq="h"),
            "wind_speed_10m": [0.5, 1.5, 3.0, 8.0, 6.0, 2.0],
            "temperature_2m": [12.0, 11.0, 10.0, 18.0, 20.0, 15.0],
            "relative_humidity_2m": [85.0, 88.0, 90.0, 70.0, 60.0, 80.0],
            "wind_direction_10m": [180.0] * n,
            "wind_gusts_10m": [3.0, 4.0, 5.0, 14.0, 12.0, 5.0],
        }
    )


class TestWindXStableAtm:
    """Calibration finding #3 — sign-flip enabler for the tree."""

    def test_feature_is_added(self):
        df = _minimal_input()
        out = ensure_base_features(df.copy())
        assert "wind_x_stable_atm" in out.columns
        assert "wind_x_stable_atm" in CORE_FEATURES
        assert "wind_x_stable_atm" in MODEL_FEATURES

    def test_zero_when_windy_or_daytime(self):
        # Daytime (is_night=0) → stable_atm=0 → wind_x_stable_atm=0 regardless of wind
        df = pd.DataFrame(
            {
                "time": pd.date_range("2026-03-01 12:00", periods=3, freq="h"),
                "wind_speed_10m": [1.0, 2.0, 4.0],
            }
        )
        out = ensure_base_features(df.copy())
        # noon-ish hours → is_night=0 → stable_atm=0
        assert (out["is_night"] == 0).all()
        assert (out["wind_x_stable_atm"] == 0.0).all()

    def test_equals_wind_when_calm_nocturnal(self):
        # Night hours (00-05, 20-23) with wind < 5 → stable_atm=1
        df = pd.DataFrame(
            {
                "time": pd.date_range("2026-03-01 02:00", periods=3, freq="h"),
                "wind_speed_10m": [1.0, 2.5, 4.5],
            }
        )
        out = ensure_base_features(df.copy())
        assert (out["is_night"] == 1).all()
        assert (out["stable_atm"] == 1).all()
        # In the trapped regime, the feature equals wind speed
        np.testing.assert_array_almost_equal(
            out["wind_x_stable_atm"].values, df["wind_speed_10m"].values
        )

    def test_zero_when_windy_nocturnal(self):
        # Night but wind ≥ 5 → stable_atm=0 → wind_x_stable_atm=0
        df = pd.DataFrame(
            {
                "time": pd.date_range("2026-03-01 02:00", periods=2, freq="h"),
                "wind_speed_10m": [6.0, 10.0],
            }
        )
        out = ensure_base_features(df.copy())
        assert (out["stable_atm"] == 0).all()
        assert (out["wind_x_stable_atm"] == 0.0).all()

    def test_falls_back_to_zero_when_wind_missing(self):
        # No wind_speed_10m column — feature exists but is zero, doesn't crash
        df = pd.DataFrame(
            {
                "time": pd.date_range("2026-03-01 02:00", periods=2, freq="h"),
                "temperature_2m": [15.0, 16.0],
            }
        )
        out = ensure_base_features(df.copy())
        assert "wind_x_stable_atm" in out.columns
        assert (out["wind_x_stable_atm"] == 0.0).all()


class TestIdempotency:
    """ensure_base_features is called in both training and forecast paths.

    The whole module's purpose depends on safe re-application without
    overwriting caller-supplied values.
    """

    def test_preserves_existing_wind_x_stable_atm(self):
        # If the caller already supplied this feature (e.g. from a custom
        # blend), don't clobber it.
        df = _minimal_input()
        df["wind_x_stable_atm"] = -999.0
        out = ensure_base_features(df.copy())
        assert (out["wind_x_stable_atm"] == -999.0).all()

    def test_repeated_calls_are_stable(self):
        df = _minimal_input()
        once = ensure_base_features(df.copy())
        twice = ensure_base_features(once.copy())
        # Same columns, same values — second call is a no-op for any
        # feature ensure_base_features already populated.
        assert set(once.columns) == set(twice.columns)
        for col in once.columns:
            if pd.api.types.is_numeric_dtype(once[col]):
                np.testing.assert_array_almost_equal(
                    once[col].values, twice[col].values, err_msg=f"col={col}"
                )
