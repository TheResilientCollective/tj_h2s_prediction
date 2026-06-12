"""Tests for the recursive forecast engine (Phase 3 of
docs/feature/rename_workplan.md).

These pin the autoregressive-feature construction at the actual/predicted
boundary BEFORE the engine is trusted with real data. Synthetic inputs are
used only to test mechanics — never for training or skill claims.

Conventions under test:
- h2s history is ordered oldest → newest; history[-1] is the last actual (t0)
- predicting lead hour h means time t0 + h
- ONE recursive pass produces all leads: each prediction is appended to the
  series, so lag features at later leads read the model's own predictions —
  including inside the nowcast window (leads 2-3), per the PR #37 decision
- short history clamps to the oldest available value (mirrors training's
  rolling(min_periods=1); training drops NaN-lag rows, inference can't)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from h2s.constants import (
    PRODUCT_FORECAST,
    PRODUCT_NEARCAST,
    PRODUCT_NOWCAST,
)
from h2s.forecasting.recursive import (
    H2S_FEATURE_COLS,
    VariantModels,
    autoregressive_features,
    run_products,
)


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------

class _RecordingRegressor:
    """Returns a constant; records every feature row it was asked to score."""

    def __init__(self, constant: float):
        self.constant = constant
        self.rows: list[np.ndarray] = []

    def predict(self, X):
        X = np.asarray(X)
        for row in X:
            self.rows.append(row.copy())
        return np.full(X.shape[0], self.constant)


class _ConstantClassifier:
    def __init__(self, p: float):
        self.p = p

    def predict_proba(self, X):
        n = np.asarray(X).shape[0]
        return np.column_stack([np.full(n, 1 - self.p), np.full(n, self.p)])


# Feature layout for the engine tests: the 5 autoregressive columns first,
# then one exogenous column so we can confirm passthrough.
_FEATURE_COLS = list(H2S_FEATURE_COLS) + ["temperature_2m"]
_IDX = {name: i for i, name in enumerate(_FEATURE_COLS)}


def _frame(n_hours: int = 24) -> pd.DataFrame:
    """Exogenous feature frame for leads 1..n: time + temperature ramp."""
    return pd.DataFrame({
        "time": pd.date_range("2026-06-12 01:00", periods=n_hours, freq="h", tz="UTC"),
        "temperature_2m": np.arange(n_hours, dtype=float) + 100.0,  # distinct per lead
    })


def _history(n: int = 24) -> list[float]:
    """Distinct actuals a1..an (1.0 .. n.0), so positions are identifiable."""
    return [float(i) for i in range(1, n + 1)]


def _run(history, constant=77.0, n_hours=24, p5=0.6, p10=0.4, p30=0.2,
         with_clf30=True):
    reg = _RecordingRegressor(constant)
    models = VariantModels(
        regression=reg,
        clf_5ppb=_ConstantClassifier(p5),
        clf_10ppb=_ConstantClassifier(p10),
        clf_30ppb=_ConstantClassifier(p30) if with_clf30 else None,
    )
    out = run_products(_frame(n_hours), history, models, _FEATURE_COLS)
    return out, reg


# ---------------------------------------------------------------------------
# autoregressive_features — the boundary arithmetic
# ---------------------------------------------------------------------------

class TestAutoregressiveFeatures:
    def test_full_history(self):
        s = _history(24)  # a1..a24, a24 = 24.0 is the last actual
        f = autoregressive_features(s)
        assert f["h2s_lag_1h"] == 24.0
        assert f["h2s_lag_3h"] == 22.0
        assert f["h2s_lag_6h"] == 19.0
        assert f["h2s_rolling_6h"] == pytest.approx(np.mean([19, 20, 21, 22, 23, 24]))
        assert f["h2s_rolling_24h"] == pytest.approx(np.mean(s))

    def test_short_history_clamps_to_oldest(self):
        s = [5.0, 7.0, 9.0]
        f = autoregressive_features(s)
        assert f["h2s_lag_1h"] == 9.0
        assert f["h2s_lag_3h"] == 5.0       # exactly 3 back
        assert f["h2s_lag_6h"] == 5.0       # clamped to oldest
        assert f["h2s_rolling_6h"] == pytest.approx(np.mean(s))   # min_periods=1
        assert f["h2s_rolling_24h"] == pytest.approx(np.mean(s))

    def test_single_value_history(self):
        f = autoregressive_features([3.0])
        assert all(v == 3.0 for v in f.values())


# ---------------------------------------------------------------------------
# Recursive mode — predictions feed back into the features
# ---------------------------------------------------------------------------

class TestRecursiveBoundary:
    """The load-bearing tests: where do actuals end and predictions begin."""

    def test_lead1_uses_actuals_only(self):
        out, reg = _run(_history(24), constant=77.0)
        # First recursive row scored = lead 1
        row = reg.rows[0]
        assert row[_IDX["h2s_lag_1h"]] == 24.0
        assert row[_IDX["h2s_lag_3h"]] == 22.0
        assert row[_IDX["h2s_lag_6h"]] == 19.0
        assert row[_IDX["h2s_rolling_24h"]] == pytest.approx(np.mean(_history(24)))

    def test_lead2_lag1_is_the_lead1_prediction(self):
        out, reg = _run(_history(24), constant=77.0)
        row = reg.rows[1]  # lead 2
        assert row[_IDX["h2s_lag_1h"]] == 77.0          # prediction, not actual
        assert row[_IDX["h2s_lag_3h"]] == 23.0          # still actual (t0-1)

    def test_lead4_lag3_is_the_lead1_prediction(self):
        out, reg = _run(_history(24), constant=77.0)
        row = reg.rows[3]  # lead 4
        assert row[_IDX["h2s_lag_3h"]] == 77.0
        assert row[_IDX["h2s_lag_6h"]] == 22.0          # t0-2, still actual

    def test_lead7_lag6_is_the_lead1_prediction(self):
        out, reg = _run(_history(24), constant=77.0)
        row = reg.rows[6]  # lead 7
        assert row[_IDX["h2s_lag_6h"]] == 77.0          # fully into predictions

    def test_rolling24_at_lead6_blends_19_actuals_5_predictions(self):
        # Predicting lead 6: series = a1..a24 + p1..p5; window = last 24
        # = a6..a24 (19 actuals) + 5 predictions
        out, reg = _run(_history(24), constant=77.0)
        row = reg.rows[5]  # lead 6
        expected = np.mean([float(i) for i in range(6, 25)] + [77.0] * 5)
        assert row[_IDX["h2s_rolling_24h"]] == pytest.approx(expected)

    def test_exogenous_passthrough(self):
        out, reg = _run(_history(24))
        # temperature ramp 100,101,... must arrive per lead untouched
        assert reg.rows[0][_IDX["temperature_2m"]] == 100.0
        assert reg.rows[9][_IDX["temperature_2m"]] == 109.0


# ---------------------------------------------------------------------------
# Nowcast window — same recursion, just the first three leads
# ---------------------------------------------------------------------------

class TestNowcastWindow:
    """Per the design decision on PR #37: the nowcast is NOT a separate
    held-persistence pass — it is leads 1-3 of the single recursion. Lead 1
    is entirely observed; leads 2-3 already read the model's predictions
    through the short lags."""

    def test_single_pass_only(self):
        out, reg = _run(_history(24), constant=77.0)
        # 24 scored rows total — no separate nowcast pass
        assert len(reg.rows) == 24

    def test_nowcast_lead2_lag1_is_the_prediction(self):
        out, reg = _run(_history(24), constant=77.0)
        row = reg.rows[1]  # lead 2, emitted as nowcast
        assert row[_IDX["h2s_lag_1h"]] == 77.0          # recursion inside 0-3h

    def test_nowcast_lag3_is_actual_through_lead3(self):
        out, reg = _run(_history(24), constant=77.0)
        # lead 1: t-3 = t0-2 → a22; lead 3: t-3 = t0 → a24 (last actual)
        assert reg.rows[0][_IDX["h2s_lag_3h"]] == 22.0
        assert reg.rows[2][_IDX["h2s_lag_3h"]] == 24.0

    def test_nowcast_rolling6_at_lead3_blends_predictions(self):
        out, reg = _run(_history(24), constant=77.0)
        row = reg.rows[2]  # lead 3
        # series = a1..a24 + [p1, p2]; last 6 = a21..a24, 77, 77
        expected = np.mean([21.0, 22.0, 23.0, 24.0, 77.0, 77.0])
        assert row[_IDX["h2s_rolling_6h"]] == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Product assembly
# ---------------------------------------------------------------------------

class TestProductAssembly:
    def test_windows_and_counts(self):
        out, _ = _run(_history(24))
        assert set(out["product"].unique()) == {
            PRODUCT_NOWCAST, PRODUCT_NEARCAST, PRODUCT_FORECAST}
        assert sorted(out[out["product"] == PRODUCT_NOWCAST]["lead_hour"]) == [1, 2, 3]
        assert sorted(out[out["product"] == PRODUCT_NEARCAST]["lead_hour"]) == [4, 5, 6]
        assert sorted(out[out["product"] == PRODUCT_FORECAST]["lead_hour"]) == list(range(7, 25))
        assert len(out) == 24

    def test_probabilities_passed_through(self):
        out, _ = _run(_history(24), p5=0.6, p10=0.4, p30=0.2)
        assert np.allclose(out["p5"], 0.6)
        assert np.allclose(out["p10"], 0.4)
        assert np.allclose(out["p30"], 0.2)

    def test_missing_clf30_yields_nan_not_crash(self):
        out, _ = _run(_history(24), with_clf30=False)
        assert out["p30"].isna().all()
        assert out["h2s_pred"].notna().all()

    def test_predictions_clipped_nonnegative(self):
        out, _ = _run(_history(24), constant=-5.0)
        assert (out["h2s_pred"] >= 0).all()

    def test_time_column_carried(self):
        out, _ = _run(_history(24))
        assert out.loc[out["lead_hour"] == 1, "time"].iloc[0] == pd.Timestamp(
            "2026-06-12 01:00", tz="UTC")

    def test_short_history_no_crash(self):
        out, _ = _run([4.0, 6.0], constant=10.0)
        assert len(out) == 24
        assert out["h2s_pred"].notna().all()
