"""Tests for the selection logic in `multi_station_trainer.train_and_select`.

The headline behaviour change: regression now defaults to `recall_30` (alert
boundary), not `R2`. This test file pins:
  - default selection_metric for regression is `recall_30`
  - eval_regressor reports recall_30 / recall_100 alongside MAE/R²
  - explicit `selection_metric='r2'` preserves the legacy R²-driven picker
  - classification stays on AUC
  - unsupported selection_metric values raise

The fits use small synthetic data — these are unit tests of the *selector
control flow*, not of model quality. Real-data validation lives in
`scripts/retrain_compare_nestor.py`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from h2s.training import multi_station_trainer as mst
from h2s.training.multi_station_trainer import (
    eval_regressor,
    train_and_select,
)


# ---------------------------------------------------------------------------
# eval_regressor: alert-aligned recall is now in the dict
# ---------------------------------------------------------------------------


class TestEvalRegressorAlertMetrics:
    def test_dict_contains_all_four_threshold_keys(self):
        """eval_regressor reports at 5/10/30/100 — categorical boundaries.

        5 and 10 added per request: complaint rate rises in the 5-10 ppb band
        (yellow_low), and 8 ppb specifically is a known complaint-trigger
        level. Recall@5 and recall@10 give visibility there.
        """
        X = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0]})
        # y = [1, 8, 15, 50, 150] — has a positive at every cut: >5 (4), >10 (3), >30 (2), >100 (1)
        y = pd.Series([1.0, 8.0, 15.0, 50.0, 150.0])

        class _IdentityRegressor:
            def predict(self, X):  # noqa: ARG002
                return np.array([1.0, 8.0, 15.0, 50.0, 150.0])

        out = eval_regressor(_IdentityRegressor(), X, y)
        # Legacy keys still present
        assert {"MAE", "RMSE", "R2"} <= set(out.keys())
        # All four threshold metric trios present
        for thr in (5, 10, 30, 100):
            assert f"recall_{thr}" in out
            assert f"precision_{thr}" in out
            assert f"n_positives_{thr}" in out
            assert out[f"recall_{thr}"] == pytest.approx(1.0)
        # Positive counts match the y distribution
        assert out["n_positives_5"] == 4    # 8, 15, 50, 150
        assert out["n_positives_10"] == 3   # 15, 50, 150
        assert out["n_positives_30"] == 2   # 50, 150
        assert out["n_positives_100"] == 1  # 150


# ---------------------------------------------------------------------------
# Selector control flow on synthetic models
# ---------------------------------------------------------------------------


class _StubModel:
    """Returns canned predictions; satisfies the duck-typed model interface."""

    def __init__(self, preds: np.ndarray, name: str):
        self._preds = preds
        self.name = name
        self.feature_importances_ = np.array([1.0])

    def predict(self, X):  # noqa: ARG002
        return self._preds

    def fit(self, *a, **kw):  # noqa: ARG002
        return self


def _build_dataset() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """7 bulk rows of 5 ppb + 3 extreme rows of 50 ppb.

    Designed so that a smooth model (perfect bulk, near-miss extremes)
    wins R² while a saturated model (predicts >30 everywhere) wins
    recall@30 — the same divergence the real-data Berry comparison
    surfaced between RF and XGB.
    """
    X_train = pd.DataFrame({"a": np.linspace(0, 1, 20)})
    X_test = pd.DataFrame({"a": np.linspace(0, 1, 10)})
    y_train = pd.Series(np.concatenate([np.full(14, 5.0), np.full(6, 50.0)]))
    y_test = pd.Series([5.0] * 7 + [50.0] * 3)
    return X_train, X_test, y_train, y_test


def _patch_trainer(monkeypatch, rf_preds: np.ndarray, xgb_preds: np.ndarray):
    """Replace RF + XGB constructors with stub models for selector-only tests."""
    monkeypatch.setattr(mst, "get_rf_regressor", lambda: _StubModel(rf_preds, "RF"))
    monkeypatch.setattr(mst, "get_xgb_regressor", lambda: _StubModel(xgb_preds, "XGB"))
    monkeypatch.setattr(mst, "HAS_XGB", True)
    # get_feature_importance reads MODEL_FEATURES — stub it out
    monkeypatch.setattr(mst, "get_feature_importance", lambda model, top_n=10: {})


class TestRegressionSelector:
    """The headline test: regression default selector must pick the higher
    recall@30 model even when R² favors the other. Mirrors the real-data
    Berry result where RF won R² but XGB won recall@100 by 32 pp.
    """

    # Predictions designed so R² and recall@30 DISAGREE:
    # y_test = [5]*7 + [50]*3, positives@30 = 3 (the three 50s)
    # RF: perfect on bulk, under-shoots extremes (smooth)
    #     → SSE = 0*7 + 25²*3 = 1875, recall@30 = 0
    # XGB: predicts 50 everywhere, perfect on extremes (saturated)
    #     → SSE = 45²*7 + 0*3 = 14175, recall@30 = 1.0
    _RF_SMOOTH = np.array([5.0]*7 + [25.0]*3)
    _XGB_SPIKY = np.array([50.0]*10)

    def test_default_metric_is_recall_30_and_picks_recall_winner(self, monkeypatch):
        _patch_trainer(monkeypatch, self._RF_SMOOTH, self._XGB_SPIKY)
        X_train, X_test, y_train, y_test = _build_dataset()

        _, choice, metrics = train_and_select(
            X_train, X_test, y_train, y_test, task="regression",
        )

        # Verify the divergence is real before testing the selector
        assert metrics["RF"]["R2"] > metrics["XGB"]["R2"], "R² did not favor RF"
        assert metrics["RF"]["recall_30"] == pytest.approx(0.0)
        assert metrics["XGB"]["recall_30"] == pytest.approx(1.0)
        # New default picks the recall winner, not the R² winner
        assert metrics["selection_metric"] == "recall_30"
        assert choice == "XGBoost"

    def test_explicit_r2_picks_r2_winner_on_same_data(self, monkeypatch):
        _patch_trainer(monkeypatch, self._RF_SMOOTH, self._XGB_SPIKY)
        X_train, X_test, y_train, y_test = _build_dataset()

        _, choice, metrics = train_and_select(
            X_train, X_test, y_train, y_test, task="regression", selection_metric="r2",
        )
        assert metrics["selection_metric"] == "r2"
        assert metrics["RF"]["R2"] > metrics["XGB"]["R2"]
        assert choice == "RandomForest"

    def test_recall_100_selector(self, monkeypatch):
        # Replace y_test to include positives@100. RF catches both 100+ ppb
        # events; XGB catches only one.
        X_train = pd.DataFrame({"a": np.linspace(0, 1, 20)})
        X_test = pd.DataFrame({"a": np.linspace(0, 1, 10)})
        y_train = pd.Series(np.concatenate([np.full(18, 5.0), np.full(2, 150.0)]))
        y_test = pd.Series([5.0]*8 + [150.0, 150.0])  # 2 positives@100

        rf_preds = np.array([0]*8 + [110, 110], dtype=float)
        xgb_preds = np.array([0]*8 + [110, 0], dtype=float)
        _patch_trainer(monkeypatch, rf_preds, xgb_preds)

        _, choice, metrics = train_and_select(
            X_train, X_test, y_train, y_test, task="regression",
            selection_metric="recall_100",
        )
        assert metrics["selection_metric"] == "recall_100"
        assert metrics["RF"]["recall_100"] == pytest.approx(1.0)
        assert metrics["XGB"]["recall_100"] == pytest.approx(0.5)
        assert choice == "RandomForest"

    def test_recall_5_selector(self, monkeypatch):
        """recall_5 added per complaint-rate analysis (5-10 ppb band)."""
        # 5 positives at the 5 ppb cut (anything > 5). RF catches 4; XGB catches 2.
        X_train = pd.DataFrame({"a": np.linspace(0, 1, 20)})
        X_test = pd.DataFrame({"a": np.linspace(0, 1, 10)})
        y_train = pd.Series(np.concatenate([np.full(15, 1.0), np.full(5, 8.0)]))
        y_test = pd.Series([1.0]*5 + [8.0]*5)  # 5 positives@5

        rf_preds = np.array([0]*5 + [6, 6, 6, 6, 1], dtype=float)   # 4 of 5 caught
        xgb_preds = np.array([0]*5 + [6, 6, 1, 1, 1], dtype=float)  # 2 of 5 caught
        _patch_trainer(monkeypatch, rf_preds, xgb_preds)

        _, choice, metrics = train_and_select(
            X_train, X_test, y_train, y_test, task="regression",
            selection_metric="recall_5",
        )
        assert metrics["selection_metric"] == "recall_5"
        assert metrics["RF"]["recall_5"] == pytest.approx(0.8)
        assert metrics["XGB"]["recall_5"] == pytest.approx(0.4)
        assert choice == "RandomForest"

    def test_recall_10_selector_accepted(self, monkeypatch):
        """recall_10 is a valid selection_metric (yellow_low/yellow_high boundary)."""
        identical = np.array([0.0]*7 + [15.0]*3)
        _patch_trainer(monkeypatch, identical.copy(), identical.copy())
        X_train, X_test, y_train, y_test = _build_dataset()
        _, _, metrics = train_and_select(
            X_train, X_test, y_train, y_test, task="regression",
            selection_metric="recall_10",
        )
        assert metrics["selection_metric"] == "recall_10"

    def test_ensemble_when_recall_within_margin(self, monkeypatch):
        # Both models catch the same positives → recall_30 ties → ensemble.
        identical_preds = np.array([0.0]*7 + [50.0]*3)
        _patch_trainer(monkeypatch, identical_preds.copy(), identical_preds.copy())

        X_train, X_test, y_train, y_test = _build_dataset()
        _, choice, _ = train_and_select(
            X_train, X_test, y_train, y_test, task="regression",
        )
        assert choice == "Ensemble"

    def test_unsupported_selection_metric_raises(self):
        X_train, X_test, y_train, y_test = _build_dataset()
        with pytest.raises(ValueError, match="unsupported for regression"):
            train_and_select(
                X_train, X_test, y_train, y_test, task="regression",
                selection_metric="auc",  # not valid for regression
            )

    def test_unsupported_selection_metric_for_classifier_raises(self):
        X_train, X_test, y_train, y_test = _build_dataset()
        y_train_bin = (y_train > 20).astype(int)
        y_test_bin = (y_test > 20).astype(int)
        with pytest.raises(ValueError, match="unsupported for 'clf_5ppb'"):
            train_and_select(
                X_train, X_test, y_train_bin, y_test_bin, task="clf_5ppb",
                selection_metric="recall_30",  # classifier branch rejects this today
            )


# ---------------------------------------------------------------------------
# Integration with a real XGBoost regressor (no stubs)
# ---------------------------------------------------------------------------


class TestRealXGBoostIntegration:
    """Ensure the new return-shape contract holds with a real fit."""

    def test_real_fit_returns_expected_keys(self):
        # Tiny but real fit — the test is for the metrics-dict shape.
        rng = np.random.default_rng(0)
        n = 80
        X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
        y = pd.Series(X["a"] * 5 + rng.normal(scale=2, size=n) + 20)
        # cut at row 60 chronologically
        X_train, X_test = X.iloc[:60], X.iloc[60:]
        y_train, y_test = y.iloc[:60], y.iloc[60:]

        _, choice, metrics = train_and_select(
            X_train, X_test, y_train, y_test, task="regression",
        )

        assert choice in {"RandomForest", "XGBoost", "Ensemble"}
        assert metrics["selection_metric"] == "recall_30"
        assert isinstance(metrics["selection_value_rf"], float)
        assert isinstance(metrics["selection_value_xgb"], float)
        # eval dicts both carry the alert keys
        for sub in (metrics["RF"], metrics["XGB"]):
            assert {"recall_30", "recall_100", "R2", "MAE"} <= set(sub.keys())


# ---------------------------------------------------------------------------
# exceed_30 target for clf_30ppb (Phase 1, docs/feature/rename_workplan.md)
# ---------------------------------------------------------------------------

_PARQUET_PATH = __import__("os").path.join(
    __import__("os").path.dirname(__file__),
    "..", "..", "..", "data", "modeldata_h2s_nofill.parquet",
)


@pytest.mark.skipif(
    not __import__("os").path.exists(_PARQUET_PATH),
    reason=f"real parquet not present at {_PARQUET_PATH}",
)
class TestExceed30Target:
    """prepare_multi_station_features must emit the exceed_30 target that
    clf_30ppb trains on — and it must equal (H2S > 30) exactly."""

    def test_exceed_30_present_and_correct(self):
        from h2s.training.multi_station_trainer import prepare_multi_station_features

        raw = pd.read_parquet(_PARQUET_PATH)
        df = prepare_multi_station_features(raw, station="NESTOR - BES")

        for col in ("exceed_5", "exceed_10", "exceed_30"):
            assert col in df.columns, f"missing target column {col}"

        expected = (df["H2S"] > 30).astype(int)
        assert (df["exceed_30"] == expected).all()
        # Berry must have a meaningful positive count to train clf_30ppb on
        assert df["exceed_30"].sum() > 300, (
            f"unexpectedly few >30ppb hours at Berry: {int(df['exceed_30'].sum())}"
        )
