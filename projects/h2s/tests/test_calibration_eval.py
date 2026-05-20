"""Tests for the calibration-aligned evaluation harness.

Unit-tests the math (Spearman, recall@threshold, persistence, chronological
split) on small known-property cases. The real-parquet integration test
is skipped unless the file is present locally.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from h2s.training.calibration_eval import (
    CalibrationReport,
    calibration_report,
    chronological_split,
    persistence_prediction,
    recall_at_threshold,
    spearman_rank,
)


# ---------------------------------------------------------------------------
# spearman_rank
# ---------------------------------------------------------------------------


class TestSpearmanRank:
    def test_monotone_nonlinear_is_perfect_rank(self):
        # Pearson < 1 but Spearman == 1 — this is the calibration point
        y_true = [1.0, 2.0, 3.0, 4.0, 5.0]
        y_pred = [1.0, 4.0, 9.0, 16.0, 25.0]
        assert spearman_rank(y_true, y_pred) == pytest.approx(1.0)

    def test_perfect_anti_rank(self):
        assert spearman_rank([1, 2, 3, 4, 5], [5, 4, 3, 2, 1]) == pytest.approx(-1.0)

    def test_nan_pairs_dropped(self):
        y_true = [1.0, 2.0, np.nan, 4.0, 5.0]
        y_pred = [1.0, 2.0, 3.0, np.nan, 5.0]
        # Only (1,1), (2,2), (5,5) survive — perfect rank
        assert spearman_rank(y_true, y_pred) == pytest.approx(1.0)

    def test_too_few_points_returns_nan(self):
        assert np.isnan(spearman_rank([1.0, 2.0], [1.0, 2.0]))


# ---------------------------------------------------------------------------
# recall_at_threshold
# ---------------------------------------------------------------------------


class TestRecallAtThreshold:
    def test_perfect_predictions(self):
        y_true = [10, 50, 5, 200, 1]
        y_pred = [10, 50, 5, 200, 1]
        out = recall_at_threshold(y_true, y_pred, 30)
        assert out["n_positives"] == 2  # 50 and 200
        assert out["recall"] == pytest.approx(1.0)
        assert out["precision"] == pytest.approx(1.0)

    def test_zero_recall_when_model_misses_extremes(self):
        # Calibration's recurring finding: exogenous predictors give 0 recall@100
        y_true = [150, 200, 175]
        y_pred = [10, 12, 9]
        out = recall_at_threshold(y_true, y_pred, 100)
        assert out["n_positives"] == 3
        assert out["recall"] == pytest.approx(0.0)
        assert out["precision"] == pytest.approx(0.0)

    def test_false_alarms_only(self):
        y_true = [1, 2, 3, 5]
        y_pred = [150, 200, 1, 1]
        out = recall_at_threshold(y_true, y_pred, 100)
        assert out["n_positives"] == 0
        assert out["recall"] == pytest.approx(0.0)
        assert out["precision"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# persistence_prediction
# ---------------------------------------------------------------------------


class TestPersistencePrediction:
    def test_within_site_shift(self):
        df = pd.DataFrame(
            {
                "time": pd.date_range("2026-01-01", periods=4, freq="h").tolist() * 2,
                "site_name": ["A"] * 4 + ["B"] * 4,
                "H2S": [10.0, 20.0, 30.0, 40.0, 100.0, 200.0, 300.0, 400.0],
            }
        )
        pred = persistence_prediction(df, lag_hours=1)
        # Per site, first row is NaN; subsequent rows are previous value.
        expected = [np.nan, 10.0, 20.0, 30.0, np.nan, 100.0, 200.0, 300.0]
        for got, want in zip(pred.values, expected):
            if np.isnan(want):
                assert np.isnan(got)
            else:
                assert got == want

    def test_no_cross_site_contamination(self):
        # B's first row must NOT borrow from A's last
        df = pd.DataFrame(
            {
                "time": pd.date_range("2026-01-01", periods=2, freq="h").tolist() * 2,
                "site_name": ["A", "A", "B", "B"],
                "H2S": [10.0, 20.0, 100.0, 200.0],
            }
        )
        pred = persistence_prediction(df, lag_hours=1)
        b_first = df.index[df["site_name"] == "B"][0]
        assert np.isnan(pred.loc[b_first])

    def test_index_alignment_preserved_under_shuffled_input(self):
        # Even if input rows are shuffled, output Series keeps original index
        df = pd.DataFrame(
            {
                "time": pd.date_range("2026-01-01", periods=4, freq="h"),
                "site_name": ["A"] * 4,
                "H2S": [10.0, 20.0, 30.0, 40.0],
            }
        )
        shuffled = df.sample(frac=1, random_state=0)
        pred = persistence_prediction(shuffled, lag_hours=1)
        # For each shuffled row, the prediction equals the time-prior H2S
        prior_lookup = dict(zip(df["time"], df["H2S"]))
        for idx, row in shuffled.iterrows():
            prior_time = row["time"] - pd.Timedelta(hours=1)
            if prior_time in prior_lookup:
                assert pred.loc[idx] == prior_lookup[prior_time]
            else:
                assert np.isnan(pred.loc[idx])


# ---------------------------------------------------------------------------
# chronological_split
# ---------------------------------------------------------------------------


class TestChronologicalSplit:
    def test_no_temporal_leakage(self):
        df = pd.DataFrame(
            {
                "time": pd.date_range("2026-01-01", periods=10, freq="h"),
                "x": range(10),
            }
        ).sample(frac=1, random_state=0)  # shuffled — sort happens inside
        train, test = chronological_split(df, train_fraction=0.7)
        assert train["time"].max() < test["time"].min()
        assert len(train) == 7
        assert len(test) == 3

    def test_invalid_fraction_raises(self):
        df = pd.DataFrame({"time": pd.date_range("2026-01-01", periods=5, freq="h")})
        with pytest.raises(ValueError):
            chronological_split(df, train_fraction=0.0)
        with pytest.raises(ValueError):
            chronological_split(df, train_fraction=1.0)


# ---------------------------------------------------------------------------
# calibration_report
# ---------------------------------------------------------------------------


class TestCalibrationReport:
    def test_perfect_predictions_yield_perfect_scores(self):
        df = pd.DataFrame(
            {
                "H2S": [1.0, 50.0, 150.0, 2.0, 60.0, 200.0],
                "site_name": ["A", "A", "A", "B", "B", "B"],
                "stable_atm": [0, 1, 1, 0, 0, 1],
            }
        )
        y_pred = df["H2S"].to_numpy()
        report = calibration_report(df, y_pred)
        assert isinstance(report, CalibrationReport)
        assert report.overall["spearman"] == pytest.approx(1.0)
        assert report.overall["thr_30"]["recall"] == pytest.approx(1.0)
        assert report.overall["thr_100"]["recall"] == pytest.approx(1.0)
        assert set(report.per_site.keys()) == {"A", "B"}
        assert set(report.per_regime.keys()) == {
            "calm_stable_atm_1",
            "windy_stable_atm_0",
        }

    def test_length_mismatch_raises(self):
        df = pd.DataFrame(
            {"H2S": [1.0, 2.0], "site_name": ["A", "A"], "stable_atm": [0, 0]}
        )
        with pytest.raises(ValueError, match="!="):
            calibration_report(df, [1.0])

    def test_regime_stratification_separates_calm_from_windy(self):
        # Calibration finding #2: rank skill differs between regimes
        df = pd.DataFrame(
            {
                "H2S": [1.0, 1.0, 1.0, 100.0, 100.0, 100.0],
                "site_name": ["A"] * 6,
                "stable_atm": [0, 0, 0, 1, 1, 1],
            }
        )
        # Model: predict 0 always (zero rank on either subset)
        y_pred = np.zeros(6)
        report = calibration_report(df, y_pred)
        # Calm subset has all the extremes; recall@30 is 0 (model never fires)
        assert report.per_regime["calm_stable_atm_1"]["n"] == 3
        assert report.per_regime["calm_stable_atm_1"]["thr_30"]["n_positives"] == 3
        assert report.per_regime["calm_stable_atm_1"]["thr_30"]["recall"] == 0.0
        # Windy subset has no extremes
        assert report.per_regime["windy_stable_atm_0"]["thr_30"]["n_positives"] == 0


# ---------------------------------------------------------------------------
# Integration — real parquet (skipped if unavailable)
# ---------------------------------------------------------------------------

_PARQUET_PATH = os.path.join(
    os.path.dirname(__file__),
    "..", "..", "..", "data", "modeldata_h2s_nofill.parquet",
)


@pytest.mark.skipif(
    not os.path.exists(_PARQUET_PATH),
    reason=f"real parquet not present at {_PARQUET_PATH}",
)
class TestRealDataPersistence:
    """Sanity-check the harness on actual H2S observations at NESTOR-BES.

    Asserts the persistence baseline produces meaningful signal — not a
    magnitude test. Calibration finding #8: persistence Spearman ≈ 0.70
    at lag-1h on Berry.
    """

    @pytest.fixture(scope="class")
    def berry_df(self) -> pd.DataFrame:
        df = pd.read_parquet(_PARQUET_PATH)
        df = df[(df["h2s_measured"] == True) & (df["site_name"] == "NESTOR - BES")].copy()
        df["H2S"] = df["H2S"].clip(lower=0)
        return df.sort_values("time").reset_index(drop=True)

    def test_persistence_spearman_is_strong(self, berry_df):
        pred = persistence_prediction(berry_df, lag_hours=1)
        rho = spearman_rank(berry_df["H2S"], pred)
        # Calibration log records ≈ 0.70 on this site. > 0.5 leaves headroom
        # while still failing if the persistence math regressed.
        assert rho > 0.5, f"Persistence Spearman {rho:.3f} on Berry looks broken"
