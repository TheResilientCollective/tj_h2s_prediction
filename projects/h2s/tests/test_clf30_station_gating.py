"""Pin the per-station clf_30ppb (P>30 / ORANGE) gating policy.

Decision (2026-06): emit P(>30 ppb) only for stations whose orange call we
trust. NESTOR-BES has enough orange training positives (~344) to learn the
class; IB_CIVIC_CTR (~51) and SAN_YSIDRO (~40) collapse at a fixed operating
point, so their P>30 is suppressed (NaN) and they fall back to the >=10 ppb
tier. See CLF_30PPB_STATIONS in constants.py and the underfitting experiment
in experiments/underfitting_results/.

These tests replicate the exact gating expression the products and daily
pipelines use (`site in CLF_30PPB_STATIONS`) and confirm it composes with the
recursive engine's documented "missing classifier -> NaN probabilities" path.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from h2s.constants import CLF_30PPB_STATIONS, STATIONS
from h2s.forecasting.recursive import VariantModels, run_products


class _ConstReg:
    def predict(self, X):
        return np.full(len(X), 12.0)


class _ConstClf:
    def __init__(self, p):
        self.p = p

    def predict_proba(self, X):
        n = len(X)
        return np.column_stack([np.full(n, 1.0 - self.p), np.full(n, self.p)])


def _feature_frame(n_hours=12):
    return pd.DataFrame({
        "time": pd.date_range("2026-06-27", periods=n_hours, freq="h"),
        **{c: 0.0 for c in (
            "h2s_lag_1h", "h2s_lag_3h", "h2s_lag_6h",
            "h2s_rolling_6h", "h2s_rolling_24h",
        )},
    })


def _models_for(site_name):
    """Build VariantModels exactly as the pipelines do: clf_30ppb is gated to
    None unless the station is in CLF_30PPB_STATIONS."""
    emit_clf30 = site_name in CLF_30PPB_STATIONS
    return VariantModels(
        regression=_ConstReg(),
        clf_5ppb=_ConstClf(0.6),
        clf_10ppb=_ConstClf(0.4),
        clf_30ppb=_ConstClf(0.2) if emit_clf30 else None,
    )


def _run_for(site_name):
    ff = _feature_frame()
    return run_products(ff, [10.0], _models_for(site_name), list(ff.columns.drop("time")))


def test_policy_nestor_only():
    """The trusted set is exactly NESTOR-BES today."""
    assert CLF_30PPB_STATIONS == frozenset({"NESTOR - BES"})
    # And it is a subset of the real stations (no typos).
    assert CLF_30PPB_STATIONS <= set(STATIONS)


def test_nestor_emits_p30():
    out = _run_for("NESTOR - BES")
    assert out["p30"].notna().all()
    assert np.allclose(out["p30"], 0.2)


def test_other_stations_suppress_p30():
    for site_name in ("SAN YSIDRO", "IB CIVIC CTR"):
        out = _run_for(site_name)
        assert out["p30"].isna().all(), f"{site_name} should emit NaN p30"
        # p5 / p10 are unaffected — those classifiers still fire.
        assert out["p5"].notna().all()
        assert out["p10"].notna().all()
