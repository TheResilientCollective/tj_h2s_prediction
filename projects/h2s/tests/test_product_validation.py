"""Tests for the forecast validation-store logic (Phase 5).

Pins the forecast↔actual join and the per-lead-hour skill curves on synthetic
inputs (mechanics only — never training/skill claims). The load-bearing
behaviours: the same target hour is kept at every lead hour, unmatched/future
rows drop out, and the two recall flavours (magnitude vs probability-call) are
computed against the right column.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from h2s.forecasting.product_validation import (
    build_validation_rows,
    skill_curves,
)

NB = "NESTOR - BES"


def _products(rows: list[dict]) -> pd.DataFrame:
    base = {
        "run_ts": "2026-06-14T1200Z", "product": "nowcast", "station": NB,
        "variant": "evidence", "model_version": "v1", "lead_hour": 1,
        "h2s_pred": 0.0, "p5": 0.1, "p10": 0.1, "p30": 0.1,
    }
    return pd.DataFrame([{**base, **r} for r in rows])


def _obs(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame([{"site_name": NB, **r} for r in rows])


# ---------------------------------------------------------------------------
# build_validation_rows
# ---------------------------------------------------------------------------

class TestBuildRows:
    def test_joins_actual_and_computes_error_and_flags(self):
        t = pd.Timestamp("2026-06-14 13:00", tz="UTC")
        prods = _products([{"time": t, "h2s_pred": 14.0, "p10": 0.7}])
        obs = _obs([{"time": t, "H2S": 11.0}])
        v = build_validation_rows(prods, obs)
        assert len(v) == 1
        row = v.iloc[0]
        assert row["actual_h2s"] == 11.0
        assert row["error"] == 3.0          # 14 - 11
        assert row["abs_error"] == 3.0
        assert bool(row["actual_ge10"]) is True
        assert bool(row["actual_ge30"]) is False

    def test_keeps_same_hour_at_multiple_lead_hours(self):
        t = pd.Timestamp("2026-06-14 13:00", tz="UTC")
        prods = _products([
            {"time": t, "lead_hour": 1, "run_ts": "2026-06-14T1200Z", "h2s_pred": 12},
            {"time": t, "lead_hour": 6, "run_ts": "2026-06-14T0700Z", "h2s_pred": 20},
        ])
        obs = _obs([{"time": t, "H2S": 13.0}])
        v = build_validation_rows(prods, obs)
        assert sorted(v["lead_hour"]) == [1, 6]
        assert set(v["actual_h2s"]) == {13.0}   # both joined to the same actual

    def test_drops_future_hours_without_actuals(self):
        t0 = pd.Timestamp("2026-06-14 13:00", tz="UTC")
        t1 = pd.Timestamp("2026-06-14 14:00", tz="UTC")
        prods = _products([{"time": t0, "h2s_pred": 9}, {"time": t1, "h2s_pred": 9}])
        obs = _obs([{"time": t0, "H2S": 8.0}])   # only t0 observed
        v = build_validation_rows(prods, obs)
        assert list(v["time"]) == [t0]

    def test_other_station_does_not_join(self):
        t = pd.Timestamp("2026-06-14 13:00", tz="UTC")
        prods = _products([{"time": t, "h2s_pred": 9}])
        obs = _obs([{"site_name": "SAN YSIDRO", "time": t, "H2S": 50.0}])
        v = build_validation_rows(prods, obs)
        assert v.empty

    def test_empty_products_returns_schema(self):
        v = build_validation_rows(_products([]).iloc[0:0], _obs([{"time": pd.Timestamp("2026-06-14", tz="UTC"), "H2S": 1.0}]))
        assert v.empty
        assert "actual_h2s" in v.columns and "abs_error" in v.columns


# ---------------------------------------------------------------------------
# skill_curves
# ---------------------------------------------------------------------------

class TestSkillCurves:
    def _validation(self, specs: list[dict]) -> pd.DataFrame:
        t0 = pd.Timestamp("2026-06-14 13:00", tz="UTC")
        prods = _products([
            {"time": t0 + pd.Timedelta(hours=i), "lead_hour": s.get("lead_hour", 1),
             "product": s.get("product", "nowcast"), "variant": s.get("variant", "evidence"),
             "h2s_pred": s["h2s_pred"], "p5": s.get("p5", 0.1), "p10": s.get("p10", 0.1),
             "p30": s.get("p30", 0.1)}
            for i, s in enumerate(specs)
        ])
        obs = _obs([
            {"time": t0 + pd.Timedelta(hours=i), "H2S": s["actual"]}
            for i, s in enumerate(specs)
        ])
        return build_validation_rows(prods, obs)

    def test_groups_by_product_variant_lead_and_counts(self):
        v = self._validation([
            {"lead_hour": 1, "h2s_pred": 5, "actual": 4},
            {"lead_hour": 1, "h2s_pred": 9, "actual": 8},
            {"lead_hour": 2, "h2s_pred": 3, "actual": 2},
        ])
        curves = skill_curves(v)
        by_lead = curves.set_index("lead_hour")["n"].to_dict()
        assert by_lead[1] == 2
        assert by_lead[2] == 1

    def test_probability_call_recall_uses_classifier_prob(self):
        # actual ≥10 at 2 hours; p10>0.5 flags one of them → recall 1/2
        v = self._validation([
            {"lead_hour": 1, "h2s_pred": 11, "p10": 0.8, "actual": 12},  # flagged + actual exceed → hit
            {"lead_hour": 1, "h2s_pred": 11, "p10": 0.2, "actual": 14},  # not flagged, actual exceed → miss
            {"lead_hour": 1, "h2s_pred": 2,  "p10": 0.1, "actual": 3},   # not flagged, no exceed
        ])
        row = skill_curves(v).iloc[0]
        assert row["n_actual_ge10"] == 2
        assert row["prob_recall_10"] == 0.5
        assert row["prob_precision_10"] == 1.0   # the one flag was correct

    def test_nan_probabilities_excluded_from_prob_metrics(self):
        v = self._validation([
            {"lead_hour": 1, "h2s_pred": 40, "p30": np.nan, "actual": 35},
        ])
        row = skill_curves(v).iloc[0]
        assert row["n_prob_30"] == 0
        assert row["prob_recall_30"] is None
        # magnitude recall still computed (40>30 predicted, 35>30 actual → hit)
        assert row["mag_recall_30"] == 1.0

    def test_empty_validation_returns_empty(self):
        assert skill_curves(pd.DataFrame()).empty
