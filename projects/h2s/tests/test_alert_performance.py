"""Tests for the observed >10 ppb "Alert Performance" machine (Phase 4).

Pins the onset/close transitions, the hours-above tally, and the
forecast-vs-actual join BEFORE the machine drives Slack or writes S3. Synthetic
observations/forecasts exercise mechanics only.
"""

from __future__ import annotations

import pandas as pd

from h2s.defs.alert_performance.logic import (
    NB_SITE,
    empty_cell,
    evaluate_yellow,
    forecast_vs_actual,
    hours_above,
    performance_summary,
)


def _obs(h2s_by_offset: list[float], end="2026-06-14 12:00") -> pd.DataFrame:
    """Hourly NESTOR-BES observations ending at `end`; list is oldest→newest."""
    n = len(h2s_by_offset)
    times = pd.date_range(end=end, periods=n, freq="h", tz="UTC")
    return pd.DataFrame({"time": times, "H2S": [float(x) for x in h2s_by_offset]})


# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------

class TestOnset:
    def test_opens_on_first_crossing_after_quiet(self):
        df = _obs([2, 3, 4, 12])  # calm then a 12 ppb reading
        now = df["time"].iloc[-1]
        verdict, cell = evaluate_yellow(df, empty_cell(), now)
        assert verdict.send_onset
        assert cell["in_event"]
        assert verdict.current_ppb == 12.0

    def test_no_onset_when_not_quiet(self):
        # a prior >=10 reading within the quiet window suppresses a fresh onset
        df = _obs([12, 5, 12])
        now = df["time"].iloc[-1]
        verdict, _ = evaluate_yellow(df, empty_cell(), now)
        assert not verdict.send_onset

    def test_no_reonset_while_in_event_updates_last_exceedance(self):
        df = _obs([2, 3, 12, 15])
        now = df["time"].iloc[-1]
        cell = empty_cell()
        cell.update({"in_event": True, "event_start": str(df["time"].iloc[2]),
                     "last_exceedance": str(df["time"].iloc[2]), "onset_sent": True})
        verdict, cell = evaluate_yellow(df, cell, now)
        assert not verdict.send_onset
        assert cell["last_exceedance"] == str(df["time"].iloc[-1])  # advanced


class TestClose:
    def test_closes_after_two_hours_below(self):
        df = _obs([12, 15, 11, 4, 4])  # peak then drop below 10
        t = list(df["time"])
        cell = empty_cell()
        cell.update({"in_event": True, "event_start": str(t[0]),
                     "last_exceedance": str(t[2]), "onset_sent": True})
        now = t[4]  # 2h after last exceedance (t[2])
        verdict, cell = evaluate_yellow(df, cell, now)
        assert verdict.send_close
        assert not cell["in_event"]
        assert cell["close_sent"]
        # event_df is the >=10 stretch
        assert list(verdict.event_df["H2S"]) == [12.0, 15.0, 11.0]

    def test_no_close_before_wait_elapsed(self):
        df = _obs([12, 15, 11, 4])
        t = list(df["time"])
        cell = empty_cell()
        cell.update({"in_event": True, "event_start": str(t[0]),
                     "last_exceedance": str(t[2]), "onset_sent": True})
        now = t[3]  # only 1h after last exceedance
        verdict, _ = evaluate_yellow(df, cell, now)
        assert not verdict.send_close

    def test_no_close_while_still_above(self):
        df = _obs([12, 15, 11, 13])
        t = list(df["time"])
        cell = empty_cell()
        cell.update({"in_event": True, "event_start": str(t[0]),
                     "last_exceedance": str(t[2]), "onset_sent": True})
        now = t[3] + pd.Timedelta(hours=3)
        verdict, _ = evaluate_yellow(df, cell, now)
        assert not verdict.send_close


# ---------------------------------------------------------------------------
# Hours above
# ---------------------------------------------------------------------------

class TestHoursAbove:
    def test_counts_yellow_and_orange_in_window(self):
        df = _obs([5, 12, 35, 8, 40, 9])
        now = df["time"].iloc[-1]
        ha = hours_above(df, now, window_h=12)
        assert ha["yellow"] == 3   # >=10: 12, 35, 40
        assert ha["orange"] == 2   # >=30: 35, 40

    def test_ignores_readings_outside_window(self):
        # 5 hourly readings ending 12:00 → 08,09,10,11,12; window_h=2 keeps 10–12 only
        df = _obs([99, 99, 5, 6, 7], end="2026-06-14 12:00")
        now = df["time"].iloc[-1]
        ha = hours_above(df, now, window_h=2)
        assert ha["yellow"] == 0   # the two 99s are outside the 2h window
        assert ha["orange"] == 0


# ---------------------------------------------------------------------------
# Forecast vs actual
# ---------------------------------------------------------------------------

def _products(rows: list[dict]) -> pd.DataFrame:
    base = {"station": NB_SITE, "variant": "evidence", "lead_hour": 1,
            "h2s_pred": 0.0, "p10": 0.1, "p30": 0.05}
    return pd.DataFrame([{**base, **r} for r in rows])


class TestForecastVsActual:
    def test_joins_earliest_run_per_hour_to_actuals(self):
        t0 = pd.Timestamp("2026-06-14 10:00", tz="UTC")
        prods = _products([
            {"run_ts": "2026-06-14T0800Z", "time": t0, "p10": 0.8, "h2s_pred": 14},  # earlier run
            {"run_ts": "2026-06-14T0900Z", "time": t0, "p10": 0.6, "h2s_pred": 11},  # later run
        ])
        obs = pd.DataFrame({"time": [t0], "H2S": [13.0]})
        fva = forecast_vs_actual(prods, obs, t0 - pd.Timedelta(hours=1), t0 + pd.Timedelta(hours=1))
        assert len(fva) == 1
        assert fva["p_exceed"].iloc[0] == 0.8          # earliest run kept
        assert fva["actual_h2s"].iloc[0] == 13.0
        assert bool(fva["actual_exceeded"].iloc[0]) is True

    def test_filters_other_station_and_out_of_window(self):
        t0 = pd.Timestamp("2026-06-14 10:00", tz="UTC")
        prods = _products([
            {"run_ts": "r", "time": t0, "p10": 0.7},
            {"run_ts": "r", "time": t0, "station": "SAN YSIDRO", "p10": 0.9},
            {"run_ts": "r", "time": t0 + pd.Timedelta(hours=5), "p10": 0.9},
        ])
        obs = pd.DataFrame({"time": [t0], "H2S": [13.0]})
        fva = forecast_vs_actual(prods, obs, t0 - pd.Timedelta(hours=1), t0 + pd.Timedelta(hours=1))
        assert len(fva) == 1
        assert fva["p_exceed"].iloc[0] == 0.7

    def test_empty_products_returns_empty(self):
        t0 = pd.Timestamp("2026-06-14 10:00", tz="UTC")
        obs = pd.DataFrame({"time": [t0], "H2S": [13.0]})
        fva = forecast_vs_actual(None, obs, t0, t0)
        assert fva.empty
        assert list(fva.columns) == ["time", "lead_hour", "p_exceed", "h2s_pred",
                                     "actual_h2s", "actual_exceeded"]

    def test_no_time_overlap_returns_empty(self):
        t0 = pd.Timestamp("2026-06-14 10:00", tz="UTC")
        prods = _products([{"run_ts": "r", "time": t0, "p10": 0.7}])
        obs = pd.DataFrame({"time": [t0 + pd.Timedelta(hours=3)], "H2S": [13.0]})
        fva = forecast_vs_actual(prods, obs, t0 - pd.Timedelta(hours=1), t0 + pd.Timedelta(hours=6))
        assert fva.empty


class TestPerformanceSummary:
    def test_precision_recall(self):
        fva = pd.DataFrame({
            "p_exceed": [0.7, 0.6, 0.3, 0.2],
            "actual_exceeded": [True, False, True, False],
        })
        s = performance_summary(fva, prob_cut=0.5)
        assert s["n_forecast_exceed"] == 2
        assert s["n_actual_exceed"] == 2
        assert s["hits"] == 1
        assert s["precision"] == 0.5
        assert s["recall"] == 0.5

    def test_empty_summary(self):
        s = performance_summary(pd.DataFrame(columns=["p_exceed", "actual_exceeded"]))
        assert s["n_hours"] == 0
        assert s["precision"] is None
