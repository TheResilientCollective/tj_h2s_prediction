"""Tests for the near-real-time H2S seed top-up (forecasting/obs_freshness).

The 2026-07 red-tier recall investigation traced 0% P(>30) recall to the
recursive engines seeding from modeldata rows that end at the last complete
day. These tests pin the merge semantics that fix it: realtime rows extend the
historical series up to the current hour, historical rows stay authoritative
where the sources overlap, and QC matches the modeldata loaders.
"""

import io
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from h2s.forecasting.obs_freshness import (
    load_realtime_h2s,
    merge_h2s_series,
    seed_gap_hours,
)


def _hist(times, values, site="NESTOR - BES"):
    return pd.DataFrame({
        "site_name": site,
        "time": pd.to_datetime(times, utc=True),
        "H2S": values,
    })


class TestMergeH2sSeries:
    def test_realtime_rows_extend_history(self):
        hist = _hist(["2026-07-12 05:00", "2026-07-12 06:00"], [1.0, 2.0])
        rt = _hist(["2026-07-13 17:00", "2026-07-13 18:00"], [102.5, 218.8])
        out = merge_h2s_series(hist, rt, "NESTOR - BES")
        assert len(out) == 4
        assert out["H2S"].tolist() == [1.0, 2.0, 102.5, 218.8]
        assert out["time"].is_monotonic_increasing

    def test_overlapping_hours_keep_historical_value(self):
        hist = _hist(["2026-07-12 05:00", "2026-07-12 06:00"], [1.0, 2.0])
        rt = _hist(["2026-07-12 06:00", "2026-07-12 07:00"], [99.0, 3.0])
        out = merge_h2s_series(hist, rt, "NESTOR - BES")
        # 06:00 stays 2.0 (historical QC'd feed authoritative); 07:00 appended
        assert out["H2S"].tolist() == [1.0, 2.0, 3.0]

    def test_other_stations_rows_ignored(self):
        hist = _hist(["2026-07-12 05:00"], [1.0])
        rt = pd.concat([
            _hist(["2026-07-13 17:00"], [50.0], site="SAN YSIDRO"),
            _hist(["2026-07-13 17:00"], [7.0]),
        ])
        out = merge_h2s_series(hist, rt, "NESTOR - BES")
        assert out["H2S"].tolist() == [1.0, 7.0]

    def test_no_realtime_feed_degrades_to_history(self):
        hist = _hist(["2026-07-12 05:00"], [1.0])
        out = merge_h2s_series(hist, None, "NESTOR - BES")
        assert out["H2S"].tolist() == [1.0]

    def test_empty_history_uses_realtime_only(self):
        hist = _hist([], [])
        rt = _hist(["2026-07-13 17:00"], [42.0])
        out = merge_h2s_series(hist, rt, "NESTOR - BES")
        assert out["H2S"].tolist() == [42.0]

    def test_unknown_site_returns_empty(self):
        hist = _hist(["2026-07-12 05:00"], [1.0])
        out = merge_h2s_series(hist, None, "NOWHERE")
        assert len(out) == 0


class TestSeedGapHours:
    def test_gap_measured_from_newest_row(self):
        series = _hist(["2026-07-13 12:00"], [1.0])
        now = pd.Timestamp("2026-07-13 14:30", tz="UTC")
        assert seed_gap_hours(series, now) == pytest.approx(2.5)

    def test_empty_series_returns_none(self):
        assert seed_gap_hours(_hist([], []), pd.Timestamp.now("UTC")) is None


class TestLoadRealtimeH2s:
    def _feed_csv(self):
        return (
            "SiteName,Site Name,Parameter,Date with time,Result\n"
            "NESTOR - BES,NESTOR - BES,07 H2S PPB,2026-07-13T10:00:00-07:00,168.3\n"
            "NESTOR - BES,NESTOR - BES,07 H2S PPB,2026-07-13T11:00:00-07:00,-0.4\n"
            "NESTOR - BES,NESTOR - BES,07 H2S PPB,2026-07-13T09:00:00-07:00,999.0\n"
            "SAN YSIDRO,SAN YSIDRO,07 H2S PPB,2026-07-13T11:00:00-07:00,1.9\n"
            "SAN YSIDRO,SAN YSIDRO,OTHER,2026-07-13T11:00:00-07:00,55.0\n"
        )

    def test_normalizes_qc_and_filters(self):
        s3 = MagicMock()
        s3.publicUrl.return_value = "https://example/hs2_lastday.csv"
        with patch("pandas.read_csv", return_value=pd.read_csv(io.StringIO(self._feed_csv()))):
            out = load_realtime_h2s(s3)
        # >500 dropped, negative clipped to 0, non-H2S parameter dropped
        nb = out[out["site_name"] == "NESTOR - BES"]
        assert len(nb) == 2
        assert nb["H2S"].tolist() == [168.3, 0.0]
        assert str(out["time"].dt.tz) == "UTC"
        sy = out[out["site_name"] == "SAN YSIDRO"]
        assert sy["H2S"].tolist() == [1.9]
