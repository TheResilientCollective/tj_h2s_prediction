"""Tests for the cascade raw-sensor observation backstop.

Pins the property the 2026-07-12 incident violated: an in-progress measured
exceedance must fire the cascade even when every model probability is blind.
"""

import pandas as pd

from h2s.defs.cascade_alerts.cascade import (
    NB_SITE,
    OBS_BACKSTOP_WINDOW_H,
    TIER_ORDER,
    evaluate_obs_backstop,
)

NOW = pd.Timestamp("2026-07-12 13:00", tz="UTC")


def _obs(times, values, site=NB_SITE):
    return pd.DataFrame({
        "site_name": site,
        "time": pd.to_datetime(times, utc=True),
        "H2S": values,
    })


def test_critical_onset_fires_all_tiers():
    # The 2026-07-12 scenario: 168.3 ppb measured within the window while the
    # model probabilities were near zero.
    obs = _obs(["2026-07-12 11:00", "2026-07-12 12:00"], [12.0, 168.3])
    out = evaluate_obs_backstop(obs, now=NOW)
    assert [out[t].fired for t in TIER_ORDER] == [True, True, True]
    assert out["tier_3"].peak_ppb == 168.3
    assert out["tier_3"].peak_time == pd.Timestamp("2026-07-12 12:00", tz="UTC")


def test_moderate_reading_fires_lower_tiers_only():
    obs = _obs(["2026-07-12 12:00"], [11.0])
    out = evaluate_obs_backstop(obs, now=NOW)
    assert out["tier_1"].fired and out["tier_2"].fired
    assert not out["tier_3"].fired


def test_quiet_readings_fire_nothing():
    obs = _obs(["2026-07-12 12:00"], [1.2])
    out = evaluate_obs_backstop(obs, now=NOW)
    assert not any(out[t].fired for t in TIER_ORDER)


def test_old_readings_outside_window_ignored():
    stale_time = NOW - pd.Timedelta(hours=OBS_BACKSTOP_WINDOW_H + 1)
    obs = _obs([stale_time.isoformat()], [200.0])
    out = evaluate_obs_backstop(obs, now=NOW)
    assert not any(out[t].fired for t in TIER_ORDER)
    assert out["tier_1"].peak_ppb is None


def test_other_station_readings_ignored():
    obs = _obs(["2026-07-12 12:00"], [200.0], site="SAN YSIDRO")
    out = evaluate_obs_backstop(obs, now=NOW)
    assert not any(out[t].fired for t in TIER_ORDER)


def test_missing_feed_degrades_to_non_firing():
    for degenerate in (None, pd.DataFrame()):
        out = evaluate_obs_backstop(degenerate, now=NOW)
        assert not any(out[t].fired for t in TIER_ORDER)


def test_accepts_h2s_ppb_column_name():
    obs = _obs(["2026-07-12 12:00"], [35.0]).rename(columns={"H2S": "h2s_ppb"})
    out = evaluate_obs_backstop(obs, now=NOW)
    assert out["tier_3"].fired
