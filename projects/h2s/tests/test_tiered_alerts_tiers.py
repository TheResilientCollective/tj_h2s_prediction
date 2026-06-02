"""Tests for tiered alert gate functions, score function, and nesting invariant."""

import pytest
import pandas as pd

from h2s.defs.tiered_alerts.tiers import (
    Horizon,
    TierResult,
    TierNestingError,
    check_nesting,
    compute_score,
    gate_tier1,
    gate_tier2,
    gate_tier3,
    HORIZON_WINDOWS_H,
)

# --- Fixtures ---

def _row(flow=20.0, anomaly=-1.0, wind=2.0, temp=15.0, dew=12.0, stable=0.7):
    return {
        "sbiwtp_flow_mgd": flow,
        "sbiwtp_anomaly": anomaly,
        "wind_speed_10m": wind,
        "temperature_2m": temp,
        "dewpoint_2m": dew,
        "stable_atm": stable,
        "stable_atm_fraction": stable,
        "temp_min": temp,
    }

_THREE_STATIONS = {"NB": _row(), "IB": _row(), "SY": _row()}


def _make_result(tier, horizon, gate, score, fire, n_stations=1, daytime=False, degraded=False):
    t = pd.Timestamp.now("UTC")
    return TierResult(
        tier=tier, label=tier.upper(), horizon=horizon,
        evaluated_at=t, window=(t, t + pd.Timedelta(hours=3)),
        gate_passed=gate, score=score, n_stations_passing_gate=n_stations,
        contributing_features={}, daytime_horizon=daytime,
        degraded=degraded, fire=fire,
    )


# ---------------------------------------------------------------------------
# Horizon enum
# ---------------------------------------------------------------------------

def test_horizon_values():
    assert Horizon.NOWCAST == "nowcast"
    assert len(HORIZON_WINDOWS_H) == 4


# ---------------------------------------------------------------------------
# Tier 1 gate (data-availability check: wind_speed_10m present and not NaN)
# ---------------------------------------------------------------------------

def test_t1_gate_passes_with_wind_data():
    rows = {"NB": _row(wind=3.0)}
    result = gate_tier1(rows)
    assert result["NB"] is True


def test_t1_gate_passes_regardless_of_sbiwtp():
    # SBIWTP at surplus (30+ MGD, positive anomaly) should no longer block
    rows = {"NB": _row(flow=33.0, anomaly=0.4, wind=2.5)}
    result = gate_tier1(rows)
    assert result["NB"] is True


def test_t1_gate_fails_missing_wind():
    rows = {"NB": {"sbiwtp_flow_mgd": 20.0, "sbiwtp_anomaly": -1.0}}
    result = gate_tier1(rows)
    assert result["NB"] is False


def test_t1_gate_fails_nan_wind():
    rows = {"NB": _row(wind=float("nan"))}
    result = gate_tier1(rows)
    assert result["NB"] is False


def test_t1_gate_multi_station():
    rows = {
        "NB": _row(wind=2.0),
        "IB": _row(wind=float("nan")),
        "SY": {},
    }
    result = gate_tier1(rows)
    assert result["NB"] is True
    assert result["IB"] is False
    assert result["SY"] is False


# ---------------------------------------------------------------------------
# Tier 2 gate
# ---------------------------------------------------------------------------

def test_t2_gate_passes_two_stations():
    t1 = {"NB": True, "IB": True, "SY": False}
    rows = _THREE_STATIONS
    result, n = gate_tier2(rows, t1)
    assert result["NB"] is True
    assert result["IB"] is True
    assert result["SY"] is False
    assert n == 2


def test_t2_gate_fails_only_one_station_passes_t1():
    t1 = {"NB": True, "IB": False, "SY": False}
    rows = _THREE_STATIONS
    result, n = gate_tier2(rows, t1)
    assert all(not v for v in result.values())
    assert n == 1


def test_t2_gate_fails_wind_above_threshold():
    t1 = {"NB": True, "IB": True, "SY": True}
    rows = {k: _row(wind=5.0) for k in ("NB", "IB", "SY")}
    result, _ = gate_tier2(rows, t1)
    assert all(not v for v in result.values())


def test_t2_gate_passes_wind_just_below():
    t1 = {"NB": True, "IB": True, "SY": True}
    rows = {k: _row(wind=3.9) for k in ("NB", "IB", "SY")}
    result, _ = gate_tier2(rows, t1)
    assert all(v for v in result.values())


# ---------------------------------------------------------------------------
# Tier 3 gate
# ---------------------------------------------------------------------------

def test_t3_gate_passes_all_conditions():
    t2 = {"NB": True, "IB": True, "SY": True}
    rows = {k: _row(temp=14.0, dew=12.0, stable=0.7) for k in ("NB", "IB", "SY")}
    result = gate_tier3(rows, t2)
    assert all(result.values())


def test_t3_gate_fails_cold_temp():
    t2 = {"NB": True}
    rows = {"NB": _row(temp=12.0, dew=12.0, stable=0.7)}
    result = gate_tier3(rows, t2)
    assert result["NB"] is False


def test_t3_gate_fails_low_dewpoint():
    t2 = {"NB": True}
    rows = {"NB": _row(temp=15.0, dew=10.0, stable=0.7)}
    result = gate_tier3(rows, t2)
    assert result["NB"] is False


def test_t3_gate_fails_low_stable_atm():
    t2 = {"NB": True}
    rows = {"NB": _row(temp=15.0, dew=12.0, stable=0.5)}
    result = gate_tier3(rows, t2)
    assert result["NB"] is False


def test_t3_gate_fails_when_t2_fails():
    t2 = {"NB": False}
    rows = {"NB": _row(temp=15.0, dew=12.0, stable=0.7)}
    result = gate_tier3(rows, t2)
    assert result["NB"] is False


# ---------------------------------------------------------------------------
# Score function
# ---------------------------------------------------------------------------

def test_score_bounded_0_to_095():
    weights = {"sbiwtp_flow_mgd": -1.44}
    stats   = {"sbiwtp_flow_mgd": {"mean": 25.0, "std": 3.0}}
    # Extreme low flow → high score
    score, _ = compute_score({"sbiwtp_flow_mgd": 1.0}, weights, stats)
    assert 0.0 < score <= 0.95

    # Flow at mean → score near 0.5
    score_mid, _ = compute_score({"sbiwtp_flow_mgd": 25.0}, weights, stats)
    assert 0.3 < score_mid < 0.7


def test_score_saturation_clip():
    weights = {"sbiwtp_flow_mgd": -100.0}
    stats   = {"sbiwtp_flow_mgd": {"mean": 25.0, "std": 1.0}}
    score, _ = compute_score({"sbiwtp_flow_mgd": 1.0}, weights, stats)
    assert score == pytest.approx(0.95)


def test_score_known_value():
    weights = {"x": 1.0}
    stats   = {"x": {"mean": 0.0, "std": 1.0}}
    # z=0 → sigmoid(0) = 0.5
    score, _ = compute_score({"x": 0.0}, weights, stats)
    assert score == pytest.approx(0.5)


def test_score_contributing_features_populated():
    weights = {"sbiwtp_flow_mgd": -1.0, "wind_speed_10m": -0.5}
    stats   = {
        "sbiwtp_flow_mgd": {"mean": 25.0, "std": 3.0},
        "wind_speed_10m":  {"mean": 4.5, "std": 2.0},
    }
    _, contrib = compute_score({"sbiwtp_flow_mgd": 20.0, "wind_speed_10m": 2.0}, weights, stats)
    assert "sbiwtp_flow_mgd" in contrib
    assert "wind_speed_10m" in contrib


def test_score_skips_missing_feature():
    weights = {"sbiwtp_flow_mgd": -1.0, "wind_speed_10m": -0.5}
    stats   = {"sbiwtp_flow_mgd": {"mean": 25.0, "std": 3.0},
               "wind_speed_10m":  {"mean": 4.5, "std": 2.0}}
    # Only sbiwtp_flow_mgd present
    _, contrib = compute_score({"sbiwtp_flow_mgd": 20.0}, weights, stats)
    assert "wind_speed_10m" not in contrib


# ---------------------------------------------------------------------------
# Tier nesting invariant
# ---------------------------------------------------------------------------

def _results_all_firing(horizon: str) -> list[TierResult]:
    return [
        _make_result("tier_1", horizon, gate=True,  score=0.8, fire=True),
        _make_result("tier_2", horizon, gate=True,  score=0.7, fire=True),
        _make_result("tier_3", horizon, gate=True,  score=0.6, fire=True),
    ]


def test_nesting_valid_all_fire():
    check_nesting(_results_all_firing("nowcast"))  # should not raise


def test_nesting_violation_t3_without_t2():
    results = [
        _make_result("tier_1", "nowcast", gate=True,  score=0.8, fire=True),
        _make_result("tier_2", "nowcast", gate=False, score=0.3, fire=False),
        _make_result("tier_3", "nowcast", gate=True,  score=0.6, fire=True),
    ]
    with pytest.raises(TierNestingError):
        check_nesting(results)


def test_nesting_violation_t3_without_t1():
    results = [
        _make_result("tier_1", "nowcast", gate=False, score=0.2, fire=False),
        _make_result("tier_2", "nowcast", gate=True,  score=0.7, fire=True),
        _make_result("tier_3", "nowcast", gate=True,  score=0.6, fire=True),
    ]
    with pytest.raises(TierNestingError):
        check_nesting(results)


def test_nesting_cross_horizon_allowed():
    """T1 fires in day_ahead but not in nowcast — T3 quiet everywhere — valid."""
    results = [
        _make_result("tier_1", "nowcast",   gate=False, score=0.2, fire=False),
        _make_result("tier_1", "day_ahead", gate=True,  score=0.8, fire=True),
        _make_result("tier_2", "nowcast",   gate=False, score=0.2, fire=False),
        _make_result("tier_2", "day_ahead", gate=False, score=0.3, fire=False),
        _make_result("tier_3", "nowcast",   gate=False, score=0.1, fire=False),
        _make_result("tier_3", "day_ahead", gate=False, score=0.2, fire=False),
    ]
    check_nesting(results)  # should not raise


def test_daytime_horizon_flag_propagates():
    result = _make_result("tier_1", "nowcast", gate=True, score=0.8, fire=True, daytime=True)
    assert result.daytime_horizon is True
