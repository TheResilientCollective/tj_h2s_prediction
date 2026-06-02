"""Tests for S3-backed tiered alert state management."""

import json
import pytest
import pandas as pd
from unittest.mock import MagicMock

from h2s.defs.tiered_alerts.state import (
    _empty_horizon_cell,
    _empty_tiers_state,
    higher_tier_active_in_horizon,
    load_state,
    mark_clear_cycle,
    mark_fired,
    should_send_onset,
    tier_last_sent_within_quiet,
)
from h2s.constants import ALERT_QUIET_HOURS


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _mock_s3(stored_json: str | None = None):
    s3 = MagicMock()
    if stored_json is None:
        s3.getFile.side_effect = Exception("not found")
    else:
        s3.getFile.return_value = stored_json.encode("utf-8")
    return s3


def _now():
    return pd.Timestamp.now("UTC")


# ---------------------------------------------------------------------------
# State loading and backward compatibility
# ---------------------------------------------------------------------------

def test_load_state_fresh():
    s3 = _mock_s3()
    state = load_state(s3)
    assert "watch" in state
    assert "critical" in state
    assert "tiers" in state
    assert "tier_1" in state["tiers"]
    assert "nowcast" in state["tiers"]["tier_1"]


def test_load_state_backward_compat_no_tiers_key():
    """A state file without 'tiers' key must read successfully."""
    existing = {"watch": {"in_event": False}, "critical": {"in_event": False}}
    s3 = _mock_s3(json.dumps(existing))
    state = load_state(s3)
    assert "tiers" in state
    assert "tier_1" in state["tiers"]
    assert "nowcast" in state["tiers"]["tier_1"]
    # Original keys preserved
    assert state["watch"]["in_event"] is False


def test_load_state_missing_horizon_defaults_to_empty():
    """Missing horizon sub-keys within tiers default to fresh inactive cell."""
    existing = {
        "watch": {},
        "critical": {},
        "tiers": {
            "tier_1": {"nowcast": {"active": True, "last_fired_at": "2026-01-01T00:00:00"}},
            "tier_2": {},
            "tier_3": {},
        },
    }
    s3 = _mock_s3(json.dumps(existing))
    state = load_state(s3)
    # tier_1/nowcast preserved
    assert state["tiers"]["tier_1"]["nowcast"]["active"] is True
    # tier_1/near defaulted
    near = state["tiers"]["tier_1"]["near"]
    assert near["active"] is False
    # tier_2 and tier_3 fully populated
    for h in ("nowcast", "near", "mid", "day_ahead"):
        assert h in state["tiers"]["tier_2"]
        assert h in state["tiers"]["tier_3"]


# ---------------------------------------------------------------------------
# Onset / clear transitions
# ---------------------------------------------------------------------------

def test_should_send_onset_inactive_cell():
    cell = _empty_horizon_cell()
    assert should_send_onset(cell, _now()) is True


def test_should_send_onset_within_quiet_period():
    cell = _empty_horizon_cell()
    now = _now()
    mark_fired(cell, 0.8, now - pd.Timedelta(hours=1))
    # 1h < ALERT_QUIET_HOURS (3h) → suppress
    assert should_send_onset(cell, now) is False


def test_should_send_onset_after_quiet_period():
    cell = _empty_horizon_cell()
    now = _now()
    mark_fired(cell, 0.8, now - pd.Timedelta(hours=ALERT_QUIET_HOURS + 0.5))
    assert should_send_onset(cell, now) is True


def test_mark_fired_updates_cell():
    cell = _empty_horizon_cell()
    now = _now()
    mark_fired(cell, 0.75, now)
    assert cell["active"] is True
    assert cell["last_score"] == pytest.approx(0.75, abs=0.001)
    assert cell["consecutive_clear_cycles"] == 0
    assert cell["rolling_7d_fires"] == 1


def test_mark_clear_cycle_closes_after_three():
    cell = _empty_horizon_cell()
    now = _now()
    mark_fired(cell, 0.8, now)
    assert mark_clear_cycle(cell) is False  # 1 cycle
    assert mark_clear_cycle(cell) is False  # 2 cycles
    closed = mark_clear_cycle(cell)          # 3 cycles
    assert closed is True
    assert cell["active"] is False


def test_mark_clear_cycle_resets_on_new_fire():
    cell = _empty_horizon_cell()
    now = _now()
    mark_fired(cell, 0.8, now)
    mark_clear_cycle(cell)
    mark_clear_cycle(cell)
    # Before closing, fire again
    mark_fired(cell, 0.9, now + pd.Timedelta(hours=1))
    assert cell["consecutive_clear_cycles"] == 0


# ---------------------------------------------------------------------------
# Within-horizon higher-tier suppression
# ---------------------------------------------------------------------------

def test_higher_tier_active_suppresses_lower():
    state = {"tiers": _empty_tiers_state()}
    state["tiers"]["tier_2"]["nowcast"]["active"] = True
    assert higher_tier_active_in_horizon(state, "tier_1", "nowcast") is True


def test_lower_tier_does_not_suppress_higher():
    state = {"tiers": _empty_tiers_state()}
    state["tiers"]["tier_1"]["nowcast"]["active"] = True
    assert higher_tier_active_in_horizon(state, "tier_2", "nowcast") is False


def test_cross_horizon_no_suppression():
    """Tier 2 active in nowcast does NOT suppress Tier 1 in day_ahead."""
    state = {"tiers": _empty_tiers_state()}
    state["tiers"]["tier_2"]["nowcast"]["active"] = True
    # tier_1/day_ahead: higher_tier_active_in_horizon checks same horizon only
    assert higher_tier_active_in_horizon(state, "tier_1", "day_ahead") is False


# ---------------------------------------------------------------------------
# Per-tier message dedup
# ---------------------------------------------------------------------------

def test_tier_dedup_suppresses_within_quiet():
    state = {"tiers": _empty_tiers_state()}
    now = _now()
    mark_fired(state["tiers"]["tier_2"]["nowcast"], 0.8, now - pd.Timedelta(hours=1))
    # 1h ago → within quiet period
    assert tier_last_sent_within_quiet(state, "tier_2", now) is True


def test_tier_dedup_allows_after_quiet():
    state = {"tiers": _empty_tiers_state()}
    now = _now()
    mark_fired(
        state["tiers"]["tier_2"]["nowcast"], 0.8,
        now - pd.Timedelta(hours=ALERT_QUIET_HOURS + 1)
    )
    assert tier_last_sent_within_quiet(state, "tier_2", now) is False


def test_tier_dedup_one_message_per_cycle_multiple_horizons():
    """If multiple horizons of the same tier fire, only one Slack message should be sent.

    This tests the state invariant: tier_last_sent_within_quiet returns True
    after any horizon for that tier fired recently.
    """
    state = {"tiers": _empty_tiers_state()}
    now = _now()
    mark_fired(state["tiers"]["tier_1"]["nowcast"], 0.9, now - pd.Timedelta(minutes=5))
    # Even if near also just fired, dedup sees the tier-level recent fire
    assert tier_last_sent_within_quiet(state, "tier_1", now) is True
