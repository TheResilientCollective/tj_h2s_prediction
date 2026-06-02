"""S3-backed state for per-tier per-horizon debounce cells.

Extends the existing alert state JSON at ALERT_STATE_S3_PATH by adding a
`tiers` key. The `watch` and `critical` keys are untouched.

Round-trip safety: a state file without a `tiers` key reads successfully
and is augmented in place; missing horizon sub-keys default to a fresh
inactive cell.
"""

import json

import pandas as pd

from h2s.constants import (
    ALERT_QUIET_HOURS,
    ALERT_STATE_S3_PATH,
)
from .tiers import HORIZON_WINDOWS_H


_TIER_KEYS = ("tier_1", "tier_2", "tier_3")
_HORIZON_KEYS = tuple(h.value for h in HORIZON_WINDOWS_H)


def _empty_horizon_cell() -> dict:
    return {
        "last_fired_at":             None,
        "last_score":                0.0,
        "active":                    False,
        "rolling_7d_fires":          0,
        "consecutive_clear_cycles":  0,
    }


def _empty_tier_state() -> dict:
    return {h: _empty_horizon_cell() for h in _HORIZON_KEYS}


def _empty_tiers_state() -> dict:
    return {t: _empty_tier_state() for t in _TIER_KEYS}


def load_state(s3) -> dict:
    """Load full alert state from S3.  Migrates missing `tiers` key in place."""
    try:
        raw = s3.getFile(ALERT_STATE_S3_PATH)
        state = json.loads(raw.decode("utf-8"))
    except Exception:
        state = {}

    # Ensure observation tiers (watch/critical) are untouched
    for key in ("watch", "critical"):
        state.setdefault(key, {})

    # Migrate / initialise the forecast tier cells
    tiers = state.setdefault("tiers", _empty_tiers_state())
    for tier_key in _TIER_KEYS:
        tier_state = tiers.setdefault(tier_key, _empty_tier_state())
        for h in _HORIZON_KEYS:
            tier_state.setdefault(h, _empty_horizon_cell())

    return state


def save_state(s3, state: dict) -> None:
    s3.putFile(
        json.dumps(state, indent=2, default=str),
        path=ALERT_STATE_S3_PATH,
        content_type="application/json",
    )


def get_cell(state: dict, tier_key: str, horizon: str) -> dict:
    return state["tiers"][tier_key][horizon]


# ---------------------------------------------------------------------------
# Debounce rules (per design §4)
# ---------------------------------------------------------------------------

def should_send_onset(cell: dict, now: pd.Timestamp) -> bool:
    """True if no re-fire within ALERT_QUIET_HOURS of the last onset for this cell."""
    if not cell["active"]:
        return True
    last = cell.get("last_fired_at")
    if last is None:
        return True
    last_ts = pd.Timestamp(last)
    if last_ts.tzinfo is None:
        last_ts = last_ts.tz_localize("UTC")
    return (now - last_ts).total_seconds() / 3600 >= ALERT_QUIET_HOURS


def mark_fired(cell: dict, score: float, now: pd.Timestamp) -> None:
    cell["active"] = True
    cell["last_fired_at"] = now.isoformat()
    cell["last_score"] = round(score, 4)
    cell["consecutive_clear_cycles"] = 0
    cell["rolling_7d_fires"] = cell.get("rolling_7d_fires", 0) + 1


def mark_clear_cycle(cell: dict) -> bool:
    """Record a clear cycle (score < 0.3).  Returns True when cell should close (3 consecutive)."""
    cell["consecutive_clear_cycles"] = cell.get("consecutive_clear_cycles", 0) + 1
    if cell["consecutive_clear_cycles"] >= 3:
        cell["active"] = False
        return True
    return False


def higher_tier_active_in_horizon(state: dict, tier_key: str, horizon: str) -> bool:
    """True if a higher tier is active in the same horizon (suppresses lower-tier onset message)."""
    tier_rank = {"tier_1": 1, "tier_2": 2, "tier_3": 3}
    rank = tier_rank.get(tier_key, 0)
    for other_key, other_rank in tier_rank.items():
        if other_rank > rank:
            other_cell = get_cell(state, other_key, horizon)
            if other_cell.get("active", False):
                return True
    return False


def tier_last_sent_within_quiet(state: dict, tier_key: str, now: pd.Timestamp) -> bool:
    """Per-tier message dedup: True if any cell for this tier fired a message within QUIET_HOURS.

    Prevents multiple messages in one cycle when multiple horizons of the same
    tier fire simultaneously — the full consolidated message already lists them.
    """
    tier_cells = state["tiers"][tier_key]
    for h in _HORIZON_KEYS:
        cell = tier_cells.get(h, {})
        last = cell.get("last_fired_at")
        if last is None:
            continue
        last_ts = pd.Timestamp(last)
        if last_ts.tzinfo is None:
            last_ts = last_ts.tz_localize("UTC")
        if (now - last_ts).total_seconds() / 3600 < ALERT_QUIET_HOURS:
            return True
    return False
