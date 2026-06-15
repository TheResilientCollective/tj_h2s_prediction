"""S3-backed debounce for the forecast cascade tiers.

Shares the single alert-state JSON at ``ALERT_STATE_S3_PATH`` with the
observation alert systems. We add only a ``cascade`` key and round-trip the
rest of the document untouched, so the ``watch`` / ``critical`` cells written by
``h2s_alert_system`` survive — and theirs preserve ``cascade`` the same way.
(Both writers read-modify-write the whole document; collisions are possible but
rare given the 6 h cascade cadence — same trade-off the retired tier system
accepted.)

Debounce is intentionally simple: one quiet window per tier. Forecast
pre-alerts refresh every 6 h, so the elaborate clear-cycle hysteresis the old
hourly system needed is gone.
"""

import json

import pandas as pd

from h2s.constants import ALERT_QUIET_HOURS, ALERT_STATE_S3_PATH

from .cascade import TIER_ORDER


def _empty_cell() -> dict:
    return {"last_sent_at": None, "last_prob": None, "n_sent": 0}


def load_state(s3) -> dict:
    """Load the full alert-state document, ensuring a populated ``cascade`` key."""
    try:
        raw = s3.getFile(ALERT_STATE_S3_PATH)
        state = json.loads(raw.decode("utf-8"))
    except Exception:
        state = {}
    cascade = state.setdefault("cascade", {})
    for tier in TIER_ORDER:
        cascade.setdefault(tier, _empty_cell())
    return state


def save_state(s3, state: dict) -> None:
    s3.putFile(
        json.dumps(state, indent=2, default=str),
        path=ALERT_STATE_S3_PATH,
        content_type="application/json",
    )


def should_send(state: dict, tier_key: str, now: pd.Timestamp) -> bool:
    """True if this tier has not sent within ALERT_QUIET_HOURS."""
    cell = state["cascade"][tier_key]
    last = cell.get("last_sent_at")
    if last is None:
        return True
    last_ts = pd.Timestamp(last)
    if last_ts.tzinfo is None:
        last_ts = last_ts.tz_localize("UTC")
    return (now - last_ts).total_seconds() / 3600 >= ALERT_QUIET_HOURS


def mark_sent(state: dict, tier_key: str, prob: float, now: pd.Timestamp) -> None:
    cell = state["cascade"][tier_key]
    cell["last_sent_at"] = now.isoformat()
    cell["last_prob"] = round(float(prob), 4) if prob == prob else None  # NaN-safe
    cell["n_sent"] = cell.get("n_sent", 0) + 1
