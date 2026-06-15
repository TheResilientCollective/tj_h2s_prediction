"""Shared-state I/O for the yellow-tier cell.

Round-trips the whole alert-state document at ``ALERT_STATE_S3_PATH``, adding
only a ``yellow`` key, so the watch/critical/cascade cells written by the other
machines survive.
"""

import json

from h2s.constants import ALERT_STATE_S3_PATH

from .logic import empty_cell


def load_state(s3) -> dict:
    try:
        raw = s3.getFile(ALERT_STATE_S3_PATH)
        state = json.loads(raw.decode("utf-8"))
    except Exception:
        state = {}
    state.setdefault("yellow", empty_cell())
    return state


def save_state(s3, state: dict) -> None:
    s3.putFile(
        json.dumps(state, indent=2, default=str),
        path=ALERT_STATE_S3_PATH,
        content_type="application/json",
    )
