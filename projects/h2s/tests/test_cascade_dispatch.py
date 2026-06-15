"""Tests for the cascade message builder and the shared-state debounce (Phase 4).

The pure trigger logic is pinned in test_cascade.py; here we pin the two side
effects: what the Slack report says, and that the debounce shares the alert
state file without clobbering the watch/critical cells.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from h2s.constants import ALERT_STATE_S3_PATH, PRODUCT_FORECAST, PRODUCT_NEARCAST, PRODUCT_NOWCAST
from h2s.defs.cascade_alerts.cascade import NB_SITE, evaluate_cascade
from h2s.defs.cascade_alerts.messages import build_cascade_message
from h2s.defs.cascade_alerts import state as cstate


# ---------------------------------------------------------------------------
# Frame builder (multi-station, multi-variant)
# ---------------------------------------------------------------------------

def _product_for_lead(lead: int) -> str:
    if lead <= 3:
        return PRODUCT_NOWCAST
    if lead <= 6:
        return PRODUCT_NEARCAST
    return PRODUCT_FORECAST


def _frame(specs: list[dict]) -> pd.DataFrame:
    """specs: list of {station, variant, prob_overrides, h2s_overrides}."""
    times = pd.date_range("2026-06-14 13:00", periods=24, freq="h", tz="UTC")
    rows = []
    for spec in specs:
        probs = spec.get("prob_overrides", {})
        h2s = spec.get("h2s_overrides", {})
        for i, lead in enumerate(range(1, 25)):
            p5, p10, p30 = probs.get(lead, (0.1, 0.1, 0.1))
            rows.append({
                "run_ts": "2026-06-14T1300Z",
                "product": _product_for_lead(lead),
                "station": spec["station"],
                "lead_hour": lead,
                "time": times[i],
                "variant": spec["variant"],
                "model_version": "v-test",
                "h2s_pred": h2s.get(lead, 0.0),
                "p5": p5, "p10": p10, "p30": p30,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Message builder
# ---------------------------------------------------------------------------

class TestMessage:
    def test_tier1_only_message_has_nowcast_and_hours(self):
        df = _frame([
            {"station": NB_SITE, "variant": "evidence",
             "prob_overrides": {2: (0.7, 0.1, 0.1)}, "h2s_overrides": {1: 8.0, 2: 6.0}},
            {"station": NB_SITE, "variant": "lean", "prob_overrides": {2: (0.6, 0.1, 0.1)}},
        ])
        result = evaluate_cascade(df)
        msg = build_cascade_message(result, df, pd.Timestamp("2026-06-14 13:00", tz="UTC"))
        assert "PLANT-SIGNAL" in msg
        assert "Nowcast" in msg
        assert "detectable H2S" in msg
        assert "E 70% / L 60%" in msg           # both variants reported
        assert "MULTI-SITE-RISK" not in msg     # tier 2 not fired → no section

    def test_all_tiers_message_has_all_station_board(self):
        specs = []
        for st in ("NESTOR - BES", "SAN YSIDRO", "IB CIVIC CTR"):
            specs.append({"station": st, "variant": "evidence",
                          "prob_overrides": {2: (0.9, 0.1, 0.1), 5: (0.1, 0.7, 0.1), 9: (0.1, 0.1, 0.6)}})
            specs.append({"station": st, "variant": "lean",
                          "prob_overrides": {9: (0.1, 0.1, 0.5)}})
        df = _frame(specs)
        result = evaluate_cascade(df)
        msg = build_cascade_message(result, df, pd.Timestamp("2026-06-14 13:00", tz="UTC"))
        assert "EXCEEDANCE-RISK" in msg
        assert "All-station forecast P(>30 ppb)" in msg
        assert "SAN YSIDRO" in msg and "IB CIVIC CTR" in msg
        assert "pending hook" in msg            # health-team future routing flagged

    def test_missing_lean_renders_na_not_crash(self):
        df = _frame([{"station": NB_SITE, "variant": "evidence",
                      "prob_overrides": {2: (0.8, 0.1, 0.1)}}])
        result = evaluate_cascade(df)
        msg = build_cascade_message(result, df, pd.Timestamp("2026-06-14 13:00", tz="UTC"))
        assert "L n/a" in msg

    def test_nan_p30_renders_na(self):
        df = _frame([{"station": NB_SITE, "variant": "evidence",
                      "prob_overrides": {2: (0.8, 0.1, 0.1)}}])
        df["p30"] = np.nan
        result = evaluate_cascade(df)
        # tier_1 fires; the forecast P(>30) isn't in a tier-1 section, so just ensure no crash
        msg = build_cascade_message(result, df, pd.Timestamp("2026-06-14 13:00", tz="UTC"))
        assert "PLANT-SIGNAL" in msg


# ---------------------------------------------------------------------------
# Debounce state
# ---------------------------------------------------------------------------

class _FakeS3:
    def __init__(self, initial_doc: dict | None = None):
        self.store: dict[str, bytes] = {}
        if initial_doc is not None:
            self.store[ALERT_STATE_S3_PATH] = json.dumps(initial_doc).encode()

    def getFile(self, path="test", bucket=None):
        if path not in self.store:
            raise Exception(f"{path} not found")
        return self.store[path]

    def putFile(self, data, path="test", content_type=None, bucket=None, metadata=None):
        self.store[path] = data.encode() if isinstance(data, str) else data
        return path


class TestDebounce:
    def test_first_send_allowed_then_suppressed_then_allowed(self):
        s3 = _FakeS3()
        state = cstate.load_state(s3)
        now = pd.Timestamp("2026-06-14 12:00", tz="UTC")
        assert cstate.should_send(state, "tier_1", now)

        cstate.mark_sent(state, "tier_1", 0.7, now)
        # 1h later — inside the 3h quiet window
        assert not cstate.should_send(state, "tier_1", now + pd.Timedelta(hours=1))
        # 3h+ later — quiet window elapsed
        assert cstate.should_send(state, "tier_1", now + pd.Timedelta(hours=3, minutes=1))

    def test_preserves_watch_and_critical_keys(self):
        initial = {
            "watch": {"in_event": True, "event_start": "2026-06-14T10:00:00Z"},
            "critical": {"in_event": False},
        }
        s3 = _FakeS3(initial)
        state = cstate.load_state(s3)
        cstate.mark_sent(state, "tier_2", 0.6, pd.Timestamp("2026-06-14 12:00", tz="UTC"))
        cstate.save_state(s3, state)

        reloaded = json.loads(s3.store[ALERT_STATE_S3_PATH].decode())
        assert reloaded["watch"]["in_event"] is True          # untouched
        assert reloaded["critical"]["in_event"] is False      # untouched
        assert reloaded["cascade"]["tier_2"]["n_sent"] == 1    # our update landed

    def test_nan_prob_marks_without_crashing(self):
        s3 = _FakeS3()
        state = cstate.load_state(s3)
        cstate.mark_sent(state, "tier_3", float("nan"), pd.Timestamp("2026-06-14 12:00", tz="UTC"))
        assert state["cascade"]["tier_3"]["last_prob"] is None
