"""Observed-exceedance "Alert Performance" machine (Phase 4).

A yellow-tier (>10 ppb) observation state machine at NESTOR-BES, separate from
the watch (30) / critical (100) machine in ``h2s_alert_system``. It opens on the
first >10 ppb reading after a quiet period and closes after readings stay below
threshold for ``ALERT_YELLOW_CLOSE_WAIT_HOURS``. The close-out report scores the
forecast cascade against what was actually measured — the bridge to the Phase-5
validation store.
"""
