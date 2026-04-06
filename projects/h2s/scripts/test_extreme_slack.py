"""Send test H2S alert messages (onset + summary) for both tiers to Slack.

Usage:
    cd projects/h2s
    uv run python scripts/test_extreme_slack.py
"""

import os
import sys

# Allow importing from the project
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from slack_sdk import WebClient

from h2s.defs.h2s_alert_system import build_onset_message, build_summary_message

import pandas as pd
import numpy as np

token = os.environ.get("SLACK_TOKEN")
channel = os.environ.get("SLACK_CHANNEL", "#test")

if not token:
    raise SystemExit("Set SLACK_TOKEN environment variable first")


# Build realistic sample data
def _make_row(h2s, wind, rh, temp, sbiwtp, flow, time_str, wd_cat="SW", wd_deg=225):
    return pd.Series({
        "time": pd.Timestamp(time_str, tz="UTC"),
        "H2S": h2s,
        "wind_speed_10m": wind,
        "wind_direction_10m": wd_deg,
        "wind_direction_categorical": wd_cat,
        "relative_humidity_2m": rh,
        "temperature_2m": temp,
        "sbiwtp_flow_mgd": sbiwtp,
        "Flow (m^3/s)--Border": flow,
    })


# Onset row: H2S just crossed 147 ppb with calm wind
onset_row = _make_row(
    h2s=147, wind=2.1, rh=88, temp=16.5, sbiwtp=20.6, flow=1.8,
    time_str="2026-04-06T04:00:00"
)
prev_row = _make_row(
    h2s=28, wind=4.2, rh=78, temp=17.1, sbiwtp=22.0, flow=1.9,
    time_str="2026-04-06T03:00:00"
)

# Event DataFrame for summary (8 hours above Watch threshold)
event_rows = [
    _make_row(147, 2.1, 88, 16.5, 20.6, 1.8, "2026-04-06T04:00:00"),
    _make_row(132, 1.8, 90, 16.2, 20.1, 1.7, "2026-04-06T05:00:00"),
    _make_row(118, 2.4, 89, 16.0, 19.8, 1.6, "2026-04-06T06:00:00"),
    _make_row(95,  2.0, 91, 15.8, 20.3, 1.7, "2026-04-06T07:00:00"),
    _make_row(72,  2.6, 87, 16.1, 21.0, 1.8, "2026-04-06T08:00:00"),
    _make_row(55,  3.1, 85, 16.8, 21.5, 1.9, "2026-04-06T09:00:00"),
    _make_row(41,  3.8, 82, 17.2, 22.0, 2.0, "2026-04-06T10:00:00"),
    _make_row(33,  4.2, 79, 17.8, 22.5, 2.1, "2026-04-06T11:00:00"),
]
event_df = pd.DataFrame(event_rows)

client = WebClient(token=token)

print("Sending test messages to Slack...\n")

# --- Watch onset ---
msg = build_onset_message("watch", onset_row, prev_row)
print("=== WATCH ONSET ===")
print(msg)
client.chat_postMessage(channel=channel, text=f"```{msg}```")

# --- Critical onset ---
msg = build_onset_message("critical", onset_row, prev_row)
print("\n=== CRITICAL ONSET ===")
print(msg)
client.chat_postMessage(channel=channel, text=f"```{msg}```")

# --- Watch summary ---
msg = build_summary_message("watch", event_df, current_ppb=8.0)
print("\n=== WATCH SUMMARY ===")
print(msg)
client.chat_postMessage(channel=channel, text=f"```{msg}```")

# --- Critical summary ---
msg = build_summary_message("critical", event_df, current_ppb=8.0)
print("\n=== CRITICAL SUMMARY ===")
print(msg)
client.chat_postMessage(channel=channel, text=f"```{msg}```")

print(f"\nAll 4 test messages sent to {channel}")
