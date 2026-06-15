"""Preview every PRESENT H2S alert in Slack (or print with --dry-run).

Posts one representative message of each current alert path, reusing the exact
production message builders (no new alert logic) on synthetic data:

  - cascade   : Tier 1–3 forecast cascade (cascade_alerts) — the
                product-probability pre-alerts on the ops channel
  - yellow    : observed >10 ppb onset + Alert-Performance close-out
                (alert_performance)
  - watch     : legacy WATCH (30) / CRITICAL (100) observation alerts
                (h2s_alert_system) — for the full ladder

Usage:
    cd projects/h2s
    # see them without posting (no token needed):
    uv run python scripts/send_sample_alerts.py --dry-run
    # post to Slack:
    set -a; source .env; set +a       # SLACK_TOKEN, SLACK_CHANNEL[, _OPS]
    uv run python scripts/send_sample_alerts.py
    # just one kind:
    uv run python scripts/send_sample_alerts.py --only cascade --dry-run
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd  # noqa: E402

from h2s.defs.cascade_alerts.cascade import NB_SITE, evaluate_cascade  # noqa: E402
from h2s.defs.cascade_alerts.messages import (  # noqa: E402
    build_cascade_blocks,
    build_cascade_message,
)
from h2s.defs.alert_performance.logic import (  # noqa: E402
    forecast_vs_actual,
    hours_above,
    performance_summary,
)
from h2s.defs.alert_performance.messages import (  # noqa: E402
    build_closeout_message,
    build_onset_message as yellow_onset_message,
)
from h2s.defs.h2s_alert_system import (  # noqa: E402
    build_onset_message as obs_onset_message,
    build_summary_message as obs_summary_message,
)

# Fixed "now" so the sample reads the same every run (UTC).
_NOW = pd.Timestamp("2026-04-06 04:00", tz="UTC")


def _product_for_lead(lead: int) -> str:
    return "nowcast" if lead <= 3 else "nearcast" if lead <= 6 else "forecast"


def _obs_row(h2s, wind, rh, temp, sbiwtp, flow, t, wd_cat="SW", wd_deg=225):
    return pd.Series({
        "time": t, "H2S": h2s,
        "wind_speed_10m": wind, "wind_direction_10m": wd_deg,
        "wind_direction_categorical": wd_cat, "relative_humidity_2m": rh,
        "temperature_2m": temp, "sbiwtp_flow_mgd": sbiwtp,
        "Flow (m^3/s)--Border": flow,
    })


# ── cascade sample ─────────────────────────────────────────────────────────────

def _cascade_products() -> pd.DataFrame:
    """3 stations × 2 variants × 24 leads. NESTOR-BES (evidence) is rigged to fire
    all three tiers: P(>5) in nowcast, P(>10) in nearcast, P(>30) in forecast."""
    times = pd.date_range(start=_NOW + pd.Timedelta(hours=1), periods=24, freq="h")
    # (p5, p10, p30, peak_h2s) per product window
    profile = {
        (NB_SITE, "evidence"): {"nowcast": (0.72, 0.45, 0.15, 14),
                                "nearcast": (0.60, 0.74, 0.30, 19),
                                "forecast": (0.50, 0.55, 0.62, 34)},
        (NB_SITE, "lean"): {"nowcast": (0.68, 0.42, 0.13, 13),
                            "nearcast": (0.57, 0.70, 0.28, 17),
                            "forecast": (0.47, 0.52, 0.58, 31)},
        ("SAN YSIDRO", "evidence"): {"nowcast": (0.30, 0.10, 0.05, 6),
                                     "nearcast": (0.34, 0.15, 0.08, 7),
                                     "forecast": (0.40, 0.20, 0.33, 12)},
        ("SAN YSIDRO", "lean"): {"nowcast": (0.28, 0.09, 0.04, 5),
                                 "nearcast": (0.31, 0.13, 0.07, 6),
                                 "forecast": (0.37, 0.18, 0.30, 11)},
        ("IB CIVIC CTR", "evidence"): {"nowcast": (0.22, 0.07, 0.03, 5),
                                       "nearcast": (0.26, 0.10, 0.05, 6),
                                       "forecast": (0.30, 0.14, 0.21, 9)},
        ("IB CIVIC CTR", "lean"): {"nowcast": (0.20, 0.06, 0.03, 4),
                                   "nearcast": (0.24, 0.09, 0.04, 5),
                                   "forecast": (0.28, 0.12, 0.19, 8)},
    }
    rows = []
    for (station, variant), prof in profile.items():
        for i, t in enumerate(times):
            lead = i + 1
            p5, p10, p30, peak = prof[_product_for_lead(lead)]
            rows.append({
                "run_ts": _NOW.strftime("%Y-%m-%dT%H%MZ"),
                "product": _product_for_lead(lead), "station": station,
                "lead_hour": lead, "time": t, "variant": variant,
                "model_version": "sample-v1",
                "h2s_pred": float(peak), "p5": p5, "p10": p10, "p30": p30,
            })
    return pd.DataFrame(rows)


def cascade_message() -> tuple[str, list]:
    df = _cascade_products()
    result = evaluate_cascade(df, station=NB_SITE, variant="evidence")
    text = build_cascade_message(result, df, _NOW)
    blocks = build_cascade_blocks(result, df, _NOW)
    return text, blocks


# ── yellow / Alert-Performance sample ───────────────────────────────────────────

def _yellow_inputs():
    """A 5 h >10 ppb event embedded in a 12 h observation window, plus the product
    forecasts that targeted those hours."""
    times = pd.date_range(end=_NOW, periods=12, freq="h")
    h2s = [3, 4, 5, 4, 6, 14, 22, 18, 12, 11, 6, 4]
    obs = pd.DataFrame({"time": times, "H2S": h2s, "wind_speed_10m": [3.0] * 12})
    event_df = obs[obs["H2S"] >= 10][["time", "H2S"]].reset_index(drop=True)
    event_start, event_end = event_df["time"].min(), event_df["time"].max()

    # Forecasts (issued ~6 h ahead) for the event hours: predicted ppb + P(>10).
    preds = list(zip(event_df["time"], [15, 20, 16, 13, 12], [0.80, 0.85, 0.70, 0.55, 0.50]))
    prods = pd.DataFrame([
        {"run_ts": "sample", "product": "forecast", "station": NB_SITE,
         "variant": "evidence", "lead_hour": i + 1, "time": t,
         "model_version": "sample-v1", "h2s_pred": hp, "p5": 0.95, "p10": p10, "p30": 0.25}
        for i, (t, hp, p10) in enumerate(preds)
    ])
    return obs, event_df, prods, event_start, event_end


def yellow_messages() -> list[str]:
    obs, event_df, prods, ev_start, ev_end = _yellow_inputs()
    onset_row = _obs_row(14, 2.1, 86, 16.5, 20.6, 1.8, ev_start)
    prev_row = _obs_row(6, 3.6, 80, 17.0, 22.0, 1.9, ev_start - pd.Timedelta(hours=1))
    onset = yellow_onset_message(onset_row, prev_row)

    hours = hours_above(obs, _NOW)
    fva = forecast_vs_actual(prods, obs, ev_start, ev_end)
    summary = performance_summary(fva)
    closeout = build_closeout_message(event_df, hours, fva, summary, current_ppb=4.0)
    return [onset, closeout]


# ── legacy watch / critical sample ──────────────────────────────────────────────

def watch_messages() -> list[str]:
    onset_row = _obs_row(147, 2.1, 88, 16.5, 20.6, 1.8, _NOW)
    prev_row = _obs_row(28, 4.2, 78, 17.1, 22.0, 1.9, _NOW - pd.Timedelta(hours=1))
    event_df = pd.DataFrame([
        _obs_row(h, w, rh, tp, sb, fl, _NOW + pd.Timedelta(hours=i))
        for i, (h, w, rh, tp, sb, fl) in enumerate([
            (147, 2.1, 88, 16.5, 20.6, 1.8), (132, 1.8, 90, 16.2, 20.1, 1.7),
            (118, 2.4, 89, 16.0, 19.8, 1.6), (95, 2.0, 91, 15.8, 20.3, 1.7),
            (72, 2.6, 87, 16.1, 21.0, 1.8), (41, 3.8, 82, 17.2, 22.0, 2.0),
        ])
    ])
    return [
        obs_onset_message("watch", onset_row, prev_row),
        obs_onset_message("critical", onset_row, prev_row),
        obs_summary_message("watch", event_df, current_ppb=8.0),
    ]


# ── dispatch ────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="Preview present H2S alerts in Slack")
    p.add_argument("--dry-run", action="store_true", help="Print messages; do not post")
    p.add_argument("--only", choices=["cascade", "yellow", "watch"], help="Post only one kind")
    p.add_argument("--channel", help="Override target channel")
    args = p.parse_args()

    default_ch = os.environ.get("SLACK_CHANNEL", "#test")
    ops_ch = args.channel or os.environ.get("SLACK_CHANNEL_OPS", default_ch)
    obs_ch = args.channel or default_ch

    cascade_text, cascade_blocks = cascade_message()
    # (channel, header, text, blocks)
    items: list[tuple[str, str, str, list | None]] = []
    if args.only in (None, "cascade"):
        items.append((ops_ch, "FORECAST CASCADE (Tiers 1–3)", cascade_text, cascade_blocks))
    if args.only in (None, "yellow"):
        y = yellow_messages()
        items.append((obs_ch, "YELLOW ONSET (>10 ppb)", f"```{y[0]}```", None))
        items.append((obs_ch, "ALERT-PERFORMANCE CLOSE-OUT", f"```{y[1]}```", None))
    if args.only in (None, "watch"):
        for label, msg in zip(["WATCH ONSET", "CRITICAL ONSET", "WATCH SUMMARY"], watch_messages()):
            items.append((obs_ch, label, f"```{msg}```", None))

    if args.dry_run:
        for ch, header, text, _blocks in items:
            print(f"\n{'=' * 70}\n[{header}]  → would post to {ch}\n{'=' * 70}")
            print(cascade_text if header.startswith("FORECAST") else text.strip("`"))
        print(f"\n[dry-run] {len(items)} message(s) — nothing sent.")
        return

    token = os.environ.get("SLACK_TOKEN")
    if not token:
        raise SystemExit("Set SLACK_TOKEN (or use --dry-run).")
    from slack_sdk import WebClient
    client = WebClient(token=token)
    for ch, header, text, blocks in items:
        kwargs = {"channel": ch, "text": text}
        if blocks:
            kwargs["blocks"] = blocks
        client.chat_postMessage(**kwargs)
        print(f"✓ sent [{header}] → {ch}")
    print(f"\nSent {len(items)} sample message(s).")


if __name__ == "__main__":
    main()
