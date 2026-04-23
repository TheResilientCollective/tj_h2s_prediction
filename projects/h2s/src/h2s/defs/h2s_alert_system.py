"""H2S two-tier alert system — NESTOR / Berry Elementary School.

Tier 1 · WATCH    (30 ppb)  → monitoring staff
Tier 2 · CRITICAL (100 ppb) → agency decision-makers

Each tier tracks its own independent state so they can overlap without
interfering.  Message flow per tier:

  1. Onset alert      — fires on first qualifying reading after QUIET_HOURS quiet
  2. Post-event summary — fires CLOSE_WAIT_HOURS after last exceedance

Dagster integration:
  - Sensor : h2s_alert_sensor     (polls every 5 min, reads obs data + state from S3)
  - Asset  : h2s_alert_dispatcher (sends Slack messages + archives summaries to S3)
  - Job    : h2s_alert_job        (wires sensor → dispatcher)
"""

import json
import os
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import dagster as dg
import numpy as np
import pandas as pd

from h2s.constants import (
    ALERT_CLOSE_WAIT_HOURS,
    ALERT_LOCAL_TZ,
    ALERT_QUIET_HOURS,
    ALERT_SBIWTP_BASELINE_MGD,
    ALERT_SITE_NAME,
    ALERT_STATE_S3_PATH,
    ALERT_SUMMARY_ARCHIVE_PATH,
    ALERT_SUMMARY_LATEST_PATH,
    ALERT_TIERS,
    OBS_DATA_PATH,
)

_KEY = lambda name: dg.AssetKey(["h2s", name])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _wind_flag(wind_ms: float) -> str:
    if wind_ms < 2.0:
        return "calm — elevated dispersion risk"
    if wind_ms < 4.0:
        return "light"
    return "moderate"


def _deficit_label(sbiwtp_flow: float) -> str:
    deficit = ALERT_SBIWTP_BASELINE_MGD - sbiwtp_flow
    if deficit <= 0:
        return f"{sbiwtp_flow:.1f} MGD  (at or above baseline)"
    return f"{sbiwtp_flow:.1f} MGD  (deficit: {deficit:.1f} MGD below baseline)"


def _fmt_time(ts: pd.Timestamp) -> str:
    return ts.tz_convert(ALERT_LOCAL_TZ).strftime("%a %b %-d, %Y  %H:%M %Z")


def _fmt_duration(start: pd.Timestamp, end: pd.Timestamp) -> str:
    total_secs = (end - start).total_seconds()
    hours = int(total_secs // 3600)
    mins = int((total_secs % 3600) // 60)
    return f"{hours} hr" if mins == 0 else f"{hours} hr {mins} min"


# ---------------------------------------------------------------------------
# S3-backed state management
# ---------------------------------------------------------------------------

def _empty_tier_state() -> dict:
    return {
        "in_event":        False,
        "event_start":     None,
        "last_exceedance": None,
        "onset_sent":      False,
        "summary_sent":    False,
    }


def _load_state(s3) -> dict:
    """Load alert state from S3.  Returns default state if not found."""
    try:
        data = s3.getFile(ALERT_STATE_S3_PATH)
        stored = json.loads(data.decode("utf-8"))
        for tier_key in ALERT_TIERS:
            if tier_key not in stored:
                stored[tier_key] = _empty_tier_state()
        return stored
    except Exception:
        return {tier_key: _empty_tier_state() for tier_key in ALERT_TIERS}


def _save_state(s3, state: dict) -> None:
    """Persist alert state to S3."""
    s3.putFile(
        json.dumps(state, indent=2, default=str),
        path=ALERT_STATE_S3_PATH,
        content_type="application/json",
    )


# ---------------------------------------------------------------------------
# S3-backed data loading
# ---------------------------------------------------------------------------

def _load_recent(s3, hours: int = 12) -> pd.DataFrame:
    """Load last N hours of Nestor data from S3 observation parquet."""
    public_bucket = os.environ.get("PUBLIC_BUCKET", s3.S3_BUCKET)
    url = s3.publicUrl(path=OBS_DATA_PATH, bucket=public_bucket)
    df = pd.read_parquet(url)
    df["time"] = pd.to_datetime(df["time"], utc=True)

    cutoff = pd.Timestamp.now("UTC") - pd.Timedelta(hours=hours)
    nestor = df[df["site_name"].str.contains(ALERT_SITE_NAME, case=False, na=False)].copy()
    return nestor[nestor["time"] >= cutoff].sort_values("time").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Message builders
# ---------------------------------------------------------------------------

def build_onset_message(tier_key: str, row: pd.Series, prev_row: pd.Series | None) -> str:
    tier = ALERT_TIERS[tier_key]
    threshold = tier["threshold"]
    label = tier["label"]
    audience = tier["audience"]

    prior_ppb = prev_row["H2S"] if prev_row is not None else float("nan")
    prior_str = f"{prior_ppb:.0f} ppb" if not pd.isna(prior_ppb) else "n/a"

    escalation = ""
    if tier_key == "watch" and row["H2S"] >= ALERT_TIERS["critical"]["threshold"]:
        escalation = f"\n  ** Already exceeds CRITICAL threshold ({ALERT_TIERS['critical']['threshold']:.0f} ppb) **"

    lines = [
        f"H2S {label} ALERT — NESTOR / Berry Elementary School",
        f"Audience: {audience}",
        "=" * 52,
        f"{'Time detected':<22} {_fmt_time(row['time'])}",
        f"{'H2S reading':<22} {row['H2S']:.0f} ppb  (threshold: {threshold:.0f} ppb){escalation}",
        f"{'Prior 1-hr reading':<22} {prior_str}",
        "-" * 52,
        f"{'Wind speed':<22} {row['wind_speed_10m']:.1f} m/s  ({_wind_flag(row['wind_speed_10m'])})",
        f"{'Wind direction':<22} {row['wind_direction_categorical']} ({row['wind_direction_10m']:.0f}°)",
        f"{'Temperature':<22} {row['temperature_2m']:.1f} °C",
        f"{'Humidity':<22} {row['relative_humidity_2m']:.0f}%",
        "-" * 52,
        f"{'SBIWTP flow':<22} {_deficit_label(row['sbiwtp_flow_mgd'])}",
        f"{'Border flow':<22} {row['Flow (m^3/s)--Border']:.2f} m³/s",
        "-" * 52,
        "Monitoring continues. Summary report will follow at event close.",
    ]
    return "\n".join(lines)


def build_summary_message(tier_key: str, event_df: pd.DataFrame, current_ppb: float) -> str:
    tier = ALERT_TIERS[tier_key]
    threshold = tier["threshold"]
    label = tier["label"]

    start = event_df["time"].iloc[0]
    end = event_df["time"].iloc[-1]
    peak = event_df["H2S"].max()
    peak_t = event_df.loc[event_df["H2S"].idxmax(), "time"]
    mean_h2s = event_df["H2S"].mean()
    n_hrs = len(event_df)

    n_watch = int((event_df["H2S"] >= ALERT_TIERS["watch"]["threshold"]).sum())
    n_critical = int((event_df["H2S"] >= ALERT_TIERS["critical"]["threshold"]).sum())

    avg_wind = event_df["wind_speed_10m"].mean()
    wd_rad = np.deg2rad(event_df["wind_direction_10m"])
    avg_dir = float(np.rad2deg(np.arctan2(np.sin(wd_rad).mean(), np.cos(wd_rad).mean())) % 360)
    avg_rh = event_df["relative_humidity_2m"].mean()
    avg_temp = event_df["temperature_2m"].mean()
    avg_flow = event_df["sbiwtp_flow_mgd"].mean()
    avg_border = event_df["Flow (m^3/s)--Border"].mean()

    tier_breakdown = (
        f"  Watch (>={ALERT_TIERS['watch']['threshold']:.0f} ppb): {n_watch} hr  |  "
        f"Critical (>={ALERT_TIERS['critical']['threshold']:.0f} ppb): {n_critical} hr"
    )

    lines = [
        f"H2S {label} EVENT CLOSED — NESTOR / Berry Elementary School",
        "=" * 52,
        f"{'Event window':<22} {_fmt_time(start)} – {end.tz_convert(ALERT_LOCAL_TZ).strftime('%H:%M %Z')}",
        f"{'Duration':<22} {_fmt_duration(start, end)}",
        f"{'Peak H2S':<22} {peak:.0f} ppb  at {peak_t.tz_convert(ALERT_LOCAL_TZ).strftime('%H:%M')}",
        f"{'Mean H2S (event)':<22} {mean_h2s:.0f} ppb",
        f"{'Hours above threshold':<22} {n_hrs}  (threshold: {threshold:.0f} ppb)",
        f"{'Tier breakdown':<22}",
        tier_breakdown,
        "-" * 52,
        f"{'Avg wind speed':<22} {avg_wind:.1f} m/s  ({_wind_flag(avg_wind)})",
        f"{'Prevailing direction':<22} {avg_dir:.0f}°",
        f"{'Avg humidity':<22} {avg_rh:.0f}%",
        f"{'Avg temperature':<22} {avg_temp:.1f} °C",
        "-" * 52,
        f"{'SBIWTP avg flow':<22} {_deficit_label(avg_flow)}",
        f"{'Border avg flow':<22} {avg_border:.2f} m³/s",
        "-" * 52,
        f"{'Current H2S':<22} {current_ppb:.0f} ppb  (returning to baseline)",
        "Event log archived. No further alerts at this time.",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 72h baseline for S3 archive
# ---------------------------------------------------------------------------

def _build_baseline_stats(s3) -> dict | None:
    """Load 72h of observation data and compute baseline statistics."""
    try:
        baseline_df = _load_recent(s3, hours=72)
        if baseline_df.empty:
            return None
        return {
            "window_hours": 72,
            "rows": len(baseline_df),
            "wind_speed_median": round(float(baseline_df["wind_speed_10m"].median()), 2),
            "humidity_median": round(float(baseline_df["relative_humidity_2m"].median()), 2),
            "sbiwtp_flow_median": round(float(baseline_df["sbiwtp_flow_mgd"].median()), 2)
                if "sbiwtp_flow_mgd" in baseline_df.columns else None,
            "sbiwtp_deficit_median": round(float(baseline_df["sbiwtp_deficit"].median()), 2)
                if "sbiwtp_deficit" in baseline_df.columns else None,
            "h2s_median": round(float(baseline_df["H2S"].median()), 2)
                if "H2S" in baseline_df.columns else None,
        }
    except Exception:
        return None


# ---------------------------------------------------------------------------
# S3 summary archival
# ---------------------------------------------------------------------------

def _archive_summary(s3, tier_key: str, summary_msg: str, baseline_stats: dict | None) -> None:
    """Archive post-event summary to S3 (timestamped + latest)."""
    ts = datetime.now(ZoneInfo("UTC"))
    summary_data = json.dumps({
        "tier": tier_key,
        "generated_at": ts.isoformat(),
        "message": summary_msg,
        "baseline_72h": baseline_stats,
    }, indent=2)

    date_str = ts.strftime("%Y-%m-%d")
    ts_str = ts.strftime("%Y-%m-%d_%H%M")

    timestamped_path = f"{ALERT_SUMMARY_ARCHIVE_PATH}/{date_str}/{ts_str}_{tier_key}_summary.json"
    s3.putFile(summary_data, path=timestamped_path, content_type="application/json")
    s3.putFile(summary_data, path=ALERT_SUMMARY_LATEST_PATH, content_type="application/json")


# ---------------------------------------------------------------------------
# Core alert logic
# ---------------------------------------------------------------------------

def _empty_result() -> dict:
    return {
        "send_onset":   False,
        "send_summary": False,
        "onset_msg":    None,
        "summary_msg":  None,
    }


def evaluate_alerts(df: pd.DataFrame, s3) -> dict:
    """Evaluate Watch and Critical tiers against the latest data window.

    Returns dict with keys:
        "watch"    : { send_onset, send_summary, onset_msg, summary_msg }
        "critical" : { send_onset, send_summary, onset_msg, summary_msg }
        "state"    : updated state dict — caller must persist via _save_state()
    """
    now = pd.Timestamp.now("UTC")
    state = _load_state(s3)
    out = {}

    if df.empty:
        return {t: _empty_result() for t in ALERT_TIERS} | {"state": state}

    latest = df.iloc[-1]
    latest_ppb = latest["H2S"]

    for tier_key, tier_cfg in ALERT_TIERS.items():
        threshold = tier_cfg["threshold"]
        ts = state[tier_key]
        result = _empty_result()
        above = latest_ppb >= threshold

        # --- Onset ---
        if above and not ts["in_event"]:
            quiet_cutoff = latest["time"] - pd.Timedelta(hours=ALERT_QUIET_HOURS)
            recent_before = df[df["time"] < latest["time"]]
            was_quiet = recent_before.empty or (
                recent_before[recent_before["time"] >= quiet_cutoff]["H2S"] < threshold
            ).all()

            if was_quiet:
                ts.update({
                    "in_event":        True,
                    "event_start":     str(latest["time"]),
                    "last_exceedance": str(latest["time"]),
                    "onset_sent":      True,
                    "summary_sent":    False,
                })
                prev_row = df.iloc[-2] if len(df) >= 2 else None
                result["send_onset"] = True
                result["onset_msg"] = build_onset_message(tier_key, latest, prev_row)

        elif above and ts["in_event"]:
            ts["last_exceedance"] = str(latest["time"])

        # --- Summary ---
        if ts["in_event"] and not ts["summary_sent"]:
            last_exc = pd.Timestamp(ts["last_exceedance"])
            if last_exc.tzinfo is None:
                last_exc = last_exc.tz_localize("UTC")
            elapsed = (now - last_exc).total_seconds() / 3600

            if elapsed >= ALERT_CLOSE_WAIT_HOURS and not above:
                event_start = pd.Timestamp(ts["event_start"])
                if event_start.tzinfo is None:
                    event_start = event_start.tz_localize("UTC")

                # Always use Watch threshold for event window so Critical
                # summary includes full context of the episode
                event_df = df[
                    (df["time"] >= event_start)
                    & (df["H2S"] >= ALERT_TIERS["watch"]["threshold"])
                ].copy()

                if not event_df.empty:
                    result["send_summary"] = True
                    result["summary_msg"] = build_summary_message(
                        tier_key, event_df, latest_ppb
                    )

                ts["in_event"] = False
                ts["summary_sent"] = True

        state[tier_key] = ts
        out[tier_key] = result

    out["state"] = state
    return out


# ---------------------------------------------------------------------------
# Dagster asset + job
# ---------------------------------------------------------------------------

class AlertConfig(dg.Config):
    watch_send_onset:      bool = False
    watch_send_summary:    bool = False
    watch_onset_msg:       str  = ""
    watch_summary_msg:     str  = ""
    critical_send_onset:   bool = False
    critical_send_summary: bool = False
    critical_onset_msg:    str  = ""
    critical_summary_msg:  str  = ""


@dg.asset(
    key_prefix="h2s",
    group_name="h2s_alerts",
    required_resource_keys={"slack", "s3"},
    kinds={"slack", "s3"},
    description="Dispatch H2S watch/critical alerts to Slack and archive summaries to S3",
)
def h2s_alert_dispatcher(
    context: dg.AssetExecutionContext,
    config: AlertConfig,
) -> None:
    """Send Watch and/or Critical onset alerts and post-event summaries."""
    slack = context.resources.slack
    s3 = context.resources.s3
    client = slack.get_client()

    sent_any = False

    for tier_key in ("watch", "critical"):
        tier_label = ALERT_TIERS[tier_key]["label"]
        send_onset = getattr(config, f"{tier_key}_send_onset")
        onset_msg = getattr(config, f"{tier_key}_onset_msg")
        send_summary = getattr(config, f"{tier_key}_send_summary")
        summary_msg = getattr(config, f"{tier_key}_summary_msg")

        if send_onset and onset_msg:
            context.log.info(f"Sending {tier_label} onset alert.")
            client.chat_postMessage(
                channel=slack.channel,
                text=f"```{onset_msg}```",
            )
            sent_any = True

        if send_summary and summary_msg:
            context.log.info(f"Sending {tier_label} event summary.")
            client.chat_postMessage(
                channel=slack.channel,
                text=f"```{summary_msg}```",
            )
            # Archive summary to S3 with 72h baseline
            baseline_stats = _build_baseline_stats(s3)
            _archive_summary(s3, tier_key, summary_msg, baseline_stats)
            context.log.info(f"Archived {tier_label} summary to S3.")
            sent_any = True

    if not sent_any:
        context.log.info("No alert conditions met — nothing sent.")


h2s_alert_job = dg.define_asset_job(
    name="h2s_alert_job",
    selection=dg.AssetSelection.assets(h2s_alert_dispatcher),
    description="Dispatch H2S alert messages triggered by the alert sensor",
)


# ---------------------------------------------------------------------------
# Dagster sensor
# ---------------------------------------------------------------------------

@dg.sensor(
    job=h2s_alert_job,
    minimum_interval_seconds=300,
    required_resource_keys={"s3"},
    description="Poll H2S observation data every 5 min; trigger alerts at watch (30 ppb) and critical (100 ppb) thresholds",
    default_status=dg.DefaultSensorStatus.RUNNING,
)
def h2s_alert_sensor(context: dg.SensorEvaluationContext):
    """Poll H2S data every 5 min; trigger dispatcher when either tier fires."""
    s3 = context.resources.s3

    try:
        df = _load_recent(s3, hours=12)
    except Exception as e:
        context.log.warning(f"Could not load H2S data: {e}")
        return

    results = evaluate_alerts(df, s3)
    _save_state(s3, results["state"])

    any_action = any(
        results[t]["send_onset"] or results[t]["send_summary"]
        for t in ALERT_TIERS
    )

    if any_action:
        yield dg.RunRequest(
            run_key=f"h2s_alert_{pd.Timestamp.utcnow().isoformat()}",
            run_config={
                "ops": {
                    "h2s__h2s_alert_dispatcher": {
                        "config": {
                            "watch_send_onset":      results["watch"]["send_onset"],
                            "watch_send_summary":    results["watch"]["send_summary"],
                            "watch_onset_msg":       results["watch"]["onset_msg"] or "",
                            "watch_summary_msg":     results["watch"]["summary_msg"] or "",
                            "critical_send_onset":   results["critical"]["send_onset"],
                            "critical_send_summary": results["critical"]["send_summary"],
                            "critical_onset_msg":    results["critical"]["onset_msg"] or "",
                            "critical_summary_msg":  results["critical"]["summary_msg"] or "",
                        }
                    }
                }
            },
        )
    else:
        context.log.debug(
            f"No alerts — latest H2S: {df.iloc[-1]['H2S']:.0f} ppb" if not df.empty else "No data"
        )
