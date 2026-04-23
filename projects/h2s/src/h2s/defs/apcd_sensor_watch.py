"""APCD public-bucket sensor watch — multi-station H2S alerts with predictions.

Polls `hs2_lastday.csv` produced by the `hs2_latest` asset in
resilient_workflows_public (stored in the `resilentpublic` public bucket),
detects per-station threshold exceedances, and dispatches:

  1. A rich Slack message containing:
       - Observed H2S value and APCD level
       - Model predictions (regression + exceedance probabilities) from
         `daily_summary.json`
       - Wind/temperature/humidity context from the observation parquet
       - Cross-station context
  2. A structured JSON event report archived to S3 at
     `tijuana/forecast/sensor_events/archive/<date>/...` along with a rolling
     index for dashboard consumption.

Two-tier thresholds match the existing `h2s_alert_system` (WATCH 30 ppb /
CRITICAL 100 ppb) but state is tracked independently per station using the
state file at `APCD_SENSOR_STATE_PATH`.

Dagster integration:
  - Sensor : apcd_sensor_watch_sensor       (polls every 5 min)
  - Asset  : apcd_sensor_alert_dispatcher   (sends Slack + archives events)
  - Job    : apcd_sensor_watch_job          (wires sensor → dispatcher)
"""

import json
import os
from datetime import datetime
from typing import Optional
from zoneinfo import ZoneInfo

import dagster as dg
import pandas as pd

from h2s.constants import (
    ALERT_CLOSE_WAIT_HOURS,
    ALERT_LOCAL_TZ,
    ALERT_QUIET_HOURS,
    ALERT_SBIWTP_BASELINE_MGD,
    ALERT_TIERS,
    APCD_H2S_PARAMETER,
    APCD_HS2_LASTDAY_PATH,
    APCD_PUBLIC_BUCKET,
    APCD_SENSOR_STATE_PATH,
    APCD_SITE_TO_STATION,
    OBS_DATA_PATH,
    SENSOR_EVENT_ARCHIVE_PATH,
    SENSOR_EVENT_INDEX_MAX,
    SENSOR_EVENT_INDEX_PATH,
    SENSOR_EVENT_LATEST_PATH,
    STATIONS,
)
from h2s.defs.h2s_alert_system import (
    _deficit_label,
    _fmt_time,
    _wind_flag,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _station_key_for_site(site_name: str) -> Optional[str]:
    """Return the STATIONS-key for an APCD site name, or None if not modeled."""
    mapped = APCD_SITE_TO_STATION.get(site_name)
    if mapped is None:
        return None
    return STATIONS[mapped]["key"]


def _station_short_for_site(site_name: str) -> Optional[str]:
    """Return the STATIONS short code (SY/NB/IB) for an APCD site name."""
    mapped = APCD_SITE_TO_STATION.get(site_name)
    if mapped is None:
        return None
    return STATIONS[mapped]["short"]


def _safe_float(value) -> Optional[float]:
    """Coerce to float, returning None for NaN/missing/unparseable values."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(f):
        return None
    return f


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_hs2_lastday(s3) -> pd.DataFrame:
    """Read hs2_lastday.csv from the public bucket via publicUrl.

    The `resilentpublic` bucket serves objects without auth, so we build the
    direct object URL with `s3.publicUrl()` and pass it to pandas. Returns a
    DataFrame sorted by (site, time) with columns normalized to:
      site_name, time (UTC), h2s_ppb, level, parameter
    """
    url = s3.publicUrl(path=APCD_HS2_LASTDAY_PATH, bucket=APCD_PUBLIC_BUCKET)
    df = pd.read_csv(url)

    # Filter to H2S rows (defensive — the file should already be filtered)
    if "Parameter" in df.columns:
        df = df[df["Parameter"] == APCD_H2S_PARAMETER]

    rename_map = {
        "Site Name": "site_name",
        "Date with time": "time",
        "Result": "h2s_ppb",
        "Parameter": "parameter",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # Parse datetimes to UTC. APCD timestamps already include timezone offset.
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    df["h2s_ppb"] = pd.to_numeric(df["h2s_ppb"], errors="coerce")
    df = df.dropna(subset=["time", "h2s_ppb", "site_name"])

    if "level" not in df.columns:
        df["level"] = None

    return df.sort_values(["site_name", "time"]).reset_index(drop=True)


def _load_predictions(s3) -> dict:
    """Load the daily_summary.json and return per-station prediction dicts.

    Returned structure is keyed by APCD site name and contains a normalized
    subset of the forecast_48h block. Missing stations return an empty dict.
    """
    try:
        raw = s3.getFile(path="latest/tijuana/forecast_data/daily_summary.json")
        summary = json.loads(raw.decode("utf-8"))
    except Exception:
        return {}

    fc = summary.get("forecast_24h", {}) or summary.get("forecast_48h", {}) or {}
    stations_obs = summary.get("stations", {}) or {}

    preds_by_site: dict = {}
    for site_name, mapped_name in APCD_SITE_TO_STATION.items():
        short = STATIONS[mapped_name]["short"]
        fc_block = fc.get(short, {})
        obs_block = stations_obs.get(short, {})
        if not fc_block and not obs_block:
            continue
        preds_by_site[site_name] = {
            "short": short,
            "station_key": STATIONS[mapped_name]["key"],
            "forecast_24h": {
                "max_h2s_ppb": fc_block.get("max_h2s"),
                "max_prob_5": fc_block.get("max_prob_5"),
                "max_prob_10": fc_block.get("max_prob_10"),
                "hours_orange": fc_block.get("hours_orange", 0),
                "hours_yellow_high": fc_block.get("hours_yellow_high", 0),
                "hours_yellow_low": fc_block.get("hours_yellow_low", 0),
                "hours_green": fc_block.get("hours_green", 0),
            },
            "recent_24h": {
                "last_h2s": obs_block.get("last_h2s"),
                "last_time": obs_block.get("last_time"),
                "mean_24h": obs_block.get("mean_24h"),
                "max_24h": obs_block.get("max_24h"),
                "pct_exceed_5": obs_block.get("pct_exceed_5"),
            },
        }
    return preds_by_site


def _load_weather_context(s3) -> dict:
    """Read last row of the obs parquet for meteorological context.

    Returned dict has the same column names as the parquet so downstream
    builders can pick fields opportunistically. Returns {} on failure.
    """
    try:
        url = s3.publicUrl(path=OBS_DATA_PATH)
        obs = pd.read_parquet(url)
    except Exception:
        return {}
    if obs is None or len(obs) == 0:
        return {}
    obs["time"] = pd.to_datetime(obs.get("time"), utc=True, errors="coerce")
    obs = obs.dropna(subset=["time"]).sort_values("time")
    if len(obs) == 0:
        return {}
    row = obs.iloc[-1].to_dict()
    # Cast numpy/pandas scalars to plain python for JSON friendliness
    clean = {}
    for k, v in row.items():
        if isinstance(v, (pd.Timestamp, datetime)):
            clean[k] = pd.Timestamp(v).tz_convert("UTC").isoformat() if getattr(v, "tzinfo", None) else str(v)
        elif pd.isna(v):
            clean[k] = None
        else:
            try:
                clean[k] = float(v)
            except (TypeError, ValueError):
                clean[k] = v
    return clean


# ---------------------------------------------------------------------------
# State management
# ---------------------------------------------------------------------------

def _empty_tier_state() -> dict:
    return {
        "in_event":        False,
        "event_start":     None,
        "last_exceedance": None,
        "onset_sent":      False,
        "summary_sent":    False,
    }


def _empty_station_state() -> dict:
    return {tier: _empty_tier_state() for tier in ALERT_TIERS}


def _load_sensor_state(s3) -> dict:
    """Load per-station per-tier state from S3. Returns fresh state on miss."""
    try:
        data = s3.getFile(path=APCD_SENSOR_STATE_PATH)
        stored = json.loads(data.decode("utf-8"))
    except Exception:
        stored = {}

    # Ensure all mapped stations + all tiers are present
    for site_name in APCD_SITE_TO_STATION:
        if site_name not in stored:
            stored[site_name] = _empty_station_state()
        else:
            for tier in ALERT_TIERS:
                if tier not in stored[site_name]:
                    stored[site_name][tier] = _empty_tier_state()
    return stored


def _save_sensor_state(s3, state: dict) -> None:
    s3.putFile(
        json.dumps(state, indent=2, default=str),
        path=APCD_SENSOR_STATE_PATH,
        content_type="application/json",
    )


# ---------------------------------------------------------------------------
# Message builder
# ---------------------------------------------------------------------------

def build_sensor_alert_message(
    site_name: str,
    tier_key: str,
    obs_row: pd.Series,
    all_latest: pd.DataFrame,
    pred: Optional[dict],
    weather: dict,
) -> str:
    """Render the rich Slack alert body (rendered inside a ``` block)."""
    tier = ALERT_TIERS[tier_key]
    threshold = tier["threshold"]
    label = tier["label"]
    audience = tier["audience"]

    h2s = float(obs_row["h2s_ppb"])
    level = obs_row.get("level") or "—"

    # Escalation hint if WATCH onset already passes CRITICAL
    escalation = ""
    if tier_key == "watch" and h2s >= ALERT_TIERS["critical"]["threshold"]:
        escalation = f"\n  ** Already exceeds CRITICAL threshold ({ALERT_TIERS['critical']['threshold']:.0f} ppb) **"

    # Prediction section
    pred_lines = []
    if pred:
        fc = pred.get("forecast_24h", {}) or pred.get("forecast_48h", {}) or {}
        max_h2s_ppb = fc.get("max_h2s_ppb") or fc.get("max_h2s")
        max_prob_5 = fc.get("max_prob_5")
        max_prob_10 = fc.get("max_prob_10")
        hours_orange = fc.get("hours_orange", 0)
        hours_yellow = fc.get("hours_yellow_high", 0) + fc.get("hours_yellow_low", 0)
        pred_lines = [
            "-" * 52,
            "Model forecast (next 24h)",
            f"  Max H2S             {max_h2s_ppb if max_h2s_ppb is not None else 'n/a'} ppb",
            f"  Max P(>5 ppb)       {max_prob_5 if max_prob_5 is not None else 'n/a'}",
            f"  Max P(>10 ppb)      {max_prob_10 if max_prob_10 is not None else 'n/a'}",
            f"  Hours orange        {hours_orange}",
            f"  Hours yellow        {hours_yellow}",
        ]
    else:
        pred_lines = [
            "-" * 52,
            "Model forecast         n/a (no prediction available for this station)",
        ]

    # Meteorological section
    def _fmt_num(v, fmt="{:.1f}"):
        f = _safe_float(v)
        return fmt.format(f) if f is not None else "n/a"

    wind_speed = _safe_float(weather.get("wind_speed_10m"))
    wind_flag = _wind_flag(wind_speed) if wind_speed is not None else "n/a"
    wind_dir = _safe_float(weather.get("wind_direction_10m"))
    wind_dir_cat = weather.get("wind_direction_categorical") or "—"
    temp = _safe_float(weather.get("temperature_2m"))
    rh = _safe_float(weather.get("relative_humidity_2m"))

    sbiwtp_flow = _safe_float(weather.get("sbiwtp_flow_mgd"))
    if sbiwtp_flow is not None:
        sbiwtp_line = _deficit_label(sbiwtp_flow)
    else:
        sbiwtp_line = "n/a"
    border_flow = _safe_float(weather.get("Flow (m^3/s)--Border"))

    # Cross-station context (latest reading per site)
    other_lines = []
    if all_latest is not None and len(all_latest):
        for _, row in all_latest.iterrows():
            if row["site_name"] == site_name:
                continue
            other_lines.append(
                f"  {row['site_name']:<14} {float(row['h2s_ppb']):>6.0f} ppb ({row.get('level') or '—'})"
            )

    lines = [
        f"H2S {label} ALERT — {site_name}",
        f"Audience: {audience}",
        "=" * 52,
        f"{'Time detected':<22} {_fmt_time(obs_row['time'])}",
        f"{'Observed H2S':<22} {h2s:.0f} ppb  (threshold: {threshold:.0f} ppb){escalation}",
        f"{'APCD level':<22} {level}",
        *pred_lines,
        "-" * 52,
        f"{'Wind speed':<22} {_fmt_num(wind_speed)} m/s  ({wind_flag})",
        f"{'Wind direction':<22} {wind_dir_cat} ({_fmt_num(wind_dir, '{:.0f}')}°)",
        f"{'Temperature':<22} {_fmt_num(temp)} °C",
        f"{'Humidity':<22} {_fmt_num(rh, '{:.0f}')}%",
        "-" * 52,
        f"{'SBIWTP flow':<22} {sbiwtp_line}",
        f"{'Border flow':<22} {_fmt_num(border_flow, '{:.2f}')} m³/s",
    ]
    if other_lines:
        lines.append("-" * 52)
        lines.append("Other stations (latest):")
        lines.extend(other_lines)
    lines.append("-" * 52)
    lines.append("Event report archived to tijuana/forecast/sensor_events/")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Event report builder + archival
# ---------------------------------------------------------------------------

def build_event_report(
    site_name: str,
    tier_key: str,
    trigger: str,  # "onset" | "summary"
    obs_row: pd.Series,
    all_latest: pd.DataFrame,
    pred: Optional[dict],
    weather: dict,
) -> dict:
    """Build a structured event report dict ready for JSON archival."""
    now_utc = datetime.now(ZoneInfo("UTC"))
    obs_time = pd.Timestamp(obs_row["time"])
    if obs_time.tzinfo is None:
        obs_time = obs_time.tz_localize("UTC")

    mapped_name = APCD_SITE_TO_STATION.get(site_name)
    station_info = STATIONS.get(mapped_name, {}) if mapped_name else {}

    event_id = (
        f"{obs_time.strftime('%Y%m%dT%H%MZ')}_"
        f"{(station_info.get('key') or site_name.replace(' ', '_'))}_"
        f"{tier_key}_{trigger}"
    )

    # Other-station snapshot
    other_stations = []
    if all_latest is not None and len(all_latest):
        for _, row in all_latest.iterrows():
            if row["site_name"] == site_name:
                continue
            other_stations.append({
                "name": row["site_name"],
                "h2s_ppb": float(row["h2s_ppb"]),
                "level": row.get("level"),
                "time_utc": pd.Timestamp(row["time"]).isoformat(),
            })

    sbiwtp_flow = _safe_float(weather.get("sbiwtp_flow_mgd"))
    sbiwtp_deficit = None
    if sbiwtp_flow is not None:
        sbiwtp_deficit = round(ALERT_SBIWTP_BASELINE_MGD - sbiwtp_flow, 2)

    meteorology = {
        "wind_speed_ms": _safe_float(weather.get("wind_speed_10m")),
        "wind_direction_deg": _safe_float(weather.get("wind_direction_10m")),
        "wind_direction_categorical": weather.get("wind_direction_categorical"),
        "temperature_c": _safe_float(weather.get("temperature_2m")),
        "relative_humidity_pct": _safe_float(weather.get("relative_humidity_2m")),
        "surface_pressure_hpa": _safe_float(weather.get("surface_pressure")),
        "cloud_cover_pct": _safe_float(weather.get("cloud_cover")),
        "precipitation_mm": _safe_float(weather.get("precipitation")),
        "stable_atmosphere": _safe_float(weather.get("stable_atm")),
    }

    flow = {
        "border_flow_cms": _safe_float(weather.get("Flow (m^3/s)--Border")),
        "sbiwtp_flow_mgd": sbiwtp_flow,
        "sbiwtp_baseline_mgd": ALERT_SBIWTP_BASELINE_MGD,
        "sbiwtp_deficit_mgd": sbiwtp_deficit,
    }

    report = {
        "schema_version": "1.0",
        "event_id": event_id,
        "generated_at": now_utc.isoformat(),
        "tier": tier_key,
        "trigger": trigger,
        "station": {
            "name": site_name,
            "key": station_info.get("key"),
            "short": station_info.get("short"),
            "lat": station_info.get("lat"),
            "lon": station_info.get("lon"),
        },
        "observation": {
            "time_utc": obs_time.isoformat(),
            "h2s_ppb": float(obs_row["h2s_ppb"]),
            "level": obs_row.get("level"),
            "source_file": f"s3://{APCD_PUBLIC_BUCKET}/{APCD_HS2_LASTDAY_PATH}",
        },
        "prediction": pred,
        "meteorology": meteorology,
        "flow": flow,
        "other_stations": other_stations,
        "thresholds": {
            "watch_ppb": ALERT_TIERS["watch"]["threshold"],
            "critical_ppb": ALERT_TIERS["critical"]["threshold"],
        },
    }
    return report


def _archive_event_report(s3, report: dict) -> str:
    """Write the event report to archive + latest paths, update index.

    Returns the archive path for reference in logs / Slack.
    """
    ts = datetime.now(ZoneInfo("UTC"))
    date_str = ts.strftime("%Y-%m-%d")
    event_id = report["event_id"]
    archive_path = f"{SENSOR_EVENT_ARCHIVE_PATH}/{date_str}/{event_id}.json"

    payload = json.dumps(report, indent=2, default=str)
    s3.putFile(payload, path=archive_path, content_type="application/json")
    s3.putFile(payload, path=SENSOR_EVENT_LATEST_PATH, content_type="application/json")

    # Update rolling index (cap at SENSOR_EVENT_INDEX_MAX entries)
    try:
        existing = json.loads(s3.getFile(path=SENSOR_EVENT_INDEX_PATH).decode("utf-8"))
        events = existing.get("events", [])
    except Exception:
        events = []

    events.insert(0, {
        "event_id": event_id,
        "generated_at": report["generated_at"],
        "tier": report["tier"],
        "trigger": report["trigger"],
        "station": report["station"]["name"],
        "h2s_ppb": report["observation"]["h2s_ppb"],
        "archive_path": archive_path,
    })
    events = events[:SENSOR_EVENT_INDEX_MAX]

    index_payload = json.dumps(
        {"updated_at": ts.isoformat(), "events": events},
        indent=2,
        default=str,
    )
    s3.putFile(index_payload, path=SENSOR_EVENT_INDEX_PATH, content_type="application/json")

    return archive_path


# ---------------------------------------------------------------------------
# Core alert evaluation
# ---------------------------------------------------------------------------

def evaluate_apcd_alerts(
    df: pd.DataFrame,
    preds: dict,
    weather: dict,
    state: dict,
) -> dict:
    """Evaluate both tiers per station.

    Returns a dict with keys:
      actions : list of {site_name, tier_key, trigger, obs_row, pred, message, report}
      state   : updated state dict (caller must persist)
    """
    actions: list = []
    if df is None or df.empty:
        return {"actions": actions, "state": state}

    now = pd.Timestamp.now(tz="UTC")

    # Latest reading per site — used for cross-station context in messages/reports
    latest_per_site = (
        df.sort_values("time")
        .groupby("site_name", as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )

    for site_name in APCD_SITE_TO_STATION:
        site_df = df[df["site_name"] == site_name].sort_values("time")
        if site_df.empty:
            continue
        latest = site_df.iloc[-1]
        latest_ppb = float(latest["h2s_ppb"])
        site_state = state.setdefault(site_name, _empty_station_state())
        site_pred = preds.get(site_name)

        for tier_key, tier_cfg in ALERT_TIERS.items():
            threshold = tier_cfg["threshold"]
            ts = site_state.setdefault(tier_key, _empty_tier_state())
            above = latest_ppb >= threshold

            # --- ONSET ---
            if above and not ts["in_event"]:
                quiet_cutoff = latest["time"] - pd.Timedelta(hours=ALERT_QUIET_HOURS)
                recent_before = site_df[site_df["time"] < latest["time"]]
                was_quiet = recent_before.empty or (
                    recent_before[recent_before["time"] >= quiet_cutoff]["h2s_ppb"] < threshold
                ).all()
                if was_quiet:
                    ts.update({
                        "in_event":        True,
                        "event_start":     str(latest["time"]),
                        "last_exceedance": str(latest["time"]),
                        "onset_sent":      True,
                        "summary_sent":    False,
                    })
                    message = build_sensor_alert_message(
                        site_name, tier_key, latest, latest_per_site, site_pred, weather
                    )
                    report = build_event_report(
                        site_name, tier_key, "onset", latest, latest_per_site, site_pred, weather
                    )
                    actions.append({
                        "site_name": site_name,
                        "tier_key":  tier_key,
                        "trigger":   "onset",
                        "message":   message,
                        "report":    report,
                    })
            elif above and ts["in_event"]:
                ts["last_exceedance"] = str(latest["time"])

            # --- SUMMARY (event close) ---
            if ts["in_event"] and not ts["summary_sent"]:
                last_exc_raw = ts.get("last_exceedance")
                if last_exc_raw:
                    last_exc = pd.Timestamp(last_exc_raw)
                    if last_exc.tzinfo is None:
                        last_exc = last_exc.tz_localize("UTC")
                    elapsed_hours = (now - last_exc).total_seconds() / 3600
                    if elapsed_hours >= ALERT_CLOSE_WAIT_HOURS and not above:
                        event_start = pd.Timestamp(ts["event_start"])
                        if event_start.tzinfo is None:
                            event_start = event_start.tz_localize("UTC")
                        message = build_sensor_alert_message(
                            site_name, tier_key, latest, latest_per_site, site_pred, weather
                        )
                        report = build_event_report(
                            site_name, tier_key, "summary", latest, latest_per_site, site_pred, weather
                        )
                        actions.append({
                            "site_name": site_name,
                            "tier_key":  tier_key,
                            "trigger":   "summary",
                            "message":   message,
                            "report":    report,
                        })
                        ts["in_event"] = False
                        ts["summary_sent"] = True

            site_state[tier_key] = ts
        state[site_name] = site_state

    return {"actions": actions, "state": state}


# ---------------------------------------------------------------------------
# Dagster dispatcher asset + job
# ---------------------------------------------------------------------------

class ApcdSensorAlertConfig(dg.Config):
    """Run config payload the sensor passes to the dispatcher.

    Actions are serialized as a JSON string (Dagster Config fields need to be
    primitives and the shape is dynamic per-run).
    """
    actions_json: str = "[]"


@dg.asset(
    key_prefix="h2s",
    group_name="h2s_alerts",
    required_resource_keys={"slack", "s3"},
    kinds={"slack", "s3"},
    description="Dispatch APCD multi-station sensor-watch alerts to Slack and archive event reports",
)
def apcd_sensor_alert_dispatcher(
    context: dg.AssetExecutionContext,
    config: ApcdSensorAlertConfig,
) -> None:
    """Post Slack messages and archive JSON event reports for each action."""
    slack = context.resources.slack
    s3 = context.resources.s3

    try:
        actions = json.loads(config.actions_json or "[]")
    except Exception as e:
        context.log.error(f"Failed to parse actions_json: {e}")
        return

    if not actions:
        context.log.info("No actions to dispatch.")
        return

    client = slack.get_client()
    for action in actions:
        site = action.get("site_name")
        tier = action.get("tier_key")
        trigger = action.get("trigger")
        message = action.get("message") or ""
        report = action.get("report") or {}

        try:
            archive_path = _archive_event_report(s3, report)
            context.log.info(f"Archived event report: {archive_path}")
        except Exception as e:
            context.log.warning(f"Failed to archive event report for {site}/{tier}: {e}")
            archive_path = None

        try:
            context.log.info(f"Sending {tier}/{trigger} Slack alert for {site}")
            client.chat_postMessage(
                channel=slack.channel,
                text=f"```{message}```",
            )
        except Exception as e:
            context.log.error(f"Slack post failed for {site}/{tier}: {e}")


apcd_sensor_watch_job = dg.define_asset_job(
    name="apcd_sensor_watch_job",
    selection=dg.AssetSelection.assets(apcd_sensor_alert_dispatcher),
    description="Dispatch APCD sensor-watch alerts triggered by apcd_sensor_watch_sensor",
)


# ---------------------------------------------------------------------------
# Dagster sensor
# ---------------------------------------------------------------------------

@dg.sensor(
    job=apcd_sensor_watch_job,
    minimum_interval_seconds=300,
    required_resource_keys={"s3"},
    description=(
        "Poll hs2_lastday.csv from the resilentpublic bucket every 5 min; "
        "fire multi-station Slack alerts with model predictions at watch "
        "(30 ppb) and critical (100 ppb) thresholds"
    ),
    default_status=dg.DefaultSensorStatus.RUNNING,
)
def apcd_sensor_watch_sensor(context: dg.SensorEvaluationContext):
    """Poll APCD H2S readings every 5 min and dispatch enriched alerts."""
    s3 = context.resources.s3

    try:
        df = _load_hs2_lastday(s3)
    except Exception as e:
        context.log.warning(f"Could not load hs2_lastday.csv: {e}")
        return

    try:
        preds = _load_predictions(s3)
    except Exception as e:
        context.log.warning(f"Could not load daily_summary.json: {e}")
        preds = {}

    try:
        weather = _load_weather_context(s3)
    except Exception as e:
        context.log.warning(f"Could not load observation parquet: {e}")
        weather = {}

    state = _load_sensor_state(s3)
    result = evaluate_apcd_alerts(df, preds, weather, state)
    _save_sensor_state(s3, result["state"])

    actions = result["actions"]
    if not actions:
        if not df.empty:
            latest_rows = (
                df.sort_values("time")
                .groupby("site_name")
                .tail(1)
                .set_index("site_name")["h2s_ppb"]
                .to_dict()
            )
            context.log.debug(f"No alerts — latest per site: {latest_rows}")
        else:
            context.log.debug("No data in hs2_lastday.csv")
        return

    yield dg.RunRequest(
        run_key=f"apcd_sensor_watch_{pd.Timestamp.now(tz='UTC').isoformat()}",
        run_config={
            "ops": {
                "h2s__apcd_sensor_alert_dispatcher": {
                    "config": {
                        "actions_json": json.dumps(actions, default=str),
                    }
                }
            }
        },
    )
