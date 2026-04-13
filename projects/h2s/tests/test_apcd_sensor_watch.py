"""Unit tests for the APCD public-bucket sensor watch.

These tests exercise the pure functions — alert evaluation, message rendering,
and event-report construction — without touching S3 or Slack. The Dagster
sensor/dispatcher wiring is validated separately by `dg check defs`.
"""

import pandas as pd
import pytest

from h2s.defs.apcd_sensor_watch import (
    _empty_station_state,
    build_event_report,
    build_sensor_alert_message,
    evaluate_apcd_alerts,
)
from h2s.constants import ALERT_TIERS, APCD_SITE_TO_STATION


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def empty_state():
    """Clean state dict covering all mapped APCD sites."""
    return {site: _empty_station_state() for site in APCD_SITE_TO_STATION}


@pytest.fixture
def sample_preds():
    """Synthetic per-site prediction dict matching _load_predictions() shape."""
    return {
        "NESTOR - BES": {
            "short": "NB",
            "station_key": "NESTOR__BES",
            "forecast_48h": {
                "max_h2s_ppb": 42.0,
                "max_prob_5": 0.88,
                "max_prob_10": 0.72,
                "hours_orange": 4,
                "hours_yellow_high": 6,
                "hours_yellow_low": 8,
                "hours_green": 30,
            },
            "recent_24h": {
                "last_h2s": 18.0,
                "last_time": "2026-04-13T13:00:00Z",
                "mean_24h": 12.5,
                "max_24h": 45.0,
                "pct_exceed_5": 0.55,
            },
        },
    }


@pytest.fixture
def sample_weather():
    """Synthetic weather-context dict matching _load_weather_context() shape."""
    return {
        "wind_speed_10m": 1.8,
        "wind_direction_10m": 225.0,
        "wind_direction_categorical": "SW",
        "temperature_2m": 18.5,
        "relative_humidity_2m": 82.0,
        "surface_pressure": 1013.2,
        "cloud_cover": 45.0,
        "precipitation": 0.0,
        "stable_atm": 1.0,
        "sbiwtp_flow_mgd": 21.5,
        "Flow (m^3/s)--Border": 2.3,
    }


def _build_df(rows):
    """Turn a list of (site_name, time_iso, h2s_ppb, level) tuples into a DF."""
    df = pd.DataFrame(rows, columns=["site_name", "time", "h2s_ppb", "level"])
    df["time"] = pd.to_datetime(df["time"], utc=True)
    return df.sort_values(["site_name", "time"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# evaluate_apcd_alerts
# ---------------------------------------------------------------------------

def test_evaluate_fires_watch_onset_on_exceedance(empty_state, sample_preds, sample_weather):
    df = _build_df([
        ("NESTOR - BES", "2026-04-13T10:00:00Z",  5.0, "green"),
        ("NESTOR - BES", "2026-04-13T11:00:00Z",  8.0, "yellow"),
        ("NESTOR - BES", "2026-04-13T12:00:00Z", 45.0, "orange"),
    ])
    result = evaluate_apcd_alerts(df, sample_preds, sample_weather, empty_state)

    watch_actions = [
        a for a in result["actions"]
        if a["site_name"] == "NESTOR - BES" and a["tier_key"] == "watch" and a["trigger"] == "onset"
    ]
    assert len(watch_actions) == 1
    assert "NESTOR - BES" in watch_actions[0]["message"]
    assert "45 ppb" in watch_actions[0]["message"]

    nestor_state = result["state"]["NESTOR - BES"]["watch"]
    assert nestor_state["in_event"] is True
    assert nestor_state["onset_sent"] is True


def test_evaluate_fires_critical_onset_on_large_exceedance(empty_state, sample_preds, sample_weather):
    df = _build_df([
        ("NESTOR - BES", "2026-04-13T10:00:00Z",   5.0, "green"),
        ("NESTOR - BES", "2026-04-13T11:00:00Z", 150.0, "orange"),
    ])
    result = evaluate_apcd_alerts(df, sample_preds, sample_weather, empty_state)

    triggers = {(a["tier_key"], a["trigger"]) for a in result["actions"] if a["site_name"] == "NESTOR - BES"}
    assert ("watch", "onset") in triggers
    assert ("critical", "onset") in triggers
    assert result["state"]["NESTOR - BES"]["critical"]["in_event"] is True


def test_evaluate_does_not_refire_when_already_in_event(empty_state, sample_preds, sample_weather):
    # Seed state: already in a watch event
    empty_state["NESTOR - BES"]["watch"] = {
        "in_event": True,
        "event_start": "2026-04-13T09:00:00Z",
        "last_exceedance": "2026-04-13T11:00:00Z",
        "onset_sent": True,
        "summary_sent": False,
    }
    df = _build_df([
        ("NESTOR - BES", "2026-04-13T12:00:00Z", 40.0, "orange"),
    ])
    result = evaluate_apcd_alerts(df, sample_preds, sample_weather, empty_state)

    onset_actions = [
        a for a in result["actions"]
        if a["site_name"] == "NESTOR - BES" and a["trigger"] == "onset"
    ]
    assert len(onset_actions) == 0
    # last_exceedance should be updated to the newest timestamp
    assert result["state"]["NESTOR - BES"]["watch"]["last_exceedance"] != "2026-04-13T11:00:00Z"


def test_evaluate_per_station_independence(empty_state, sample_preds, sample_weather):
    df = _build_df([
        ("NESTOR - BES", "2026-04-13T12:00:00Z",  4.0, "green"),
        ("SAN YSIDRO",   "2026-04-13T12:00:00Z", 50.0, "orange"),
        ("IB CIVIC CTR", "2026-04-13T12:00:00Z", 10.0, "yellow"),
    ])
    result = evaluate_apcd_alerts(df, sample_preds, sample_weather, empty_state)

    # Only SAN YSIDRO should have fired
    firing_sites = {a["site_name"] for a in result["actions"] if a["trigger"] == "onset"}
    assert firing_sites == {"SAN YSIDRO"}

    assert result["state"]["SAN YSIDRO"]["watch"]["in_event"] is True
    assert result["state"]["NESTOR - BES"]["watch"]["in_event"] is False
    assert result["state"]["IB CIVIC CTR"]["watch"]["in_event"] is False


def test_evaluate_handles_empty_dataframe(empty_state, sample_preds, sample_weather):
    df = _build_df([])
    result = evaluate_apcd_alerts(df, sample_preds, sample_weather, empty_state)
    assert result["actions"] == []


# ---------------------------------------------------------------------------
# build_sensor_alert_message
# ---------------------------------------------------------------------------

def test_build_sensor_alert_message_renders_all_sections(sample_preds, sample_weather):
    df = _build_df([
        ("NESTOR - BES", "2026-04-13T12:00:00Z", 45.0, "orange"),
        ("SAN YSIDRO",   "2026-04-13T12:00:00Z", 12.0, "yellow"),
        ("IB CIVIC CTR", "2026-04-13T12:00:00Z",  3.0, "green"),
    ])
    obs = df[df["site_name"] == "NESTOR - BES"].iloc[-1]
    msg = build_sensor_alert_message(
        "NESTOR - BES", "watch", obs, df, sample_preds["NESTOR - BES"], sample_weather,
    )

    assert "H2S WATCH ALERT" in msg
    assert "NESTOR - BES" in msg
    assert "45 ppb" in msg
    assert "threshold: 30 ppb" in msg
    # Model forecast block populated
    assert "Model forecast" in msg
    assert "42" in msg  # max_h2s_ppb
    # Meteorology
    assert "Wind speed" in msg
    assert "1.8 m/s" in msg
    assert "SW" in msg
    # Cross-station block
    assert "Other stations" in msg
    assert "SAN YSIDRO" in msg
    assert "IB CIVIC CTR" in msg


def test_build_sensor_alert_message_missing_prediction(sample_weather):
    df = _build_df([
        ("SAN YSIDRO", "2026-04-13T12:00:00Z", 50.0, "orange"),
    ])
    obs = df.iloc[-1]
    msg = build_sensor_alert_message("SAN YSIDRO", "watch", obs, df, None, sample_weather)
    assert "n/a" in msg  # prediction block falls back
    assert "50 ppb" in msg


def test_build_sensor_alert_message_critical_escalation_hint(sample_preds, sample_weather):
    df = _build_df([
        ("NESTOR - BES", "2026-04-13T12:00:00Z", 150.0, "orange"),
    ])
    obs = df.iloc[-1]
    msg = build_sensor_alert_message(
        "NESTOR - BES", "watch", obs, df, sample_preds["NESTOR - BES"], sample_weather,
    )
    # Watch tier alert that also exceeds critical should mention it
    assert "CRITICAL" in msg


# ---------------------------------------------------------------------------
# build_event_report
# ---------------------------------------------------------------------------

def test_build_event_report_schema(sample_preds, sample_weather):
    df = _build_df([
        ("NESTOR - BES", "2026-04-13T12:00:00Z", 45.0, "orange"),
        ("SAN YSIDRO",   "2026-04-13T12:00:00Z", 12.0, "yellow"),
    ])
    obs = df[df["site_name"] == "NESTOR - BES"].iloc[-1]
    report = build_event_report(
        "NESTOR - BES", "watch", "onset", obs, df,
        sample_preds["NESTOR - BES"], sample_weather,
    )

    assert report["schema_version"] == "1.0"
    assert report["tier"] == "watch"
    assert report["trigger"] == "onset"
    assert report["station"]["name"] == "NESTOR - BES"
    assert report["station"]["key"] == "NESTOR__BES"
    assert report["observation"]["h2s_ppb"] == 45.0
    assert report["observation"]["source_file"].startswith("s3://resilentpublic/")
    assert report["thresholds"]["watch_ppb"] == ALERT_TIERS["watch"]["threshold"]
    assert report["thresholds"]["critical_ppb"] == ALERT_TIERS["critical"]["threshold"]

    # Meteorology is fully populated
    met = report["meteorology"]
    assert met["wind_speed_ms"] == 1.8
    assert met["wind_direction_deg"] == 225.0
    assert met["wind_direction_categorical"] == "SW"
    assert met["temperature_c"] == 18.5

    # Flow with deficit derived from baseline
    flow = report["flow"]
    assert flow["sbiwtp_flow_mgd"] == 21.5
    assert flow["sbiwtp_deficit_mgd"] is not None

    # Other stations snapshot
    names = {other["name"] for other in report["other_stations"]}
    assert "SAN YSIDRO" in names
    assert "NESTOR - BES" not in names


def test_build_event_report_handles_missing_weather():
    df = _build_df([
        ("SAN YSIDRO", "2026-04-13T12:00:00Z", 50.0, "orange"),
    ])
    obs = df.iloc[-1]
    report = build_event_report(
        "SAN YSIDRO", "watch", "onset", obs, df, pred=None, weather={},
    )
    assert report["meteorology"]["wind_speed_ms"] is None
    assert report["flow"]["sbiwtp_flow_mgd"] is None
    assert report["prediction"] is None


# ---------------------------------------------------------------------------
# Empty-state helpers
# ---------------------------------------------------------------------------

def test_empty_station_state_has_all_tiers():
    state = _empty_station_state()
    for tier in ALERT_TIERS:
        assert tier in state
        assert state[tier]["in_event"] is False
        assert state[tier]["onset_sent"] is False
