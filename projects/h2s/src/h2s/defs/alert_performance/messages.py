"""Slack messages for the observed >10 ppb machine — onset + performance close-out.

Reuses the formatting helpers from ``h2s_alert_system`` so the yellow-tier
messages read consistently with watch/critical.
"""

from __future__ import annotations

import pandas as pd

from h2s.constants import (
    ALERT_LOCAL_TZ,
    ALERT_YELLOW_THRESHOLD,
    H2S_THRESHOLD_HIGH,
)
from h2s.defs.h2s_alert_system import _fmt_duration, _fmt_time, _wind_flag

_MAX_FVA_ROWS = 12


def build_onset_message(row: pd.Series, prev_row: pd.Series | None) -> str:
    prior = prev_row["H2S"] if prev_row is not None else float("nan")
    prior_str = f"{prior:.0f} ppb" if not pd.isna(prior) else "n/a"
    lines = [
        "H2S YELLOW (>10 ppb) — NESTOR / Berry Elementary School",
        "Observation-based exceedance event opened.",
        "=" * 52,
        f"{'Time detected':<22} {_fmt_time(row['time'])}",
        f"{'H2S reading':<22} {row['H2S']:.0f} ppb  (threshold: {ALERT_YELLOW_THRESHOLD:.0f} ppb)",
        f"{'Prior 1-hr reading':<22} {prior_str}",
    ]
    if "wind_speed_10m" in row and not pd.isna(row.get("wind_speed_10m")):
        lines.append(
            f"{'Wind speed':<22} {row['wind_speed_10m']:.1f} m/s  ({_wind_flag(row['wind_speed_10m'])})"
        )
    lines += [
        "-" * 52,
        "Monitoring continues. A forecast-performance report will follow at event close.",
    ]
    return "\n".join(lines)


def _fva_table(fva: pd.DataFrame) -> list[str]:
    if fva.empty:
        return ["  (no stored product forecasts covered these hours — comparison unavailable)"]
    rows = ["  time(PT)   P(>10)   pred    actual   flagged?"]
    for _, r in fva.head(_MAX_FVA_ROWS).iterrows():
        t = pd.Timestamp(r["time"]).tz_convert(ALERT_LOCAL_TZ).strftime("%a %H:%M")
        p = r["p_exceed"]
        p_str = "n/a" if pd.isna(p) else f"{round(p * 100):d}%"
        flagged = (not pd.isna(p)) and p > 0.5
        actual_mark = "✓" if bool(r["actual_exceeded"]) else " "
        flag_mark = "✓" if flagged else "·"
        rows.append(
            f"  {t:<9} {p_str:>5}   {r['h2s_pred']:>4.0f}   {r['actual_h2s']:>4.0f}{actual_mark}    {flag_mark}"
        )
    if len(fva) > _MAX_FVA_ROWS:
        rows.append(f"  … {len(fva) - _MAX_FVA_ROWS} more hours")
    return rows


def _skill_line(summary: dict) -> str:
    if summary["n_hours"] == 0:
        return "Forecast skill: no overlapping forecasts to score."
    prec = summary["precision"]
    rec = summary["recall"]
    prec_s = "n/a" if prec is None else f"{round(prec * 100):d}%"
    rec_s = "n/a" if rec is None else f"{round(rec * 100):d}%"
    return (
        f"Forecast skill: flagged {summary['hits']}/{summary['n_actual_exceed']} "
        f"exceedance hours (recall {rec_s}); {summary['hits']}/{summary['n_forecast_exceed']} "
        f"flags were correct (precision {prec_s})."
    )


def build_closeout_message(
    event_df: pd.DataFrame,
    hours: dict,
    fva: pd.DataFrame,
    summary: dict,
    current_ppb: float,
) -> str:
    start = event_df["time"].iloc[0]
    end = event_df["time"].iloc[-1]
    peak = event_df["H2S"].max()
    peak_t = event_df.loc[event_df["H2S"].idxmax(), "time"]
    mean_h2s = event_df["H2S"].mean()

    lines = [
        "H2S YELLOW EVENT CLOSED — Alert Performance — NESTOR / Berry Elementary School",
        "=" * 60,
        f"{'Event window':<24} {_fmt_time(start)} – "
        f"{end.tz_convert(ALERT_LOCAL_TZ).strftime('%H:%M %Z')}",
        f"{'Duration':<24} {_fmt_duration(start, end)}",
        f"{'Peak H2S':<24} {peak:.0f} ppb  at {peak_t.tz_convert(ALERT_LOCAL_TZ).strftime('%H:%M')}",
        f"{'Mean H2S (event)':<24} {mean_h2s:.0f} ppb",
        f"{'Current H2S':<24} {current_ppb:.0f} ppb  (returned below threshold)",
        "-" * 60,
        f"Hours above threshold (past {hours['window_h']}h):",
        f"  Yellow (≥{ALERT_YELLOW_THRESHOLD:.0f} ppb): {hours['yellow']} hr   |   "
        f"Orange (≥{H2S_THRESHOLD_HIGH:.0f} ppb): {hours['orange']} hr",
        "-" * 60,
        "Forecast performance — P(>10 ppb) issued ahead vs measured:",
    ] + _fva_table(fva) + [
        "-" * 60,
        _skill_line(summary),
        "Event log archived.",
    ]
    return "\n".join(lines)
