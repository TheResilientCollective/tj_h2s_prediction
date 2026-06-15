"""Pure logic for the observed >10 ppb state machine — no Dagster / S3 / Slack.

Three testable pieces:

- :func:`evaluate_yellow` — onset/close transitions for a single yellow-tier
  cell, given a recent observation window. Mirrors the watch/critical machine
  (``h2s_alert_system.evaluate_alerts``) but at the 10 ppb threshold with a
  2 h close wait.
- :func:`hours_above` — tally of hours ≥ yellow (10) and ≥ orange (30) in the
  recent look-back window, for the close-out report.
- :func:`forecast_vs_actual` — join the stored product forecasts that targeted
  the event hours against what was actually measured, so the close-out can
  report predicted P(exceed) vs the truth ("Alert Performance"). One forecast
  per hour, taking the earliest run that predicted it (the most-ahead, hardest
  test).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from h2s.constants import (
    ALERT_PERF_HOURS_WINDOW,
    ALERT_QUIET_HOURS,
    ALERT_YELLOW_CLOSE_WAIT_HOURS,
    ALERT_YELLOW_THRESHOLD,
    H2S_THRESHOLD_HIGH,
    H2S_THRESHOLD_MED,
)

# NESTOR-BES, as it appears in both the observation `site_name` and the product
# `station` columns.
NB_SITE = "NESTOR - BES"

_FVA_COLS = ["time", "lead_hour", "p_exceed", "h2s_pred", "actual_h2s", "actual_exceeded"]


def empty_cell() -> dict:
    return {
        "in_event": False,
        "event_start": None,
        "last_exceedance": None,
        "onset_sent": False,
        "close_sent": False,
    }


@dataclass
class YellowVerdict:
    send_onset: bool = False
    send_close: bool = False
    current_ppb: float | None = None
    onset_row: pd.Series | None = None
    prev_row: pd.Series | None = None
    event_df: pd.DataFrame | None = None
    extras: dict = field(default_factory=dict)


def _as_utc(ts) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    return t.tz_localize("UTC") if t.tzinfo is None else t


def evaluate_yellow(
    df: pd.DataFrame,
    cell: dict,
    now: pd.Timestamp,
    threshold: float = ALERT_YELLOW_THRESHOLD,
    close_wait_h: float = ALERT_YELLOW_CLOSE_WAIT_HOURS,
    quiet_h: float = ALERT_QUIET_HOURS,
) -> tuple[YellowVerdict, dict]:
    """Advance the yellow-tier cell against the latest observation window.

    ``df`` is recent NESTOR-BES observations sorted oldest→newest with ``time``
    and ``H2S`` columns. Returns the verdict and the mutated cell (caller
    persists it).
    """
    verdict = YellowVerdict()
    if df.empty:
        return verdict, cell

    latest = df.iloc[-1]
    latest_ppb = float(latest["H2S"])
    verdict.current_ppb = latest_ppb
    above = latest_ppb >= threshold

    # --- Onset ---
    if above and not cell["in_event"]:
        quiet_cutoff = latest["time"] - pd.Timedelta(hours=quiet_h)
        before = df[df["time"] < latest["time"]]
        was_quiet = before.empty or (
            before[before["time"] >= quiet_cutoff]["H2S"] < threshold
        ).all()
        if was_quiet:
            cell.update({
                "in_event": True,
                "event_start": str(latest["time"]),
                "last_exceedance": str(latest["time"]),
                "onset_sent": True,
                "close_sent": False,
            })
            verdict.send_onset = True
            verdict.onset_row = latest
            verdict.prev_row = df.iloc[-2] if len(df) >= 2 else None
    elif above and cell["in_event"]:
        cell["last_exceedance"] = str(latest["time"])

    # --- Close ---
    if cell["in_event"] and not cell["close_sent"]:
        last_exc = _as_utc(cell["last_exceedance"])
        elapsed_h = (now - last_exc).total_seconds() / 3600
        if elapsed_h >= close_wait_h and not above:
            event_start = _as_utc(cell["event_start"])
            event_df = df[(df["time"] >= event_start) & (df["H2S"] >= threshold)].copy()
            verdict.send_close = True
            verdict.event_df = event_df
            cell["in_event"] = False
            cell["close_sent"] = True

    return verdict, cell


def hours_above(
    df: pd.DataFrame,
    now: pd.Timestamp,
    window_h: int = ALERT_PERF_HOURS_WINDOW,
    yellow: float = H2S_THRESHOLD_MED,
    orange: float = H2S_THRESHOLD_HIGH,
) -> dict:
    """Count observation hours ≥ yellow and ≥ orange over the look-back window."""
    if df.empty:
        return {"window_h": window_h, "yellow": 0, "orange": 0}
    cutoff = now - pd.Timedelta(hours=window_h)
    recent = df[df["time"] >= cutoff]
    return {
        "window_h": window_h,
        "yellow": int((recent["H2S"] >= yellow).sum()),
        "orange": int((recent["H2S"] >= orange).sum()),
    }


def forecast_vs_actual(
    products_df: pd.DataFrame | None,
    obs_df: pd.DataFrame,
    event_start: pd.Timestamp,
    event_end: pd.Timestamp,
    station: str = NB_SITE,
    variant: str = "evidence",
    prob_col: str = "p10",
) -> pd.DataFrame:
    """Per-hour predicted P(exceed) vs measured H2S over the event window.

    Takes the earliest run that forecast each hour (most-ahead = hardest test).
    Returns columns [time, lead_hour, p_exceed, h2s_pred, actual_h2s,
    actual_exceeded]; empty frame (same columns) when no overlap exists.
    """
    if products_df is None or len(products_df) == 0:
        return pd.DataFrame(columns=_FVA_COLS)

    p = products_df[
        (products_df["station"] == station) & (products_df["variant"] == variant)
    ].copy()
    if p.empty or prob_col not in p.columns:
        return pd.DataFrame(columns=_FVA_COLS)

    p["time"] = pd.to_datetime(p["time"], utc=True)
    p = p[(p["time"] >= event_start) & (p["time"] <= event_end)]
    if p.empty:
        return pd.DataFrame(columns=_FVA_COLS)

    # One forecast per target hour: the earliest run that predicted it.
    sort_cols = ["run_ts", "time"] if "run_ts" in p.columns else ["time"]
    p = p.sort_values(sort_cols).drop_duplicates("time", keep="first")

    o = obs_df[["time", "H2S"]].copy()
    o["time"] = pd.to_datetime(o["time"], utc=True)
    o = o.rename(columns={"H2S": "actual_h2s"})

    merged = p.merge(o, on="time", how="inner")
    if merged.empty:
        return pd.DataFrame(columns=_FVA_COLS)

    merged = merged.rename(columns={prob_col: "p_exceed"})
    merged["actual_exceeded"] = merged["actual_h2s"] >= H2S_THRESHOLD_MED
    return merged[_FVA_COLS].sort_values("time").reset_index(drop=True)


def performance_summary(fva: pd.DataFrame, prob_cut: float = 0.5) -> dict:
    """Precision/recall of the P(exceed)>cut forecast against measured >10 ppb."""
    if fva.empty:
        return {"n_hours": 0, "n_forecast_exceed": 0, "n_actual_exceed": 0,
                "hits": 0, "precision": None, "recall": None}
    forecast_exceed = fva["p_exceed"] > prob_cut
    actual_exceed = fva["actual_exceeded"].astype(bool)
    hits = int((forecast_exceed & actual_exceed).sum())
    n_fc = int(forecast_exceed.sum())
    n_act = int(actual_exceed.sum())
    return {
        "n_hours": int(len(fva)),
        "n_forecast_exceed": n_fc,
        "n_actual_exceed": n_act,
        "hits": hits,
        "precision": (hits / n_fc) if n_fc else None,
        "recall": (hits / n_act) if n_act else None,
    }
