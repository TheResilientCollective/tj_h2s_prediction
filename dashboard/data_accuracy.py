"""Accuracy rollup loader for the Panel dashboard.

Reads the JSON artifacts written by `accuracy_reporting_pipeline` from the
public MinIO bucket (no auth needed) and returns tidy DataFrames the Panel
components can plot.
"""

from __future__ import annotations

import json
import time
from typing import Any

import pandas as pd

from .constants import ACCURACY_REPORTS_URL

_cache: dict[str, tuple[float, Any]] = {}
_TTL = 300


def _fetch_json(url: str) -> dict[str, Any] | None:
    """Fetch JSON from a URL. Works in both server (CPython) and browser (Pyodide)."""
    try:
        try:
            from pyodide.http import open_url  # type: ignore[import-not-found]
            return json.loads(open_url(url).read())
        except ImportError:
            from urllib.request import urlopen
            with urlopen(url, timeout=10) as resp:
                return json.loads(resp.read())
    except Exception:  # noqa: BLE001 — report-missing UX lives in the view
        return None


def _cached_fetch(key: str, url: str) -> dict[str, Any] | None:
    now = time.monotonic()
    if key in _cache:
        ts, val = _cache[key]
        if now - ts < _TTL:
            return val
    val = _fetch_json(url)
    _cache[key] = (now, val)
    return val


def load_latest_scorecards() -> dict[str, Any] | None:
    return _cached_fetch("latest", f"{ACCURACY_REPORTS_URL}/latest.json")


def load_rolling_scorecard(window_days: int) -> dict[str, Any] | None:
    return _cached_fetch(f"rolling_{window_days}", f"{ACCURACY_REPORTS_URL}/rolling/{window_days}d/scorecard.json")


def load_alert_performance() -> dict[str, Any] | None:
    return _cached_fetch("alert_perf", f"{ACCURACY_REPORTS_URL}/alert_performance/30d.json")


def sites_dataframe(scorecard: dict[str, Any] | None) -> pd.DataFrame:
    """Flatten `scorecard['sites']` into a tidy DataFrame."""
    if not scorecard or not scorecard.get("sites"):
        return pd.DataFrame(
            columns=[
                "site",
                "n_predictions",
                "n_matched_observations",
                "balanced_accuracy",
                "orange_recall",
                "orange_precision",
                "false_alarm_rate",
            ]
        )
    rows = []
    for site, card in scorecard["sites"].items():
        rows.append(
            {
                "site": site,
                **{k: card.get(k) for k in (
                    "n_predictions",
                    "n_matched_observations",
                    "balanced_accuracy",
                    "orange_recall",
                    "orange_precision",
                    "false_alarm_rate",
                )},
            }
        )
    return pd.DataFrame(rows).sort_values("site").reset_index(drop=True)
