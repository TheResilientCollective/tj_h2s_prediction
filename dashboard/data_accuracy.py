"""Accuracy rollup loader for the Panel dashboard.

Reads the JSON artifacts written by `accuracy_reporting_pipeline` from the
public MinIO bucket (no auth needed) and returns tidy DataFrames the Panel
components can plot.
"""

from __future__ import annotations

import json
from typing import Any
from urllib.request import urlopen

import pandas as pd
import panel as pn

from .constants import ACCURACY_REPORTS_URL


def _fetch_json(url: str) -> dict[str, Any] | None:
    try:
        with urlopen(url, timeout=10) as resp:
            return json.loads(resp.read())
    except Exception:  # noqa: BLE001 — report-missing UX lives in the view
        return None


@pn.cache(ttl=300)
def load_latest_scorecards() -> dict[str, Any] | None:
    return _fetch_json(f"{ACCURACY_REPORTS_URL}/latest.json")


@pn.cache(ttl=300)
def load_rolling_scorecard(window_days: int) -> dict[str, Any] | None:
    return _fetch_json(f"{ACCURACY_REPORTS_URL}/rolling/{window_days}d/scorecard.json")


@pn.cache(ttl=300)
def load_alert_performance() -> dict[str, Any] | None:
    return _fetch_json(f"{ACCURACY_REPORTS_URL}/alert_performance/30d.json")


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
