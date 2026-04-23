"""Accuracy tab for the Panel dashboard.

Shows the rolling-window scorecard from the accuracy reporting pipeline:
headline numbers, per-site table, and a per-level alert-performance chart.
"""

from __future__ import annotations

import panel as pn
import pandas as pd

from ..constants import CATEGORY_COLORS
from ..data_accuracy import (
    load_alert_performance,
    load_rolling_scorecard,
    sites_dataframe,
)

_BAR_COLORS = {
    "green": CATEGORY_COLORS["green"],
    "yellow": CATEGORY_COLORS["yellow"],
    "orange": CATEGORY_COLORS["orange"],
}


def _headline_card(label: str, value: str, hint: str = "") -> pn.pane.HTML:
    return pn.pane.HTML(
        f"""
        <div style="padding:16px;border-radius:8px;background:#fafafa;
             box-shadow:0 1px 2px rgba(0,0,0,0.08);">
          <div style="font-size:12px;color:#666;text-transform:uppercase;
               letter-spacing:0.05em;">{label}</div>
          <div style="font-size:28px;font-weight:600;color:#222;
               margin-top:4px;">{value}</div>
          <div style="font-size:11px;color:#888;margin-top:2px;">{hint}</div>
        </div>
        """,
        sizing_mode="stretch_width",
    )


def _pct(v: float | None) -> str:
    return "—" if v is None else f"{100 * v:.1f}%"


def _build_alert_bar(alert: dict | None) -> pn.viewable.Viewable:
    if not alert:
        return pn.pane.Markdown("_No alert-performance data available yet._")
    rows = []
    for level in ("green", "yellow", "orange"):
        metrics = (alert.get("by_level") or {}).get(level, {})
        rows.append(
            {
                "level": level,
                "precision": metrics.get("precision") or 0.0,
                "recall": metrics.get("recall") or 0.0,
                "f1": metrics.get("f1") or 0.0,
            }
        )
    df = pd.DataFrame(rows).set_index("level")
    return pn.pane.DataFrame(
        df.style.format("{:.2f}").apply(
            lambda s: [f"background-color:{_BAR_COLORS.get(i, '#eee')}22" for i in s.index],
            axis=0,
        ),
        sizing_mode="stretch_width",
    )


def create_accuracy_view() -> pn.viewable.Viewable:
    window_selector = pn.widgets.RadioButtonGroup(
        name="Window",
        options=["7d", "30d", "90d"],
        value="30d",
        button_type="default",
    )

    def _render(window: str) -> pn.viewable.Viewable:
        scorecard = load_rolling_scorecard(int(window.rstrip("d")))
        alert = load_alert_performance()

        if not scorecard:
            return pn.pane.Markdown(
                "### Accuracy reports unavailable\n\n"
                "The rolling scorecard has not been written to S3 yet. "
                "Run the `accuracy_reporting_job` in Dagster once per-day "
                "`metrics.json` files have accumulated."
            )

        overall = scorecard.get("overall") or {}
        headline = pn.Row(
            _headline_card(
                "Balanced accuracy",
                _pct(overall.get("balanced_accuracy")),
                hint=f"across {overall.get('n_sites', 0)} sites",
            ),
            _headline_card(
                "Orange recall",
                _pct(overall.get("orange_recall")),
                hint="share of true exceedances caught",
            ),
            _headline_card(
                "False-alarm rate",
                _pct(overall.get("false_alarm_rate")),
                hint="orange predicted, not exceedance",
            ),
            _headline_card(
                "Matched obs.",
                str(overall.get("n_matched_observations", 0)),
                hint=f"window: {scorecard.get('period_start')} → {scorecard.get('period_end')}",
            ),
            sizing_mode="stretch_width",
        )

        site_df = sites_dataframe(scorecard)
        pretty = site_df.copy()
        for col in ("balanced_accuracy", "orange_recall", "orange_precision",
                    "false_alarm_rate"):
            if col in pretty:
                pretty[col] = pretty[col].map(_pct)
        per_site = pn.widgets.Tabulator(
            pretty,
            disabled=True,
            show_index=False,
            sizing_mode="stretch_width",
            height=260,
            configuration={"columnDefaults": {"headerHozAlign": "left"}},
        )

        return pn.Column(
            headline,
            pn.pane.Markdown("### Per-site scorecard"),
            per_site,
            pn.pane.Markdown("### Alert-level performance (30d)"),
            _build_alert_bar(alert),
            sizing_mode="stretch_width",
        )

    return pn.Column(
        pn.Row(pn.pane.Markdown("### Model accuracy"), window_selector),
        pn.bind(_render, window=window_selector),
        sizing_mode="stretch_width",
    )
