"""Timeseries range tool with linked minimap for H2S dashboard."""

import holoviews as hv
import numpy as np
import pandas as pd
import panel as pn
from holoviews.plotting.links import RangeToolLink

from dashboard.constants import CATEGORY_COLORS, COLOR_GRAY, H2S_GREEN_MAX, H2S_YELLOW_MAX

hv.extension("bokeh")

# Detail charts share this kdim so their x-axes are linked automatically
_XDIM = hv.Dimension("x", label="Date")
# Minimap uses a separate kdim so it keeps its own independent range
_MINIMAP_XDIM = hv.Dimension("xm", label="Date")


def create_timeseries(
    h2s_df: pd.DataFrame,
    complaints_df: pd.DataFrame | None = None,
) -> pn.pane.HoloViews:
    """Build linked timeseries charts with a minimap range selector.

    Four stacked detail charts sharing an x-axis:
      1. Hourly H2S line chart with colored background zones
      2. Hourly streamflow line chart
      3. Daily stacked bar chart of yellow/orange H2S counts
      4. Daily odor complaints bar chart

    A minimap at the bottom drives x-range via RangeToolLink.
    """
    if h2s_df.empty:
        return pn.pane.HTML(
            "<div style='text-align:center;color:#666;padding:40px;'>"
            "No data available for selected filters</div>",
            height=500,
            sizing_mode="stretch_width",
        )

    h2s_df = h2s_df.copy()

    # --- Chart 1: Hourly H2S line with background zones ---
    times = h2s_df.groupby("time")["H2S"].mean().sort_index()
    times.index = times.index.tz_localize(None) if times.index.tz else times.index

    h2s_curve = hv.Curve(
        (times.index, times.values), kdims=_XDIM, vdims="H2S"
    ).opts(color="black", line_width=1)

    yellow_box = hv.HSpan(H2S_GREEN_MAX, H2S_YELLOW_MAX).opts(
        color=CATEGORY_COLORS["yellow"], alpha=0.25
    )
    orange_box = hv.HSpan(H2S_YELLOW_MAX, max(times.max() * 1.1, H2S_YELLOW_MAX + 5)).opts(
        color=CATEGORY_COLORS["orange"], alpha=0.25
    )

    h2s_chart = (yellow_box * orange_box * h2s_curve).opts(
        height=150,
        responsive=True,
        xaxis=None,
        ylabel="H2S (ppb)",
        toolbar="above",
        title="",
    )

    # --- Chart 2: Hourly streamflow ---
    flow_col = None
    for candidate in ["Flow (m^3/s)--Border", "flow_rate_cms", "Flow"]:
        if candidate in h2s_df.columns:
            flow_col = candidate
            break

    if flow_col:
        flow_ts = h2s_df.groupby("time")[flow_col].mean().sort_index()
        flow_ts.index = flow_ts.index.tz_localize(None) if flow_ts.index.tz else flow_ts.index
        flow_chart = hv.Curve(
            (flow_ts.index, flow_ts.values), kdims=_XDIM, vdims="Flow"
        ).opts(
            color="#1f77b4",
            line_width=1,
            height=120,
            responsive=True,
            xaxis=None,
            ylabel="Flow (m\u00b3/s)",
            title="",
        )
    else:
        flow_chart = hv.Curve([], kdims=_XDIM, vdims="Flow").opts(
            height=120, responsive=True, xaxis=None, ylabel="Flow (m\u00b3/s)", title=""
        )

    # --- Chart 3: Daily stacked yellow/orange H2S counts ---
    time_naive = h2s_df["time"].dt.tz_localize(None)
    h2s_df["date_dt"] = time_naive.dt.normalize()

    hazard = h2s_df[h2s_df["h2s_category"].isin(["yellow", "orange"])]
    daily_cats = (
        hazard.groupby(["date_dt", "h2s_category"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=["yellow", "orange"], fill_value=0)
    )

    if daily_cats.empty:
        daily_cats = pd.DataFrame(
            {"yellow": [0], "orange": [0]},
            index=pd.DatetimeIndex([times.index.min()], name="date_dt"),
        )

    daily_cats.index = pd.to_datetime(daily_cats.index)

    yellow_bars = hv.Bars(
        (daily_cats.index, daily_cats["yellow"]),
        kdims=_XDIM, vdims="Count",
    ).opts(color=CATEGORY_COLORS["yellow"])

    orange_bars = hv.Bars(
        (daily_cats.index, daily_cats["orange"]),
        kdims=_XDIM, vdims="Count",
    ).opts(color=CATEGORY_COLORS["orange"])

    hazard_chart = (yellow_bars * orange_bars).opts(
        height=120,
        responsive=True,
        xaxis=None,
        ylabel="Hazard Count",
        title="",
    )

    # --- Chart 4: Daily odor complaints ---
    if complaints_df is not None and not complaints_df.empty:
        odor = complaints_df[
            complaints_df["nature_of_complaint"].str.contains("Odor", case=False, na=False)
        ].copy()
        odor["date_dt"] = pd.to_datetime(odor["date"])
        daily_complaints = odor.groupby("date_dt").size()
        daily_complaints.index = pd.to_datetime(daily_complaints.index)

        complaints_bars = hv.Bars(
            (daily_complaints.index, daily_complaints.values),
            kdims=_XDIM, vdims="Complaints",
        ).opts(
            color="#8856a7",
            height=100,
            responsive=True,
            xaxis=None,
            ylabel="Complaints",
            title="",
        )
    else:
        complaints_bars = hv.Bars(
            (daily_cats.index, np.zeros(len(daily_cats))),
            kdims=_XDIM, vdims="Complaints",
        ).opts(
            color=COLOR_GRAY,
            height=100,
            responsive=True,
            xaxis=None,
            ylabel="Complaints",
            title="",
        )

    # --- Minimap: hourly mean H2S (separate kdim so it keeps own range) ---
    minimap = hv.Curve(
        (times.index, times.values),
        kdims=_MINIMAP_XDIM, vdims="H2S",
    ).opts(
        height=80,
        responsive=True,
        yaxis=None,
        default_tools=[],
        title="",
        color=CATEGORY_COLORS["orange"],
        line_width=1,
    )

    # Single RangeToolLink from minimap → first detail chart.
    # The detail charts share x-range via shared_axes (default=True),
    # so controlling one controls all.
    RangeToolLink(minimap, h2s_chart, axes=["x"])

    # Detail charts share _XDIM so they get linked x-axes automatically.
    # Minimap uses _MINIMAP_XDIM so it stays independent.
    # merge_tools=False keeps toolbars separate.
    layout = (h2s_chart + flow_chart + hazard_chart + complaints_bars + minimap).cols(1).opts(
        merge_tools=False,
    )

    return pn.pane.HoloViews(layout, height=650, sizing_mode="stretch_width")
