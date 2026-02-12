"""H2S Monitoring Dashboard — Panel/HoloViz application.

Run with:
    panel serve dashboard/app.py --port 5006 --show --autoreload
"""

import sys
from pathlib import Path

# Ensure the project root is on sys.path so 'dashboard' package imports work
# when Panel serves this file as a standalone script.
_root = str(Path(__file__).resolve().parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

import panel as pn

from dashboard.constants import SITES
from dashboard.data import filter_data, load_complaints, load_h2s_data, load_locations
from dashboard.components.map_view import create_map
from dashboard.components.polar_heatmap import create_polar_heatmap
from dashboard.components.timeseries_range import create_timeseries

pn.extension("tabulator", sizing_mode="stretch_width")


def build_dashboard() -> pn.template.FastListTemplate:
    # Load data once
    h2s_df = load_h2s_data()
    locations_df = load_locations()
    complaints_df = load_complaints()

    # Derive year range from data
    years = sorted(h2s_df["year"].unique())
    min_year, max_year = int(years[0]), int(years[-1])

    # --- Sidebar widgets ---
    logo_placeholder = pn.pane.HTML(
        "<div style='height:60px;background:#eee;display:flex;"
        "align-items:center;justify-content:center;border-radius:8px;"
        "margin-bottom:16px;color:#999;'>Logo</div>",
        sizing_mode="stretch_width",
    )

    year_slider = pn.widgets.DiscreteSlider(
        name="Year",
        options={str(y): y for y in range(min_year, max_year + 1)},
        value=max_year,
        stylesheets=[
            ":host { --design-unit: 12; }",
            ":host .bk-slider-title { font-size: 16px; margin-bottom: 8px; }",
        ],
    )

    site_selector = pn.widgets.CheckButtonGroup(
        name="Sites",
        options=SITES,
        value=list(SITES),
        button_type="success",
        orientation="vertical",
    )

    # --- Reactive components ---
    def _filtered_data(year: int, sites: list[str]):
        if not sites:
            return h2s_df.iloc[0:0]
        return filter_data(h2s_df, year, sites)

    def _map_view(year: int, sites: list[str]):
        df = _filtered_data(year, sites)
        return create_map(df, locations_df)

    def _polar_view(year: int, sites: list[str]):
        df = _filtered_data(year, sites)
        return create_polar_heatmap(df)

    def _timeseries_view(year: int, sites: list[str]):
        df = _filtered_data(year, sites)
        # Filter complaints to selected year
        cdf = complaints_df[complaints_df["year"] == year] if not complaints_df.empty else None
        return create_timeseries(df, cdf)

    bound_map = pn.bind(_map_view, year=year_slider, sites=site_selector)
    bound_polar = pn.bind(_polar_view, year=year_slider, sites=site_selector)
    bound_ts = pn.bind(_timeseries_view, year=year_slider, sites=site_selector)

    # --- Layout ---
    map_pane = pn.panel(bound_map, height=300, sizing_mode="stretch_width")
    polar_pane = pn.panel(bound_polar, sizing_mode="stretch_width")

    top_row = pn.Row(
        pn.Column(map_pane, sizing_mode="stretch_width", max_width=450),
        pn.Column(polar_pane, sizing_mode="stretch_width"),
        sizing_mode="stretch_width",
    )

    bottom_row = pn.Column(
        pn.panel(bound_ts, height=650, sizing_mode="stretch_width"),
        sizing_mode="stretch_width",
    )

    template = pn.template.FastListTemplate(
        title="",
        sidebar=[logo_placeholder, year_slider, site_selector],
        main=[top_row, bottom_row],
        main_max_width="1720px",
        accent_base_color="#FF5722",
        header_background="#FF5722",
    )

    return template


# Panel entry point
dashboard = build_dashboard()
dashboard.servable()
