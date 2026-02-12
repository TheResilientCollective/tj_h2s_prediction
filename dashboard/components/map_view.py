"""Folium map component showing H2S station status."""

import folium
import pandas as pd
import panel as pn

from dashboard.constants import CATEGORY_COLORS, COLOR_GRAY


def create_map(h2s_df: pd.DataFrame, locations_df: pd.DataFrame) -> pn.pane.plot.Folium:
    """Build a Folium map with station markers colored by latest H2S category.

    Parameters
    ----------
    h2s_df : DataFrame
        Filtered H2S data (already filtered by year/sites).
    locations_df : DataFrame
        Site locations with latitude, longitude, site_name columns.
    """
    # Compute latest H2S category per site from filtered data
    latest_per_site: dict[str, dict] = {}
    if not h2s_df.empty:
        idx = h2s_df.groupby("site_name")["time"].idxmax()
        for _, row in h2s_df.loc[idx].iterrows():
            latest_per_site[row["site_name"]] = {
                "category": str(row["h2s_category"]),
                "h2s": row["H2S"],
                "time": row["time"],
            }

    # Center map on mean of all locations
    center_lat = locations_df["lat"].mean()
    center_lon = locations_df["lon"].mean()

    m = folium.Map(
        location=[center_lat, center_lon],
        tiles="CartoDB positron",
        width="100%",
        height="100%",
    )

    # Fit bounds to show all stations with padding
    sw = [locations_df["lat"].min() - 0.02, locations_df["lon"].min() - 0.02]
    ne = [locations_df["lat"].max() + 0.02, locations_df["lon"].max() + 0.02]
    m.fit_bounds([sw, ne])

    for _, loc in locations_df.iterrows():
        site = loc["site_name"]
        info = latest_per_site.get(site)

        if info:
            color = CATEGORY_COLORS.get(info["category"], COLOR_GRAY)
            popup_text = (
                f"<b>{site}</b><br>"
                f"H2S: {info['h2s']:.1f} ppb<br>"
                f"{info['time'].strftime('%Y-%m-%d %H:%M')}"
            )
        else:
            color = COLOR_GRAY
            popup_text = f"<b>{site}</b><br>No data"

        folium.CircleMarker(
            location=[loc["lat"], loc["lon"]],
            radius=10,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.8,
            popup=folium.Popup(popup_text, max_width=200),
        ).add_to(m)

    return pn.pane.plot.Folium(m, height=300, sizing_mode="stretch_width")
