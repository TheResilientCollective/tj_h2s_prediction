"""Polar heatmap component replicating openair polarPlot style (Fig. 2)."""

import io

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import panel as pn
from matplotlib.colors import BoundaryNorm, ListedColormap

from dashboard.constants import CATEGORY_COLORS, H2S_GREEN_MAX, H2S_YELLOW_MAX


# Discrete colormap matching H2S thresholds: green <5, yellow 5-30, orange >=30
_CMAP = ListedColormap([CATEGORY_COLORS["green"], CATEGORY_COLORS["yellow"], CATEGORY_COLORS["orange"]])
_BOUNDS = [0, H2S_GREEN_MAX, H2S_YELLOW_MAX, 100]
_NORM = BoundaryNorm(_BOUNDS, _CMAP.N)


def _render_polar_figure(h2s_df: pd.DataFrame) -> bytes | None:
    """Render polar heatmap to PNG bytes. Returns None if no data."""
    if h2s_df.empty:
        return None

    sites = sorted(h2s_df["site_name"].unique())
    n_sites = len(sites)

    if n_sites <= 2:
        nrows, ncols = 1, n_sites
    else:
        nrows, ncols = 1, 3

    fig, axes = plt.subplots(
        nrows, ncols,
        subplot_kw={"projection": "polar"},
        figsize=(3.5 * ncols + 0.8, 4.5),
        dpi=120,
    )
    if n_sites == 1:
        axes = [axes]
    else:
        axes = list(np.array(axes).flat)

    # Bins for the polar grid
    n_dir_bins = 36
    n_spd_bins = 10
    dir_bins = np.linspace(0, 2 * np.pi, n_dir_bins + 1)
    max_wind_speed = 15  # m/s cap
    spd_bins = np.linspace(0, max_wind_speed, n_spd_bins + 1)

    mesh = None
    for i, site in enumerate(sites):
        ax = axes[i]
        site_df = h2s_df[h2s_df["site_name"] == site].copy()

        # Filter to latest 12 hours for this site
        if not site_df.empty:
            latest = site_df["time"].max()
            cutoff = latest - pd.Timedelta(hours=12)
            site_df = site_df[site_df["time"] >= cutoff]

        if site_df.empty or "wind_direction_10m" not in site_df.columns:
            ax.text(0, 0, "No data", ha="center", va="center", fontsize=10, color="#666")
            ax.set_yticklabels([])
            ax.set_xticklabels([])
            continue

        # Convert wind direction from meteorological degrees to radians
        wd_rad = np.deg2rad(90 - site_df["wind_direction_10m"].values) % (2 * np.pi)
        ws = site_df["wind_speed_10m"].values.clip(0, max_wind_speed)
        h2s = site_df["H2S"].values

        # Bin the data
        grid = np.full((n_spd_bins, n_dir_bins), np.nan)
        counts = np.zeros((n_spd_bins, n_dir_bins), dtype=int)

        dir_idx = np.clip(np.digitize(wd_rad, dir_bins) - 1, 0, n_dir_bins - 1)
        spd_idx = np.clip(np.digitize(ws, spd_bins) - 1, 0, n_spd_bins - 1)

        for j in range(len(h2s)):
            di, si = dir_idx[j], spd_idx[j]
            if np.isnan(h2s[j]):
                continue
            if counts[si, di] == 0:
                grid[si, di] = h2s[j]
            else:
                grid[si, di] = (grid[si, di] * counts[si, di] + h2s[j]) / (counts[si, di] + 1)
            counts[si, di] += 1

        theta, r = np.meshgrid(dir_bins, spd_bins)
        mesh = ax.pcolormesh(theta, r, grid, cmap=_CMAP, norm=_NORM, shading="flat")

        # North at top, clockwise
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_yticklabels([])
        ax.set_xticks(np.linspace(0, 2 * np.pi, 8, endpoint=False))
        ax.set_xticklabels(["N", "", "E", "", "S", "", "W", ""])
        ax.tick_params(axis="x", labelsize=8)
        ax.set_xlabel(site, fontsize=9, labelpad=10)

    # Hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    if mesh is not None:
        cb = fig.colorbar(mesh, ax=axes[:n_sites], label="H2S (ppb)", shrink=0.7, pad=0.08)
        cb.set_ticks([2.5, 17.5, 65])
        cb.set_ticklabels(["<5\nGreen", "5–30\nYellow", "≥30\nOrange"])

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def create_polar_heatmap(h2s_df: pd.DataFrame) -> pn.pane.PNG | pn.pane.HTML:
    """Build polar pcolormesh plots of H2S by wind direction and speed."""
    img_bytes = _render_polar_figure(h2s_df)

    if img_bytes is None:
        return pn.pane.HTML(
            "<div style='text-align:center;color:#666;padding:40px;'>"
            "No data available</div>",
            height=420,
            sizing_mode="stretch_width",
        )

    return pn.pane.PNG(
        io.BytesIO(img_bytes),
        sizing_mode="stretch_width",
    )
