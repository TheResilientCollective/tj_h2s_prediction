"""Dispersion model visualization generators for S3 storage.

Generates heatmaps and source maps for Gaussian forward forecasts.
Returns plots as BytesIO objects for direct S3 upload.
"""

from io import BytesIO
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap


def generate_concentration_heatmap(
    grid_data: dict,
    title: str = "H2S Concentration Forecast",
    vmax: Optional[float] = None,
    bounds: Optional[dict] = None,
) -> BytesIO:
    """Generate concentration heatmap from GeoDemic GridData.

    Args:
        grid_data: GridData dict with 'data', 'lat_centers', 'lon_centers', 'bounds'.
            bounds dict uses keys: 'north', 'south', 'east', 'west'
        title: Plot title
        vmax: Maximum value for colorbar (auto-scales if None)
        bounds: Optional bounds override (uses grid_data bounds if None).
            Dict with keys: 'north', 'south', 'east', 'west'

    Returns:
        BytesIO object with PNG image data
    """
    concentration = np.array(grid_data["data"])
    lat_centers = np.array(grid_data["lat_centers"])
    lon_centers = np.array(grid_data["lon_centers"])

    # Use provided bounds or fall back to grid_data bounds
    plot_bounds = bounds if bounds is not None else grid_data["bounds"]

    # Auto-scale vmax if not provided
    if vmax is None:
        vmax = max(np.percentile(concentration[concentration > 0], 95), 10.0) if concentration.max() > 0 else 30.0

    # Create custom colormap: white → yellow → orange → red
    colors = ['white', '#FFFF99', '#FFD700', '#FFA500', '#FF4500', '#8B0000']
    n_bins = 100
    cmap = LinearSegmentedColormap.from_list('h2s', colors, N=n_bins)

    fig, ax = plt.subplots(figsize=(12, 10))

    # Plot concentration
    lon_mesh, lat_mesh = np.meshgrid(lon_centers, lat_centers)
    im = ax.pcolormesh(
        lon_mesh, lat_mesh, concentration,
        cmap=cmap,
        vmin=0,
        vmax=vmax,
        alpha=0.9,
        shading='auto',
    )

    # Set extent and aspect (bounds uses north/south/east/west keys)
    ax.set_xlim(plot_bounds["west"], plot_bounds["east"])
    ax.set_ylim(plot_bounds["south"], plot_bounds["north"])
    ax.set_aspect('equal', adjustable='box')

    # Labels and grid
    ax.set_xlabel('Longitude', fontsize=11)
    ax.set_ylabel('Latitude', fontsize=11)
    ax.grid(True, alpha=0.3, linewidth=0.5)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, orientation='vertical', pad=0.02, shrink=0.8)
    cbar.set_label('H₂S Concentration (ppb)', fontsize=12)

    # Title and metadata
    metadata = grid_data.get("metadata", {})
    time_str = metadata.get("time", "")
    wind_speed = metadata.get("wind_speed_ms", "")
    wind_dir = metadata.get("wind_direction_deg", "")

    subtitle = f"Time: {time_str}"
    if wind_speed and wind_dir:
        subtitle += f" | Wind: {wind_speed} m/s @ {wind_dir}°"

    ax.set_title(f"{title}\n{subtitle}", fontsize=13, fontweight='bold', pad=15)

    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return buf


def generate_source_emission_map(
    sources: dict,
    emission_rates: dict,
    sensors: Optional[dict] = None,
    title: str = "H2S Source Emission Rates",
    bounds: Optional[dict] = None,
) -> BytesIO:
    """Generate map showing source locations with emission rates.

    Args:
        sources: Dict of {source_name: {"lat": ..., "lon": ..., "name": ...}}
        emission_rates: Dict of {source_name: emission_rate_g_s}
        sensors: Optional dict of {sensor_name: {"lat": ..., "lon": ...}}
        title: Plot title
        bounds: Optional bounds dict with keys: 'north', 'south', 'east', 'west'
            (auto-computed from source/sensor locations if not provided)

    Returns:
        BytesIO object with PNG image data
    """
    # Auto-compute bounds if not provided
    if bounds is None:
        lats = [s["lat"] for s in sources.values()]
        lons = [s["lon"] for s in sources.values()]
        if sensors:
            lats.extend([s["lat"] for s in sensors.values()])
            lons.extend([s["lon"] for s in sensors.values()])

        lat_range = max(lats) - min(lats)
        lon_range = max(lons) - min(lons)
        bounds = {
            "south": min(lats) - 0.15 * lat_range,
            "north": max(lats) + 0.15 * lat_range,
            "west": min(lons) - 0.15 * lon_range,
            "east": max(lons) + 0.15 * lon_range,
        }

    fig, ax = plt.subplots(figsize=(12, 10))

    # Background color for land/ocean effect
    ax.set_facecolor('#E0F7FF')  # Light blue background

    # Plot sources with size proportional to emission rate
    max_rate = max(emission_rates.values()) if emission_rates else 1.0
    min_rate = min([v for v in emission_rates.values() if v > 0], default=0.0)

    for src_name, src_info in sources.items():
        rate = emission_rates.get(src_name, 0.0)
        if rate <= 0:
            continue

        # Size: scale from 100 to 1000
        if max_rate > min_rate:
            size = 100 + 900 * (rate - min_rate) / (max_rate - min_rate)
        else:
            size = 500

        # Color by magnitude
        if rate > 20:
            color = '#8B0000'  # Dark red (high)
        elif rate > 10:
            color = '#FF4500'  # Orange-red (medium)
        elif rate > 5:
            color = '#FFA500'  # Orange (moderate)
        else:
            color = '#FFD700'  # Gold (low)

        ax.scatter(
            src_info["lon"], src_info["lat"],
            s=size,
            c=color,
            alpha=0.7,
            edgecolors='black',
            linewidths=1.5,
            zorder=5,
        )

        # Label with source name and rate
        display_name = src_info.get('name', src_name)
        if len(display_name) > 20:  # Truncate long names
            display_name = display_name[:17] + '...'
        label = f"{display_name}\n{rate:.1f} g/s"
        ax.text(
            src_info["lon"], src_info["lat"] + 0.002,
            label,
            fontsize=7,
            ha='center',
            va='bottom',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85, edgecolor='gray', linewidth=0.5),
            zorder=6,
        )

    # Plot sensors if provided
    if sensors:
        for sensor_name, sensor_info in sensors.items():
            ax.scatter(
                sensor_info["lon"], sensor_info["lat"],
                s=200,
                c='blue',
                marker='^',
                alpha=0.9,
                edgecolors='black',
                linewidths=1.5,
                zorder=7,
                label='Sensor' if sensor_name == list(sensors.keys())[0] else "",
            )
            ax.text(
                sensor_info["lon"], sensor_info["lat"] - 0.002,
                sensor_name.replace(' - ', '\n'),
                fontsize=8,
                ha='center',
                va='top',
                fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.95, edgecolor='blue', linewidth=1),
                zorder=8,
            )

    # Set extent and aspect
    ax.set_xlim(bounds["west"], bounds["east"])
    ax.set_ylim(bounds["south"], bounds["north"])
    ax.set_aspect('equal', adjustable='box')

    # Labels and grid
    ax.set_xlabel('Longitude', fontsize=11)
    ax.set_ylabel('Latitude', fontsize=11)
    ax.grid(True, alpha=0.4, linewidth=0.5, linestyle='--')

    # Legend for source sizes
    legend_elements = [
        mpatches.Patch(color='#8B0000', alpha=0.7, label='> 20 g/s'),
        mpatches.Patch(color='#FF4500', alpha=0.7, label='10-20 g/s'),
        mpatches.Patch(color='#FFA500', alpha=0.7, label='5-10 g/s'),
        mpatches.Patch(color='#FFD700', alpha=0.7, label='< 5 g/s'),
    ]
    if sensors:
        legend_elements.append(mpatches.Patch(color='blue', label='Sensors'))

    ax.legend(handles=legend_elements, loc='upper right', fontsize=9, framealpha=0.9)

    # Title
    total_emission = sum(emission_rates.values())
    subtitle = f"Total Emission Rate: {total_emission:.1f} g/s"
    ax.set_title(f"{title}\n{subtitle}", fontsize=13, fontweight='bold', pad=15)

    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return buf


def generate_peak_concentration_timeseries(
    forecast_result,
    title: str = "Peak H2S Concentration Forecast",
    threshold_lines: Optional[dict] = None,
) -> BytesIO:
    """Generate timeseries plot of peak concentrations at each sensor.

    Args:
        forecast_result: ForwardModelResult object
        title: Plot title
        threshold_lines: Optional dict of {label: threshold_ppb} for horizontal lines

    Returns:
        BytesIO object with PNG image data
    """
    if threshold_lines is None:
        threshold_lines = {"Watch": 30.0, "Critical": 100.0}

    df = forecast_result.to_dataframe()

    fig, ax = plt.subplots(figsize=(14, 6))

    # Plot each sensor
    colors = {'NESTOR - BES': '#1f77b4', 'IB CIVIC CTR': '#ff7f0e', 'SAN YSIDRO': '#2ca02c'}
    for sensor in df['sensor'].unique():
        sensor_data = df[df['sensor'] == sensor].copy()
        ax.plot(
            sensor_data['time'],
            sensor_data['predicted_ppb'],
            label=sensor,
            color=colors.get(sensor, 'gray'),
            linewidth=2,
            marker='o',
            markersize=3,
        )

    # Add threshold lines
    for label, threshold in threshold_lines.items():
        ax.axhline(
            threshold,
            color='red' if 'critical' in label.lower() else 'orange',
            linestyle='--',
            linewidth=1.5,
            alpha=0.7,
            label=f"{label} ({threshold} ppb)",
        )

    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('H₂S Concentration (ppb)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return buf