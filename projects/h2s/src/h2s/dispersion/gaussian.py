"""
Gaussian plume forward model for H2S dispersion forecasting.

Implements a time-varying Gaussian plume with:
  - Multiple simultaneous point sources (East/West/South zones)
  - Pasquill-Gifford stability class derived from wind speed + is_night
  - Hourly met updates from a forecast or observation DataFrame
  - Sensor-point concentration extraction at NB, IB, SY

Emission rates must be in g/s throughout. Always derive Q from backward
inversion (Lagrangian ensemble or HYSPLIT cdump) before operational use.
Default calibration values: east=20, west=10, south=137 g/s (March 13 2026 event).

Adapted from modeling_sources/gaussian_forward.py for use as a library module.
Key changes:
  - Dagster @asset removed (lives in h2s_dispersion_pipeline.py)
  - Emission rate units canonicalized to g/s (removing g/hr label confusion)
  - ForwardModelResult.to_json() uses "emission_rates_g_s" key
"""

import json
import argparse
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional


# --- Sensor locations ---

SENSORS = {
    "NESTOR - BES": {"lat": 32.567097, "lon": -117.090656},
    "IB CIVIC CTR": {"lat": 32.576139, "lon": -117.115361},
    "SAN YSIDRO":   {"lat": 32.552794, "lon": -117.047286},
}

# Three emission source zones (COARSE: for fast regional forecasts)
SOURCES = {
    "east":  {"lat": 32.541, "lon": -117.058, "name": "Stewart's Drain / Dairy Mart"},
    "west":  {"lat": 32.570, "lon": -117.127, "name": "Oneonta Slough / PS"},
    "south": {"lat": 32.537, "lon": -117.099, "name": "Goat Canyon / cross-border"},
}

# Sixteen candidate sources (DETAILED: for high-resolution spatial forecasts)
# Same source locations as lagrangian.py backward model
CANDIDATE_SOURCES = {
    "stewarts_drain":       {"lat": 32.54064,  "lon": -117.05801,  "name": "Stewart's Drain"},
    "smugglers_gulch":      {"lat": 32.5377,   "lon": -117.08623,  "name": "Smuggler's Gulch"},
    "hollister_ps":         {"lat": 32.5476,   "lon": -117.088374, "name": "Hollister PS"},
    "goat_canyon":          {"lat": 32.5369,   "lon": -117.09916,  "name": "Goat Canyon"},
    "goat_canyon_ps":       {"lat": 32.543476, "lon": -117.108026, "name": "Goat Canyon PS"},
    "del_sol_canyon":       {"lat": 32.5393,   "lon": -117.06885,  "name": "Del Sol Canyon"},
    "silva_drain":          {"lat": 32.539743, "lon": -117.064269, "name": "Silva Drain"},
    "saturn_blvd_bridge":   {"lat": 32.559383, "lon": -117.092992, "name": "Saturn Blvd Bridge"},
    "hollister_bridge_n":   {"lat": 32.554177, "lon": -117.084135, "name": "Hollister Bridge N"},
    "hollister_bridge_s":   {"lat": 32.551466, "lon": -117.084021, "name": "Hollister Bridge S"},
    "dairy_mart_bridge":    {"lat": 32.548531, "lon": -117.064293, "name": "Dairy Mart Bridge"},
    "oneonta_slough":       {"lat": 32.570082, "lon": -117.126724, "name": "Oneonta Slough"},
    "tijuana_beach_outlet": {"lat": 32.556206, "lon": -117.126178, "name": "Tijuana Beach Outlet"},
    "tj_crossing_cdlp_w":   {"lat": 32.542103, "lon": -117.054117, "name": "TJ Crossing CDLP W"},
    "tj_crossing_cdlp_e":   {"lat": 32.542166, "lon": -117.050325, "name": "TJ Crossing CDLP E"},
    "sd_bay_otay_outlet":   {"lat": 32.594557, "lon": -117.113542, "name": "SD Bay Otay Outlet"},
    "sd_bay_fruitdale":     {"lat": 32.595305, "lon": -117.091869, "name": "SD Bay Fruitdale"},
}

# Pasquill-Gifford dispersion coefficients (Slade 1968, rural)
# sigma_y = a * x^b, sigma_z = c * x^d  (x in km, sigma in m)
PG_PARAMS = {
    "A": (0.22, 0.894, 0.20, 0.894),
    "B": (0.16, 0.894, 0.12, 0.894),
    "C": (0.11, 0.894, 0.08, 0.894),
    "D": (0.08, 0.894, 0.06, 0.894),
    "E": (0.06, 0.894, 0.03, 0.894),
    "F": (0.04, 0.894, 0.016, 0.894),
}

M_PER_DEG_LAT = 111_320.0
M_PER_DEG_LON = 111_320.0 * np.cos(np.radians(32.55))
MW_H2S = 34.08
MOLAR_VOL_STP = 24.45  # L/mol at 20°C


def stability_class(wind_speed_ms: float, is_night: bool) -> str:
    if is_night:
        if wind_speed_ms < 2.0:
            return "F"
        elif wind_speed_ms < 3.0:
            return "E"
        else:
            return "D"
    else:
        if wind_speed_ms < 2.0:
            return "B"
        elif wind_speed_ms < 5.0:
            return "C"
        else:
            return "D"


def pg_sigmas(stab: str, x_km: float) -> tuple[float, float]:
    a_y, b_y, a_z, b_z = PG_PARAMS[stab]
    # Near-field floor of 100 m (was 10 m). At < 100 m the Gaussian plume
    # ground-level equation is not physical: σ_y, σ_z shrink below a cell-width
    # and concentration blows up. This bounds the grid cell co-located with a
    # source to a realistic value. Sensor receptors are always > 300 m from
    # the nearest source so timeseries forecasts are unaffected.
    x_km = max(x_km, 0.1)
    sigma_y = a_y * (x_km ** b_y) * 1000.0
    sigma_z = a_z * (x_km ** b_z) * 1000.0
    return sigma_y, sigma_z


def ug_m3_to_ppb(conc_ug_m3: float, temp_c: float = 20.0) -> float:
    molar_vol = MOLAR_VOL_STP * (273.15 + temp_c) / 293.15
    return conc_ug_m3 * molar_vol / MW_H2S


def gaussian_plume_concentration(
    source_lat: float, source_lon: float,
    emission_rate_g_s: float,
    receptor_lat: float, receptor_lon: float,
    wind_u: float, wind_v: float,
    stab: str,
    stack_height: float = 2.0,
    receptor_height: float = 1.5,
    sigma_theta_deg: float = 20.0,
) -> float:
    """
    Steady-state Gaussian plume concentration (μg/m³) at receptor.

    Includes Gifford (1961) wind meandering correction for light winds — critical
    for stable nocturnal conditions (class E/F) where slow wind direction
    oscillations broaden the time-averaged footprint.

    Args:
        emission_rate_g_s: Emission rate in g/s.

    Returns 0 if receptor is upwind of source.
    """
    wind_speed = np.sqrt(wind_u**2 + wind_v**2)
    wind_speed_eff = max(wind_speed, 0.3)

    if wind_speed < 0.1:
        dist_m = max(np.sqrt(
            ((receptor_lat - source_lat) * M_PER_DEG_LAT) ** 2 +
            ((receptor_lon - source_lon) * M_PER_DEG_LON) ** 2
        ), 50.0)
        mixing_height = 50.0
        conc = emission_rate_g_s * 1e6 / (dist_m * dist_m * mixing_height)
        return max(conc, 0.0)

    u_hat = np.array([wind_u, wind_v]) / wind_speed
    dx = (receptor_lon - source_lon) * M_PER_DEG_LON
    dy = (receptor_lat - source_lat) * M_PER_DEG_LAT
    r = np.array([dx, dy])

    x_down = float(np.dot(r, u_hat))
    if x_down <= 0:
        return 0.0

    y_cross = float(u_hat[0] * r[1] - u_hat[1] * r[0])
    x_km = x_down / 1000.0
    sigma_y, sigma_z = pg_sigmas(stab, x_km)

    sigma_theta_rad = np.radians(sigma_theta_deg)
    sigma_y_eff = np.sqrt(sigma_y**2 + (sigma_theta_rad * x_down)**2)

    Q = emission_rate_g_s * 1e6
    exp_y = np.exp(-0.5 * (y_cross / sigma_y_eff) ** 2)
    exp_z_direct  = np.exp(-0.5 * ((receptor_height - stack_height) / sigma_z) ** 2)
    exp_z_reflect = np.exp(-0.5 * ((receptor_height + stack_height) / sigma_z) ** 2)

    conc = (Q / (np.pi * wind_speed_eff * sigma_y_eff * sigma_z)) * exp_y * (exp_z_direct + exp_z_reflect)
    return max(conc, 0.0)


@dataclass
class ForwardModelResult:
    times: list
    concentrations: dict   # sensor_name → list of ppb values
    emission_rates_g_s: dict   # zone → g/s
    metadata: dict = field(default_factory=dict)

    def to_dataframe(self) -> pd.DataFrame:
        rows = []
        for i, t in enumerate(self.times):
            for sensor, ppb_vals in self.concentrations.items():
                rows.append({"time": t, "sensor": sensor, "predicted_ppb": ppb_vals[i]})
        return pd.DataFrame(rows)

    def to_json(self) -> str:
        d = {
            "emission_rates_g_s": self.emission_rates_g_s,
            "metadata": self.metadata,
            "timeseries": {},
        }
        for sensor, vals in self.concentrations.items():
            d["timeseries"][sensor] = [
                {"time": str(t), "predicted_ppb": round(v, 3) if not np.isnan(v) else None}
                for t, v in zip(self.times, vals)
            ]
        return json.dumps(d, indent=2)


def run_forward_model(
    df: pd.DataFrame,
    emission_rates_g_s: dict[str, float],
    start_time: pd.Timestamp,
    hours: int = 72,
    ref_sensor: str = "NESTOR - BES",
) -> ForwardModelResult:
    """
    Run Gaussian plume forward model over a time window.

    Args:
        df: DataFrame with time, site_name, wind_speed_10m, wind_direction_10m,
            temperature_2m, is_night columns. Can be forecast or observation data.
        emission_rates_g_s: {"east": Q, "west": Q, "south": Q} in g/s.
        start_time: Forecast start (tz-aware).
        hours: Forecast duration in hours.
        ref_sensor: Sensor to use for met when per-sensor data is unavailable.
    """
    times = pd.date_range(start_time, periods=hours, freq="1h")
    concentrations = {sname: [] for sname in SENSORS}

    for t in times:
        for sensor_name, sensor_coords in SENSORS.items():
            row = df[(df["time"] == t) & (df["site_name"] == sensor_name)]
            if row.empty:
                row = df[(df["time"] == t) & (df["site_name"] == ref_sensor)]
            if row.empty:
                concentrations[sensor_name].append(np.nan)
                continue

            ws = float(row["wind_speed_10m"].iloc[0])
            wd_deg = float(row["wind_direction_10m"].iloc[0])
            temp_c = float(row["temperature_2m"].iloc[0])
            is_night = bool(row["is_night"].iloc[0])

            wd_rad = np.radians(wd_deg)
            u = -ws * np.sin(wd_rad)
            v = -ws * np.cos(wd_rad)
            stab = stability_class(ws, is_night)

            total_ug_m3 = 0.0
            for zone, src in SOURCES.items():
                q = emission_rates_g_s.get(zone, 0.0)
                if q <= 0:
                    continue
                total_ug_m3 += gaussian_plume_concentration(
                    source_lat=src["lat"],
                    source_lon=src["lon"],
                    emission_rate_g_s=q,
                    receptor_lat=sensor_coords["lat"],
                    receptor_lon=sensor_coords["lon"],
                    wind_u=u, wind_v=v,
                    stab=stab,
                )

            concentrations[sensor_name].append(round(float(ug_m3_to_ppb(total_ug_m3, temp_c)), 3))

    return ForwardModelResult(
        times=list(times),
        concentrations=concentrations,
        emission_rates_g_s=emission_rates_g_s,
        metadata={
            "model": "Gaussian plume (Pasquill-Gifford, Slade 1968)",
            "start_time": str(start_time),
            "hours": hours,
            "sources": SOURCES,
        },
    )


def run_forward_model_detailed(
    df: pd.DataFrame,
    emission_rates_per_source_g_s: dict[str, float],
    start_time: pd.Timestamp,
    hours: int = 72,
    ref_sensor: str = "NESTOR - BES",
) -> ForwardModelResult:
    """
    Run Gaussian plume forward model with 16 individual candidate sources.

    This provides a more spatially accurate representation of the source
    distribution compared to the 3-zone aggregated model.

    Args:
        df: DataFrame with time, site_name, wind_speed_10m, wind_direction_10m,
            temperature_2m, is_night columns. Can be forecast or observation data.
        emission_rates_per_source_g_s: {"stewarts_drain": Q, "goat_canyon": Q, ...} in g/s.
        start_time: Forecast start (tz-aware).
        hours: Forecast duration in hours.
        ref_sensor: Sensor to use for met when per-sensor data is unavailable.
    """
    times = pd.date_range(start_time, periods=hours, freq="1h")
    concentrations = {sname: [] for sname in SENSORS}

    for t in times:
        for sensor_name, sensor_coords in SENSORS.items():
            row = df[(df["time"] == t) & (df["site_name"] == sensor_name)]
            if row.empty:
                row = df[(df["time"] == t) & (df["site_name"] == ref_sensor)]
            if row.empty:
                concentrations[sensor_name].append(np.nan)
                continue

            ws = float(row["wind_speed_10m"].iloc[0])
            wd_deg = float(row["wind_direction_10m"].iloc[0])
            temp_c = float(row["temperature_2m"].iloc[0])
            is_night = bool(row["is_night"].iloc[0])

            wd_rad = np.radians(wd_deg)
            u = -ws * np.sin(wd_rad)
            v = -ws * np.cos(wd_rad)
            stab = stability_class(ws, is_night)

            total_ug_m3 = 0.0
            for src_name, src in CANDIDATE_SOURCES.items():
                q = emission_rates_per_source_g_s.get(src_name, 0.0)
                if q <= 0:
                    continue
                total_ug_m3 += gaussian_plume_concentration(
                    source_lat=src["lat"],
                    source_lon=src["lon"],
                    emission_rate_g_s=q,
                    receptor_lat=sensor_coords["lat"],
                    receptor_lon=sensor_coords["lon"],
                    wind_u=u, wind_v=v,
                    stab=stab,
                )

            concentrations[sensor_name].append(round(float(ug_m3_to_ppb(total_ug_m3, temp_c)), 3))

    return ForwardModelResult(
        times=list(times),
        concentrations=concentrations,
        emission_rates_g_s=emission_rates_per_source_g_s,
        metadata={
            "model": "Gaussian plume (Pasquill-Gifford, Slade 1968, 16-source detailed)",
            "start_time": str(start_time),
            "hours": hours,
            "n_sources": len(CANDIDATE_SOURCES),
            "sources": CANDIDATE_SOURCES,
        },
    )


def _build_grid_data(
    concentration_grid: np.ndarray,
    confidence_grid: np.ndarray,
    lat_centers: np.ndarray,
    lon_centers: np.ndarray,
    bounds: dict,
    resolution_meters: int,
    metadata: dict,
) -> dict:
    """Build a GeoDemic-compatible GridData dict.

    Matches the frontend ``GridData`` interface::

        { bounds, data[][], confidence[][], lat_centers[], lon_centers[],
          resolution_meters, metadata }
    """
    return {
        "bounds": bounds,
        "data": concentration_grid.tolist(),
        "confidence": confidence_grid.tolist(),
        "lat_centers": np.round(lat_centers, 6).tolist(),
        "lon_centers": np.round(lon_centers, 6).tolist(),
        "resolution_meters": resolution_meters,
        "metadata": metadata,
    }


def _confidence_from_distance(
    lat_grid: np.ndarray,
    lon_grid: np.ndarray,
    sensor_coords: list[tuple[float, float]],
    forecast_lead_h: int = 0,
) -> np.ndarray:
    """Heuristic confidence score (0-1) based on distance to nearest sensor and forecast lead time.

    Close to sensors → high confidence; far away → lower confidence.
    Longer forecast lead → lower confidence.
    """
    min_dist_m = np.full(lat_grid.shape, np.inf)
    for s_lat, s_lon in sensor_coords:
        d_lat = (lat_grid - s_lat) * M_PER_DEG_LAT
        d_lon = (lon_grid - s_lon) * M_PER_DEG_LON
        dist = np.sqrt(d_lat ** 2 + d_lon ** 2)
        min_dist_m = np.minimum(min_dist_m, dist)

    # Distance decay: 1.0 at sensor, 0.3 at 10 km
    dist_conf = np.clip(1.0 - 0.07 * (min_dist_m / 1000.0), 0.3, 1.0)

    # Lead-time decay: -0.02 per hour of forecast lead
    lead_decay = max(1.0 - 0.02 * forecast_lead_h, 0.3)

    return dist_conf * lead_decay


def run_forward_model_gridded(
    df: pd.DataFrame,
    emission_rates_g_s: dict[str, float],
    start_time: pd.Timestamp,
    hours: int = 72,
    ref_sensor: str = "NESTOR - BES",
) -> list[dict]:
    """Run Gaussian plume forward model and return per-hour GeoDemic GridData dicts.

    Evaluates ``gaussian_plume_concentration()`` at every cell of the unified
    grid (from ``grid_config``) for each forecast hour.  Returns a list of
    GridData dicts (one per hour) suitable for direct upload to S3 and
    consumption by GeoDemic's ``HeatMapLayer``.

    Args:
        df: Met DataFrame (same format as ``run_forward_model``).
        emission_rates_g_s: {"east": Q, "west": Q, "south": Q} in g/s.
        start_time: Forecast start (tz-aware).
        hours: Forecast duration in hours.
        ref_sensor: Fallback sensor for met when per-sensor data is missing.

    Returns:
        List of GridData dicts, one per forecast hour.
    """
    from h2s.dispersion.grid_config import (
        GRID_BOUNDS,
        GRID_LAT_CENTERS,
        GRID_LON_CENTERS,
        GRID_NROWS,
        GRID_NCOLS,
        GRID_RESOLUTION_METERS,
    )

    times = pd.date_range(start_time, periods=hours, freq="1h")

    # Pre-build 2-D coordinate meshes (rows=lat, cols=lon)
    lon_mesh, lat_mesh = np.meshgrid(GRID_LON_CENTERS, GRID_LAT_CENTERS)

    sensor_coords = [(s["lat"], s["lon"]) for s in SENSORS.values()]

    frames: list[dict] = []

    for hour_idx, t in enumerate(times):
        # Get met for this timestep (use ref_sensor as representative)
        row = df[(df["time"] == t) & (df["site_name"] == ref_sensor)]
        if row.empty:
            # Try any sensor for this timestep
            row = df[df["time"] == t]
        if row.empty:
            # No met available — produce an empty grid for this hour
            frames.append(_build_grid_data(
                concentration_grid=np.zeros((GRID_NROWS, GRID_NCOLS)),
                confidence_grid=np.full((GRID_NROWS, GRID_NCOLS), 0.1),
                lat_centers=GRID_LAT_CENTERS,
                lon_centers=GRID_LON_CENTERS,
                bounds=GRID_BOUNDS,
                resolution_meters=GRID_RESOLUTION_METERS,
                metadata={"time": str(t), "hour_index": hour_idx, "status": "no_met"},
            ))
            continue

        row = row.iloc[0]
        ws = float(row["wind_speed_10m"])
        wd_deg = float(row["wind_direction_10m"])
        temp_c = float(row.get("temperature_2m", 20.0))
        is_night_val = bool(row.get("is_night", 0))

        wd_rad = np.radians(wd_deg)
        wind_u = -ws * np.sin(wd_rad)
        wind_v = -ws * np.cos(wd_rad)
        stab = stability_class(ws, is_night_val)

        # Accumulate concentration from all source zones
        total_ppb = np.zeros((GRID_NROWS, GRID_NCOLS))

        for zone, src in SOURCES.items():
            q = emission_rates_g_s.get(zone, 0.0)
            if q <= 0:
                continue

            # Vectorized plume evaluation over the grid
            conc_ug = _gaussian_plume_grid(
                source_lat=src["lat"],
                source_lon=src["lon"],
                emission_rate_g_s=q,
                lat_grid=lat_mesh,
                lon_grid=lon_mesh,
                wind_u=wind_u,
                wind_v=wind_v,
                stab=stab,
            )
            total_ppb += _ug_grid_to_ppb(conc_ug, temp_c)

        confidence = _confidence_from_distance(lat_mesh, lon_mesh, sensor_coords, forecast_lead_h=hour_idx)

        frames.append(_build_grid_data(
            concentration_grid=np.round(total_ppb, 3),
            confidence_grid=np.round(confidence, 3),
            lat_centers=GRID_LAT_CENTERS,
            lon_centers=GRID_LON_CENTERS,
            bounds=GRID_BOUNDS,
            resolution_meters=GRID_RESOLUTION_METERS,
            metadata={
                "time": str(t),
                "hour_index": hour_idx,
                "wind_speed_ms": round(ws, 2),
                "wind_direction_deg": round(wd_deg, 1),
                "stability_class": stab,
                "emission_rates_g_s": {k: float(v) for k, v in emission_rates_g_s.items()},
            },
        ))

    return frames


GRID_BASELINE_SCALE: float = 0.1
"""Default scale applied to per-source emission rates for the *gridded* forecast.

The inversion (``emission_rate_inversion``) calibrates emission rates against
acute events (e.g. March 13 2026, 394 ppb @ NESTOR-BES) so sensor-level
timeseries match observed peaks. Those same rates produce visually saturated
grid maps under typical (non-event) conditions. The grid product multiplies
rates by this factor to show a baseline-like spatial footprint while leaving
the sensor-level timeseries forecast untouched. Set to 1.0 to render at the
full event-calibrated level.
"""

GRID_PPB_CLIP: float = 500.0
"""Upper clip applied to every grid cell in the gridded forecast (ppb).

Guards against unphysical near-source singularities and keeps the tile color
ramp legible. 500 ppb is ~5× the worst recorded event-level value at a
community sensor; anything above is numerical artefact, not signal.
"""


def run_forward_model_gridded_detailed(
    df: pd.DataFrame,
    emission_rates_per_source_g_s: dict[str, float],
    start_time: pd.Timestamp,
    hours: int = 72,
    ref_sensor: str = "NESTOR - BES",
    baseline_scale: float = GRID_BASELINE_SCALE,
    ppb_clip: float = GRID_PPB_CLIP,
) -> list[dict]:
    """Run Gaussian plume forward model with 16 sources and return per-hour GeoDemic GridData dicts.

    This is the detailed version of ``run_forward_model_gridded()`` that uses
    individual candidate sources instead of 3 aggregate zones. Provides more
    spatially accurate concentration fields at the cost of ~5× slower execution.

    Evaluates ``gaussian_plume_concentration()`` at every cell of the unified
    grid (from ``grid_config``) for each forecast hour.  Returns a list of
    GridData dicts (one per hour) suitable for direct upload to S3 and
    consumption by GeoDemic's ``HeatMapLayer``.

    Args:
        df: Met DataFrame (same format as ``run_forward_model``).
        emission_rates_per_source_g_s: {"stewarts_drain": Q, "goat_canyon": Q, ...} in g/s.
        start_time: Forecast start (tz-aware).
        hours: Forecast duration in hours.
        ref_sensor: Fallback sensor for met when per-sensor data is missing.
        baseline_scale: Multiplier applied to each per-source rate before plume
            evaluation. Default 0.1 produces a non-event baseline (see
            ``GRID_BASELINE_SCALE``). Pass 1.0 to render at event-calibrated
            levels.
        ppb_clip: Per-cell ppb clip to suppress near-source singularities.
            Default 500 ppb (see ``GRID_PPB_CLIP``).

    Returns:
        List of GridData dicts, one per forecast hour.
    """
    from h2s.dispersion.grid_config import (
        GRID_BOUNDS,
        GRID_LAT_CENTERS,
        GRID_LON_CENTERS,
        GRID_NROWS,
        GRID_NCOLS,
        GRID_RESOLUTION_METERS,
    )

    times = pd.date_range(start_time, periods=hours, freq="1h")

    # Pre-build 2-D coordinate meshes (rows=lat, cols=lon)
    lon_mesh, lat_mesh = np.meshgrid(GRID_LON_CENTERS, GRID_LAT_CENTERS)

    sensor_coords = [(s["lat"], s["lon"]) for s in SENSORS.values()]

    frames: list[dict] = []

    for hour_idx, t in enumerate(times):
        # Get met for this timestep (use ref_sensor as representative)
        row = df[(df["time"] == t) & (df["site_name"] == ref_sensor)]
        if row.empty:
            # Try any sensor for this timestep
            row = df[df["time"] == t]
        if row.empty:
            # No met available — produce an empty grid for this hour
            frames.append(_build_grid_data(
                concentration_grid=np.zeros((GRID_NROWS, GRID_NCOLS)),
                confidence_grid=np.full((GRID_NROWS, GRID_NCOLS), 0.1),
                lat_centers=GRID_LAT_CENTERS,
                lon_centers=GRID_LON_CENTERS,
                bounds=GRID_BOUNDS,
                resolution_meters=GRID_RESOLUTION_METERS,
                metadata={"time": str(t), "hour_index": hour_idx, "status": "no_met"},
            ))
            continue

        row = row.iloc[0]
        ws = float(row["wind_speed_10m"])
        wd_deg = float(row["wind_direction_10m"])
        temp_c = float(row.get("temperature_2m", 20.0))
        is_night_val = bool(row.get("is_night", 0))

        wd_rad = np.radians(wd_deg)
        wind_u = -ws * np.sin(wd_rad)
        wind_v = -ws * np.cos(wd_rad)
        stab = stability_class(ws, is_night_val)

        # Accumulate concentration from all 16 candidate sources
        total_ppb = np.zeros((GRID_NROWS, GRID_NCOLS))

        for src_name, src in CANDIDATE_SOURCES.items():
            q = emission_rates_per_source_g_s.get(src_name, 0.0) * baseline_scale
            if q <= 0:
                continue

            # Vectorized plume evaluation over the grid
            conc_ug = _gaussian_plume_grid(
                source_lat=src["lat"],
                source_lon=src["lon"],
                emission_rate_g_s=q,
                lat_grid=lat_mesh,
                lon_grid=lon_mesh,
                wind_u=wind_u,
                wind_v=wind_v,
                stab=stab,
            )
            total_ppb += _ug_grid_to_ppb(conc_ug, temp_c)

        # Clip near-source singularities to a physically defensible ceiling.
        np.minimum(total_ppb, ppb_clip, out=total_ppb)

        confidence = _confidence_from_distance(lat_mesh, lon_mesh, sensor_coords, forecast_lead_h=hour_idx)

        # Compute top contributing sources for this hour (for metadata)
        source_contributions = {}
        for src_name, src in CANDIDATE_SOURCES.items():
            q = emission_rates_per_source_g_s.get(src_name, 0.0)
            if q > 0:
                source_contributions[src_name] = round(q, 2)
        top_3_sources = dict(sorted(source_contributions.items(), key=lambda x: -x[1])[:3])

        frames.append(_build_grid_data(
            concentration_grid=np.round(total_ppb, 3),
            confidence_grid=np.round(confidence, 3),
            lat_centers=GRID_LAT_CENTERS,
            lon_centers=GRID_LON_CENTERS,
            bounds=GRID_BOUNDS,
            resolution_meters=GRID_RESOLUTION_METERS,
            metadata={
                "time": str(t),
                "hour_index": hour_idx,
                "wind_speed_ms": round(ws, 2),
                "wind_direction_deg": round(wd_deg, 1),
                "stability_class": stab,
                "n_sources": len(CANDIDATE_SOURCES),
                "top_3_sources_g_s": top_3_sources,
                "emission_rates_per_source_g_s": {k: float(v) for k, v in emission_rates_per_source_g_s.items() if v > 0},
                "baseline_scale": baseline_scale,
                "ppb_clip": ppb_clip,
            },
        ))

    return frames


def footprint_to_grid_data(
    footprint: pd.DataFrame,
    metadata: Optional[dict] = None,
) -> dict:
    """Resample a Lagrangian footprint DataFrame to the unified grid and wrap as GridData.

    The footprint has lat-index and lon-columns (120x160, different bounds).
    We resample it to the shared grid via nearest-neighbor interpolation.

    Args:
        footprint: DataFrame from ``build_footprint()`` — lat index, lon columns,
            values are probability densities.
        metadata: Extra metadata to include in the GridData dict.

    Returns:
        GeoDemic-compatible GridData dict.
    """
    from h2s.dispersion.grid_config import (
        GRID_BOUNDS,
        GRID_LAT_CENTERS,
        GRID_LON_CENTERS,
        GRID_NROWS,
        GRID_NCOLS,
        GRID_RESOLUTION_METERS,
    )
    from scipy.interpolate import RegularGridInterpolator

    src_lats = footprint.index.to_numpy(dtype=float)
    src_lons = footprint.columns.to_numpy(dtype=float)
    src_data = footprint.values.astype(float)

    # RegularGridInterpolator requires strictly increasing coordinates
    if src_lats[0] > src_lats[-1]:
        src_lats = src_lats[::-1]
        src_data = src_data[::-1, :]

    interp = RegularGridInterpolator(
        (src_lats, src_lons), src_data,
        method="nearest", bounds_error=False, fill_value=0.0,
    )

    lon_mesh, lat_mesh = np.meshgrid(GRID_LON_CENTERS, GRID_LAT_CENTERS)
    pts = np.column_stack([lat_mesh.ravel(), lon_mesh.ravel()])
    resampled = interp(pts).reshape(GRID_NROWS, GRID_NCOLS)

    # Re-normalize so values sum to 1
    total = resampled.sum()
    if total > 0:
        resampled = resampled / total

    # Confidence: high where footprint has data, low where it's zero
    confidence = np.where(resampled > 0, 0.8, 0.2)

    from h2s.dispersion.lagrangian import CANDIDATE_SOURCES

    merged_metadata = metadata or {}
    merged_metadata["candidate_sources"] = {
        name: {"lat": src["lat"], "lon": src["lon"]}
        for name, src in CANDIDATE_SOURCES.items()
    }

    return _build_grid_data(
        concentration_grid=np.round(resampled, 8),
        confidence_grid=confidence,
        lat_centers=GRID_LAT_CENTERS,
        lon_centers=GRID_LON_CENTERS,
        bounds=GRID_BOUNDS,
        resolution_meters=GRID_RESOLUTION_METERS,
        metadata=merged_metadata,
    )


def _gaussian_plume_grid(
    source_lat: float,
    source_lon: float,
    emission_rate_g_s: float,
    lat_grid: np.ndarray,
    lon_grid: np.ndarray,
    wind_u: float,
    wind_v: float,
    stab: str,
    stack_height: float = 2.0,
    receptor_height: float = 1.5,
    sigma_theta_deg: float = 20.0,
) -> np.ndarray:
    """Vectorized Gaussian plume over a 2-D lat/lon mesh. Returns μg/m³ array."""
    wind_speed = np.sqrt(wind_u ** 2 + wind_v ** 2)
    wind_speed_eff = max(wind_speed, 0.3)

    dx = (lon_grid - source_lon) * M_PER_DEG_LON
    dy = (lat_grid - source_lat) * M_PER_DEG_LAT

    if wind_speed < 0.1:
        # Calm-wind fallback: isotropic dispersion
        dist_m = np.maximum(np.sqrt(dx ** 2 + dy ** 2), 50.0)
        mixing_height = 50.0
        conc = emission_rate_g_s * 1e6 / (dist_m * dist_m * mixing_height)
        return np.maximum(conc, 0.0)

    u_hat = np.array([wind_u, wind_v]) / wind_speed
    x_down = dx * u_hat[0] + dy * u_hat[1]
    y_cross = u_hat[0] * dy - u_hat[1] * dx

    # Mask upwind cells
    valid = x_down > 0
    conc = np.zeros_like(x_down)

    x_km = x_down[valid] / 1000.0
    a_y, b_y, a_z, b_z = PG_PARAMS[stab]
    x_km_safe = np.maximum(x_km, 0.01)
    sigma_y = a_y * (x_km_safe ** b_y) * 1000.0
    sigma_z = a_z * (x_km_safe ** b_z) * 1000.0

    sigma_theta_rad = np.radians(sigma_theta_deg)
    sigma_y_eff = np.sqrt(sigma_y ** 2 + (sigma_theta_rad * x_down[valid]) ** 2)

    Q = emission_rate_g_s * 1e6
    exp_y = np.exp(-0.5 * (y_cross[valid] / sigma_y_eff) ** 2)
    exp_z_direct = np.exp(-0.5 * ((receptor_height - stack_height) / sigma_z) ** 2)
    exp_z_reflect = np.exp(-0.5 * ((receptor_height + stack_height) / sigma_z) ** 2)

    conc[valid] = (Q / (np.pi * wind_speed_eff * sigma_y_eff * sigma_z)) * exp_y * (exp_z_direct + exp_z_reflect)
    return np.maximum(conc, 0.0)


def _ug_grid_to_ppb(conc_ug_m3: np.ndarray, temp_c: float = 20.0) -> np.ndarray:
    """Convert μg/m³ array to ppb."""
    molar_vol = MOLAR_VOL_STP * (273.15 + temp_c) / 293.15
    return conc_ug_m3 * molar_vol / MW_H2S


if __name__ == "__main__":
    import os
    from pathlib import Path

    def _parse_rates(s: str) -> dict[str, float]:
        result = {}
        for part in s.split(","):
            k, v = part.split("=")
            result[k.strip()] = float(v.strip())
        return result

    parser = argparse.ArgumentParser(description="Gaussian forward H2S dispersion model")
    parser.add_argument("--data", default=os.environ.get("H2S_DATA_PATH", "modeldata_h2s_nofill.csv"))
    parser.add_argument("--emission_rates", default="east=20,west=10,south=137",
                        help="zone=Q(g/s) pairs, comma separated")
    parser.add_argument("--start", default=None, help="Forecast start ISO (local time). Default: now.")
    parser.add_argument("--hours", type=int, default=72)
    parser.add_argument("--output", default="./forward_output")
    args = parser.parse_args()

    df = pd.read_csv(args.data) if args.data.endswith(".csv") else pd.read_parquet(args.data)
    df["time"] = pd.to_datetime(df["time"], utc=True).dt.tz_convert("America/Los_Angeles")

    if args.start:
        start = pd.Timestamp(args.start).tz_localize("America/Los_Angeles")
    else:
        start = pd.Timestamp.now(tz="America/Los_Angeles").floor("1h")

    rates = _parse_rates(args.emission_rates)
    print(f"Forward model: sources={rates} g/s, start={start}, hours={args.hours}")

    result = run_forward_model(df, rates, start, args.hours)

    Path(args.output).mkdir(parents=True, exist_ok=True)
    out_path = Path(args.output) / f"forecast_{start.strftime('%Y%m%d_%H')}.json"
    out_path.write_text(result.to_json())
    print(f"Saved → {out_path}")

    df_out = result.to_dataframe()
    print("\nPeak predicted concentrations (ppb):")
    print(df_out.groupby("sensor")["predicted_ppb"].max().round(1).to_string())
