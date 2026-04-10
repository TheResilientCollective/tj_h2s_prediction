"""
Backward Lagrangian particle model for H2S source attribution.

Residence-time (influence function) footprint approach. Accumulates ALL
particle positions throughout the backward trajectory, producing a dense
influence function covering the full upwind path (~37 hits/grid cell at
2000 particles / 4h run vs ~0.1 hits/cell for endpoint-only).

Key improvements over endpoint-only approach:
  - Residence-time footprint is non-zero across the entire upwind transport path
  - Adaptive backward duration: scales with mean wind speed so particles
    don't fly out of the source domain
  - Multi-sensor concentration-weighted combination: footprints from NB, IB,
    and SY are summed weighted by observed H2S, reinforcing sources visible
    to multiple sensors

Mathematically equivalent to the adjoint/STILT approach (Lin et al. 2003).

Adapted from modeling_sources/lagrangian_residence.py for use as a library module.
Key change: run_inversion_window() accepts a pre-loaded DataFrame and returns
results in-memory (no disk I/O) — caller handles S3 upload.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Union


# --- Domain & source configuration ---

SENSORS = {
    "NESTOR - BES": {"key": "NB",  "lat": 32.567097, "lon": -117.090656},
    "IB CIVIC CTR": {"key": "IB",  "lat": 32.576139, "lon": -117.115361},
    "SAN YSIDRO":   {"key": "SY",  "lat": 32.552794, "lon": -117.047286},
}

CANDIDATE_SOURCES = {
    "stewarts_drain":       {"lat": 32.54064,  "lon": -117.05801},
    "smugglers_gulch":      {"lat": 32.5377,   "lon": -117.08623},
    "hollister_ps":         {"lat": 32.5476,   "lon": -117.088374},
    "goat_canyon":          {"lat": 32.5369,   "lon": -117.09916},
    "goat_canyon_ps":       {"lat": 32.543476, "lon": -117.108026},
    "del_sol_canyon":       {"lat": 32.5393,   "lon": -117.06885},
    "silva_drain":          {"lat": 32.539743, "lon": -117.064269},
    "saturn_blvd_bridge":   {"lat": 32.559383, "lon": -117.092992},
    "hollister_bridge_n":   {"lat": 32.554177, "lon": -117.084135},
    "hollister_bridge_s":   {"lat": 32.551466, "lon": -117.084021},
    "dairy_mart_bridge":    {"lat": 32.548531, "lon": -117.064293},
    "oneonta_slough":       {"lat": 32.570082, "lon": -117.126724},
    "tijuana_beach_outlet": {"lat": 32.556206, "lon": -117.126178},
    "tj_crossing_cdlp_w":   {"lat": 32.542103, "lon": -117.054117},
    "tj_crossing_cdlp_e":   {"lat": 32.542166, "lon": -117.050325},
    "sd_bay_otay_outlet":   {"lat": 32.594557, "lon": -117.113542},
    "sd_bay_fruitdale":     {"lat": 32.595305, "lon": -117.091869},
}

# Grid for footprint output
GRID = {
    "lat_min": 32.50, "lat_max": 32.62,
    "lon_min": -117.18, "lon_max": -117.02,
    "nlat": 120, "nlon": 160,  # ~100m cells
}

# Degrees per meter at ~32.5°N
M_PER_DEG_LAT = 111_320.0
M_PER_DEG_LON = 111_320.0 * np.cos(np.radians(32.55))

# Domain radius (m) — used for adaptive backward duration
DOMAIN_RADIUS_M = 6_000.0

# H2S background (ppb) — subtract before weighting to avoid noise bias
H2S_BACKGROUND_PPB = 1.0

# Source kernel radius for attribution (m)
SOURCE_KERNEL_M = 500.0


@dataclass
class LagrangianConfig:
    n_particles: int = 2000
    dt_seconds: float = 60.0
    hours_back: float = 4.0        # max backward duration (h); pipeline-compatible name
    min_hours_back: float = 0.5    # minimum backward duration (h)
    # Turbulence (stable nocturnal BL)
    sigma_u: float = 0.3           # along-wind turbulence std (m/s)
    sigma_v: float = 0.3           # cross-wind turbulence std (m/s)
    tau_lagrangian: float = 300.0  # Lagrangian timescale (s)
    # Residence-time sampling: save positions every N steps (5min snapshots at dt=60s)
    save_every_n: int = 5
    # Domain bounds for particle reflection
    lat_min: float = 32.45
    lat_max: float = 32.65
    lon_min: float = -117.25
    lon_max: float = -117.00


# ---------------------------------------------------------------------------
# Met handling
# ---------------------------------------------------------------------------

def load_met_at_time(
    df: pd.DataFrame,
    dt: pd.Timestamp,
    sensor: str,
    hours_back: float,
) -> pd.DataFrame:
    """Extract met obs for the backward window, sorted oldest-to-newest."""
    t_start = dt - pd.Timedelta(hours=hours_back)
    mask = (
        (df["time"] >= t_start) & (df["time"] <= dt)
        & (df["site_name"] == sensor)
    )
    return df[mask].copy().sort_values("time")


def interpolate_wind(met: pd.DataFrame, t: pd.Timestamp) -> tuple[float, float]:
    """Linear-interpolate u,v to time t from met DataFrame."""
    if met.empty:
        return 0.0, 0.0
    ws  = met["wind_speed_10m"].values
    wd  = np.radians(met["wind_direction_10m"].values)
    u   = -ws * np.sin(wd)
    v   = -ws * np.cos(wd)
    tvals = met["time"].values.astype("int64")
    t_int = np.int64(pd.Timestamp(t).value)
    if len(tvals) == 1 or t_int <= tvals[0]:
        return float(u[0]), float(v[0])
    if t_int >= tvals[-1]:
        return float(u[-1]), float(v[-1])
    i = np.searchsorted(tvals, t_int) - 1
    frac = (t_int - tvals[i]) / (tvals[i + 1] - tvals[i])
    return float(u[i] + frac * (u[i + 1] - u[i])), float(v[i] + frac * (v[i + 1] - v[i]))


def _mean_wind_speed(met: pd.DataFrame) -> float:
    if met.empty:
        return 1.0
    return float(met["wind_speed_10m"].mean())


def _adaptive_hours_back(
    met: pd.DataFrame,
    cfg: LagrangianConfig,
    domain_radius_m: float = DOMAIN_RADIUS_M,
) -> float:
    """
    Choose backward duration so mean-wind transport ≈ domain radius.

    At high wind speeds, particles exit the source domain quickly — a long
    backward run wastes compute. At low wind speeds (stable events), longer
    runs keep particles in the source zone.

    Returns hours, clamped to [min_hours_back, hours_back].
    """
    ws_mean = max(_mean_wind_speed(met), 0.3)
    t_hours = (domain_radius_m / ws_mean) / 3600.0
    return float(np.clip(t_hours, cfg.min_hours_back, cfg.hours_back))


# ---------------------------------------------------------------------------
# Core particle model — residence-time footprint
# ---------------------------------------------------------------------------

def _reflect(vals: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Reflect particles off domain boundaries (prevents edge accumulation)."""
    out = vals.copy()
    below = out < lo
    out[below] = 2 * lo - out[below]
    above = out > hi
    out[above] = 2 * hi - out[above]
    return np.clip(out, lo, hi)


def run_backward_particles(
    receptor_lat: float,
    receptor_lon: float,
    met: pd.DataFrame,
    event_time: pd.Timestamp,
    cfg: LagrangianConfig,
    rng: np.random.Generator,
    hours_back: Optional[float] = None,
) -> np.ndarray:
    """
    Release particles at receptor, integrate backward, accumulate ALL positions.

    Produces the residence-time (influence function) footprint: a 2D histogram
    of how much time particles collectively spent at each grid cell. Cells that
    particles pass through frequently are likely upwind source regions.

    Parameters
    ----------
    hours_back : float or None
        Backward duration. If None, uses adaptive duration based on wind speed.

    Returns
    -------
    footprint : np.ndarray  shape (nlat, nlon)
        Normalised residence-time footprint (sums to 1.0).
    """
    if hours_back is None:
        hours_back = _adaptive_hours_back(met, cfg)

    n      = cfg.n_particles
    dt     = cfg.dt_seconds
    n_steps = int(hours_back * 3600 / dt)
    tau    = cfg.tau_lagrangian
    alpha  = np.exp(-dt / tau)

    lats   = np.full(n, receptor_lat)
    lons   = np.full(n, receptor_lon)
    u_turb = rng.normal(0, cfg.sigma_u, n)
    v_turb = rng.normal(0, cfg.sigma_v, n)

    lat_edges = np.linspace(GRID["lat_min"], GRID["lat_max"], GRID["nlat"] + 1)
    lon_edges = np.linspace(GRID["lon_min"], GRID["lon_max"], GRID["nlon"] + 1)

    H = np.zeros((GRID["nlat"], GRID["nlon"]), dtype=np.float64)
    current_time = event_time

    for step in range(n_steps):
        u_mean, v_mean = interpolate_wind(met, current_time)

        # Langevin turbulence update
        u_turb = alpha * u_turb + np.sqrt(1 - alpha**2) * rng.normal(0, cfg.sigma_u, n)
        v_turb = alpha * v_turb + np.sqrt(1 - alpha**2) * rng.normal(0, cfg.sigma_v, n)

        # Backward displacement: negate the mean wind
        u_total = -(u_mean + u_turb)
        v_total = -(v_mean + v_turb)

        lats += v_total * dt / M_PER_DEG_LAT
        lons += u_total * dt / M_PER_DEG_LON

        # Reflect particles back into domain (prevents artificial edge accumulation)
        lats = _reflect(lats, cfg.lat_min, cfg.lat_max)
        lons = _reflect(lons, cfg.lon_min, cfg.lon_max)

        current_time -= pd.Timedelta(seconds=dt)

        if step % cfg.save_every_n == 0:
            h, _, _ = np.histogram2d(lats, lons, bins=[lat_edges, lon_edges])
            H += h

    total = H.sum()
    return H / total if total > 0 else H


def build_footprint(fp_array: np.ndarray) -> pd.DataFrame:
    """
    Convert a residence-time footprint ndarray to a labeled DataFrame.

    Returns a DataFrame with lat bin centers as the index and lon bin centers
    as columns. Values are probability densities (sum ≈ 1.0).
    Suitable for CSV/parquet export and footprint_to_grid_data().
    """
    lat_edges = np.linspace(GRID["lat_min"], GRID["lat_max"], GRID["nlat"] + 1)
    lon_edges = np.linspace(GRID["lon_min"], GRID["lon_max"], GRID["nlon"] + 1)
    lat_centers = np.round(0.5 * (lat_edges[:-1] + lat_edges[1:]), 6)
    lon_centers = np.round(0.5 * (lon_edges[:-1] + lon_edges[1:]), 6)
    return pd.DataFrame(fp_array, index=lat_centers, columns=lon_centers)


# ---------------------------------------------------------------------------
# Multi-sensor concentration-weighted combination
# ---------------------------------------------------------------------------

def _combine_footprints(
    footprints: dict[str, np.ndarray],
    h2s_obs: dict[str, float],
    background_ppb: float = H2S_BACKGROUND_PPB,
) -> np.ndarray:
    """
    Concentration-weighted sum of per-sensor footprints.

    Weight = max(C_obs - C_background, 0). Sensors with elevated H2S
    contribute proportionally more; sources visible to multiple elevated
    sensors are reinforced; false positives in a single sensor cancel.
    """
    combined = np.zeros_like(next(iter(footprints.values())), dtype=np.float64)
    total_weight = 0.0

    for sensor, fp in footprints.items():
        c_obs = h2s_obs.get(sensor, 0.0)
        weight = max(c_obs - background_ppb, 0.0)
        combined += weight * fp
        total_weight += weight

    if total_weight > 0:
        combined /= total_weight

    total = combined.sum()
    return combined / total if total > 0 else combined


# ---------------------------------------------------------------------------
# Source attribution
# ---------------------------------------------------------------------------

def source_attribution(
    footprint: Union[pd.DataFrame, np.ndarray],
    kernel_radius_m: float = SOURCE_KERNEL_M,
) -> dict[str, float]:
    """
    Compute relative contribution of each candidate source from a footprint.

    Accepts either a DataFrame (lat index, lon columns) from build_footprint()
    or a raw ndarray from run_backward_particles().

    Returns dict sorted by contribution (highest first).
    """
    if isinstance(footprint, pd.DataFrame):
        arr = footprint.values
        lat_centers = footprint.index.to_numpy(dtype=float)
        lon_centers = footprint.columns.to_numpy(dtype=float)
    else:
        arr = footprint
        lat_edges = np.linspace(GRID["lat_min"], GRID["lat_max"], GRID["nlat"] + 1)
        lon_edges = np.linspace(GRID["lon_min"], GRID["lon_max"], GRID["nlon"] + 1)
        lat_centers = 0.5 * (lat_edges[:-1] + lat_edges[1:])
        lon_centers = 0.5 * (lon_edges[:-1] + lon_edges[1:])

    sigma_lat = kernel_radius_m / M_PER_DEG_LAT
    sigma_lon = kernel_radius_m / M_PER_DEG_LON
    lons_g, lats_g = np.meshgrid(lon_centers, lat_centers)

    contributions = {}
    for name, src in CANDIDATE_SOURCES.items():
        kernel = np.exp(-0.5 * (
            ((lats_g - src["lat"]) / sigma_lat) ** 2 +
            ((lons_g - src["lon"]) / sigma_lon) ** 2
        ))
        contributions[name] = float(np.sum(arr * kernel))

    total = sum(contributions.values())
    if total > 0:
        contributions = {k: v / total for k, v in contributions.items()}

    return dict(sorted(contributions.items(), key=lambda x: x[1], reverse=True))


# ---------------------------------------------------------------------------
# Event processing
# ---------------------------------------------------------------------------

def _process_event(
    event_time: pd.Timestamp,
    df: pd.DataFrame,
    cfg: LagrangianConfig,
    rng: np.random.Generator,
) -> tuple[dict, np.ndarray | None]:
    """
    Run residence-time attribution for one event time across all sensors.

    Returns (result_dict, combined_footprint_ndarray) or ({}, None) on failure.
    """
    tag = event_time.strftime("%Y%m%d_%H%M")

    # Collect H2S at event time across sensors
    h2s_obs = {}
    for sname in SENSORS:
        row = df[(df["time"] == event_time) & (df["site_name"] == sname)]
        if not row.empty and not pd.isna(row["H2S"].iloc[0]):
            h2s_obs[sname] = float(row["H2S"].iloc[0])
        else:
            h2s_obs[sname] = 0.0

    if max(h2s_obs.values()) < 1.0:
        return {}, None

    # Run per-sensor backward models
    sensor_footprints: dict[str, np.ndarray] = {}
    for sname, sc in SENSORS.items():
        if h2s_obs[sname] < H2S_BACKGROUND_PPB:
            continue

        met = load_met_at_time(df, event_time, sname, cfg.hours_back)
        if len(met) < 2:
            continue

        h_back = _adaptive_hours_back(met, cfg)
        fp = run_backward_particles(sc["lat"], sc["lon"], met, event_time, cfg, rng, hours_back=h_back)
        sensor_footprints[sname] = fp

    if not sensor_footprints:
        return {}, None

    combined = _combine_footprints(sensor_footprints, h2s_obs)
    attr = source_attribution(combined)

    result = {
        "tag": tag,
        "time": event_time.isoformat(),
        "h2s_obs": h2s_obs,
        "n_sensors_used": len(sensor_footprints),
        "top_sources": {k: round(v, 4) for k, v in list(attr.items())[:5]},
        "all_sources": attr,
    }
    return result, combined


# ---------------------------------------------------------------------------
# Pipeline-facing API
# ---------------------------------------------------------------------------

def run_inversion_window(
    df: pd.DataFrame,
    cfg: LagrangianConfig,
    date_start: str = "2026-02-01",
    date_end: str = "2026-04-01",
    h2s_threshold: float = 30.0,
    max_events: Optional[int] = None,
    seed: int = 42,
) -> tuple[dict[str, dict], pd.DataFrame | None]:
    """
    Run residence-time attribution for all qualifying events in inversion window.

    Qualifying events: H2S ≥ threshold at any sensor, stable_atm = 1.
    For each qualifying hour, all sensors are processed simultaneously and
    their footprints are combined with concentration weighting.

    Args:
        df: Pre-loaded observation DataFrame. Must have columns:
            time (tz-aware), H2S, site_name, stable_atm,
            wind_speed_10m, wind_direction_10m.
        cfg: Lagrangian configuration.
        date_start, date_end: Inversion window bounds.
        h2s_threshold: Minimum H2S (ppb) for event selection.
        max_events: Cap on events processed (None = all).
        seed: RNG seed for reproducibility.

    Returns:
        (results_dict, ensemble_footprint_dataframe)
        ensemble_footprint_dataframe is None if no events were processed.
        DataFrame has lat bin centers as index and lon bin centers as columns.
    """
    rng = np.random.default_rng(seed)

    # Find event times: any sensor above threshold in window with stable_atm
    mask = (
        (df["time"] >= date_start) & (df["time"] <= date_end)
        & (df["H2S"] >= h2s_threshold) & df["H2S"].notna()
        & (df["stable_atm"] == 1)
    )
    event_times = df[mask]["time"].drop_duplicates().sort_values(ascending=False)
    if max_events:
        event_times = event_times.head(max_events)

    results: dict[str, dict] = {}
    combined_fps: list[np.ndarray] = []

    for et in event_times:
        result, fp = _process_event(et, df, cfg, rng)
        if result and fp is not None:
            results[result["tag"]] = result
            combined_fps.append(fp)

    if not combined_fps:
        return results, None

    ensemble = np.mean(combined_fps, axis=0)
    total = ensemble.sum()
    if total > 0:
        ensemble /= total

    return results, build_footprint(ensemble)