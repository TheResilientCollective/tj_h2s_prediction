"""
Backward Lagrangian particle model for H2S source attribution.

Traces particles backward in time from receptor locations using observed
wind fields. Builds a source footprint (probability density of origin locations)
over the inversion window. Cross-correlates footprint with candidate source
locations to estimate relative source contributions.

This model is self-contained (no HYSPLIT binary required) and pipeline-friendly.
~5s per event on a single core.

Adapted from modeling_sources/lagrangian_backward.py for use as a library module.
Key change: run_inversion_window() accepts a pre-loaded DataFrame and returns
results in-memory (no disk I/O) — caller handles S3 upload.
"""

import json
import argparse
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional


# --- Domain & source configuration ---

SENSORS = {
    "NESTOR - BES": {"lat": 32.567097, "lon": -117.090656},
    "IB CIVIC CTR": {"lat": 32.576139, "lon": -117.115361},
    "SAN YSIDRO":   {"lat": 32.552794, "lon": -117.047286},
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


@dataclass
class LagrangianConfig:
    n_particles: int = 2000
    dt_seconds: float = 60.0
    hours_back: int = 6

    # Wind-dependent diffusion parameterization (NEW)
    # Atmospheric turbulence scales as sigma ~ U^exponent
    # For Tijuana River Valley (mixed suburban/open terrain):
    use_wind_dependent_diffusion: bool = True
    sigma_u_coeff: float = 0.15   # baseline horizontal diffusion coefficient (m/s)
    sigma_v_coeff: float = 0.15   # baseline horizontal diffusion coefficient (m/s)
    sigma_w_coeff: float = 0.05   # baseline vertical diffusion coefficient (m/s)
    sigma_u_exponent: float = 0.5  # wind speed scaling exponent (horizontal)
    sigma_v_exponent: float = 0.5  # wind speed scaling exponent (horizontal)
    sigma_w_exponent: float = 0.3  # wind speed scaling exponent (vertical, weaker)
    min_wind_speed: float = 0.5    # minimum wind speed for diffusion calc (m/s, avoid div-by-zero)

    # Fixed diffusion (LEGACY - kept for backward compatibility)
    # Only used if use_wind_dependent_diffusion=False
    sigma_u: float = 0.3
    sigma_v: float = 0.3
    sigma_w: float = 0.05

    h2s_decay_hr: float = 1e6
    lat_min: float = 32.45
    lat_max: float = 32.65
    lon_min: float = -117.25
    lon_max: float = -117.00


def load_met_at_time(
    df: pd.DataFrame,
    dt: pd.Timestamp,
    sensor: str,
    hours_back: int = 6,
) -> pd.DataFrame:
    t_end = dt
    t_start = dt - pd.Timedelta(hours=hours_back)
    mask = (
        (df["time"] >= t_start)
        & (df["time"] <= t_end)
        & (df["site_name"] == sensor)
    )
    return df[mask].copy().sort_values("time")


def interpolate_wind(met: pd.DataFrame, t: pd.Timestamp) -> tuple[float, float]:
    if met.empty:
        return 0.0, 0.0
    ws = met["wind_speed_10m"].values
    wd = np.radians(met["wind_direction_10m"].values)
    u = -ws * np.sin(wd)
    v = -ws * np.cos(wd)
    times = met["time"].values.astype("int64")
    t_int = np.int64(pd.Timestamp(t).value)
    if len(times) == 1 or t_int <= times[0]:
        return float(u[0]), float(v[0])
    if t_int >= times[-1]:
        return float(u[-1]), float(v[-1])
    i = np.searchsorted(times, t_int) - 1
    frac = (t_int - times[i]) / (times[i + 1] - times[i])
    return float(u[i] + frac * (u[i + 1] - u[i])), float(v[i] + frac * (v[i + 1] - v[i]))


def run_backward_particles(
    receptor_lat: float,
    receptor_lon: float,
    met: pd.DataFrame,
    event_time: pd.Timestamp,
    cfg: LagrangianConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    n = cfg.n_particles
    dt = cfg.dt_seconds
    n_steps = int(cfg.hours_back * 3600 / dt)

    lats = np.full(n, receptor_lat)
    lons = np.full(n, receptor_lon)

    tau = 300.0
    alpha = np.exp(-dt / tau)

    # Initialize turbulence (use wind-dependent or fixed diffusion)
    if cfg.use_wind_dependent_diffusion:
        # Initial wind speed for sigma calculation
        u_init, v_init = interpolate_wind(met, event_time)
        U_init = max(np.sqrt(u_init**2 + v_init**2), cfg.min_wind_speed)
        sigma_u_init = cfg.sigma_u_coeff * (U_init ** cfg.sigma_u_exponent)
        sigma_v_init = cfg.sigma_v_coeff * (U_init ** cfg.sigma_v_exponent)
        u_turb = rng.normal(0, sigma_u_init, n)
        v_turb = rng.normal(0, sigma_v_init, n)
    else:
        # Fixed diffusion (legacy)
        u_turb = rng.normal(0, cfg.sigma_u, n)
        v_turb = rng.normal(0, cfg.sigma_v, n)

    current_time = event_time

    for _ in range(n_steps):
        u_mean, v_mean = interpolate_wind(met, current_time)

        # Compute wind-dependent diffusion coefficients
        if cfg.use_wind_dependent_diffusion:
            U = max(np.sqrt(u_mean**2 + v_mean**2), cfg.min_wind_speed)
            sigma_u_local = cfg.sigma_u_coeff * (U ** cfg.sigma_u_exponent)
            sigma_v_local = cfg.sigma_v_coeff * (U ** cfg.sigma_v_exponent)
        else:
            sigma_u_local = cfg.sigma_u
            sigma_v_local = cfg.sigma_v

        # Autoregressive turbulence evolution with wind-dependent sigma
        u_turb = alpha * u_turb + np.sqrt(1 - alpha**2) * rng.normal(0, sigma_u_local, n)
        v_turb = alpha * v_turb + np.sqrt(1 - alpha**2) * rng.normal(0, sigma_v_local, n)

        u_total = -(u_mean + u_turb)
        v_total = -(v_mean + v_turb)
        lats += v_total * dt / M_PER_DEG_LAT
        lons += u_total * dt / M_PER_DEG_LON
        current_time -= pd.Timedelta(seconds=dt)
        lats = np.clip(lats, cfg.lat_min, cfg.lat_max)
        lons = np.clip(lons, cfg.lon_min, cfg.lon_max)

    return lats, lons


def build_footprint(lats: np.ndarray, lons: np.ndarray) -> pd.DataFrame:
    """
    Convert final particle positions to a normalized 2D footprint DataFrame.

    Returns a DataFrame with lat bin centers as the index and lon bin centers
    as columns. Values are probability densities (sum = 1.0).
    Suitable for direct inspection, CSV export, or parquet upload.
    """
    lat_edges = np.linspace(GRID["lat_min"], GRID["lat_max"], GRID["nlat"] + 1)
    lon_edges = np.linspace(GRID["lon_min"], GRID["lon_max"], GRID["nlon"] + 1)
    H, _, _ = np.histogram2d(lats, lons, bins=[lat_edges, lon_edges])
    total = H.sum()
    H_norm = H / total if total > 0 else H

    lat_centers = np.round(0.5 * (lat_edges[:-1] + lat_edges[1:]), 6)
    lon_centers = np.round(0.5 * (lon_edges[:-1] + lon_edges[1:]), 6)
    return pd.DataFrame(H_norm, index=lat_centers, columns=lon_centers)


def source_attribution(footprint: pd.DataFrame) -> dict[str, float]:
    """
    Compute relative contribution of each candidate source from a footprint DataFrame.

    Args:
        footprint: DataFrame returned by build_footprint() — lat index, lon columns.

    Returns:
        Dict of source_name → fraction (0–1), sorted by contribution descending.
    """
    arr = footprint.values
    lat_centers = footprint.index.to_numpy()
    lon_centers = footprint.columns.to_numpy()
    lons_g, lats_g = np.meshgrid(lon_centers, lat_centers)

    sigma_m = 500.0
    sigma_lat = sigma_m / M_PER_DEG_LAT
    sigma_lon = sigma_m / M_PER_DEG_LON

    contributions = {}
    for name, src in CANDIDATE_SOURCES.items():
        kernel = np.exp(
            -0.5 * (
                ((lats_g - src["lat"]) / sigma_lat) ** 2 +
                ((lons_g - src["lon"]) / sigma_lon) ** 2
            )
        )
        contributions[name] = float(np.sum(arr * kernel))

    total = sum(contributions.values())
    if total > 0:
        contributions = {k: v / total for k, v in contributions.items()}
    return dict(sorted(contributions.items(), key=lambda x: x[1], reverse=True))


def _process_event(
    row: pd.Series,
    df_met: pd.DataFrame,
    cfg: LagrangianConfig,
    rng: np.random.Generator,
) -> tuple[dict, pd.DataFrame | None]:
    """Run backward particle model for a single event. Returns (result_dict, footprint_array)."""
    sensor_name = row["site_name"]
    sensor = SENSORS.get(sensor_name)
    if sensor is None:
        return {}, None

    dt = pd.Timestamp(row["time"])
    h2s_obs = float(row["H2S"])
    tag = f"{dt.strftime('%Y%m%d_%H%M')}_{sensor_name.replace(' ', '_').replace('-', '')}"

    met = load_met_at_time(df_met, dt, sensor_name, cfg.hours_back)
    if len(met) < 2:
        return {}, None

    final_lats, final_lons = run_backward_particles(
        sensor["lat"], sensor["lon"], met, dt, cfg, rng
    )
    footprint = build_footprint(final_lats, final_lons)
    attribution = source_attribution(footprint)

    result = {
        "tag": tag,
        "time": dt.isoformat(),
        "sensor": sensor_name,
        "h2s_obs_ppb": h2s_obs,
        "top_sources": {k: round(v, 4) for k, v in list(attribution.items())[:5]},
        "all_sources": attribution,
    }
    return result, footprint


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
    Run backward Lagrangian model for all qualifying events in inversion window.

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
        (results_dict, ensemble_footprint_array)
        ensemble_footprint_array is None if no events were processed.
    """
    rng = np.random.default_rng(seed)

    mask = (
        (df["time"] >= date_start)
        & (df["time"] <= date_end)
        & (df["H2S"] >= h2s_threshold)
        & df["H2S"].notna()
        & (df["stable_atm"] == 1)
    )
    events = df[mask].sort_values("H2S", ascending=False)
    if max_events:
        events = events.head(max_events)

    results: dict[str, dict] = {}
    footprints_all: list[pd.DataFrame] = []

    for _, row in events.iterrows():
        result, footprint = _process_event(row, df, cfg, rng)
        if result and footprint is not None:
            results[result["tag"]] = result
            footprints_all.append(footprint)

    if not footprints_all:
        return results, None

    # Element-wise mean over all event footprints, preserving lat/lon labels
    ensemble_footprint = pd.concat(footprints_all).groupby(level=0).mean()
    return results, ensemble_footprint


if __name__ == "__main__":
    import os
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Backward Lagrangian particle model for H2S attribution")
    parser.add_argument("--data", default=os.environ.get("H2S_DATA_PATH", "modeldata_h2s_nofill.csv"))
    parser.add_argument("--output", default="./lagrangian_output")
    parser.add_argument("--date_start", default="2026-02-01")
    parser.add_argument("--date_end", default="2026-04-01")
    parser.add_argument("--h2s_min", type=float, default=30.0)
    parser.add_argument("--n_particles", type=int, default=2000)
    parser.add_argument("--hours_back", type=int, default=6)
    parser.add_argument("--max_events", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_csv(args.data) if args.data.endswith(".csv") else pd.read_parquet(args.data)
    df["time"] = pd.to_datetime(df["time"], utc=True).dt.tz_convert("America/Los_Angeles")

    cfg = LagrangianConfig(n_particles=args.n_particles, hours_back=args.hours_back)
    results, ensemble = run_inversion_window(
        df, cfg,
        date_start=args.date_start,
        date_end=args.date_end,
        h2s_threshold=args.h2s_min,
        max_events=args.max_events,
        seed=args.seed,
    )

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    if ensemble is not None:
        np.save(out_dir / "footprint_ensemble.npy", ensemble)
        attrib = source_attribution(ensemble)
        with open(out_dir / "source_attribution_ensemble.json", "w") as f:
            json.dump({"n_events": len(results), "ensemble_source_fractions": attrib}, f, indent=2)
        print(f"\nTop sources ({len(results)} events):")
        for k, v in list(attrib.items())[:6]:
            print(f"  {k:30s} {v:.3f}")
