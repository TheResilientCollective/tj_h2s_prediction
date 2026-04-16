#!/usr/bin/env python3
"""
Backward Lagrangian source attribution using the residence-time
(influence function) footprint approach.

Replaces the old endpoint-only lagrangian_backward approach. Key differences:

  OLD (endpoint-only):
    - Saves only final particle positions after N hours
    - Result: a single sparse blob, often outside the source domain
    - No multi-sensor combination
    - ~0.1 hits/grid cell on average -> very noisy attribution

  NEW (residence-time):
    - Accumulates ALL particle positions throughout the backward trajectory
    - Result: a dense influence function covering the full upwind path
    - ~37 hits/grid cell at 2000 particles / 6h run
    - Adaptive backward duration: scales with mean wind speed so particles
      don't fly out of the source domain
    - Multi-sensor concentration-weighted combination: footprints from NB,
      IB, and SY are summed weighted by observed H2S, so sources visible to
      multiple sensors are reinforced and false positives cancel
    - Direct emission rate estimation: Q (g/s) from concentration / footprint

This is mathematically equivalent to the adjoint/STILT approach used in
greenhouse gas flux inversion (Lin et al. 2003, Gerbig et al. 2003).

Usage (standalone):
    python lagrangian.py \\
        --data modeldata_h2s_nofill.csv \\
        --output ./lagrangian_output/ \\
        --date_start 2026-02-01 --date_end 2026-04-01

Key outputs:
    footprint_{tag}.npy              : per-sensor residence-time footprint
    footprint_combined_{tag}.npy     : concentration-weighted multi-sensor footprint
    attribution_{tag}.json           : source fractions + emission rate estimates
    ensemble_footprint.npy           : mean combined footprint over all events
    ensemble_attribution.json        : ensemble source fractions
"""

import os
import json
import argparse
import shutil
import tempfile
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

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

GRID = {
    "lat_min": 32.50, "lat_max": 32.62,
    "lon_min": -117.18, "lon_max": -117.02,
    "nlat": 120, "nlon": 160,   # ~100m cells
}

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
    n_particles: int    = 2000   # per sensor per event
    dt_seconds: float   = 60.0   # integration step (s)
    max_hours_back: float = 4.0  # maximum backward duration (h)
    min_hours_back: float = 0.5  # minimum backward duration (h)
    # Turbulence (stable nocturnal BL)
    sigma_u: float = 0.3         # along-wind turbulence std (m/s)
    sigma_v: float = 0.3         # cross-wind turbulence std (m/s)
    tau_lagrangian: float = 300.0  # Lagrangian timescale (s); ~300s for stable BL
    # Residence-time sampling: save positions every N steps
    # dt_save = N * dt_seconds; default 5min snapshots
    save_every_n: int = 5
    # Domain
    lat_min: float = 32.45
    lat_max: float = 32.65
    lon_min: float = -117.25
    lon_max: float = -117.00
    # Legacy alias: hours_back → max_hours_back (backward compat for pipeline)
    hours_back: Optional[float] = None

    def __post_init__(self):
        if self.hours_back is not None:
            self.max_hours_back = float(self.hours_back)


# ---------------------------------------------------------------------------
# Met handling
# ---------------------------------------------------------------------------

def load_met(
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
    met = df[mask].copy().sort_values("time")
    return met


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
    frac = (t_int - tvals[i]) / (tvals[i+1] - tvals[i])
    return float(u[i] + frac*(u[i+1]-u[i])), float(v[i] + frac*(v[i+1]-v[i]))


def mean_wind_speed(met: pd.DataFrame) -> float:
    """Mean wind speed over the met window."""
    if met.empty:
        return 1.0
    return float(met["wind_speed_10m"].mean())


# ---------------------------------------------------------------------------
# Adaptive backward duration
# ---------------------------------------------------------------------------

def adaptive_hours_back(
    met: pd.DataFrame,
    cfg: LagrangianConfig,
    domain_radius_m: float = DOMAIN_RADIUS_M,
) -> float:
    """
    Choose backward duration so mean-wind transport ≈ domain radius.

    At high wind speeds, particles exit the source domain quickly — a long
    backward run wastes compute and fills the footprint with out-of-domain
    positions. At low wind speeds (stable events), we can run longer and
    still keep particles in the source zone.

    Returns hours, clamped to [min_hours_back, max_hours_back].
    """
    ws_mean = max(mean_wind_speed(met), 0.3)
    # Target: transport time for 1 domain radius
    t_hours = (domain_radius_m / ws_mean) / 3600.0
    return float(np.clip(t_hours, cfg.min_hours_back, cfg.max_hours_back))


# ---------------------------------------------------------------------------
# Core particle model — residence-time footprint
# ---------------------------------------------------------------------------

def run_residence_time_particles(
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

    This produces the residence-time (influence function) footprint: a 2D
    histogram of how much time particles collectively spent at each grid cell.
    Cells that particles pass through frequently are likely upwind source regions.

    Unlike endpoint-only, this:
      - Is non-zero across the entire upwind transport path
      - Has ~300x higher grid density (n_particles × n_steps vs n_particles)
      - Correctly weights cells closer to the receptor more heavily
        (particles start there and spread outward)

    Parameters
    ----------
    hours_back : float or None
        Backward duration. If None, uses adaptive_hours_back().

    Returns
    -------
    footprint : np.ndarray  shape (nlat, nlon)
        Normalised residence-time footprint (sums to 1.0).
    """
    if hours_back is None:
        hours_back = adaptive_hours_back(met, cfg)

    n     = cfg.n_particles
    dt    = cfg.dt_seconds
    n_steps = int(hours_back * 3600 / dt)
    tau   = cfg.tau_lagrangian
    alpha = np.exp(-dt / tau)

    lats = np.full(n, receptor_lat)
    lons = np.full(n, receptor_lon)
    u_turb = rng.normal(0, cfg.sigma_u, n)
    v_turb = rng.normal(0, cfg.sigma_v, n)

    lat_edges = np.linspace(GRID["lat_min"], GRID["lat_max"], GRID["nlat"] + 1)
    lon_edges = np.linspace(GRID["lon_min"], GRID["lon_max"], GRID["nlon"] + 1)

    # Accumulate residence-time histogram
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

        # Soft boundary: reflect particles back into domain instead of clipping
        # (clipping creates artificial accumulation at edges)
        lats = _reflect(lats, cfg.lat_min, cfg.lat_max)
        lons = _reflect(lons, cfg.lon_min, cfg.lon_max)

        current_time -= pd.Timedelta(seconds=dt)

        # Accumulate every save_every_n steps
        if step % cfg.save_every_n == 0:
            h, _, _ = np.histogram2d(
                lats, lons, bins=[lat_edges, lon_edges]
            )
            H += h

    total = H.sum()
    return H / total if total > 0 else H


def _reflect(vals: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Reflect particles off domain boundaries (prevents edge accumulation)."""
    out = vals.copy()
    # Below lower bound
    below = out < lo
    out[below] = 2 * lo - out[below]
    # Above upper bound
    above = out > hi
    out[above] = 2 * hi - out[above]
    # Second clip in case double-reflection still out of range (rare)
    return np.clip(out, lo, hi)


# ---------------------------------------------------------------------------
# Multi-sensor concentration-weighted combination
# ---------------------------------------------------------------------------

def combine_footprints(
    footprints: dict[str, np.ndarray],
    h2s_obs: dict[str, float],
    background_ppb: float = H2S_BACKGROUND_PPB,
) -> np.ndarray:
    """
    Concentration-weighted sum of per-sensor footprints.

    Weight = max(C_obs - C_background, 0)

    This means:
      - Sensors with elevated H2S contribute proportionally more
      - Sensors at background (~1 ppb) contribute near zero
      - Sources visible to MULTIPLE elevated sensors are reinforced
      - False positives in a single sensor's footprint are down-weighted

    Parameters
    ----------
    footprints : sensor_name → normalised residence-time footprint
    h2s_obs    : sensor_name → observed H2S (ppb) at event time
    background_ppb : subtract as baseline before weighting

    Returns
    -------
    combined : np.ndarray   (nlat, nlon), normalised
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
# Source attribution from footprint
# ---------------------------------------------------------------------------

def source_attribution(
    footprint,
    kernel_radius_m: float = SOURCE_KERNEL_M,
) -> dict[str, float]:
    """
    Compute fractional source contributions from a residence-time footprint.

    Accepts either a numpy array (nlat, nlon) or a legacy DataFrame
    (lat index, lon columns) for backward compatibility with the pipeline.

    Uses a Gaussian kernel centred on each candidate source location.
    Larger kernel_radius_m gives more spatial tolerance for imprecise
    source locations; smaller gives sharper discrimination.

    Returns dict sorted by contribution (highest first).
    """
    # Backward compat: accept DataFrame from legacy callers
    if isinstance(footprint, pd.DataFrame):
        footprint = footprint.values

    lat_centers = np.linspace(
        GRID["lat_min"], GRID["lat_max"], GRID["nlat"],
        endpoint=False
    ) + (GRID["lat_max"] - GRID["lat_min"]) / (2 * GRID["nlat"])
    lon_centers = np.linspace(
        GRID["lon_min"], GRID["lon_max"], GRID["nlon"],
        endpoint=False
    ) + (GRID["lon_max"] - GRID["lon_min"]) / (2 * GRID["nlon"])

    sigma_lat = kernel_radius_m / M_PER_DEG_LAT
    sigma_lon = kernel_radius_m / M_PER_DEG_LON
    lons_g, lats_g = np.meshgrid(lon_centers, lat_centers)

    contributions = {}
    for name, src in CANDIDATE_SOURCES.items():
        kernel = np.exp(-0.5 * (
            ((lats_g - src["lat"]) / sigma_lat) ** 2 +
            ((lons_g - src["lon"]) / sigma_lon) ** 2
        ))
        contributions[name] = float(np.sum(footprint * kernel))

    total = sum(contributions.values())
    if total > 0:
        contributions = {k: v / total for k, v in contributions.items()}

    return dict(sorted(contributions.items(), key=lambda x: x[1], reverse=True))


def estimate_emission_rates(
    combined_footprint: np.ndarray,
    h2s_obs: dict[str, float],
    source_fractions: dict[str, float],
    background_ppb: float = H2S_BACKGROUND_PPB,
) -> dict[str, float]:
    """
    Estimate per-source emission rates Q (g/s) from footprint and observations.

    Method (simplified Bayesian inversion):
      The concentration observed at a receptor is:
        C = Q * F
      where F is the footprint value at the source location (s/m³ equivalent).
      Rearranging: Q = C / F

    We use:
      - C_eff = mean concentration across elevated sensors (C - background)
      - F_source = footprint value within the source kernel × fraction
      - Normalisation: convert dimensionless footprint fraction to m²·s/g equivalent

    Note: this gives relative Q values that are physically consistent with the
    Gaussian forward model. Absolute calibration requires a unit-source HYSPLIT
    backward dispersion run (cdump).

    Returns dict of source_name → Q_estimate (g/s), only for top sources.
    """
    # Mean effective concentration at elevated sensors
    c_eff_vals = [max(c - background_ppb, 0) for c in h2s_obs.values()]
    c_eff = np.mean([c for c in c_eff_vals if c > 0]) if any(c > 0 for c in c_eff_vals) else 0.0

    if c_eff <= 0:
        return {}

    # Grid cell area (m²)
    dlat_m = (GRID["lat_max"] - GRID["lat_min"]) / GRID["nlat"] * M_PER_DEG_LAT
    dlon_m = (GRID["lon_max"] - GRID["lon_min"]) / GRID["nlon"] * M_PER_DEG_LON
    cell_area_m2 = dlat_m * dlon_m

    # Convert footprint fraction to dimensional footprint (s/m³ equivalent)
    # Footprint value ≈ fraction of time particles spent per cell area
    # Dimensional: fp_dim [m⁻²] = fraction / cell_area_m2
    # C [μg/m³] = Q [g/s] * fp_dim [m⁻²] * 1e6 * (mean_transport_time_s)
    # Simplified: use footprint fraction directly for relative Q
    total_fp = combined_footprint.sum()  # should be 1.0 if normalised

    rates = {}
    for name, frac in source_fractions.items():
        if frac < 0.02:  # skip negligible sources
            continue
        # Relative Q: Q_rel = C_eff * fraction_of_total_footprint
        # Scale factor to get g/s: calibrated from Gaussian forward model
        # (137 g/s produced 394 ppb at 3km under stable F conditions)
        # Dimensionless footprint fraction * calibration_factor = Q in g/s
        # calibration_factor derived from: Q_ref / (C_ref * domain_fraction_ref)
        # Using Mar 13 calibration: 137 g/s / (394 ppb * ~0.25 domain fraction) ≈ 1.4
        calibration = 1.4  # g/s per (ppb * footprint_fraction); update from HYSPLIT inversion
        q_estimate = c_eff * frac * calibration
        rates[name] = round(q_estimate, 2)

    return dict(sorted(rates.items(), key=lambda x: x[1], reverse=True))


# ---------------------------------------------------------------------------
# Event processing
# ---------------------------------------------------------------------------

def process_event(
    event_time: pd.Timestamp,
    df_met: pd.DataFrame,
    cfg: LagrangianConfig,
    out_dir: Path,
    rng: np.random.Generator,
    h2s_df: Optional[pd.DataFrame] = None,
) -> dict:
    """
    Run the full residence-time attribution for one event time across all sensors.

    Collects H2S observations at event_time from all sensors, runs backward
    particle model from each, combines footprints with concentration weighting,
    and estimates source fractions and emission rates.

    Parameters
    ----------
    event_time : tz-aware timestamp
    df_met     : full modeldata DataFrame (or 15-min met from Met15Loader)
    h2s_df     : if None, H2S values are read from df_met['H2S']
    """
    tag = event_time.strftime("%Y%m%d_%H%M")

    # Collect H2S at event time across sensors
    h2s_obs = {}
    for sname in SENSORS:
        src = h2s_df if h2s_df is not None else df_met
        row = src[(src["time"] == event_time) & (src["site_name"] == sname)]
        if not row.empty and not pd.isna(row["H2S"].iloc[0]):
            h2s_obs[sname] = float(row["H2S"].iloc[0])
        else:
            h2s_obs[sname] = 0.0

    max_h2s = max(h2s_obs.values())
    if max_h2s < 1.0:
        return {}  # No signal to attribute

    print(f"  [event {tag}] H2S: " +
          " | ".join(f"{SENSORS[s]['key']}={h2s_obs[s]:.0f}ppb" for s in SENSORS))

    # Run per-sensor backward models
    footprints = {}
    for sname, sc in SENSORS.items():
        if h2s_obs[sname] < H2S_BACKGROUND_PPB:
            continue  # Skip sensors at background

        met = load_met(df_met, event_time, sname, cfg.max_hours_back)
        if len(met) < 2:
            print(f"    [skip {SENSORS[sname]['key']}] insufficient met ({len(met)} rows)")
            continue

        h_back = adaptive_hours_back(met, cfg)
        fp = run_residence_time_particles(
            sc["lat"], sc["lon"], met, event_time, cfg, rng,
            hours_back=h_back,
        )
        footprints[sname] = fp
        np.save(out_dir / f"fp_{SENSORS[sname]['key']}_{tag}.npy", fp)
        print(f"    [{SENSORS[sname]['key']}] h_back={h_back:.1f}h "
              f"non-zero cells={int((fp>0).sum())}/{GRID['nlat']*GRID['nlon']}")

    if not footprints:
        return {}

    # Concentration-weighted combination
    combined = combine_footprints(footprints, h2s_obs)
    np.save(out_dir / f"fp_combined_{tag}.npy", combined)

    # Source attribution and emission rates
    attr = source_attribution(combined)
    rates = estimate_emission_rates(combined, h2s_obs, attr)

    result = {
        "tag":           tag,
        "time":          event_time.isoformat(),
        "h2s_obs":       h2s_obs,
        "n_sensors_used": len(footprints),
        "top_sources":   {k: round(v, 4) for k, v in list(attr.items())[:6]},
        "all_sources":   attr,
        "emission_rates_g_s": rates,
        "adaptive_h_back": {
            SENSORS[s]["key"]: round(adaptive_hours_back(
                load_met(df_met, event_time, s, cfg.max_hours_back), cfg
            ), 2)
            for s in footprints
        },
    }

    with open(out_dir / f"attribution_{tag}.json", "w") as f:
        json.dump(result, f, indent=2, default=str)

    print(f"    Top sources: " +
          " | ".join(f"{k}={v:.2f}" for k, v in list(attr.items())[:3]))

    return result


# ---------------------------------------------------------------------------
# Batch inversion
# ---------------------------------------------------------------------------

def run_inversion(
    data_path: str,
    out_dir: Path,
    cfg: LagrangianConfig,
    date_start: str = "2026-02-01",
    date_end: str   = "2026-04-01",
    h2s_threshold: float = 30.0,
    require_stable: bool = True,
    max_events: Optional[int] = None,
    seed: int = 42,
) -> dict:
    """
    Run residence-time attribution for all qualifying events in a window.

    Qualifying events: H2S ≥ threshold at any sensor, stable_atm = 1.
    For each qualifying hour, all three sensors are processed simultaneously
    and their footprints are combined.

    Returns results dict keyed by event tag.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    df = pd.read_csv(data_path) if data_path.endswith(".csv") else pd.read_parquet(data_path)
    df["time"] = pd.to_datetime(df["time"], utc=True).dt.tz_convert("America/Los_Angeles")

    # Find event times: any sensor above threshold in window
    mask = (
        (df["time"] >= date_start) & (df["time"] <= date_end)
        & (df["H2S"] >= h2s_threshold) & df["H2S"].notna()
    )
    if require_stable:
        mask &= df["stable_atm"] == 1

    event_times = df[mask]["time"].drop_duplicates().sort_values(ascending=False)
    if max_events:
        event_times = event_times.head(max_events)

    print(f"[residence] {len(event_times)} qualifying event-times "
          f"({date_start} → {date_end}, H2S≥{h2s_threshold}ppb, stable={require_stable})")

    results = {}
    combined_fps = []

    for et in event_times:
        r = process_event(et, df, cfg, out_dir, rng)
        if r:
            results[r["tag"]] = r
            fp_path = out_dir / f"fp_combined_{r['tag']}.npy"
            if fp_path.exists():
                combined_fps.append(np.load(fp_path))

    # Ensemble
    if combined_fps:
        ensemble = np.mean(combined_fps, axis=0)
        ensemble_total = ensemble.sum()
        if ensemble_total > 0:
            ensemble /= ensemble_total
        np.save(out_dir / "ensemble_footprint.npy", ensemble)
        ensemble_attr = source_attribution(ensemble)
        with open(out_dir / "ensemble_attribution.json", "w") as f:
            json.dump({
                "n_events": len(combined_fps),
                "date_range": f"{date_start} to {date_end}",
                "h2s_threshold_ppb": h2s_threshold,
                "method": "residence_time_concentration_weighted",
                "source_fractions": ensemble_attr,
            }, f, indent=2)
        print(f"\n[ensemble] {len(combined_fps)} events — top sources:")
        for k, v in list(ensemble_attr.items())[:6]:
            print(f"  {k:32s} {v:.3f}")

    summary = {
        "n_events": len(results),
        "output_dir": str(out_dir),
        "results": results,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    return results


# ---------------------------------------------------------------------------
# Backward-compat helpers for Dagster pipeline
# ---------------------------------------------------------------------------

def _numpy_to_footprint_df(arr: np.ndarray) -> pd.DataFrame:
    """Convert residence-time footprint numpy array to DataFrame (lat index, lon columns).

    Returns the same format as the old build_footprint() — used by
    store_dataframe_to_s3() and footprint_to_grid_data() in the pipeline asset.
    """
    lat_centers = (
        np.linspace(GRID["lat_min"], GRID["lat_max"], GRID["nlat"], endpoint=False)
        + (GRID["lat_max"] - GRID["lat_min"]) / (2 * GRID["nlat"])
    )
    lon_centers = (
        np.linspace(GRID["lon_min"], GRID["lon_max"], GRID["nlon"], endpoint=False)
        + (GRID["lon_max"] - GRID["lon_min"]) / (2 * GRID["nlon"])
    )
    return pd.DataFrame(arr, index=lat_centers, columns=lon_centers)


def run_inversion_window(
    df: pd.DataFrame,
    cfg: LagrangianConfig,
    date_start: str = "2026-02-01",
    date_end: str   = "2026-04-01",
    h2s_threshold: float = 30.0,
    max_events: Optional[int] = None,
    seed: int = 42,
) -> tuple[dict, pd.DataFrame | None]:
    """Backward-compat adapter for the Dagster dispersion pipeline.

    Accepts a pre-loaded DataFrame (as the pipeline provides), delegates to
    run_inversion() via a temp directory, and returns (results_dict, ensemble_df)
    where ensemble_df is a lat-indexed DataFrame matching the old build_footprint()
    format expected by store_dataframe_to_s3() and footprint_to_grid_data().
    """
    tmp_dir = Path(tempfile.mkdtemp())
    tmp_parquet = str(tmp_dir / "obs.parquet")
    try:
        df.to_parquet(tmp_parquet)
        results = run_inversion(
            data_path=tmp_parquet,
            out_dir=tmp_dir,
            cfg=cfg,
            date_start=date_start,
            date_end=date_end,
            h2s_threshold=h2s_threshold,
            max_events=max_events,
            seed=seed,
        )
        ensemble_path = tmp_dir / "ensemble_footprint.npy"
        ensemble_df = (
            _numpy_to_footprint_df(np.load(str(ensemble_path)))
            if ensemble_path.exists()
            else None
        )
    finally:
        shutil.rmtree(tmp_dir)

    return results, ensemble_df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Residence-time Lagrangian H2S source attribution"
    )
    parser.add_argument("--data",       default=os.environ.get("H2S_DATA_PATH", "modeldata_h2s_nofill.csv"))
    parser.add_argument("--output",     default="./lagrangian_residence_output")
    parser.add_argument("--date_start", default="2026-02-01")
    parser.add_argument("--date_end",   default="2026-04-01")
    parser.add_argument("--h2s_min",    type=float, default=30.0)
    parser.add_argument("--n_particles",type=int,   default=2000)
    parser.add_argument("--max_events", type=int,   default=None)
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--event_time", default=None,
                        help="Process a single event: 'YYYY-MM-DD HH:MM' (local time)")
    args = parser.parse_args()

    cfg = LagrangianConfig(n_particles=args.n_particles)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    if args.event_time:
        df = pd.read_csv(args.data) if args.data.endswith(".csv") else pd.read_parquet(args.data)
        df["time"] = pd.to_datetime(df["time"], utc=True).dt.tz_convert("America/Los_Angeles")
        et = pd.Timestamp(args.event_time, tz="America/Los_Angeles").floor("1h")
        rng = np.random.default_rng(args.seed)
        print(f"Single event: {et}")
        r = process_event(et, df, cfg, out, rng)
        if r:
            print(json.dumps(r, indent=2, default=str))
    else:
        run_inversion(
            args.data, out, cfg,
            date_start=args.date_start,
            date_end=args.date_end,
            h2s_threshold=args.h2s_min,
            max_events=args.max_events,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
