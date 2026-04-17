"""
Channel-snapped H2S emission inversion.

Closes the loop between the backward Lagrangian footprint (dispersion/lagrangian.py)
and the forward Gaussian plume (dispersion/gaussian.py) by solving for channel-
segment emission rates Q (g/s) that reconcile sensor observations with the
forward model.

Pipeline (three mathematical steps kept deliberately separate):

  1. LOCATE  — project the combined backward residence-time footprint onto the
               Tijuana River / estuary channel grid with a Gaussian kernel.
               Produces a spatial prior f_j over channel segments.

  2. INVERT  — build sensitivity matrix A[sensor, segment] from the Gaussian
               plume kernel. Solve Q = argmin ‖A·Q − C_obs‖² + λ₁‖Q‖₁ subject
               to Q ≥ 0 via scipy.optimize.nnls on the augmented system.

  3. ITERATE — compute forward residuals C_obs − A·Q; re-weight footprints by
               positive residuals and add a correction ΔQ. Converges in 1–3
               iterations on validated events.

For operational use, `batch_inversion_stacked()` stacks per-sensor × per-timestep
rows from a rolling window of qualifying events, yielding an overdetermined
system even with only 3 sensors (~200–600 rows vs. ~100 channel segments).

Ported from a validated standalone prototype that inverted Apr 4 2026 peak
events — 15:00 localized to ~415 m of Saturn Blvd Bridge, 16:00 clustered 80–
425 m of Dairy Mart Bridge — consistent with wind-shift-driven attribution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import nnls

from h2s.dispersion.gaussian import (
    gaussian_plume_concentration,
    stability_class,
    ug_m3_to_ppb,
)
from h2s.dispersion.lagrangian import (
    GRID,
    M_PER_DEG_LAT,
    M_PER_DEG_LON,
    SENSORS,
)


# ---------------------------------------------------------------------------
# Channel waypoints — Tijuana River main stem + tributaries + estuary
# ---------------------------------------------------------------------------
# Ordered roughly NW-to-SE following the channel centerline. Points are
# interpolated at `segment_spacing_m` resolution by build_channel_grid() to
# yield ~100 segments. Updating waypoints changes the inversion's spatial
# basis — keep anchored to physical landmarks (bridges, outfalls, canyon
# confluences).

CHANNEL_WAYPOINTS: list[tuple[float, float]] = [
    # Upper estuary / beach outlet
    (32.570082, -117.127240),   # Oneonta Slough
    (32.563000, -117.127000),   # Estuary mid
    (32.556206, -117.126178),   # Tijuana River Beach Outlet
    # Main stem west
    (32.559383, -117.092992),   # Saturn Blvd Bridge
    (32.558000, -117.088000),   # Channel between Saturn and Hollister
    (32.554177, -117.084135),   # Hollister Bridge N
    (32.551466, -117.084021),   # Hollister Bridge S
    (32.548000, -117.084000),   # Channel south of Hollister
    # Hollister PS branch
    (32.547600, -117.088374),   # Hollister PS
    (32.546000, -117.090000),   # Branch connection
    # Main stem continuing SE
    (32.541000, -117.078000),   # Channel bend
    (32.541000, -117.070000),   # Mid-reach
    (32.539743, -117.064269),   # Silva Drain
    (32.548531, -117.064293),   # Dairy Mart Bridge
    (32.542103, -117.054117),   # TJ Crossing CDLP W
    (32.542166, -117.050325),   # TJ Crossing CDLP E
    # Smugglers Gulch / south tributaries
    (32.538600, -117.086230),   # Smugglers Gulch
    (32.538000, -117.092000),   # Canyon reach
    (32.536900, -117.099160),   # Goat Canyon
    (32.543476, -117.108026),   # Goat Canyon PS
    (32.540000, -117.115000),   # Lower canyon
    # Del Sol Canyon
    (32.539300, -117.068850),   # Del Sol Canyon
]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class InversionConfig:
    """Channel-snapped inversion parameters."""

    # Spatial basis
    segment_spacing_m: float  = 150.0   # channel grid resolution → ~100 segments
    footprint_kernel_m: float = 350.0   # Gaussian kernel for footprint→channel projection

    # Gaussian forward sensitivity matrix
    gauss_meandering_deg: float = 20.0  # Gifford (1961) wind meandering σ for stable BL

    # Inversion regularisation
    lambda_l1: float     = 0.3          # L1 sparsity — favors a few real sources
    lambda_smooth: float = 0.0          # along-channel first-difference smoothness (0 = off)

    # Residual-iteration loop
    max_iter: int        = 8
    tol_ppb: float       = 20.0         # convergence threshold on max sensor residual

    # Background
    background_ppb: float = 1.0


# ---------------------------------------------------------------------------
# Channel grid construction
# ---------------------------------------------------------------------------

def build_channel_grid(
    segment_spacing_m: float = 150.0,
    waypoints: Optional[list[tuple[float, float]]] = None,
) -> list[tuple[float, float]]:
    """Interpolate channel waypoints into evenly-spaced segment centroids.

    Uses local metres-per-degree conversions (``M_PER_DEG_LAT/LON`` from the
    lagrangian module, evaluated at the centroid latitude of the Tijuana
    valley) so segment lengths are metric-accurate without invoking a full
    geodesic library.

    Returns
    -------
    list[(lat, lon)]
        Segment centroid coordinates, one per ~`segment_spacing_m` metres
        along the channel.
    """
    pts = waypoints if waypoints is not None else CHANNEL_WAYPOINTS
    segments: list[tuple[float, float]] = []
    for i in range(len(pts) - 1):
        la0, lo0 = pts[i]
        la1, lo1 = pts[i + 1]
        dy_m = (la1 - la0) * M_PER_DEG_LAT
        dx_m = (lo1 - lo0) * M_PER_DEG_LON
        seg_len_m = float(np.sqrt(dy_m * dy_m + dx_m * dx_m))
        n = max(1, int(seg_len_m / segment_spacing_m))
        for k in range(n):
            t = (k + 0.5) / n
            segments.append((la0 + t * (la1 - la0), lo0 + t * (lo1 - lo0)))
    return segments


# ---------------------------------------------------------------------------
# LOCATE: project 2-D footprint onto channel grid
# ---------------------------------------------------------------------------

def project_footprint_to_channel(
    combined_footprint: np.ndarray,
    channel_segments: list[tuple[float, float]],
    kernel_m: float = 350.0,
) -> np.ndarray:
    """Project a 2-D residence-time footprint onto the channel grid.

    For each channel segment centroid, sums the footprint weighted by an
    isotropic Gaussian kernel of radius ``kernel_m``. The result is
    normalised to a probability distribution along the channel, giving a
    spatial prior for the inversion (where on the channel was the event
    likely sourced).

    Parameters
    ----------
    combined_footprint
        2-D array of shape ``(GRID["nlat"], GRID["nlon"])`` — the
        concentration-weighted multi-sensor footprint from
        ``lagrangian.combine_footprints``.
    channel_segments
        Output of :func:`build_channel_grid`.
    kernel_m
        Gaussian kernel radius in metres. 350 m matches the ~100 m grid
        cell scale with enough smoothing to bridge single-cell gaps.

    Returns
    -------
    np.ndarray
        1-D vector of length ``len(channel_segments)``, sums to 1 (or to
        ``1/n`` if the footprint is all-zero — uniform fallback).
    """
    lat_edges = np.linspace(GRID["lat_min"], GRID["lat_max"], GRID["nlat"] + 1)
    lon_edges = np.linspace(GRID["lon_min"], GRID["lon_max"], GRID["nlon"] + 1)
    lat_c = (lat_edges[:-1] + lat_edges[1:]) / 2.0
    lon_c = (lon_edges[:-1] + lon_edges[1:]) / 2.0
    lon_grid, lat_grid = np.meshgrid(lon_c, lat_c)

    sigma_lat_deg = kernel_m / M_PER_DEG_LAT
    sigma_lon_deg = kernel_m / M_PER_DEG_LON

    weights = np.zeros(len(channel_segments), dtype=float)
    for j, (la, lo) in enumerate(channel_segments):
        kernel = np.exp(
            -0.5 * (
                ((lat_grid - la) / sigma_lat_deg) ** 2
                + ((lon_grid - lo) / sigma_lon_deg) ** 2
            )
        )
        weights[j] = float(np.sum(combined_footprint * kernel))

    total = float(weights.sum())
    if total > 0:
        return weights / total
    return np.ones_like(weights) / len(weights)


# ---------------------------------------------------------------------------
# INVERT: Gaussian sensitivity matrix + NNLS solver
# ---------------------------------------------------------------------------

def build_sensitivity_matrix(
    channel_segments: list[tuple[float, float]],
    met_row: pd.Series,
    sensor_names: list[str],
    cfg: InversionConfig,
) -> np.ndarray:
    """Build ``A[sensor, segment]`` = ppb at sensor i from 1 g/s at segment j.

    Uses the same Gaussian plume kernel as ``dispersion.gaussian`` to
    guarantee self-consistency: a Q field inverted with this A and then
    replayed through ``run_forward_model_from_Q_field`` reconstructs the
    same concentrations.

    Parameters
    ----------
    channel_segments
        Output of :func:`build_channel_grid`.
    met_row
        Single-row pandas Series with ``wind_speed_10m``,
        ``wind_direction_10m``, ``temperature_2m``, ``is_night``.
    sensor_names
        Ordered list of sensor keys (matching ``SENSORS``). Row i of A
        corresponds to ``sensor_names[i]``.

    Returns
    -------
    A : (n_sensors, n_segments) ndarray, units ppb per (g/s).
    """
    ws = float(met_row.get("wind_speed_10m", 1.0))
    wd_deg = float(met_row.get("wind_direction_10m", 180.0))
    temp_c = float(met_row.get("temperature_2m", 18.0))
    is_night = bool(met_row.get("is_night", 1))
    stab = stability_class(ws, is_night)

    wd_rad = np.radians(wd_deg)
    u = -ws * np.sin(wd_rad)
    v = -ws * np.cos(wd_rad)

    n_sensors = len(sensor_names)
    n_segments = len(channel_segments)
    A = np.zeros((n_sensors, n_segments), dtype=float)

    for j, (src_lat, src_lon) in enumerate(channel_segments):
        for i, sname in enumerate(sensor_names):
            sc = SENSORS[sname]
            c_ug = gaussian_plume_concentration(
                source_lat=src_lat,
                source_lon=src_lon,
                emission_rate_g_s=1.0,
                receptor_lat=sc["lat"],
                receptor_lon=sc["lon"],
                wind_u=u,
                wind_v=v,
                stab=stab,
                sigma_theta_deg=cfg.gauss_meandering_deg,
            )
            A[i, j] = ug_m3_to_ppb(c_ug, temp_c)

    return A


def solve_nnls(
    A: np.ndarray,
    C_obs: np.ndarray,
    cfg: InversionConfig,
) -> np.ndarray:
    """Solve Q ≥ 0 : argmin ‖A·Q − C_obs‖² + λ₁‖Q‖² + λ_s‖D·Q‖² via NNLS.

    Ridge L1-like regularisation is approximated here by an identity
    penalty (``λ₁·‖Q‖²``). True L1 requires NNLS + projection; the
    identity penalty is convex, handled natively by scipy.optimize.nnls
    on the augmented system, and in practice produces sparse-enough
    solutions once combined with the NNLS non-negativity constraint.

    Column-scaling is applied so the solver operates on a well-
    conditioned ``A_scaled`` — segments far from sensors (tiny A columns)
    aren't artificially penalized by the identity term.

    Parameters
    ----------
    A : (n_rows, n_segments) ndarray
        Sensitivity rows from one or more (sensor, timestep) events stacked.
    C_obs : (n_rows,) ndarray
        Observed ppb at matching rows (background already subtracted).
    cfg : InversionConfig

    Returns
    -------
    Q : (n_segments,) ndarray, units g/s, all non-negative.
    """
    n_seg = A.shape[1]
    col_scale = np.maximum(A.max(axis=0), 1e-8)
    A_scaled = A / col_scale[np.newaxis, :]

    l1_block = cfg.lambda_l1 * np.eye(n_seg)
    blocks = [A_scaled, l1_block]
    rhs = [C_obs, np.zeros(n_seg)]

    if cfg.lambda_smooth > 0 and n_seg > 1:
        D = np.zeros((n_seg - 1, n_seg))
        for i in range(n_seg - 1):
            D[i, i] = 1.0
            D[i, i + 1] = -1.0
        blocks.append(cfg.lambda_smooth * D)
        rhs.append(np.zeros(n_seg - 1))

    A_aug = np.vstack(blocks)
    b_aug = np.concatenate(rhs)

    Q_scaled, _ = nnls(A_aug, b_aug)
    return Q_scaled / col_scale


# ---------------------------------------------------------------------------
# ITERATE: calibration loop for a single event
# ---------------------------------------------------------------------------

def calibration_loop(
    footprints: dict[str, np.ndarray],
    h2s_obs: dict[str, float],
    met_row: pd.Series,
    channel_segments: list[tuple[float, float]],
    cfg: InversionConfig,
) -> dict:
    """Run LOCATE → INVERT → ITERATE for one event timestep.

    Steps
    -----
    1. Footprint projection onto the channel grid (concentration-weighted
       multi-sensor combination, then Gaussian kernel smoothing).
    2. Initial NNLS inversion on ``A·Q = C_obs``.
    3. Residual loop: while ``max|C_obs − A·Q| > tol_ppb`` and
       ``iter < max_iter``, re-project positive residuals onto the channel
       and add a correction ΔQ.

    Returns
    -------
    dict with keys:
      - ``Q``          : (n_segments,) emission rates in g/s
      - ``Q_total_g_s``: ΣQ
      - ``active_sources``: list of {lat, lon, Q_g_s, fraction, channel_index}
        for segments carrying ≥1 % of ΣQ (and ≥0.5 g/s)
      - ``predicted_ppb``, ``residual_ppb``: per-sensor dicts
      - ``converged``, ``iterations``, ``history``
      - ``channel_prior``: initial footprint projection (list for JSON)
      - ``sensors_used``: ordered list of sensors that had signal
    """
    # --- Signal sensors and C_obs (background subtracted) ---
    sensors_used = [
        s for s in SENSORS
        if h2s_obs.get(s, 0.0) > cfg.background_ppb and s in footprints
    ]
    if not sensors_used:
        return {
            "Q": np.zeros(len(channel_segments)),
            "Q_total_g_s": 0.0,
            "active_sources": [],
            "predicted_ppb": {},
            "residual_ppb": {},
            "converged": False,
            "iterations": 0,
            "history": [],
            "channel_prior": [],
            "sensors_used": [],
            "reason": "no_signal",
        }

    C_obs = np.array(
        [max(h2s_obs[s] - cfg.background_ppb, 0.0) for s in sensors_used],
        dtype=float,
    )

    # --- LOCATE: concentration-weighted combined footprint → channel prior ---
    fp_weights = C_obs / max(C_obs.sum(), 1e-6)
    fp_shape = footprints[sensors_used[0]].shape
    combined_fp = np.zeros(fp_shape, dtype=float)
    for w, s in zip(fp_weights, sensors_used):
        combined_fp += w * footprints[s]

    channel_prior = project_footprint_to_channel(
        combined_fp, channel_segments, cfg.footprint_kernel_m
    )

    # --- INVERT: build A, solve initial NNLS ---
    A = build_sensitivity_matrix(channel_segments, met_row, sensors_used, cfg)

    if A.max() < 1e-6:
        return {
            "Q": np.zeros(len(channel_segments)),
            "Q_total_g_s": 0.0,
            "active_sources": [],
            "predicted_ppb": {s: round(cfg.background_ppb, 1) for s in sensors_used},
            "residual_ppb": {s: round(h2s_obs[s] - cfg.background_ppb, 1) for s in sensors_used},
            "converged": False,
            "iterations": 0,
            "history": [],
            "channel_prior": channel_prior.tolist(),
            "sensors_used": sensors_used,
            "reason": "zero_sensitivity",
        }

    Q = solve_nnls(A, C_obs, cfg)

    # --- ITERATE: residual-weighted correction ---
    history = []
    converged = False
    A_max = float(A.max())

    for iteration in range(cfg.max_iter):
        C_pred = A @ Q
        residuals = C_obs - C_pred
        max_resid = float(np.max(np.abs(residuals)))

        history.append({
            "iter": iteration,
            "Q_total_g_s": float(Q.sum()),
            "max_residual_ppb": max_resid,
            "C_pred": [float(x) for x in C_pred],
            "C_obs":  [float(x) for x in C_obs],
        })

        if max_resid < cfg.tol_ppb:
            converged = True
            break

        pos_res = np.clip(residuals, 0.0, None)
        pos_norm = pos_res / (float(np.abs(residuals).max()) + 1e-6)

        if pos_norm.sum() > 0:
            resid_fp = np.zeros(fp_shape, dtype=float)
            for w, s in zip(pos_norm, sensors_used):
                resid_fp += w * footprints[s]
            r_total = float(resid_fp.sum())
            if r_total > 0:
                resid_fp /= r_total
            resid_channel = project_footprint_to_channel(
                resid_fp, channel_segments, cfg.footprint_kernel_m
            )
            resid_mean_ppb = float(np.mean(pos_res))
            dQ = resid_channel * resid_mean_ppb / max(A_max, 0.01)
            Q = np.maximum(Q + dQ, 0.0)
        else:
            # Over-prediction: scale current solution down proportionally
            scale = 1.0 - float(np.clip(max_resid / (float(C_obs.mean()) + 1.0), 0.0, 0.3))
            Q = Q * scale

    # --- Active sources summary ---
    Q_total = float(Q.sum())
    q_threshold = max(0.01 * Q_total, 0.5)
    active_sources = [
        {
            "lat": float(channel_segments[i][0]),
            "lon": float(channel_segments[i][1]),
            "Q_g_s": round(float(Q[i]), 2),
            "fraction": round(float(Q[i] / Q_total), 4) if Q_total > 0 else 0.0,
            "channel_index": i,
        }
        for i in range(len(channel_segments))
        if Q[i] > q_threshold
    ]
    active_sources.sort(key=lambda x: -x["Q_g_s"])

    C_pred_final = A @ Q
    predicted = {
        s: round(float(C_pred_final[i] + cfg.background_ppb), 1)
        for i, s in enumerate(sensors_used)
    }
    residual = {
        s: round(float(h2s_obs[s] - (C_pred_final[i] + cfg.background_ppb)), 1)
        for i, s in enumerate(sensors_used)
    }

    return {
        "Q": Q,
        "Q_total_g_s": round(Q_total, 1),
        "active_sources": active_sources,
        "predicted_ppb": predicted,
        "residual_ppb": residual,
        "converged": converged,
        "iterations": len(history),
        "history": history,
        "channel_prior": channel_prior.tolist(),
        "sensors_used": sensors_used,
    }


def invert_event(
    event_time: pd.Timestamp,
    h2s_obs: dict[str, float],
    met_row: pd.Series,
    footprints: dict[str, np.ndarray],
    cfg: InversionConfig,
    channel_segments: Optional[list[tuple[float, float]]] = None,
) -> dict:
    """Thin wrapper around :func:`calibration_loop` that also annotates met.

    Builds the channel grid lazily if one isn't supplied. The returned
    dict gains ``tag``, ``time``, ``h2s_obs``, and ``met`` fields suitable
    for JSON serialisation.
    """
    segments = channel_segments if channel_segments is not None else build_channel_grid(
        cfg.segment_spacing_m
    )

    result = calibration_loop(
        footprints=footprints,
        h2s_obs=h2s_obs,
        met_row=met_row,
        channel_segments=segments,
        cfg=cfg,
    )

    result["tag"] = event_time.strftime("%Y%m%d_%H%M")
    result["time"] = event_time.isoformat()
    result["h2s_obs"] = h2s_obs
    result["met"] = {
        "wind_speed_10m":     round(float(met_row.get("wind_speed_10m", 0.0)), 2),
        "wind_direction_10m": round(float(met_row.get("wind_direction_10m", 0.0)), 1),
        "temperature_2m":     round(float(met_row.get("temperature_2m", 0.0)), 1),
        "is_night":           bool(met_row.get("is_night", 1)),
        "stability_class":    stability_class(
            float(met_row.get("wind_speed_10m", 1.0)),
            bool(met_row.get("is_night", 1)),
        ),
    }
    return result


# ---------------------------------------------------------------------------
# Rolling-window block NNLS (OPERATIONAL MODE)
# ---------------------------------------------------------------------------

def batch_inversion_stacked(
    events: list[dict],
    channel_segments: list[tuple[float, float]],
    cfg: InversionConfig,
) -> dict:
    """Block NNLS over a rolling window of events — operational inversion path.

    Stacks per-sensor × per-timestep rows from multiple event timesteps
    into a single tall NNLS problem. With 3 sensors × ~70 signal-bearing
    timesteps in a 7-day window we get ~200 rows against ~100 channel
    segments — well-conditioned, unlike the single-event 3-row system.

    The result is a **time-averaged** Q field representative of the window.
    Diel (day/night) decomposition can be layered on later by column-
    doubling the A matrix; not implemented here.

    Parameters
    ----------
    events : list of dicts, one per event timestep, each with:
        - ``time``     : pd.Timestamp (tz-aware)
        - ``h2s_obs``  : dict[sensor_name -> ppb]
        - ``met_row``  : pandas Series (wind/temp/is_night columns)
        - ``footprints``: dict[sensor_name -> 2-D ndarray]
    channel_segments : output of :func:`build_channel_grid`
    cfg : InversionConfig

    Returns
    -------
    dict with keys:
      - ``Q`` : (n_segments,) ndarray, g/s (time-averaged)
      - ``Q_total_g_s``, ``active_sources``
      - ``n_events``, ``n_rows``  : diagnostic counts
      - ``sensor_rmse_ppb``      : per-sensor reconstruction RMSE on stacked rows
      - ``per_event_predictions``: list of {time, sensor, obs_ppb, pred_ppb}
    """
    n_seg = len(channel_segments)

    rows_A: list[np.ndarray] = []
    rows_C: list[float] = []
    row_meta: list[dict] = []   # parallel list for per-row diagnostics

    for ev in events:
        h2s_obs = ev["h2s_obs"]
        met_row = ev["met_row"]
        footprints = ev["footprints"]
        event_time = ev["time"]

        sensors_present = [
            s for s in SENSORS
            if h2s_obs.get(s, 0.0) > cfg.background_ppb and s in footprints
        ]
        if not sensors_present:
            continue

        A_ev = build_sensitivity_matrix(
            channel_segments, met_row, sensors_present, cfg,
        )
        if A_ev.max() < 1e-6:
            continue

        for i, sname in enumerate(sensors_present):
            rows_A.append(A_ev[i, :])
            c_bg = max(h2s_obs[sname] - cfg.background_ppb, 0.0)
            rows_C.append(c_bg)
            row_meta.append({
                "time": event_time,
                "sensor": sname,
                "C_obs_ppb": float(h2s_obs[sname]),
            })

    if not rows_A:
        return {
            "Q": np.zeros(n_seg),
            "Q_total_g_s": 0.0,
            "active_sources": [],
            "n_events": 0,
            "n_rows": 0,
            "sensor_rmse_ppb": {},
            "per_event_predictions": [],
            "reason": "no_rows",
        }

    A_stack = np.vstack(rows_A)
    C_stack = np.array(rows_C, dtype=float)

    Q = solve_nnls(A_stack, C_stack, cfg)
    C_pred = A_stack @ Q

    # --- Per-sensor reconstruction RMSE ---
    sensor_rmse: dict[str, float] = {}
    by_sensor: dict[str, list[tuple[float, float]]] = {s: [] for s in SENSORS}
    for meta, c_obs, c_pred in zip(row_meta, C_stack, C_pred):
        by_sensor.setdefault(meta["sensor"], []).append((float(c_obs), float(c_pred)))
    for sname, pairs in by_sensor.items():
        if not pairs:
            continue
        diffs = np.array([o - p for o, p in pairs])
        sensor_rmse[sname] = round(float(np.sqrt(np.mean(diffs * diffs))), 2)

    # --- Active sources ---
    Q_total = float(Q.sum())
    q_threshold = max(0.01 * Q_total, 0.5)
    active_sources = [
        {
            "lat": float(channel_segments[i][0]),
            "lon": float(channel_segments[i][1]),
            "Q_g_s": round(float(Q[i]), 2),
            "fraction": round(float(Q[i] / Q_total), 4) if Q_total > 0 else 0.0,
            "channel_index": i,
        }
        for i in range(n_seg)
        if Q[i] > q_threshold
    ]
    active_sources.sort(key=lambda x: -x["Q_g_s"])

    per_event = [
        {
            "time": str(meta["time"]),
            "sensor": meta["sensor"],
            "obs_ppb": round(meta["C_obs_ppb"], 1),
            "pred_ppb": round(float(c_pred + cfg.background_ppb), 1),
        }
        for meta, c_pred in zip(row_meta, C_pred)
    ]

    # Unique event times
    n_events = len({str(m["time"]) for m in row_meta})

    return {
        "Q": Q,
        "Q_total_g_s": round(Q_total, 1),
        "active_sources": active_sources,
        "n_events": n_events,
        "n_rows": len(rows_C),
        "sensor_rmse_ppb": sensor_rmse,
        "per_event_predictions": per_event,
    }


# ---------------------------------------------------------------------------
# Forward-model handoff
# ---------------------------------------------------------------------------

def inversion_to_forward_sources(
    result: dict,
    min_fraction: float = 0.05,
) -> list[dict]:
    """Convert an inversion result to the source-list format the Gaussian
    forward model expects.

    Filters to segments carrying at least ``min_fraction`` of the total Q
    (default 5%) so the forward model runs over a handful of dominant
    sources rather than the full 100+ channel segments — typically 2–8
    point sources per event.

    Output dicts have keys ``name``, ``lat``, ``lon``, ``Q_g_s`` suitable
    for passing to ``run_forward_model_from_Q_field()``.
    """
    return [
        {
            "name": f"channel_{src['channel_index']:03d}_{src['lat']:.4f}_{abs(src['lon']):.4f}",
            "lat":  src["lat"],
            "lon":  src["lon"],
            "Q_g_s": src["Q_g_s"],
        }
        for src in result.get("active_sources", [])
        if src.get("fraction", 0.0) >= min_fraction
    ]


def q_field_to_parquet_rows(
    result: dict,
    channel_segments: list[tuple[float, float]],
) -> list[dict]:
    """Flatten a full Q vector into per-segment dicts for parquet storage.

    Includes all segments (even Q=0) so the parquet has a stable schema
    with constant segment_idx semantics across runs.
    """
    Q = result.get("Q")
    if Q is None:
        return []
    rows = []
    for i, (lat, lon) in enumerate(channel_segments):
        rows.append({
            "segment_idx": i,
            "lat": float(lat),
            "lon": float(lon),
            "Q_g_s": round(float(Q[i]), 4),
        })
    return rows


# Public constants for downstream callers / diagnostics
__all__ = [
    "CHANNEL_WAYPOINTS",
    "InversionConfig",
    "build_channel_grid",
    "project_footprint_to_channel",
    "build_sensitivity_matrix",
    "solve_nnls",
    "calibration_loop",
    "invert_event",
    "batch_inversion_stacked",
    "inversion_to_forward_sources",
    "q_field_to_parquet_rows",
]
