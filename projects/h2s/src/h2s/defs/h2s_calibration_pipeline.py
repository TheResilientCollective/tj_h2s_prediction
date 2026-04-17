"""H2S emission calibration pipeline — rolling-window channel-snapped NNLS.

Inverts a time-averaged H2S emission Q field along the Tijuana River channel
grid by stacking per-sensor × per-timestep footprint+sensitivity rows from a
rolling window of qualifying events (≥ 30 ppb at any sensor, stable BL).

Three assets run nightly:

  1. ``rolling_footprint_matrix``    — per-event residence-time footprints
                                       for each sensor/timestep in the window.
  2. ``channel_emission_inversion``  — stacked-block NNLS over the window.
                                       Writes Q_field parquet + latest pointer.
  3. ``calibration_diagnostics``     — leave-one-sensor-out CV + forward
                                       reconstruction RMSE + Σ Q sanity
                                       check. Writes diagnostics JSON.

Output Q_field parquet is the input that
``gaussian_forward_forecast_detailed`` prefers over ``EMISSION_RATES_PATH``.
"""
import json
from dataclasses import asdict
from typing import Optional

import dagster as dg
import numpy as np
import pandas as pd

from h2s.constants import (
    CALIBRATION_BASE_PATH,
    OBS_DATA_PATH,
    Q_FIELD_DIAGNOSTICS_LATEST_PATH,
    Q_FIELD_DIAGNOSTICS_PATH,
    Q_FIELD_LATEST_JSON_PATH,
    Q_FIELD_LATEST_PATH,
    Q_FIELD_PATH,
    Q_FIELD_VIZ_BUDGET_LATEST_PATH,
    Q_FIELD_VIZ_BUDGET_PATH,
    Q_FIELD_VIZ_CV_LATEST_PATH,
    Q_FIELD_VIZ_CV_PATH,
    Q_FIELD_VIZ_MAP_LATEST_PATH,
    Q_FIELD_VIZ_MAP_PATH,
)
from h2s.dispersion.emission_inversion import (
    InversionConfig,
    batch_inversion_stacked,
    build_channel_grid,
    build_sensitivity_matrix,
    q_field_to_parquet_rows,
    solve_nnls,
)
from h2s.dispersion.lagrangian import (
    LagrangianConfig,
    SENSORS,
    adaptive_hours_back,
    load_met,
    run_residence_time_particles,
)


# ==============================================================================
# Config
# ==============================================================================

class CalibrationConfig(dg.Config):
    """Runtime config for the rolling emission calibration."""

    # Rolling window
    window_days: int = 7              # look back this many days from now (or date_end)
    date_end: Optional[str] = None    # ISO date (inclusive). Default: most recent obs time.

    # Event gating
    h2s_threshold_ppb: float = 30.0
    require_stable: bool = True       # stable_atm == 1 (calm nocturnal BL)
    max_events: int = 48              # cap timesteps processed per window

    # Lagrangian footprints (per-sensor per-timestep)
    n_particles: int = 1500           # reduced from default 2000 for batch speed
    hours_back: float = 2.0           # valley-scale sources (1-7 km)

    # Channel grid
    segment_spacing_m: float = 150.0

    # Inversion regularization
    lambda_l1: float = 0.3
    lambda_smooth: float = 0.0
    background_ppb: float = 1.0

    # Diagnostics
    min_rows_for_inversion: int = 9   # ≥ 3 events × 3 sensors before we trust NNLS


# ==============================================================================
# Helpers
# ==============================================================================

def _get_event_times(
    df: pd.DataFrame,
    date_end: pd.Timestamp,
    window_days: int,
    threshold_ppb: float,
    require_stable: bool,
    max_events: int,
) -> list[pd.Timestamp]:
    """Find qualifying timesteps (elevated H2S, stable BL) in the rolling window."""
    date_start = date_end - pd.Timedelta(days=window_days)
    mask = (
        (df["time"] >= date_start)
        & (df["time"] <= date_end)
        & df["H2S"].notna()
        & (df["H2S"] >= threshold_ppb)
    )
    if require_stable and "stable_atm" in df.columns:
        mask &= df["stable_atm"] == 1
    times = (
        df[mask]["time"]
        .drop_duplicates()
        .sort_values(ascending=False)
        .head(max_events)
        .tolist()
    )
    return times


def _collect_event_row(
    df: pd.DataFrame,
    event_time: pd.Timestamp,
    cfg: LagrangianConfig,
    rng: np.random.Generator,
    background_ppb: float,
    log,
) -> Optional[dict]:
    """Build one event dict (time, h2s_obs, met_row, footprints) or None if no signal."""
    h2s_obs: dict[str, float] = {}
    for sname in SENSORS:
        row = df[(df["time"] == event_time) & (df["site_name"] == sname)]
        if not row.empty and pd.notna(row["H2S"].iloc[0]):
            h2s_obs[sname] = float(row["H2S"].iloc[0])
        else:
            h2s_obs[sname] = 0.0

    if max(h2s_obs.values()) < background_ppb:
        return None

    footprints: dict[str, np.ndarray] = {}
    met_row_used: Optional[pd.Series] = None

    for sname, sc in SENSORS.items():
        if h2s_obs[sname] < background_ppb:
            continue
        met = load_met(df, event_time, sname, cfg.max_hours_back)
        if len(met) < 2:
            continue
        h_back = adaptive_hours_back(met, cfg)
        fp = run_residence_time_particles(
            sc["lat"], sc["lon"], met, event_time, cfg, rng, hours_back=h_back,
        )
        footprints[sname] = fp
        if met_row_used is None:
            met_row_used = met.iloc[-1]  # last met row before event

    if not footprints or met_row_used is None:
        return None

    return {
        "time": event_time,
        "h2s_obs": h2s_obs,
        "met_row": met_row_used,
        "footprints": footprints,
    }


# ==============================================================================
# Asset 1: rolling_footprint_matrix
# ==============================================================================

@dg.asset(
    key_prefix="h2s",
    group_name="h2s_calibration",
    required_resource_keys={"s3"},
    kinds={"python", "s3"},
    description=(
        "Per-event Lagrangian residence-time footprints over a rolling window "
        "(default 7 days) of H2S events (≥30 ppb, stable BL). Returns a list "
        "of event dicts passed downstream to channel_emission_inversion."
    ),
)
def rolling_footprint_matrix(
    context: dg.AssetExecutionContext,
    config: CalibrationConfig,
) -> dg.Output[list]:
    log = context.log
    s3 = context.resources.s3

    log.info(f"Loading obs data from S3: {OBS_DATA_PATH}")
    url = s3.get_presigned_url(OBS_DATA_PATH)
    df = pd.read_parquet(url)
    df["time"] = pd.to_datetime(df["time"], utc=True).dt.tz_convert("America/Los_Angeles")
    log.info(f"Loaded {len(df)} obs rows; range {df['time'].min()} → {df['time'].max()}")

    if config.date_end:
        date_end = pd.Timestamp(config.date_end).tz_localize("America/Los_Angeles") \
            if pd.Timestamp(config.date_end).tzinfo is None \
            else pd.Timestamp(config.date_end).tz_convert("America/Los_Angeles")
    else:
        date_end = df["time"].max()

    event_times = _get_event_times(
        df,
        date_end=date_end,
        window_days=config.window_days,
        threshold_ppb=config.h2s_threshold_ppb,
        require_stable=config.require_stable,
        max_events=config.max_events,
    )
    log.info(
        f"Found {len(event_times)} qualifying event timesteps in window "
        f"({config.window_days}d ending {date_end}, H2S ≥ {config.h2s_threshold_ppb} ppb, "
        f"stable={config.require_stable})"
    )

    lcfg = LagrangianConfig(
        n_particles=config.n_particles,
        hours_back=config.hours_back,
    )
    rng = np.random.default_rng(42)

    events: list[dict] = []
    for i, et in enumerate(event_times):
        ev = _collect_event_row(df, et, lcfg, rng, config.background_ppb, log)
        if ev is not None:
            events.append(ev)
        if (i + 1) % 5 == 0:
            log.info(f"  processed {i + 1}/{len(event_times)} event timesteps")

    n_rows = sum(len(ev["footprints"]) for ev in events)
    log.info(f"Assembled {len(events)} events → ~{n_rows} (sensor,timestep) rows for NNLS")

    # Detach raw H2S/timestamps/met summary for metadata only; footprints stay in-process
    preview = [
        {
            "time": str(ev["time"]),
            "n_sensors": len(ev["footprints"]),
            "max_h2s_ppb": round(max(ev["h2s_obs"].values()), 1),
        }
        for ev in events[:10]
    ]

    return dg.Output(
        events,
        metadata={
            "n_events":             dg.MetadataValue.int(len(events)),
            "n_nnls_rows":          dg.MetadataValue.int(n_rows),
            "window_days":          dg.MetadataValue.int(config.window_days),
            "date_end":             dg.MetadataValue.text(str(date_end)),
            "h2s_threshold_ppb":    dg.MetadataValue.float(config.h2s_threshold_ppb),
            "n_particles":          dg.MetadataValue.int(config.n_particles),
            "hours_back":           dg.MetadataValue.float(config.hours_back),
            "events_preview":       dg.MetadataValue.json(preview),
        },
    )


# ==============================================================================
# Asset 2: channel_emission_inversion
# ==============================================================================

@dg.asset(
    key_prefix="h2s",
    group_name="h2s_calibration",
    required_resource_keys={"s3"},
    kinds={"python", "s3"},
    description=(
        "Stacked-block NNLS inversion: solves Q ≥ 0 : argmin ‖A·Q − C_obs‖² "
        "+ λ₁‖Q‖² over a rolling-window batch of events. Writes a channel-snapped "
        "Q field parquet (segment_idx, lat, lon, Q_g_s) plus a _latest pointer."
    ),
    deps=[dg.AssetKey(["h2s", "rolling_footprint_matrix"])],
)
def channel_emission_inversion(
    context: dg.AssetExecutionContext,
    config: CalibrationConfig,
    rolling_footprint_matrix: list,
) -> dg.MaterializeResult:
    log = context.log
    s3 = context.resources.s3

    events = rolling_footprint_matrix
    log.info(f"Inverting {len(events)} events ({sum(len(e['footprints']) for e in events)} rows)")

    channel_segments = build_channel_grid(segment_spacing_m=config.segment_spacing_m)
    log.info(f"Channel grid: {len(channel_segments)} segments @ {config.segment_spacing_m} m spacing")

    cfg = InversionConfig(
        segment_spacing_m=config.segment_spacing_m,
        lambda_l1=config.lambda_l1,
        lambda_smooth=config.lambda_smooth,
        background_ppb=config.background_ppb,
    )

    n_rows_expected = sum(len(e["footprints"]) for e in events)
    if n_rows_expected < config.min_rows_for_inversion:
        log.warning(
            f"Only {n_rows_expected} rows available — below min_rows_for_inversion="
            f"{config.min_rows_for_inversion}. Skipping inversion this cycle."
        )
        return dg.MaterializeResult(metadata={
            "n_events":      dg.MetadataValue.int(len(events)),
            "n_rows":        dg.MetadataValue.int(n_rows_expected),
            "status":        dg.MetadataValue.text("skipped_insufficient_rows"),
        })

    result = batch_inversion_stacked(events, channel_segments, cfg)

    if result.get("reason") == "no_rows":
        log.warning("batch_inversion_stacked returned no rows — skipping S3 write")
        return dg.MaterializeResult(metadata={
            "n_events": dg.MetadataValue.int(0),
            "status":   dg.MetadataValue.text("no_rows"),
        })

    q_total = float(result["Q_total_g_s"])
    active = result.get("active_sources", [])
    log.info(
        f"Inversion complete: Σ Q = {q_total:.1f} g/s, {len(active)} active segments "
        f"({len(channel_segments)} total)"
    )
    if active:
        top = active[:5]
        log.info("Top 5 active segments: " + ", ".join(
            f"seg_{s['channel_index']:03d}@({s['lat']:.4f},{s['lon']:.4f})={s['Q_g_s']}g/s"
            for s in top
        ))

    # --- Serialize Q field to parquet (all segments, stable schema) ---
    rows = q_field_to_parquet_rows(result, channel_segments)
    q_df = pd.DataFrame(rows)

    run_tag = pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M")
    versioned_path = Q_FIELD_PATH.format(run_tag=run_tag)

    import io as _io
    buf = _io.BytesIO()
    q_df.to_parquet(buf, index=False)
    parquet_bytes = buf.getvalue()

    s3.putFile(parquet_bytes, path=versioned_path, content_type="application/octet-stream")
    s3.putFile(parquet_bytes, path=Q_FIELD_LATEST_PATH, content_type="application/octet-stream")
    log.info(f"Uploaded Q field → {versioned_path} and {Q_FIELD_LATEST_PATH}")

    # --- GeoDemic-friendly JSON sidecar (active segments only) ---
    json_payload = {
        "timestamp":    pd.Timestamp.utcnow().isoformat(),
        "run_tag":      run_tag,
        "n_segments":   len(channel_segments),
        "n_active":     len(active),
        "Q_total_g_s":  round(q_total, 2),
        "n_events":     int(result.get("n_events", 0)),
        "n_rows":       int(result.get("n_rows", 0)),
        "sensor_rmse_ppb": result.get("sensor_rmse_ppb", {}),
        "active_sources": active,
        "config": {
            "segment_spacing_m": config.segment_spacing_m,
            "lambda_l1":         config.lambda_l1,
            "lambda_smooth":     config.lambda_smooth,
            "window_days":       config.window_days,
            "h2s_threshold_ppb": config.h2s_threshold_ppb,
        },
    }
    s3.putFile(
        json.dumps(json_payload, indent=2).encode(),
        path=Q_FIELD_LATEST_JSON_PATH,
        content_type="application/json",
    )
    log.info(f"Uploaded Q field JSON summary → {Q_FIELD_LATEST_JSON_PATH}")

    return dg.MaterializeResult(metadata={
        "n_events":        dg.MetadataValue.int(int(result.get("n_events", 0))),
        "n_nnls_rows":     dg.MetadataValue.int(int(result.get("n_rows", 0))),
        "n_segments":      dg.MetadataValue.int(len(channel_segments)),
        "n_active":        dg.MetadataValue.int(len(active)),
        "Q_total_g_s":     dg.MetadataValue.float(round(q_total, 2)),
        "sensor_rmse_ppb": dg.MetadataValue.json(result.get("sensor_rmse_ppb", {})),
        "s3_versioned":    dg.MetadataValue.text(versioned_path),
        "s3_latest":       dg.MetadataValue.text(Q_FIELD_LATEST_PATH),
        "run_tag":         dg.MetadataValue.text(run_tag),
    })


# ==============================================================================
# Asset 3: calibration_diagnostics
# ==============================================================================

@dg.asset(
    key_prefix="h2s",
    group_name="h2s_calibration",
    required_resource_keys={"s3"},
    kinds={"python", "s3"},
    description=(
        "Calibration sanity checks: leave-one-sensor-out CV RMSE, forward-"
        "reconstruction self-consistency, Σ Q budget check. Writes "
        "inversion_diagnostics_{run_tag}.json. Flags Slack-worthy failures via "
        "metadata but does not fail the asset."
    ),
    deps=[dg.AssetKey(["h2s", "channel_emission_inversion"])],
)
def calibration_diagnostics(
    context: dg.AssetExecutionContext,
    config: CalibrationConfig,
    rolling_footprint_matrix: list,
) -> dg.MaterializeResult:
    log = context.log
    s3 = context.resources.s3

    events = rolling_footprint_matrix
    channel_segments = build_channel_grid(segment_spacing_m=config.segment_spacing_m)
    cfg = InversionConfig(
        segment_spacing_m=config.segment_spacing_m,
        lambda_l1=config.lambda_l1,
        lambda_smooth=config.lambda_smooth,
        background_ppb=config.background_ppb,
    )

    diagnostics: dict = {
        "timestamp": pd.Timestamp.utcnow().isoformat(),
        "n_events": len(events),
        "gates": {},
    }

    # --- Gate 1: leave-one-sensor-out CV ---
    sensor_names = list(SENSORS.keys())
    loo_rmse: dict[str, dict] = {}

    for held_out in sensor_names:
        train_rows_A: list[np.ndarray] = []
        train_rows_C: list[float] = []
        test_rows_A: list[np.ndarray] = []
        test_rows_C: list[float] = []

        for ev in events:
            h2s_obs = ev["h2s_obs"]
            met_row = ev["met_row"]
            footprints = ev["footprints"]

            sensors_present = [
                s for s in SENSORS
                if h2s_obs.get(s, 0.0) > cfg.background_ppb and s in footprints
            ]
            if not sensors_present:
                continue

            A_ev = build_sensitivity_matrix(channel_segments, met_row, sensors_present, cfg)
            if A_ev.max() < 1e-6:
                continue

            for i, sname in enumerate(sensors_present):
                c_bg = max(h2s_obs[sname] - cfg.background_ppb, 0.0)
                if sname == held_out:
                    test_rows_A.append(A_ev[i, :])
                    test_rows_C.append(c_bg)
                else:
                    train_rows_A.append(A_ev[i, :])
                    train_rows_C.append(c_bg)

        if not train_rows_A or not test_rows_A:
            loo_rmse[held_out] = {"rmse_ppb": None, "bias_ppb": None, "n_test": 0,
                                  "reason": "insufficient_rows"}
            continue

        A_train = np.vstack(train_rows_A)
        c_train = np.array(train_rows_C, dtype=float)
        Q_train = solve_nnls(A_train, c_train, cfg)

        A_test = np.vstack(test_rows_A)
        c_test = np.array(test_rows_C, dtype=float)
        c_pred = A_test @ Q_train
        diffs = c_test - c_pred

        rmse = float(np.sqrt(np.mean(diffs * diffs)))
        bias = float(np.mean(diffs))
        c_std = float(np.std(c_test)) if len(c_test) > 1 else 0.0

        loo_rmse[held_out] = {
            "rmse_ppb":    round(rmse, 2),
            "bias_ppb":    round(bias, 2),
            "c_obs_std":   round(c_std, 2),
            "n_test":      len(c_test),
            "rmse_over_std": round(rmse / c_std, 3) if c_std > 0 else None,
            "c_obs_ppb":   [round(float(v), 3) for v in c_test],
            "c_pred_ppb":  [round(float(v), 3) for v in c_pred],
        }

    gate1_pass = all(
        (v.get("rmse_over_std") is not None and v["rmse_over_std"] < 1.0
         and abs(v.get("bias_ppb") or 0.0) < 10.0)
        for v in loo_rmse.values()
        if v.get("n_test", 0) > 0
    )
    diagnostics["leave_one_sensor_out"] = loo_rmse
    diagnostics["gates"]["leave_one_sensor_out_pass"] = bool(gate1_pass)

    # --- Gate 2: budget sanity ---
    try:
        latest_parquet = s3.getFile(Q_FIELD_LATEST_PATH)
        import io as _io
        q_df = pd.read_parquet(_io.BytesIO(latest_parquet))
        q_total = float(q_df["Q_g_s"].sum())
    except Exception as exc:
        log.warning(f"Could not read Q_field_latest for budget gate: {exc}")
        q_total = 0.0

    budget_low, budget_high = 30.0, 500.0
    gate2_pass = budget_low <= q_total <= budget_high
    diagnostics["budget_sanity"] = {
        "Q_total_g_s":     round(q_total, 2),
        "allowed_low":     budget_low,
        "allowed_high":    budget_high,
        "anchor_g_s":      167.0,  # March 13 2026 calibration anchor
    }
    diagnostics["gates"]["budget_sanity_pass"] = bool(gate2_pass)

    # --- Summary ---
    all_gates = [diagnostics["gates"][k] for k in diagnostics["gates"]]
    diagnostics["gates"]["all_pass"] = bool(all(all_gates)) if all_gates else False

    log.info(
        f"Diagnostics summary: LOO={gate1_pass} | Budget={gate2_pass} (Σ Q={q_total:.1f} g/s) "
        f"| all_pass={diagnostics['gates']['all_pass']}"
    )

    run_tag = pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M")
    versioned_path = Q_FIELD_DIAGNOSTICS_PATH.format(run_tag=run_tag)
    payload = json.dumps(diagnostics, indent=2, default=str).encode()
    s3.putFile(payload, path=versioned_path, content_type="application/json")
    s3.putFile(payload, path=Q_FIELD_DIAGNOSTICS_LATEST_PATH, content_type="application/json")
    log.info(f"Uploaded diagnostics → {versioned_path}")

    return dg.MaterializeResult(metadata={
        "Q_total_g_s":          dg.MetadataValue.float(round(q_total, 2)),
        "loo_pass":             dg.MetadataValue.bool(bool(gate1_pass)),
        "budget_pass":          dg.MetadataValue.bool(bool(gate2_pass)),
        "all_gates_pass":       dg.MetadataValue.bool(bool(diagnostics["gates"]["all_pass"])),
        "loo_rmse_ppb":         dg.MetadataValue.json({k: v.get("rmse_ppb") for k, v in loo_rmse.items()}),
        "s3_versioned":         dg.MetadataValue.text(versioned_path),
        "s3_latest":            dg.MetadataValue.text(Q_FIELD_DIAGNOSTICS_LATEST_PATH),
        "run_tag":              dg.MetadataValue.text(run_tag),
    })


# ==============================================================================
# Asset 4: calibration_viz
# ==============================================================================

@dg.asset(
    key_prefix="h2s",
    group_name="h2s_calibration",
    required_resource_keys={"s3"},
    kinds={"python", "s3", "matplotlib"},
    description=(
        "Verification visualizations for the nightly calibration. Renders: "
        "(1) channel segment map colored by Q_g_s with SENSORS overlay, "
        "(2) leave-one-sensor-out CV predicted-vs-observed scatter, "
        "(3) Σ Q budget bar vs the 30/167/500 g/s reference lines. "
        "Uploads PNGs under tijuana/dispersion/calibration/viz/."
    ),
    deps=[
        dg.AssetKey(["h2s", "channel_emission_inversion"]),
        dg.AssetKey(["h2s", "calibration_diagnostics"]),
    ],
)
def calibration_viz(
    context: dg.AssetExecutionContext,
) -> dg.MaterializeResult:
    import io as _io

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    from h2s.dispersion.lagrangian import SENSORS

    log = context.log
    s3 = context.resources.s3

    # --- Load latest artifacts from S3 ---
    q_df = pd.read_parquet(_io.BytesIO(s3.getFile(Q_FIELD_LATEST_PATH)))
    diagnostics = json.loads(s3.getFile(Q_FIELD_DIAGNOSTICS_LATEST_PATH))

    loo = diagnostics.get("leave_one_sensor_out", {}) or {}
    budget = diagnostics.get("budget_sanity", {}) or {}
    q_total = float(budget.get("Q_total_g_s", q_df["Q_g_s"].sum()))

    run_tag = pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M")
    uploaded: dict[str, str] = {}

    # --- Plot 1: channel segment map ---
    fig, ax = plt.subplots(figsize=(9, 7))
    active = q_df[q_df["Q_g_s"] > 0]
    inactive = q_df[q_df["Q_g_s"] == 0]
    ax.scatter(
        inactive["lon"], inactive["lat"],
        c="lightgray", s=18, marker="o", alpha=0.6, label="inactive segment",
    )
    if not active.empty:
        sc = ax.scatter(
            active["lon"], active["lat"],
            c=active["Q_g_s"], s=60 + 8 * active["Q_g_s"],
            cmap="YlOrRd", edgecolor="black", linewidth=0.3, alpha=0.9,
            label="active Q (g/s)",
        )
        plt.colorbar(sc, ax=ax, label="Q (g/s)")

    for sname, sc_info in SENSORS.items():
        ax.plot(sc_info["lon"], sc_info["lat"], marker="^",
                markersize=14, color="blue", markeredgecolor="white")
        ax.annotate(sname, (sc_info["lon"], sc_info["lat"]),
                    xytext=(6, 6), textcoords="offset points", fontsize=9, color="blue")

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(
        f"Channel-snapped Q field — {len(active)}/{len(q_df)} active segments, "
        f"Σ Q = {q_total:.1f} g/s  ({run_tag})"
    )
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.3)
    legend_elts = [
        Line2D([], [], marker="o", color="w", markerfacecolor="lightgray",
               markersize=8, label="inactive"),
        Line2D([], [], marker="o", color="w", markerfacecolor="orange",
               markeredgecolor="black", markersize=10, label="active Q>0"),
        Line2D([], [], marker="^", color="w", markerfacecolor="blue",
               markersize=12, label="sensor"),
    ]
    ax.legend(handles=legend_elts, loc="upper left", fontsize=9)

    buf = _io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    png_bytes = buf.getvalue()
    versioned = Q_FIELD_VIZ_MAP_PATH.format(run_tag=run_tag)
    s3.putFile(png_bytes, path=versioned, content_type="image/png")
    s3.putFile(png_bytes, path=Q_FIELD_VIZ_MAP_LATEST_PATH, content_type="image/png")
    uploaded["map"] = versioned
    log.info(f"Uploaded Q field map → {versioned}")

    # --- Plot 2: leave-one-sensor-out CV scatter ---
    fig, ax = plt.subplots(figsize=(7, 7))
    colors = {"NESTOR - BES": "tab:blue", "IB CIVIC CTR": "tab:orange",
              "SAN YSIDRO": "tab:green"}
    max_val = 1.0
    for sname, data in loo.items():
        obs = data.get("c_obs_ppb") or []
        pred = data.get("c_pred_ppb") or []
        if not obs or not pred:
            continue
        rmse = data.get("rmse_ppb")
        bias = data.get("bias_ppb")
        ax.scatter(obs, pred, s=40, alpha=0.75,
                   c=colors.get(sname, "tab:gray"),
                   edgecolor="black", linewidth=0.3,
                   label=f"{sname} (RMSE={rmse}, bias={bias})")
        max_val = max(max_val, max(obs + pred))
    lim = max_val * 1.1
    ax.plot([0, lim], [0, lim], color="black", linestyle="--", linewidth=1, label="1:1")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("Observed C (ppb, held-out sensor)")
    ax.set_ylabel("Predicted C (ppb, from other sensors)")
    ax.set_title(
        f"Leave-one-sensor-out CV — gate pass: "
        f"{diagnostics.get('gates', {}).get('leave_one_sensor_out_pass', False)}"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)
    ax.set_aspect("equal")

    buf = _io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    png_bytes = buf.getvalue()
    versioned = Q_FIELD_VIZ_CV_PATH.format(run_tag=run_tag)
    s3.putFile(png_bytes, path=versioned, content_type="image/png")
    s3.putFile(png_bytes, path=Q_FIELD_VIZ_CV_LATEST_PATH, content_type="image/png")
    uploaded["cv_scatter"] = versioned
    log.info(f"Uploaded CV scatter → {versioned}")

    # --- Plot 3: Σ Q budget bar ---
    fig, ax = plt.subplots(figsize=(8, 3.5))
    budget_low = float(budget.get("allowed_low", 30.0))
    budget_high = float(budget.get("allowed_high", 500.0))
    anchor = float(budget.get("anchor_g_s", 167.0))
    gate_pass = bool(diagnostics.get("gates", {}).get("budget_sanity_pass", False))

    ax.axvspan(budget_low, budget_high, color="lightgreen", alpha=0.35, label="allowed")
    ax.axvline(anchor, color="gray", linestyle="--", linewidth=1.5,
               label=f"anchor ({anchor:.0f} g/s)")
    ax.axvline(q_total, color="green" if gate_pass else "red",
               linewidth=3.5, label=f"Σ Q = {q_total:.1f} g/s")
    ax.set_xlim(0, max(budget_high * 1.1, q_total * 1.1))
    ax.set_xlabel("Σ Q (g/s)")
    ax.set_yticks([])
    ax.set_title(f"Budget sanity — gate pass: {gate_pass}")
    ax.grid(True, axis="x", alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)

    buf = _io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    png_bytes = buf.getvalue()
    versioned = Q_FIELD_VIZ_BUDGET_PATH.format(run_tag=run_tag)
    s3.putFile(png_bytes, path=versioned, content_type="image/png")
    s3.putFile(png_bytes, path=Q_FIELD_VIZ_BUDGET_LATEST_PATH, content_type="image/png")
    uploaded["budget"] = versioned
    log.info(f"Uploaded budget bar → {versioned}")

    return dg.MaterializeResult(metadata={
        "run_tag":        dg.MetadataValue.text(run_tag),
        "map_png":        dg.MetadataValue.text(Q_FIELD_VIZ_MAP_LATEST_PATH),
        "cv_scatter_png": dg.MetadataValue.text(Q_FIELD_VIZ_CV_LATEST_PATH),
        "budget_png":     dg.MetadataValue.text(Q_FIELD_VIZ_BUDGET_LATEST_PATH),
        "n_active":       dg.MetadataValue.int(int((q_df["Q_g_s"] > 0).sum())),
        "Q_total_g_s":    dg.MetadataValue.float(round(q_total, 2)),
        "uploaded":       dg.MetadataValue.json(uploaded),
    })


# ==============================================================================
# Exported asset list (for Definitions)
# ==============================================================================

calibration_assets = [
    rolling_footprint_matrix,
    channel_emission_inversion,
    calibration_diagnostics,
    calibration_viz,
]
