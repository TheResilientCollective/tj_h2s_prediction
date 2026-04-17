"""H2S emission calibration pipeline — weekly-partitioned channel-snapped NNLS.

Inverts a time-averaged H2S emission Q field along the Tijuana River channel
grid by stacking per-sensor × per-timestep footprint + sensitivity rows across
the qualifying events (≥ 5 ppb at any sensor, stable BL — the smell-
detection / resident-complaint threshold) inside a 7-day partition window.

Four assets run per weekly partition (partition key = week-start Monday):

  1. ``rolling_footprint_matrix``    — per-event residence-time footprints
                                       for each sensor/timestep in the partition
                                       (events ≥ 5 ppb, stable BL).
  2. ``channel_emission_inversion``  — stacked-block NNLS over the partition.
                                       Writes Q_field parquet (per-partition +
                                       `_latest` pointer when the partition is
                                       recent). Skips the inversion when the
                                       week has fewer than
                                       ``min_events_per_week`` events.
  3. ``calibration_diagnostics``     — two CV sub-gates + Σ Q sanity. Gate 1
                                       has two parts that must both pass:
                                       leave-one-sensor-out (tests sensor
                                       disagreement) and leave-one-time-fold-out
                                       (random k-fold, adaptive k; tests
                                       temporal stability of Q). Writes
                                       per-partition diagnostics JSON.
  4. ``calibration_viz``             — four verification PNGs (Q-field map,
                                       LOSO scatter, LOTO scatter, budget bar).

Supports historical backfills (2025-onward) by re-running the job against
any prior weekly partition. The `_latest` pointer only moves when the
partition's end is within the last ``Q_FIELD_LATEST_MAX_AGE_DAYS`` days so
backfills don't stamp stale historical Q fields onto the live forecast.
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
    Q_FIELD_LATEST_MAX_AGE_DAYS,
    Q_FIELD_LATEST_PATH,
    Q_FIELD_PATH,
    Q_FIELD_VIZ_BUDGET_LATEST_PATH,
    Q_FIELD_VIZ_BUDGET_PATH,
    Q_FIELD_VIZ_CV_LATEST_PATH,
    Q_FIELD_VIZ_CV_PATH,
    Q_FIELD_VIZ_LOTO_LATEST_PATH,
    Q_FIELD_VIZ_MAP_LATEST_PATH,
    Q_FIELD_VIZ_MAP_PATH,
    Q_FIELD_WEEKLY_DIAGNOSTICS_PATH,
    Q_FIELD_WEEKLY_JSON_PATH,
    Q_FIELD_WEEKLY_PATH,
    Q_FIELD_WEEKLY_VIZ_BUDGET_PATH,
    Q_FIELD_WEEKLY_VIZ_CV_PATH,
    Q_FIELD_WEEKLY_VIZ_LOTO_PATH,
    Q_FIELD_WEEKLY_VIZ_MAP_PATH,
)


# ==============================================================================
# Partition Definition
# ==============================================================================
# Weekly partitions, Monday-start, starting with the first Monday of 2025.
# end_offset=0 means a partition is only available once its week has completed;
# the weekly schedule fires on the following Monday and materializes the
# previous week.

CALIBRATION_WEEKLY_PARTITIONS = dg.WeeklyPartitionsDefinition(
    start_date="2025-01-06",  # First Monday of 2025 (2025-01-01 was Wednesday)
    day_offset=1,             # 1 = Monday-start week
    timezone="UTC",
    end_offset=0,
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
    date_end: Optional[str] = None    # ISO date (inclusive). Ignored for partitioned
                                      # runs (partition key drives window bounds).

    # Event gating — 5 ppb matches the community smell-detection threshold
    # (residents report nuisance at 5-10 ppb, well below the 30 ppb ORANGE alert).
    # Lower threshold also helps the NNLS: more rows/week → better conditioning.
    h2s_threshold_ppb: float = 5.0
    require_stable: bool = True       # stable_atm == 1 (calm nocturnal BL)
    max_events: int = 48              # cap timesteps processed per window
    min_events_per_week: int = 3      # skip weeks with fewer qualifying events than this

    # Lagrangian footprints (per-sensor per-timestep)
    n_particles: int = 1500           # reduced from default 2000 for batch speed
    hours_back: float = 2.0           # valley-scale sources (1-7 km)

    # Channel grid
    segment_spacing_m: float = 150.0

    # Inversion regularization
    lambda_l1: float = 0.3
    lambda_smooth: float = 0.0
    background_ppb: float = 1.0

    # Gaussian plume sensitivity-matrix geometry
    gauss_meandering_deg: float = 20.0  # Gifford (1961) wind meandering σ for stable BL;
                                        # lower values concentrate the plume (raises max_A)

    # Geometry-plausibility pre-filter — events needing more than this much Q
    # on the single strongest segment under the current kernel are skipped
    # before NNLS (they can't be reproduced by any forward-model Q).
    q_required_max_g_s: float = 500.0

    # Diagnostics
    min_rows_for_inversion: int = 9   # ≥ 3 events × 3 sensors before we trust NNLS


# ==============================================================================
# Helpers
# ==============================================================================

def _partition_window(
    context: dg.AssetExecutionContext,
    config: CalibrationConfig,
    df_time_max: pd.Timestamp,
) -> tuple[pd.Timestamp, pd.Timestamp, Optional[str]]:
    """Resolve [window_start, window_end, partition_key] for this materialization.

    When a weekly partition key is attached to the run, the partition's
    [start, end) window drives the selection (7 days, ignoring ``window_days`` /
    ``date_end`` config). Otherwise we fall back to the legacy rolling-window
    behavior (window_days ending at ``date_end`` or the most recent obs time).
    """
    partition_key: Optional[str] = None
    try:
        partition_key = context.partition_key
    except Exception:
        partition_key = None

    if partition_key:
        start = pd.Timestamp(partition_key).tz_localize("UTC").tz_convert("America/Los_Angeles")
        end = start + pd.Timedelta(days=7)
        return start, end, partition_key

    if config.date_end:
        end_raw = pd.Timestamp(config.date_end)
        end = (
            end_raw.tz_localize("America/Los_Angeles")
            if end_raw.tzinfo is None
            else end_raw.tz_convert("America/Los_Angeles")
        )
    else:
        end = df_time_max
    start = end - pd.Timedelta(days=config.window_days)
    return start, end, None


def _is_partition_recent(partition_key: Optional[str], max_age_days: int) -> bool:
    """True when the partition's end is within max_age_days of today (UTC)."""
    if not partition_key:
        return True  # legacy unpartitioned run: always recent
    end = pd.Timestamp(partition_key).tz_localize("UTC") + pd.Timedelta(days=7)
    now_utc = pd.Timestamp.now(tz="UTC")
    return (now_utc - end).days <= max_age_days


def _load_inversion_sidecar(
    s3,
    partition_key: Optional[str],
    fallback_cfg: "InversionConfig",
    log,
) -> tuple["InversionConfig", dict]:
    """Read the Q_field.json sidecar written by channel_emission_inversion.

    Returns (InversionConfig, full_sidecar_payload_dict).  The sidecar is the
    single source of truth so that diagnostics evaluate CV under the exact
    same regularization as was used to fit Q.  Without this, an operator
    overriding `lambda_l1` on `calibration_diagnostics` alone (or forgetting
    to set it there) would silently fit Q under one lambda and evaluate
    gates under another.

    Falls back to `(fallback_cfg, {})` if the sidecar is missing / unparseable
    (legacy unpartitioned runs, or the first materialization before
    `channel_emission_inversion` wrote the enriched sidecar).
    """
    from dataclasses import fields

    if partition_key:
        sidecar_path = Q_FIELD_WEEKLY_JSON_PATH.format(partition=partition_key)
    else:
        sidecar_path = Q_FIELD_LATEST_JSON_PATH

    try:
        payload = json.loads(s3.getFile(sidecar_path))
    except Exception as exc:
        log.warning(
            f"Could not read inversion sidecar {sidecar_path} ({exc}); "
            f"diagnostics falling back to op-level config."
        )
        return fallback_cfg, {}

    sidecar = payload.get("inversion_config") or {}
    if not sidecar:
        log.warning(
            f"Sidecar {sidecar_path} has no inversion_config block — "
            f"falling back to op-level config. Rerun channel_emission_inversion "
            f"to populate it."
        )
        return fallback_cfg, payload

    valid_keys = {f.name for f in fields(InversionConfig)}
    loaded = InversionConfig(**{k: v for k, v in sidecar.items() if k in valid_keys})

    if abs(loaded.lambda_l1 - fallback_cfg.lambda_l1) > 1e-9:
        log.warning(
            f"Op-level config.lambda_l1={fallback_cfg.lambda_l1} differs from "
            f"sidecar lambda_l1={loaded.lambda_l1}. Using sidecar (single source "
            f"of truth — set lambda_l1 only on channel_emission_inversion)."
        )
    return loaded, payload


def _get_event_times(
    df: pd.DataFrame,
    date_start: pd.Timestamp,
    date_end: pd.Timestamp,
    threshold_ppb: float,
    require_stable: bool,
    max_events: int,
) -> list[pd.Timestamp]:
    """Find qualifying timesteps (elevated H2S, stable BL) within [start, end]."""
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


def _time_fold_cv(
    per_event_rows: list,
    cfg,
    *,
    seed: int = 0,
) -> dict:
    """Random k-fold leave-one-time-fold-out CV over per-event rows.

    Each entry of ``per_event_rows`` is a dict with keys ``time``, ``sensors``
    (list of sensor names), ``A`` (row block for this event, shape
    (n_sensors, n_segments)), and ``c_bg`` (background-subtracted obs vector
    of length n_sensors).

    Returns a dict shaped like the one described in the plan file:
    ``{n_folds, k_rule, fold_type, seed, n_events_total, overall, per_fold,
    per_sensor}``.  ``per_sensor`` entries carry the same ``c_obs_ppb`` /
    ``c_pred_ppb`` arrays that the LOSO block produces, so the scatter
    plotter can consume either without a shape change.
    """
    n_events = len(per_event_rows)
    if n_events < 2:
        return {"status": "skipped_insufficient_events", "n_events_total": n_events}

    k = max(2, min(5, n_events // 3))
    rng = np.random.default_rng(seed)
    order = np.arange(n_events)
    rng.shuffle(order)
    folds = [list(f) for f in np.array_split(order, k)]

    per_sensor: dict[str, dict] = {}
    per_fold: list[dict] = []
    all_obs: list[float] = []
    all_pred: list[float] = []

    for i, test_idx in enumerate(folds):
        train_idx = [j for j in range(n_events) if j not in set(test_idx)]
        if not train_idx or not test_idx:
            continue

        A_train = np.vstack([per_event_rows[j]["A"] for j in train_idx])
        c_train = np.concatenate([per_event_rows[j]["c_bg"] for j in train_idx])
        if A_train.size == 0 or A_train.max() < 1e-6:
            continue
        Q_train = solve_nnls(A_train, c_train, cfg)

        fold_obs: list[float] = []
        fold_pred: list[float] = []
        for j in test_idx:
            ev = per_event_rows[j]
            c_pred = ev["A"] @ Q_train
            for sname, c_obs, cp in zip(ev["sensors"], ev["c_bg"], c_pred):
                sd = per_sensor.setdefault(
                    sname, {"c_obs_ppb": [], "c_pred_ppb": []}
                )
                sd["c_obs_ppb"].append(round(float(c_obs), 3))
                sd["c_pred_ppb"].append(round(float(cp), 3))
                fold_obs.append(float(c_obs))
                fold_pred.append(float(cp))
                all_obs.append(float(c_obs))
                all_pred.append(float(cp))

        if fold_obs:
            diffs = np.array(fold_obs) - np.array(fold_pred)
            per_fold.append({
                "fold":      i,
                "n_events":  len(test_idx),
                "n_test":    len(fold_obs),
                "rmse_ppb":  round(float(np.sqrt(np.mean(diffs * diffs))), 2),
                "bias_ppb":  round(float(np.mean(diffs)), 2),
            })

    for sname, sd in per_sensor.items():
        obs_arr = np.array(sd["c_obs_ppb"])
        pred_arr = np.array(sd["c_pred_ppb"])
        diffs = obs_arr - pred_arr
        sd["n_test"] = int(len(obs_arr))
        sd["rmse_ppb"] = round(float(np.sqrt(np.mean(diffs * diffs))), 2) if len(diffs) else None
        sd["bias_ppb"] = round(float(np.mean(diffs)), 2) if len(diffs) else None

    obs_arr = np.array(all_obs)
    pred_arr = np.array(all_pred)
    if len(obs_arr) == 0:
        return {"status": "no_test_rows", "n_events_total": n_events}

    diffs = obs_arr - pred_arr
    rmse = float(np.sqrt(np.mean(diffs * diffs)))
    bias = float(np.mean(diffs))
    c_std = float(np.std(obs_arr)) if len(obs_arr) > 1 else 0.0

    return {
        "n_folds":          len(per_fold),
        "k_rule":           "max(2, min(5, n_events // 3))",
        "fold_type":        "random",
        "seed":             int(seed),
        "n_events_total":   n_events,
        "overall": {
            "rmse_ppb":        round(rmse, 2),
            "bias_ppb":        round(bias, 2),
            "c_obs_std":       round(c_std, 2),
            "rmse_over_std":   round(rmse / c_std, 3) if c_std > 0 else None,
            "n_test":          int(len(obs_arr)),
        },
        "per_fold":         per_fold,
        "per_sensor":       per_sensor,
    }


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
    partitions_def=CALIBRATION_WEEKLY_PARTITIONS,
    description=(
        "Per-event Lagrangian residence-time footprints over a weekly partition "
        "of H2S events (≥5 ppb, stable BL — community smell-detection threshold). "
        "Partition key = week-start Monday; the 7-day window is "
        "[partition_key, partition_key + 7 days)."
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

    date_start, date_end, partition_key = _partition_window(context, config, df["time"].max())
    log.info(
        f"Window: {date_start} → {date_end} (partition={partition_key or 'none'})"
    )

    event_times = _get_event_times(
        df,
        date_start=date_start,
        date_end=date_end,
        threshold_ppb=config.h2s_threshold_ppb,
        require_stable=config.require_stable,
        max_events=config.max_events,
    )
    log.info(
        f"Found {len(event_times)} qualifying event timesteps "
        f"(H2S ≥ {config.h2s_threshold_ppb} ppb, stable={config.require_stable})"
    )

    # Early skip: week doesn't have enough events to constrain the NNLS.
    # Returning an empty list lets downstream assets short-circuit cleanly
    # without running particle simulations.
    if len(event_times) < config.min_events_per_week:
        log.warning(
            f"Only {len(event_times)} events in window — below "
            f"min_events_per_week={config.min_events_per_week}. "
            f"Skipping footprint generation for this partition."
        )
        return dg.Output(
            [],
            metadata={
                "status":               dg.MetadataValue.text("skipped_insufficient_events"),
                "n_events":             dg.MetadataValue.int(0),
                "n_candidate_events":   dg.MetadataValue.int(len(event_times)),
                "min_events_per_week":  dg.MetadataValue.int(config.min_events_per_week),
                "date_start":           dg.MetadataValue.text(str(date_start)),
                "date_end":             dg.MetadataValue.text(str(date_end)),
                "partition":            dg.MetadataValue.text(partition_key or "none"),
            },
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

    sufficient = len(events) >= config.min_events_per_week

    return dg.Output(
        events,
        metadata={
            "status":               dg.MetadataValue.text(
                "ok" if sufficient else "skipped_insufficient_events"
            ),
            "n_events":             dg.MetadataValue.int(len(events)),
            "n_candidate_events":   dg.MetadataValue.int(len(event_times)),
            "n_nnls_rows":          dg.MetadataValue.int(n_rows),
            "min_events_per_week":  dg.MetadataValue.int(config.min_events_per_week),
            "partition":            dg.MetadataValue.text(partition_key or "none"),
            "date_start":           dg.MetadataValue.text(str(date_start)),
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
    partitions_def=CALIBRATION_WEEKLY_PARTITIONS,
    description=(
        "Stacked-block NNLS inversion for one weekly partition: solves "
        "Q ≥ 0 : argmin ‖A·Q − C_obs‖² + λ₁‖Q‖² across all qualifying events "
        "in the week. Writes Q_field.parquet under weekly/{partition}/ and "
        "updates the _latest pointer only when the partition is recent."
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
    partition_key: Optional[str] = None
    try:
        partition_key = context.partition_key
    except Exception:
        partition_key = None

    log.info(f"Inverting {len(events)} events ({sum(len(e['footprints']) for e in events)} rows)")

    # Gate 0: week has fewer events than the configured minimum.
    # Matches the short-circuit in rolling_footprint_matrix.
    if len(events) < config.min_events_per_week:
        log.warning(
            f"Week has {len(events)} events — below "
            f"min_events_per_week={config.min_events_per_week}. "
            f"Skipping inversion for partition={partition_key or 'none'}."
        )
        return dg.MaterializeResult(metadata={
            "status":              dg.MetadataValue.text("skipped_insufficient_events"),
            "n_events":             dg.MetadataValue.int(len(events)),
            "min_events_per_week":  dg.MetadataValue.int(config.min_events_per_week),
            "partition":            dg.MetadataValue.text(partition_key or "none"),
        })

    channel_segments = build_channel_grid(segment_spacing_m=config.segment_spacing_m)
    log.info(f"Channel grid: {len(channel_segments)} segments @ {config.segment_spacing_m} m spacing")

    cfg = InversionConfig(
        segment_spacing_m=config.segment_spacing_m,
        lambda_l1=config.lambda_l1,
        lambda_smooth=config.lambda_smooth,
        background_ppb=config.background_ppb,
        gauss_meandering_deg=config.gauss_meandering_deg,
        q_required_max_g_s=config.q_required_max_g_s,
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
            "partition":     dg.MetadataValue.text(partition_key or "none"),
        })

    result = batch_inversion_stacked(events, channel_segments, cfg)

    if result.get("reason") == "no_rows":
        log.warning("batch_inversion_stacked returned no rows — skipping S3 write")
        return dg.MaterializeResult(metadata={
            "n_events": dg.MetadataValue.int(0),
            "status":   dg.MetadataValue.text("no_rows"),
            "partition": dg.MetadataValue.text(partition_key or "none"),
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

    # --- A-scale summary: is the sensitivity matrix physically capable of
    # reproducing the peak observed concentrations?  q_required_peak_g_s is
    # the Q on the single strongest segment that would reproduce the event's
    # peak sensor reading.  If many events need > ~200 g/s per event, the
    # Gaussian plume kernel is spreading the plume too wide and lambda_l1
    # tuning will not unlock mass — retuning gauss_meandering_deg, stability
    # class, or hours_back is the fix.
    per_event_sens = result.get("per_event_sensitivity", []) or []
    n_skipped_geom = int(result.get("n_events_skipped_geometry", 0))
    a_summary: dict = {
        "n_events":              len(per_event_sens),
        "n_events_skipped_geometry": n_skipped_geom,
        "q_required_max_g_s":    cfg.q_required_max_g_s,
    }
    if per_event_sens:
        req = [e["q_required_peak_g_s"] for e in per_event_sens
               if e.get("q_required_peak_g_s") is not None]
        max_a = [e["max_A_ppb_per_g_s"] for e in per_event_sens]
        peak_obs = [e["max_obs_ppb"] for e in per_event_sens]
        a_summary.update({
            "max_A_ppb_per_g_s_p50":   round(float(np.median(max_a)), 5),
            "max_A_ppb_per_g_s_max":   round(float(np.max(max_a)), 5),
            "peak_obs_ppb_max":        round(float(np.max(peak_obs)), 1),
            "q_required_peak_g_s_p50": round(float(np.median(req)), 1) if req else None,
            "q_required_peak_g_s_max": round(float(np.max(req)), 1) if req else None,
            "n_events_needing_gt_500_g_s": int(sum(1 for r in req if r > 500)),
        })
        log.info(
            f"A-scale: max_A p50={a_summary['max_A_ppb_per_g_s_p50']} ppb/(g/s), "
            f"peak obs max={a_summary['peak_obs_ppb_max']} ppb, "
            f"Q-required median={a_summary.get('q_required_peak_g_s_p50')} g/s, "
            f"events needing >500 g/s: {a_summary['n_events_needing_gt_500_g_s']}/"
            f"{len(per_event_sens)} | geometry-skipped: {n_skipped_geom} "
            f"(threshold q_required_max_g_s={cfg.q_required_max_g_s})"
        )

    # --- Serialize Q field to parquet (all segments, stable schema) ---
    rows = q_field_to_parquet_rows(result, channel_segments)
    q_df = pd.DataFrame(rows)

    run_tag = pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M")

    import io as _io
    buf = _io.BytesIO()
    q_df.to_parquet(buf, index=False)
    parquet_bytes = buf.getvalue()

    uploaded: dict[str, str] = {}

    # Per-partition path (canonical for weekly runs).
    if partition_key:
        weekly_path = Q_FIELD_WEEKLY_PATH.format(partition=partition_key)
        s3.putFile(parquet_bytes, path=weekly_path, content_type="application/octet-stream")
        uploaded["weekly"] = weekly_path
        log.info(f"Uploaded Q field → {weekly_path}")
    else:
        # Legacy unpartitioned fallback: keep the run_tag path.
        legacy_path = Q_FIELD_PATH.format(run_tag=run_tag)
        s3.putFile(parquet_bytes, path=legacy_path, content_type="application/octet-stream")
        uploaded["legacy"] = legacy_path
        log.info(f"Uploaded Q field → {legacy_path}")

    # _latest pointer — gated to recent partitions so backfills don't
    # clobber the live forecast's Q field with a stale 2025 field.
    is_recent = _is_partition_recent(partition_key, Q_FIELD_LATEST_MAX_AGE_DAYS)
    if is_recent:
        s3.putFile(parquet_bytes, path=Q_FIELD_LATEST_PATH, content_type="application/octet-stream")
        uploaded["latest"] = Q_FIELD_LATEST_PATH
        log.info(f"Updated {Q_FIELD_LATEST_PATH} (partition within {Q_FIELD_LATEST_MAX_AGE_DAYS}d)")
    else:
        log.info(
            f"Partition {partition_key} is older than {Q_FIELD_LATEST_MAX_AGE_DAYS}d — "
            f"leaving {Q_FIELD_LATEST_PATH} untouched."
        )

    # --- GeoDemic-friendly JSON sidecar (active segments only) ---
    # `inversion_config` is the machine-readable contract: downstream assets
    # (calibration_diagnostics) reconstruct InversionConfig from this block so
    # Q is fit and evaluated under the exact same regularization.  Leaving the
    # human-facing `config` block alongside it for ops dashboards.
    json_payload = {
        "timestamp":    pd.Timestamp.utcnow().isoformat(),
        "run_tag":      run_tag,
        "partition":    partition_key,
        "n_segments":   len(channel_segments),
        "n_active":     len(active),
        "Q_total_g_s":  round(q_total, 2),
        "n_events":     int(result.get("n_events", 0)),
        "n_rows":       int(result.get("n_rows", 0)),
        "sensor_rmse_ppb": result.get("sensor_rmse_ppb", {}),
        "active_sources": active,
        "sensitivity_diagnostics": {
            "summary":    a_summary,
            "per_event":  per_event_sens,
        },
        "inversion_config": asdict(cfg),
        "footprint_config": {
            "hours_back":        config.hours_back,
            "n_particles":       config.n_particles,
            "h2s_threshold_ppb": config.h2s_threshold_ppb,
            "require_stable":    config.require_stable,
            "max_events":        config.max_events,
            "min_events_per_week": config.min_events_per_week,
        },
        "config": {
            "segment_spacing_m":    config.segment_spacing_m,
            "lambda_l1":            config.lambda_l1,
            "lambda_smooth":        config.lambda_smooth,
            "gauss_meandering_deg": config.gauss_meandering_deg,
            "window_days":          config.window_days,
            "h2s_threshold_ppb":    config.h2s_threshold_ppb,
            "min_events_per_week":  config.min_events_per_week,
            "hours_back":           config.hours_back,
            "n_particles":          config.n_particles,
        },
    }
    json_bytes = json.dumps(json_payload, indent=2).encode()

    if partition_key:
        weekly_json_path = Q_FIELD_WEEKLY_JSON_PATH.format(partition=partition_key)
        s3.putFile(json_bytes, path=weekly_json_path, content_type="application/json")
        uploaded["weekly_json"] = weekly_json_path

    if is_recent:
        s3.putFile(json_bytes, path=Q_FIELD_LATEST_JSON_PATH, content_type="application/json")
        uploaded["latest_json"] = Q_FIELD_LATEST_JSON_PATH

    return dg.MaterializeResult(metadata={
        "status":          dg.MetadataValue.text("ok"),
        "partition":       dg.MetadataValue.text(partition_key or "none"),
        "n_events":        dg.MetadataValue.int(int(result.get("n_events", 0))),
        "n_events_skipped_geometry": dg.MetadataValue.int(n_skipped_geom),
        "n_nnls_rows":     dg.MetadataValue.int(int(result.get("n_rows", 0))),
        "n_segments":      dg.MetadataValue.int(len(channel_segments)),
        "n_active":        dg.MetadataValue.int(len(active)),
        "Q_total_g_s":     dg.MetadataValue.float(round(q_total, 2)),
        "sensor_rmse_ppb": dg.MetadataValue.json(result.get("sensor_rmse_ppb", {})),
        "a_scale":         dg.MetadataValue.json(a_summary),
        "latest_updated":  dg.MetadataValue.bool(bool(is_recent)),
        "uploaded":        dg.MetadataValue.json(uploaded),
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
    partitions_def=CALIBRATION_WEEKLY_PARTITIONS,
    description=(
        "Per-partition sanity checks: leave-one-sensor-out CV (sensor "
        "disagreement), leave-one-time-fold-out CV (random k-fold, adaptive k "
        "— temporal stability of Q), Σ Q budget check. Writes "
        "weekly/{partition}/diagnostics.json. Skips cleanly when the "
        "upstream inversion was skipped."
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
    partition_key: Optional[str] = None
    try:
        partition_key = context.partition_key
    except Exception:
        partition_key = None

    if len(events) < config.min_events_per_week:
        log.warning(
            f"Skipping diagnostics — only {len(events)} events "
            f"(min {config.min_events_per_week}) for partition={partition_key or 'none'}."
        )
        return dg.MaterializeResult(metadata={
            "status":              dg.MetadataValue.text("skipped_insufficient_events"),
            "n_events":             dg.MetadataValue.int(len(events)),
            "min_events_per_week":  dg.MetadataValue.int(config.min_events_per_week),
            "partition":            dg.MetadataValue.text(partition_key or "none"),
        })

    fallback_cfg = InversionConfig(
        segment_spacing_m=config.segment_spacing_m,
        lambda_l1=config.lambda_l1,
        lambda_smooth=config.lambda_smooth,
        background_ppb=config.background_ppb,
        gauss_meandering_deg=config.gauss_meandering_deg,
    )
    # Pull the exact InversionConfig that channel_emission_inversion used to
    # fit Q — avoids a CV-under-different-lambda footgun.  Also mirrors the
    # footprint_config block (hours_back, n_particles, ...) from the sidecar
    # into diagnostics.json so the operator can see every knob that drove
    # this partition's Q field in one file.
    cfg, sidecar_payload = _load_inversion_sidecar(s3, partition_key, fallback_cfg, log)
    channel_segments = build_channel_grid(segment_spacing_m=cfg.segment_spacing_m)

    footprint_cfg = sidecar_payload.get("footprint_config") or {
        "hours_back":           config.hours_back,
        "n_particles":          config.n_particles,
        "h2s_threshold_ppb":    config.h2s_threshold_ppb,
        "require_stable":       config.require_stable,
        "max_events":           config.max_events,
        "min_events_per_week":  config.min_events_per_week,
    }

    diagnostics: dict = {
        "timestamp": pd.Timestamp.utcnow().isoformat(),
        "n_events": len(events),
        "inversion_config": asdict(cfg),
        "footprint_config": footprint_cfg,
        "gates": {},
    }

    # --- Precompute per-event sensitivity rows (shared by both Gate 1 CVs) ---
    # Applies the same geometry-plausibility filter that channel_emission_inversion
    # applies, so CV evaluates over the same row set that NNLS actually fit.
    sensor_names = list(SENSORS.keys())
    q_threshold = getattr(cfg, "q_required_max_g_s", None)
    per_event_rows: list[dict] = []
    n_diag_skipped = 0
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
            n_diag_skipped += 1
            continue

        max_A_ev = float(A_ev.max())
        max_obs_ev = float(max(h2s_obs[s] for s in sensors_present) - cfg.background_ppb)
        q_req_ev = (max_obs_ev / max_A_ev) if max_A_ev > 1e-6 else None
        if (
            q_threshold is not None and q_threshold > 0
            and (q_req_ev is None or q_req_ev > q_threshold)
        ):
            n_diag_skipped += 1
            continue

        c_bg = np.array(
            [max(h2s_obs[s] - cfg.background_ppb, 0.0) for s in sensors_present],
            dtype=float,
        )
        per_event_rows.append({
            "time":    ev["time"],
            "sensors": sensors_present,
            "A":       A_ev,
            "c_bg":    c_bg,
        })

    diagnostics["n_events_skipped_geometry"] = n_diag_skipped
    log.info(
        f"Diagnostics per-event rows: {len(per_event_rows)} kept, "
        f"{n_diag_skipped} geometry-skipped (q_required_max_g_s={q_threshold})"
    )

    # --- Gate 1a: leave-one-sensor-out CV ---
    loo_rmse: dict[str, dict] = {}

    for held_out in sensor_names:
        train_rows_A: list[np.ndarray] = []
        train_rows_C: list[float] = []
        test_rows_A: list[np.ndarray] = []
        test_rows_C: list[float] = []

        for ev_row in per_event_rows:
            for i, sname in enumerate(ev_row["sensors"]):
                row_A = ev_row["A"][i, :]
                row_c = float(ev_row["c_bg"][i])
                if sname == held_out:
                    test_rows_A.append(row_A)
                    test_rows_C.append(row_c)
                else:
                    train_rows_A.append(row_A)
                    train_rows_C.append(row_c)

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

    gate1a_pass = all(
        (v.get("rmse_over_std") is not None and v["rmse_over_std"] < 1.0
         and abs(v.get("bias_ppb") or 0.0) < 10.0)
        for v in loo_rmse.values()
        if v.get("n_test", 0) > 0
    )
    diagnostics["leave_one_sensor_out"] = loo_rmse
    diagnostics["gates"]["leave_one_sensor_out_pass"] = bool(gate1a_pass)

    # --- Gate 1b: leave-one-time-fold-out CV ---
    # Random k-fold over events (adaptive k); all 3 sensors stay in both
    # train and test so spatial coverage is the same on both sides. Tests
    # temporal stability of Q rather than cross-sensor agreement.
    loto = _time_fold_cv(per_event_rows, cfg, seed=0)
    diagnostics["leave_one_time_fold_out"] = loto
    loto_overall = loto.get("overall") or {}
    rmse_over_std = loto_overall.get("rmse_over_std")
    bias_ppb = loto_overall.get("bias_ppb")
    gate1b_pass = (
        rmse_over_std is not None
        and rmse_over_std < 1.0
        and abs(bias_ppb or 0.0) < 10.0
    )
    diagnostics["gates"]["leave_one_time_fold_out_pass"] = bool(gate1b_pass)

    # --- Gate 2: budget sanity ---
    # Prefer the per-partition Q field parquet; fall back to _latest when
    # running legacy unpartitioned.
    q_total = 0.0
    try:
        import io as _io
        if partition_key:
            weekly_parquet_path = Q_FIELD_WEEKLY_PATH.format(partition=partition_key)
            q_bytes = s3.getFile(weekly_parquet_path)
        else:
            q_bytes = s3.getFile(Q_FIELD_LATEST_PATH)
        q_df = pd.read_parquet(_io.BytesIO(q_bytes))
        q_total = float(q_df["Q_g_s"].sum())
    except Exception as exc:
        log.warning(f"Could not read Q field for budget gate: {exc}")

    budget_low, budget_high = 30.0, 500.0
    gate2_pass = budget_low <= q_total <= budget_high
    diagnostics["partition"] = partition_key
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
        f"Diagnostics summary: LOSO={gate1a_pass} | LOTO={gate1b_pass} | "
        f"Budget={gate2_pass} (Σ Q={q_total:.1f} g/s) "
        f"| all_pass={diagnostics['gates']['all_pass']}"
    )

    run_tag = pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M")
    payload = json.dumps(diagnostics, indent=2, default=str).encode()
    uploaded: dict[str, str] = {}

    if partition_key:
        weekly_diag_path = Q_FIELD_WEEKLY_DIAGNOSTICS_PATH.format(partition=partition_key)
        s3.putFile(payload, path=weekly_diag_path, content_type="application/json")
        uploaded["weekly"] = weekly_diag_path
        log.info(f"Uploaded diagnostics → {weekly_diag_path}")
    else:
        legacy_path = Q_FIELD_DIAGNOSTICS_PATH.format(run_tag=run_tag)
        s3.putFile(payload, path=legacy_path, content_type="application/json")
        uploaded["legacy"] = legacy_path
        log.info(f"Uploaded diagnostics → {legacy_path}")

    is_recent = _is_partition_recent(partition_key, Q_FIELD_LATEST_MAX_AGE_DAYS)
    if is_recent:
        s3.putFile(payload, path=Q_FIELD_DIAGNOSTICS_LATEST_PATH, content_type="application/json")
        uploaded["latest"] = Q_FIELD_DIAGNOSTICS_LATEST_PATH

    return dg.MaterializeResult(metadata={
        "status":               dg.MetadataValue.text("ok"),
        "partition":            dg.MetadataValue.text(partition_key or "none"),
        "Q_total_g_s":          dg.MetadataValue.float(round(q_total, 2)),
        "loso_pass":            dg.MetadataValue.bool(bool(gate1a_pass)),
        "loto_pass":            dg.MetadataValue.bool(bool(gate1b_pass)),
        "budget_pass":          dg.MetadataValue.bool(bool(gate2_pass)),
        "all_gates_pass":       dg.MetadataValue.bool(bool(diagnostics["gates"]["all_pass"])),
        "loso_rmse_ppb":        dg.MetadataValue.json({k: v.get("rmse_ppb") for k, v in loo_rmse.items()}),
        "loto_rmse_ppb":        dg.MetadataValue.float(float(loto_overall.get("rmse_ppb") or 0.0)),
        "loto_rmse_over_std":   dg.MetadataValue.float(float(loto_overall.get("rmse_over_std") or 0.0)),
        "loto_n_folds":         dg.MetadataValue.int(int(loto.get("n_folds") or 0)),
        "latest_updated":       dg.MetadataValue.bool(bool(is_recent)),
        "uploaded":             dg.MetadataValue.json(uploaded),
        "run_tag":               dg.MetadataValue.text(run_tag),
    })


# ==============================================================================
# Asset 4: calibration_viz
# ==============================================================================

@dg.asset(
    key_prefix="h2s",
    group_name="h2s_calibration",
    required_resource_keys={"s3"},
    kinds={"python", "s3", "matplotlib"},
    partitions_def=CALIBRATION_WEEKLY_PARTITIONS,
    description=(
        "Per-partition verification visualizations. Renders: "
        "(1) channel segment map colored by Q_g_s with SENSORS overlay, "
        "(2) leave-one-sensor-out CV predicted-vs-observed scatter, "
        "(3) leave-one-time-fold-out CV predicted-vs-observed scatter, "
        "(4) Σ Q budget bar vs the 30/167/500 g/s reference lines. "
        "Uploads PNGs under weekly/{partition}/ (and _latest when recent)."
    ),
    deps=[
        dg.AssetKey(["h2s", "channel_emission_inversion"]),
        dg.AssetKey(["h2s", "calibration_diagnostics"]),
    ],
)
def calibration_viz(
    context: dg.AssetExecutionContext,
    config: CalibrationConfig,
    rolling_footprint_matrix: list,
) -> dg.MaterializeResult:
    import io as _io

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    from h2s.dispersion.lagrangian import SENSORS

    log = context.log
    s3 = context.resources.s3

    partition_key: Optional[str] = None
    try:
        partition_key = context.partition_key
    except Exception:
        partition_key = None

    # Skip if upstream had insufficient events (no Q field was written).
    if len(rolling_footprint_matrix) < config.min_events_per_week:
        log.warning(
            f"Skipping viz — partition={partition_key or 'none'} has "
            f"{len(rolling_footprint_matrix)} events (min {config.min_events_per_week})."
        )
        return dg.MaterializeResult(metadata={
            "status":              dg.MetadataValue.text("skipped_insufficient_events"),
            "n_events":             dg.MetadataValue.int(len(rolling_footprint_matrix)),
            "min_events_per_week":  dg.MetadataValue.int(config.min_events_per_week),
            "partition":            dg.MetadataValue.text(partition_key or "none"),
        })

    # --- Load artifacts: prefer per-partition paths, fall back to _latest ---
    if partition_key:
        q_path = Q_FIELD_WEEKLY_PATH.format(partition=partition_key)
        diag_path = Q_FIELD_WEEKLY_DIAGNOSTICS_PATH.format(partition=partition_key)
    else:
        q_path = Q_FIELD_LATEST_PATH
        diag_path = Q_FIELD_DIAGNOSTICS_LATEST_PATH

    q_df = pd.read_parquet(_io.BytesIO(s3.getFile(q_path)))
    diagnostics = json.loads(s3.getFile(diag_path))

    loo = diagnostics.get("leave_one_sensor_out", {}) or {}
    loto = diagnostics.get("leave_one_time_fold_out", {}) or {}
    budget = diagnostics.get("budget_sanity", {}) or {}
    q_total = float(budget.get("Q_total_g_s", q_df["Q_g_s"].sum()))

    run_tag = pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M")
    is_recent = _is_partition_recent(partition_key, Q_FIELD_LATEST_MAX_AGE_DAYS)
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
    if partition_key:
        weekly_map = Q_FIELD_WEEKLY_VIZ_MAP_PATH.format(partition=partition_key)
        s3.putFile(png_bytes, path=weekly_map, content_type="image/png")
        uploaded["map_weekly"] = weekly_map
    else:
        legacy_map = Q_FIELD_VIZ_MAP_PATH.format(run_tag=run_tag)
        s3.putFile(png_bytes, path=legacy_map, content_type="image/png")
        uploaded["map_legacy"] = legacy_map
    if is_recent:
        s3.putFile(png_bytes, path=Q_FIELD_VIZ_MAP_LATEST_PATH, content_type="image/png")
        uploaded["map_latest"] = Q_FIELD_VIZ_MAP_LATEST_PATH
    log.info(f"Uploaded Q field map (partition={partition_key or 'none'})")

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
    if partition_key:
        weekly_cv = Q_FIELD_WEEKLY_VIZ_CV_PATH.format(partition=partition_key)
        s3.putFile(png_bytes, path=weekly_cv, content_type="image/png")
        uploaded["cv_weekly"] = weekly_cv
    else:
        legacy_cv = Q_FIELD_VIZ_CV_PATH.format(run_tag=run_tag)
        s3.putFile(png_bytes, path=legacy_cv, content_type="image/png")
        uploaded["cv_legacy"] = legacy_cv
    if is_recent:
        s3.putFile(png_bytes, path=Q_FIELD_VIZ_CV_LATEST_PATH, content_type="image/png")
        uploaded["cv_latest"] = Q_FIELD_VIZ_CV_LATEST_PATH
    log.info(f"Uploaded LOSO CV scatter (partition={partition_key or 'none'})")

    # --- Plot 3: leave-one-time-fold-out CV scatter ---
    loto_per_sensor = (loto.get("per_sensor") if isinstance(loto, dict) else None) or {}
    loto_gate_pass = bool(diagnostics.get("gates", {}).get("leave_one_time_fold_out_pass", False))
    fig, ax = plt.subplots(figsize=(7, 7))
    max_val = 1.0
    for sname, data in loto_per_sensor.items():
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
    ax.set_xlabel("Observed C (ppb, held-out time fold)")
    ax.set_ylabel("Predicted C (ppb, from other folds)")
    ax.set_title(
        f"Leave-one-time-fold-out CV (k={loto.get('n_folds', '?')}) — gate pass: "
        f"{loto_gate_pass}"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)
    ax.set_aspect("equal")

    buf = _io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    png_bytes = buf.getvalue()
    if partition_key:
        weekly_loto = Q_FIELD_WEEKLY_VIZ_LOTO_PATH.format(partition=partition_key)
        s3.putFile(png_bytes, path=weekly_loto, content_type="image/png")
        uploaded["loto_weekly"] = weekly_loto
    if is_recent:
        s3.putFile(png_bytes, path=Q_FIELD_VIZ_LOTO_LATEST_PATH, content_type="image/png")
        uploaded["loto_latest"] = Q_FIELD_VIZ_LOTO_LATEST_PATH
    log.info(f"Uploaded LOTO CV scatter (partition={partition_key or 'none'})")

    # --- Plot 4: Σ Q budget bar ---
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
    if partition_key:
        weekly_budget = Q_FIELD_WEEKLY_VIZ_BUDGET_PATH.format(partition=partition_key)
        s3.putFile(png_bytes, path=weekly_budget, content_type="image/png")
        uploaded["budget_weekly"] = weekly_budget
    else:
        legacy_budget = Q_FIELD_VIZ_BUDGET_PATH.format(run_tag=run_tag)
        s3.putFile(png_bytes, path=legacy_budget, content_type="image/png")
        uploaded["budget_legacy"] = legacy_budget
    if is_recent:
        s3.putFile(png_bytes, path=Q_FIELD_VIZ_BUDGET_LATEST_PATH, content_type="image/png")
        uploaded["budget_latest"] = Q_FIELD_VIZ_BUDGET_LATEST_PATH
    log.info(f"Uploaded budget bar (partition={partition_key or 'none'})")

    return dg.MaterializeResult(metadata={
        "status":         dg.MetadataValue.text("ok"),
        "partition":      dg.MetadataValue.text(partition_key or "none"),
        "run_tag":        dg.MetadataValue.text(run_tag),
        "latest_updated": dg.MetadataValue.bool(bool(is_recent)),
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
