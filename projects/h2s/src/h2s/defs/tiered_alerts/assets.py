"""Dagster assets: tiered_alert_features, tier_N_scores, tier_alert_dispatcher."""

import os
import dagster as dg
import pandas as pd

from h2s.constants import ALERT_TIERS, STATIONS

from .features import compute_horizon_features, load_forecast_df
from .messages import build_tier_blocks, build_tier_message
from .state import (
    get_cell,
    higher_tier_active_in_horizon,
    load_state,
    mark_clear_cycle,
    mark_fired,
    save_state,
    should_send_onset,
    tier_last_sent_within_quiet,
)
from .tiers import (
    TierResult,
    TierNestingError,
    check_nesting,
    compute_score,
    gate_tier1,
    gate_tier2,
    gate_tier3,
    load_config,
    HORIZON_ORDER,
    HORIZON_WINDOWS_H,
)

_KEY = lambda name: dg.AssetKey(["h2s", name])

# NESTOR-BES is the bellwether for per-horizon score computation (design §2.2)
_NB_SITE = "NESTOR - BES"
_IB_SITE = "IB CIVIC CTR"


# ---------------------------------------------------------------------------
# tiered_alert_features
# ---------------------------------------------------------------------------

@dg.asset(
    key_prefix="h2s",
    group_name="tiered_alerts",
    required_resource_keys={"s3"},
    kinds={"python", "s3"},
    description="Per-horizon forecast feature aggregates for all four windows × three stations",
)
def tiered_alert_features(
    context: dg.AssetExecutionContext,
) -> dict:
    """Load latest forecast and compute horizon-windowed feature aggregates.

    Returns dict keyed by (horizon_key, site_name) → feature dict.
    """
    s3 = context.resources.s3
    df = load_forecast_df(s3)

    if df.empty:
        context.log.warning("Forecast data is empty — returning empty feature dict")
        return {}

    t = pd.Timestamp.now("UTC")
    cell_features, degraded = compute_horizon_features(df, t)

    context.log.info(
        f"Computed {len(cell_features)} horizon×station feature cells "
        f"(t={t.isoformat()}, degraded={degraded})"
    )
    context.add_output_metadata({
        "n_cells": len(cell_features),
        "evaluated_at": t.isoformat(),
        "degraded": degraded,
    })
    return cell_features


# ---------------------------------------------------------------------------
# Tier scoring helpers
# ---------------------------------------------------------------------------

def _score_cell(
    tier_key: str,
    features: dict,
    config: dict,
    single_station: bool = False,
) -> tuple[float, dict]:
    """Compute risk score for a single (tier, horizon) aggregate feature dict."""
    weights = config["tiers"][tier_key]["score_weights"]
    if single_station and "single_station_quiet_night_stats" in config:
        stats = config["single_station_quiet_night_stats"]
    else:
        stats = config["quiet_night_stats"]
    return compute_score(features, weights, stats)


def _n_h2s_active(rows_by_station: dict[str, dict]) -> int:
    """Count stations with genuine H2S observations (_h2s_active set before replication)."""
    return sum(1 for row in rows_by_station.values() if row.get("_h2s_active", False))


def _build_tier_results(
    tier_key: str,
    label: str,
    cell_features: dict,
    gate_by_horizon: dict[str, bool],
    n_stations_by_horizon: dict[str, int],
    scores_by_horizon: dict[str, tuple[float, dict]],
    evaluated_at: pd.Timestamp,
    thresholds_by_horizon: dict[str, float] | None = None,
) -> list[TierResult]:
    results = []
    for horizon in HORIZON_ORDER:
        t_start = evaluated_at + pd.Timedelta(hours=HORIZON_WINDOWS_H[horizon][0])
        t_end   = evaluated_at + pd.Timedelta(hours=HORIZON_WINDOWS_H[horizon][1])
        gate = gate_by_horizon.get(horizon, False)
        score, contributing = scores_by_horizon.get(horizon, (0.0, {}))
        n_stations = n_stations_by_horizon.get(horizon, 0)
        threshold = (thresholds_by_horizon or {}).get(horizon, 0.5)

        # Get degraded / daytime flags from any cell for this horizon
        sample_cell = next(
            (v for (h, _), v in cell_features.items() if h == horizon), {}
        )
        degraded = bool(sample_cell.get("_degraded", False))
        daytime  = bool(sample_cell.get("_daytime_horizon", False))

        results.append(TierResult(
            tier=tier_key,
            label=label,
            horizon=horizon,
            evaluated_at=evaluated_at,
            window=(t_start, t_end),
            gate_passed=gate,
            score=score,
            n_stations_passing_gate=n_stations,
            contributing_features=contributing,
            daytime_horizon=daytime,
            degraded=degraded,
            fire=gate and score >= threshold,
        ))
    return results


def _bellwether_features(cell_features: dict, horizon: str) -> dict:
    """Return NB features for a horizon, falling back to IB."""
    for site in (_NB_SITE, _IB_SITE):
        f = cell_features.get((horizon, site), {})
        if f:
            return f
    return {}


# ---------------------------------------------------------------------------
# Tier 1 scores
# ---------------------------------------------------------------------------

@dg.asset(
    key_prefix="h2s",
    group_name="tiered_alerts",
    kinds={"python"},
    description="Tier 1 (Plant-Signal) risk scores across all four forecast horizons",
    ins={"tiered_alert_features": dg.AssetIn(key=_KEY("tiered_alert_features"))},
)
def tier_1_scores(
    context: dg.AssetExecutionContext,
    tiered_alert_features: dict,
) -> list:
    config = load_config()
    evaluated_at = pd.Timestamp.now("UTC")
    label = ALERT_TIERS["tier_1"]["label"]

    gate_by_horizon: dict[str, bool] = {}
    n_stations_by_horizon: dict[str, int] = {}
    scores_by_horizon: dict[str, tuple] = {}

    thresholds_by_horizon: dict[str, float] = {}
    score_thresholds = config.get("score_thresholds", {})
    for horizon in HORIZON_ORDER:
        rows_by_station = {
            site: tiered_alert_features.get((horizon, site), {})
            for site in STATIONS
        }
        t1_gates = gate_tier1(rows_by_station)
        n_passing = sum(t1_gates.values())
        n_stations_by_horizon[horizon] = n_passing
        gate_by_horizon[horizon] = any(t1_gates.values())

        single = _n_h2s_active(rows_by_station) <= 1
        thresholds_by_horizon[horizon] = score_thresholds.get(
            "single_station" if single else "multi_station", 0.5
        )
        bw = _bellwether_features(tiered_alert_features, horizon)
        score, contrib = _score_cell("tier_1", bw, config, single_station=single) if bw else (0.0, {})
        scores_by_horizon[horizon] = (score, contrib)

    results = _build_tier_results(
        "tier_1", label, tiered_alert_features,
        gate_by_horizon, n_stations_by_horizon, scores_by_horizon, evaluated_at,
        thresholds_by_horizon=thresholds_by_horizon,
    )
    firing = sum(1 for r in results if r.fire)
    context.log.info(f"Tier 1: {firing}/4 horizons firing")
    return results


# ---------------------------------------------------------------------------
# Tier 2 scores
# ---------------------------------------------------------------------------

@dg.asset(
    key_prefix="h2s",
    group_name="tiered_alerts",
    kinds={"python"},
    description="Tier 2 (Multi-Site-Risk) risk scores across all four forecast horizons",
    ins={"tiered_alert_features": dg.AssetIn(key=_KEY("tiered_alert_features"))},
)
def tier_2_scores(
    context: dg.AssetExecutionContext,
    tiered_alert_features: dict,
) -> list:
    config = load_config()
    evaluated_at = pd.Timestamp.now("UTC")
    label = ALERT_TIERS["tier_2"]["label"]

    gate_by_horizon: dict[str, bool] = {}
    n_stations_by_horizon: dict[str, int] = {}
    scores_by_horizon: dict[str, tuple] = {}

    thresholds_by_horizon: dict[str, float] = {}
    score_thresholds = config.get("score_thresholds", {})
    for horizon in HORIZON_ORDER:
        rows_by_station = {
            site: tiered_alert_features.get((horizon, site), {})
            for site in STATIONS
        }
        t1_gates = gate_tier1(rows_by_station)
        t2_gates, n_t1 = gate_tier2(rows_by_station, t1_gates)
        n_stations_by_horizon[horizon] = n_t1
        gate_by_horizon[horizon] = any(t2_gates.values())

        single = _n_h2s_active(rows_by_station) <= 1
        thresholds_by_horizon[horizon] = score_thresholds.get(
            "single_station" if single else "multi_station", 0.5
        )
        bw = _bellwether_features(tiered_alert_features, horizon)
        score, contrib = _score_cell("tier_2", bw, config, single_station=single) if bw else (0.0, {})
        scores_by_horizon[horizon] = (score, contrib)

    results = _build_tier_results(
        "tier_2", label, tiered_alert_features,
        gate_by_horizon, n_stations_by_horizon, scores_by_horizon, evaluated_at,
        thresholds_by_horizon=thresholds_by_horizon,
    )
    firing = sum(1 for r in results if r.fire)
    context.log.info(f"Tier 2: {firing}/4 horizons firing")
    return results


# ---------------------------------------------------------------------------
# Tier 3 scores
# ---------------------------------------------------------------------------

@dg.asset(
    key_prefix="h2s",
    group_name="tiered_alerts",
    kinds={"python"},
    description="Tier 3 (Exceedance-Risk) risk scores across all four forecast horizons",
    ins={"tiered_alert_features": dg.AssetIn(key=_KEY("tiered_alert_features"))},
)
def tier_3_scores(
    context: dg.AssetExecutionContext,
    tiered_alert_features: dict,
) -> list:
    config = load_config()
    evaluated_at = pd.Timestamp.now("UTC")
    label = ALERT_TIERS["tier_3"]["label"]

    gate_by_horizon: dict[str, bool] = {}
    n_stations_by_horizon: dict[str, int] = {}
    scores_by_horizon: dict[str, tuple] = {}

    thresholds_by_horizon: dict[str, float] = {}
    score_thresholds = config.get("score_thresholds", {})
    for horizon in HORIZON_ORDER:
        rows_by_station = {
            site: tiered_alert_features.get((horizon, site), {})
            for site in STATIONS
        }
        t1_gates = gate_tier1(rows_by_station)
        t2_gates, n_t1 = gate_tier2(rows_by_station, t1_gates)
        t3_gates = gate_tier3(rows_by_station, t2_gates)
        n_stations_by_horizon[horizon] = n_t1
        gate_by_horizon[horizon] = any(t3_gates.values())

        single = _n_h2s_active(rows_by_station) <= 1
        thresholds_by_horizon[horizon] = score_thresholds.get(
            "single_station" if single else "multi_station", 0.5
        )
        bw = _bellwether_features(tiered_alert_features, horizon)
        score, contrib = _score_cell("tier_3", bw, config, single_station=single) if bw else (0.0, {})
        scores_by_horizon[horizon] = (score, contrib)

    results = _build_tier_results(
        "tier_3", label, tiered_alert_features,
        gate_by_horizon, n_stations_by_horizon, scores_by_horizon, evaluated_at,
        thresholds_by_horizon=thresholds_by_horizon,
    )
    firing = sum(1 for r in results if r.fire)
    context.log.info(f"Tier 3: {firing}/4 horizons firing")
    return results


# ---------------------------------------------------------------------------
# tier_alert_dispatcher
# ---------------------------------------------------------------------------

@dg.asset(
    key_prefix="h2s",
    group_name="tiered_alerts",
    required_resource_keys={"slack", "s3"},
    kinds={"slack", "s3"},
    description="Dispatch Tier 1–3 pre-alert messages to Slack ops channel; manage debounce state",
    ins={
        "tier_1_scores": dg.AssetIn(key=_KEY("tier_1_scores")),
        "tier_2_scores": dg.AssetIn(key=_KEY("tier_2_scores")),
        "tier_3_scores": dg.AssetIn(key=_KEY("tier_3_scores")),
    },
)
def tier_alert_dispatcher(
    context: dg.AssetExecutionContext,
    tier_1_scores: list,
    tier_2_scores: list,
    tier_3_scores: list,
) -> None:
    """Consolidate tier scores, apply debounce, and dispatch to Slack.

    Shadow mode: set TIERED_ALERTS_SHADOW=true to skip Slack dispatch while
    still writing state and logs.
    """
    s3 = context.resources.s3
    slack = context.resources.slack
    shadow = os.environ.get("TIERED_ALERTS_SHADOW", "false").lower() == "true"

    now = pd.Timestamp.now("UTC")
    state = load_state(s3)

    scores_by_tier: dict[str, list[TierResult]] = {
        "tier_1": tier_1_scores,
        "tier_2": tier_2_scores,
        "tier_3": tier_3_scores,
    }

    # Enforce nesting invariant before any dispatch
    all_results: list[TierResult] = tier_1_scores + tier_2_scores + tier_3_scores
    try:
        check_nesting(all_results)
    except TierNestingError as e:
        context.log.error(f"Tier nesting invariant violated: {e}")
        raise

    ops_channel = os.environ.get("SLACK_CHANNEL_OPS", slack.channel)

    for tier_key, results in scores_by_tier.items():
        firing = [r for r in results if r.fire]
        evaluated_at = results[0].evaluated_at if results else now

        # Update state for all cells (firing and clearing)
        cells_with_new_onset: list[str] = []
        for r in results:
            cell = get_cell(state, tier_key, r.horizon)
            if r.fire:
                if should_send_onset(cell, now):
                    cells_with_new_onset.append(r.horizon)
                mark_fired(cell, r.score, now)
            elif r.score < 0.3 and cell.get("active", False):
                closed = mark_clear_cycle(cell)
                if closed:
                    context.log.info(
                        f"{tier_key}/{r.horizon}: cleared after 3 consecutive clear cycles"
                    )

        # Per-tier message dedup: only one message per tier per QUIET_HOURS window
        if not cells_with_new_onset:
            context.log.debug(f"{tier_key}: no new onset cells this cycle")
            continue

        if tier_last_sent_within_quiet(state, tier_key, now):
            # New horizons fired but a tier-level message was already sent recently.
            # Update state but suppress Slack message.
            context.log.info(
                f"{tier_key}: new onset in {cells_with_new_onset} but "
                "per-tier dedup suppresses message (already sent within quiet period)"
            )
            save_state(s3, state)
            continue

        # Check within-horizon suppression for all firing cells
        # (higher tier active in same horizon → suppress lower tier onset message)
        suppressed_by_higher = all(
            higher_tier_active_in_horizon(state, tier_key, h)
            for h in cells_with_new_onset
        )
        if suppressed_by_higher:
            context.log.info(
                f"{tier_key}: onset suppressed by higher-tier active in same horizons "
                f"{cells_with_new_onset}"
            )
            save_state(s3, state)
            continue

        # Build and dispatch the consolidated message
        label = ALERT_TIERS[tier_key]["label"]
        context.log.info(
            f"Dispatching {tier_key} ({label}): firing horizons={[r.horizon for r in firing]}"
        )

        if not shadow:
            client = slack.get_client()
            blocks = build_tier_blocks(tier_key, results, evaluated_at)
            text = build_tier_message(tier_key, results, evaluated_at)
            client.chat_postMessage(
                channel=ops_channel,
                text=text,
                blocks=blocks,
            )
            context.log.info(f"Sent {tier_key} message to {ops_channel}")
        else:
            context.log.info(
                f"[SHADOW] Would send {tier_key} message to {ops_channel} — suppressed by TIERED_ALERTS_SHADOW=true"
            )

        save_state(s3, state)

    # Final state save for any clear-cycle updates that didn't trigger dispatch
    save_state(s3, state)
    context.log.info("State saved to S3")
