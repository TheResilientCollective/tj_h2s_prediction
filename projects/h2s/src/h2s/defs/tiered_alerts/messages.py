"""Slack Block Kit message templates for Tiers 1–3.

One consolidated message per tier per cycle — all four horizon states listed.
Tier 4 (WATCH) and Tier 5 (CRITICAL) message templates are unchanged.
"""

import pandas as pd

from h2s.constants import ALERT_TIERS
from h2s.defs.h2s_alert_system import _deficit_label, _fmt_time, _wind_flag
from .tiers import TierResult, HORIZON_ORDER, HORIZON_LABELS

_TIER_EMOJI = {"tier_1": "🟢", "tier_2": "🟡", "tier_3": "🔴"}

_SUGGESTED_RESPONSE = {
    "tier_1": (
        "SBIWTP flow is in the plant-signal regime. Monitor station readings "
        "and confirm SBIWTP operational status."
    ),
    "tier_2": (
        "Verify monitoring station status. Pre-position field response if NB peak "
        "exceeds 20 ppb within 6 hours."
    ),
    "tier_3": (
        "Conditions are in the exceedance-risk regime. Alert monitoring staff. "
        "Pre-position for possible WATCH-level response within the forecast window."
    ),
}


def _horizon_line(result: TierResult | None, horizon: str) -> str:
    label = HORIZON_LABELS[horizon]
    if result is None or not result.gate_passed:
        reason = ""
        if result is not None and not result.gate_passed:
            # Build a brief reason string from the worst feature
            cf = result.contributing_features
            if cf:
                top_feat, (val, z, _w) = sorted(cf.items(), key=lambda kv: abs(kv[1][1]))[0]
                reason = f" (gate failed — {top_feat.replace('_', ' ')} {val:.1f})"
        score_str = f"score {result.score:.2f}" if result is not None else "no data"
        return f"     {label:<28} {score_str}{reason}"

    n = result.n_stations_passing_gate
    station_note = f"gate at {n} station{'s' if n != 1 else ''}"
    return f"  ⚠️  {label:<28} score {result.score:.2f}  ← FIRING ({station_note})"


def _top_features_section(results: list[TierResult]) -> str:
    firing = [r for r in results if r.fire]
    if not firing:
        return ""
    merged: dict[str, tuple[float, float, float]] = {}
    for r in firing:
        for feat, (val, z, w) in r.contributing_features.items():
            if feat not in merged or abs(z) > abs(merged[feat][1]):
                merged[feat] = (val, z, w)

    top = sorted(merged.items(), key=lambda kv: abs(kv[1][1]), reverse=True)[:4]
    lines = ["*Top contributing factors (firing horizons):*"]
    for feat, (val, _z, _w) in top:
        if feat == "sbiwtp_flow_mgd":
            lines.append(f"  • SBIWTP forecast flow: {val:.1f} MGD  ({_deficit_label(val)})")
        elif feat == "wind_speed_10m":
            lines.append(f"  • Forecast wind speed: {val:.1f} m/s  ({_wind_flag(val)})")
        elif feat == "sbiwtp_anomaly":
            lines.append(f"  • SBIWTP anomaly: {val:.2f}")
        elif feat == "dewpoint_2m":
            lines.append(f"  • Dewpoint: {val:.1f}°C")
        elif feat == "temperature_2m":
            lines.append(f"  • Temperature: {val:.1f}°C")
        else:
            lines.append(f"  • {feat.replace('_', ' ').title()}: {val:.2f}")
    return "\n".join(lines)


def build_tier_message(tier_key: str, results: list[TierResult], evaluated_at: pd.Timestamp) -> str:
    """Build a consolidated plain-text tier message for all four horizon states.

    One message per tier per cycle. Lists all four horizons, firing and non-firing.
    """
    tier_cfg = ALERT_TIERS[tier_key]
    label = tier_cfg["label"]
    emoji = _TIER_EMOJI.get(tier_key, "⚠️")

    result_by_horizon = {r.horizon: r for r in results}

    time_str = _fmt_time(evaluated_at.tz_localize("UTC") if evaluated_at.tzinfo is None else evaluated_at)

    horizon_lines = []
    any_daytime = False
    any_degraded = False
    for h in HORIZON_ORDER:
        r = result_by_horizon.get(h)
        horizon_lines.append(_horizon_line(r, h))
        if r and r.daytime_horizon:
            any_daytime = True
        if r and r.degraded:
            any_degraded = True

    features_section = _top_features_section(results)

    interpretation = _build_interpretation(tier_key, results)
    suggested = _SUGGESTED_RESPONSE.get(tier_key, "")

    lines = [
        f"{emoji} *Tier {tier_key[-1]} — {label}*",
        f"Evaluated: {time_str}",
        "",
        "*Horizon states:*",
    ] + horizon_lines + [
        "",
    ]

    if features_section:
        lines += [features_section, ""]

    if interpretation:
        lines += [f"*Interpretation:* {interpretation}", ""]

    if suggested:
        lines += [f"*Suggested response:* {suggested}", ""]

    lines.append("_Reference: docs/tiered-alert-system-design.md §3.4, §3.6_")

    if any_daytime:
        lines.append(
            "_Daytime-horizon scoring is advisory — weights are nightly-calibrated "
            "(see design §8.6)._"
        )
    if any_degraded:
        lines.append(
            "_NESTOR-BES unavailable; met inputs from IB Civic Center this cycle._"
        )

    return "\n".join(lines)


def _build_interpretation(tier_key: str, results: list[TierResult]) -> str:
    firing = [r for r in results if r.fire]
    if not firing:
        return ""

    horizon_names = [HORIZON_LABELS[r.horizon] for r in firing]
    horizons_str = " and ".join(horizon_names) if len(horizon_names) <= 2 else ", ".join(horizon_names[:-1]) + f", and {horizon_names[-1]}"

    if tier_key == "tier_1":
        return (
            f"SBIWTP throughput drops into the plant-signal regime in the {horizons_str} window. "
            "This is a pre-alert; no exceedance is yet observed."
        )
    if tier_key == "tier_2":
        return (
            f"Plant throughput drops into the multi-site detection regime in the {horizons_str} window, "
            "with light winds limiting dispersion. This is a pre-alert; no exceedance is yet observed."
        )
    if tier_key == "tier_3":
        return (
            f"Conditions align for a potential exceedance event in the {horizons_str} window: "
            "low plant throughput, light winds, stable atmosphere, and warm/humid boundary layer. "
            "This is a forecast-based pre-alert."
        )
    return ""


def build_tier_blocks(tier_key: str, results: list[TierResult], evaluated_at: pd.Timestamp) -> list[dict]:
    """Build Slack Block Kit blocks for a tier message."""
    text = build_tier_message(tier_key, results, evaluated_at)
    return [{"type": "section", "text": {"type": "mrkdwn", "text": text}}]
