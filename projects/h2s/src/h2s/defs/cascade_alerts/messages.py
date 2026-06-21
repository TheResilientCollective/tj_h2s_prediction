"""Slack report builders for the forecast cascade (Tiers 1–3).

One consolidated message per cascade run lists a section for every tier that
fired, escalating in audience and horizon:

  Tier 1 (PLANT-SIGNAL)   → nowcast + nearcast peaks + estimated hours of H2S
  Tier 2 (MULTI-SITE-RISK)→ adds the forecast-window outlook
  Tier 3 (EXCEEDANCE-RISK)→ adds the all-station P(>30 ppb) board + health-team flag

Evidence drives the triggers; Lean probabilities are shown beside them (``E`` /
``L``) so a reader can see the two feature sets agree — the published
"not overdetermined" check.
"""

from __future__ import annotations

import math

import pandas as pd

from h2s.constants import (
    ALERT_TIERS,
    PRODUCT_FORECAST,
    PRODUCT_NEARCAST,
    PRODUCT_NOWCAST,
    STATIONS,
)
from h2s.defs.cascade_alerts.cascade import CascadeResult, NB_SITE
from h2s.defs.h2s_alert_system import _fmt_time

_TIER_EMOJI = {"tier_1": "🟢", "tier_2": "🟡", "tier_3": "🔴"}

_SUGGESTED = {
    "tier_1": "Monitor station readings and confirm SBIWTP operational status.",
    "tier_2": (
        "Verify monitoring-station status and pre-position field response. "
        "_(Health-team email routing: pending hook.)_"
    ),
    "tier_3": (
        "Alert the health team and prepare for a possible WATCH-level exceedance. "
        "_(Health-team routing: pending hook.)_"
    ),
}


def _fmt_pct(p: float | None) -> str:
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return "n/a"
    return f"{round(p * 100):d}%"


def _hhmm(ts) -> str:
    try:
        return pd.Timestamp(ts).tz_convert("America/Los_Angeles").strftime("%H:%M")
    except Exception:
        return str(ts)


def _peak(df: pd.DataFrame, station: str, variant: str, product: str, prob_col: str) -> dict | None:
    """Peak-probability row for a (station, variant, product) window."""
    w = df[
        (df["station"] == station)
        & (df["variant"] == variant)
        & (df["product"] == product)
    ]
    if w.empty or prob_col not in w.columns:
        return None
    probs = w[prob_col].dropna()
    if probs.empty:
        # No probability, but we can still surface the magnitude peak.
        if w["h2s_pred"].notna().any():
            row = w.loc[w["h2s_pred"].idxmax()]
            return {"prob": float("nan"), "h2s": float(row["h2s_pred"]),
                    "time": row["time"], "lead": int(row["lead_hour"])}
        return None
    row = w.loc[probs.idxmax()]
    return {"prob": float(row[prob_col]), "h2s": float(row["h2s_pred"]),
            "time": row["time"], "lead": int(row["lead_hour"])}


def _both_variants_pct(df: pd.DataFrame, station: str, product: str, prob_col: str) -> str:
    """``E 62% / L 58%`` for one (station, product, prob_col)."""
    ev = _peak(df, station, "evidence", product, prob_col)
    ln = _peak(df, station, "lean", product, prob_col)
    return f"E {_fmt_pct(ev['prob'] if ev else None)} / L {_fmt_pct(ln['prob'] if ln else None)}"


def _tier1_lines(df: pd.DataFrame, result: CascadeResult) -> list[str]:
    ev = result.evaluations["tier_1"]
    lines = [
        f"{_TIER_EMOJI['tier_1']} *Tier 1 — {ALERT_TIERS['tier_1']['label']}*  "
        f"(P(>5 ppb) nowcast = {_fmt_pct(ev.trigger_prob)} > {_fmt_pct(ev.prob_cutoff)})",
        f"   Next 6 h: ~{result.est_hours_h2s_6h} h of detectable H2S (≥5 ppb) predicted",
    ]
    now_pk = _peak(df, NB_SITE, "evidence", PRODUCT_NOWCAST, "p5")
    near_pk = _peak(df, NB_SITE, "evidence", PRODUCT_NEARCAST, "p10")
    if now_pk:
        lines.append(
            f"   Nowcast (0–3h):  peak {now_pk['h2s']:.0f} ppb @ {_hhmm(now_pk['time'])}   "
            f"P(>5) {_both_variants_pct(df, NB_SITE, PRODUCT_NOWCAST, 'p5')}"
        )
    if near_pk:
        lines.append(
            f"   Nearcast (3–6h): peak {near_pk['h2s']:.0f} ppb @ {_hhmm(near_pk['time'])}   "
            f"P(>10) {_both_variants_pct(df, NB_SITE, PRODUCT_NEARCAST, 'p10')}"
        )
    lines.append(f"   → {_SUGGESTED['tier_1']}")
    return lines


def _tier2_lines(df: pd.DataFrame, result: CascadeResult) -> list[str]:
    ev = result.evaluations["tier_2"]
    lines = [
        f"{_TIER_EMOJI['tier_2']} *Tier 2 — {ALERT_TIERS['tier_2']['label']}*  "
        f"(P(>10 ppb) nearcast = {_fmt_pct(ev.trigger_prob)} > {_fmt_pct(ev.prob_cutoff)})",
    ]
    fc_pk = _peak(df, NB_SITE, "evidence", PRODUCT_FORECAST, "p10")
    if fc_pk:
        lines.append(
            f"   Forecast (6–24h): peak {fc_pk['h2s']:.0f} ppb @ {_hhmm(fc_pk['time'])}   "
            f"P(>10) {_both_variants_pct(df, NB_SITE, PRODUCT_FORECAST, 'p10')}   "
            f"P(>30) {_both_variants_pct(df, NB_SITE, PRODUCT_FORECAST, 'p30')}"
        )
    lines.append(f"   → {_SUGGESTED['tier_2']}")
    return lines


def _tier3_lines(df: pd.DataFrame, result: CascadeResult) -> list[str]:
    ev = result.evaluations["tier_3"]
    lines = [
        f"{_TIER_EMOJI['tier_3']} *Tier 3 — {ALERT_TIERS['tier_3']['label']}*  "
        f"(P(>30 ppb) forecast = {_fmt_pct(ev.trigger_prob)} > {_fmt_pct(ev.prob_cutoff)})",
        "   All-station forecast P(>30 ppb):",
    ]
    for site_name, info in STATIONS.items():
        pk = _peak(df, site_name, "evidence", PRODUCT_FORECAST, "p30")
        peak_note = f"   (peak {pk['h2s']:.0f} ppb @ {_hhmm(pk['time'])})" if pk else ""
        lines.append(
            f"     {info['short']:<3} {site_name:<13} "
            f"{_both_variants_pct(df, site_name, PRODUCT_FORECAST, 'p30')}{peak_note}"
        )
    lines.append(f"   → {_SUGGESTED['tier_3']}")
    return lines


_TIER_LINE_BUILDERS = {
    "tier_1": _tier1_lines,
    "tier_2": _tier2_lines,
    "tier_3": _tier3_lines,
}


def build_cascade_message(
    result: CascadeResult,
    products_df: pd.DataFrame,
    evaluated_at: pd.Timestamp,
) -> str:
    """Consolidated plain-text cascade report for all fired tiers."""
    fired = result.fired_tiers
    labels = ", ".join(ALERT_TIERS[t]["label"] for t in fired) or "none"
    header = [
        "🌊 *H2S Forecast Cascade — NESTOR / Berry Elementary School*",
        f"Evaluated: {_fmt_time(evaluated_at)}",
        f"Forecast run: {result.run_ts or 'n/a'}   ·   Model: {result.model_version or 'n/a'}",
        f"Tiers firing: *{labels}*",
        "─" * 48,
    ]
    body: list[str] = []
    for tier in fired:  # ascending: tier_1 → tier_3
        body += _TIER_LINE_BUILDERS[tier](products_df, result)
        body.append("")
    footer = [
        "_Evidence variant drives triggers; Lean (L) shown for the "
        "not-overdetermined comparison. Cutoffs are pre-validation starting "
        "points (Phase 5 will tune them)._",
    ]
    return "\n".join(header + body + footer)


def _mrkdwn_section(text: str) -> dict:
    """A section block, truncated to Slack's 3000-char text limit."""
    if len(text) > 2990:
        text = text[:2985] + "\n…"
    return {"type": "section", "text": {"type": "mrkdwn", "text": text}}


def build_cascade_blocks(
    result: CascadeResult,
    products_df: pd.DataFrame,
    evaluated_at: pd.Timestamp,
    image_url: str | None = None,
) -> list[dict]:
    """Block Kit payload for the cascade report.

    One section block per fired tier (never one giant block) so the consolidated
    message can never blow Slack's 3000-char-per-section cap, plus an optional
    NESTOR hazard-timeline image and a context footer.
    """
    fired = result.fired_tiers
    labels = ", ".join(ALERT_TIERS[t]["label"] for t in fired) or "none"
    header = "\n".join([
        "🌊 *H2S Forecast Cascade — NESTOR / Berry Elementary School*",
        f"Evaluated: {_fmt_time(evaluated_at)}",
        f"Forecast run: {result.run_ts or 'n/a'}   ·   Model: {result.model_version or 'n/a'}",
        f"Tiers firing: *{labels}*",
    ])

    blocks: list[dict] = [_mrkdwn_section(header), {"type": "divider"}]
    for tier in fired:  # ascending: tier_1 → tier_3
        blocks.append(_mrkdwn_section("\n".join(_TIER_LINE_BUILDERS[tier](products_df, result))))

    if image_url:
        blocks.append({
            "type": "image",
            "image_url": image_url,
            "alt_text": "NESTOR-BES 24 h H2S forecast — hazard timeline and log-scale sparkline",
        })

    blocks.append({
        "type": "context",
        "elements": [{
            "type": "mrkdwn",
            "text": (
                "Evidence variant drives triggers; Lean (L) shown for the "
                "not-overdetermined comparison. Cutoffs are pre-validation "
                "starting points (Phase 5 will tune them)."
            ),
        }],
    })
    return blocks
