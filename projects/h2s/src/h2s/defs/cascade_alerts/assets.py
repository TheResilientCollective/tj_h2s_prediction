"""Dagster asset: cascade_alert_dispatcher.

Runs downstream of ``h2s_products`` (the recursive product rows). Evaluates the
Tier 1–3 cascade off NESTOR-BES Evidence probabilities, applies a per-tier
quiet-window debounce against the shared alert state, and posts one
consolidated report to the ops Slack channel when a tier fires.

No shadow mode: the retired gate/sigmoid system was never vetted against the
calibration principles, so this cuts straight over to the classifier-driven
triggers.
"""

import math
import os

import dagster as dg
import pandas as pd

from .cascade import NB_SITE, TIER_ORDER, TRIGGER_VARIANT, evaluate_cascade
from .messages import build_cascade_blocks, build_cascade_message
from .state import load_state, mark_sent, save_state, should_send

_KEY = lambda name: dg.AssetKey(["h2s", name])


def _md_prob(prob: float):
    """Metadata-safe probability: rounded float, or 'n/a' for NaN."""
    if prob is None or (isinstance(prob, float) and math.isnan(prob)):
        return "n/a"
    return round(float(prob), 4)


@dg.asset(
    key_prefix="h2s",
    group_name="cascade_alerts",
    required_resource_keys={"slack", "s3"},
    kinds={"slack", "s3"},
    description="Evaluate the Tier 1–3 forecast cascade off NESTOR-BES product probabilities; dispatch to the ops Slack channel",
    ins={"h2s_products": dg.AssetIn(key=_KEY("h2s_products"))},
)
def cascade_alert_dispatcher(
    context: dg.AssetExecutionContext,
    h2s_products: pd.DataFrame,
) -> None:
    s3 = context.resources.s3
    slack = context.resources.slack
    now = pd.Timestamp.now("UTC")

    result = evaluate_cascade(h2s_products, station=NB_SITE, variant=TRIGGER_VARIANT)
    for tier in TIER_ORDER:
        ev = result.evaluations[tier]
        context.log.info(
            f"{tier} ({ev.product}/{ev.prob_col}): peak={ev.trigger_prob:.3f} "
            f"cutoff={ev.prob_cutoff} → fired={ev.fired}"
        )

    fired = result.fired_tiers
    base_md = {
        "fired_tiers": ", ".join(fired) or "none",
        "highest_fired": result.highest_fired or "none",
        "est_hours_h2s_6h": result.est_hours_h2s_6h,
        "run_ts": result.run_ts or "n/a",
        **{f"{t}_peak_prob": _md_prob(result.evaluations[t].trigger_prob) for t in TIER_ORDER},
    }

    state = load_state(s3)

    if not fired:
        context.log.info("No cascade tiers firing this run.")
        save_state(s3, state)  # round-trips any first-time `cascade` migration
        context.add_output_metadata(base_md | {"dispatched": False})
        return

    sendable = [t for t in fired if should_send(state, t, now)]
    if not sendable:
        context.log.info(
            f"Tiers firing {fired} but all within the quiet window — suppressed."
        )
        context.add_output_metadata(base_md | {"dispatched": False, "suppressed_by_debounce": True})
        return

    ops_channel = os.environ.get("SLACK_CHANNEL_OPS", slack.channel)
    text = build_cascade_message(result, h2s_products, now)
    blocks = build_cascade_blocks(result, h2s_products, now)
    slack.get_client().chat_postMessage(channel=ops_channel, text=text, blocks=blocks)
    context.log.info(f"Dispatched cascade {fired} (new: {sendable}) to {ops_channel}")

    # Mark every fired tier so the consolidated send debounces all of them.
    for tier in fired:
        mark_sent(state, tier, result.evaluations[tier].trigger_prob, now)
    save_state(s3, state)

    context.add_output_metadata(base_md | {"dispatched": True, "channel": ops_channel})
