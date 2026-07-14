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

from h2s.constants import PRODUCTS_PATH
from h2s.forecasting.obs_freshness import load_realtime_h2s
from h2s.predictor.visualizations import generate_forecast_hazard_chart

from .cascade import (
    NB_SITE,
    TIER_ORDER,
    TRIGGER_VARIANT,
    evaluate_cascade,
    evaluate_obs_backstop,
)
from .messages import (
    build_cascade_blocks,
    build_cascade_message,
    build_obs_backstop_blocks,
    build_obs_backstop_message,
)
from .state import load_state, mark_sent, save_state, should_send

_KEY = lambda name: dg.AssetKey(["h2s", name])

# Latest-mirror for the cascade hazard chart (stable URL for dashboards).
_CASCADE_CHART_LATEST = "latest/tijuana/forecast_data/cascade_nestor_latest.png"


def _md_prob(prob: float):
    """Metadata-safe probability: rounded float, or 'n/a' for NaN."""
    if prob is None or (isinstance(prob, float) and math.isnan(prob)):
        return "n/a"
    return round(float(prob), 4)


def _render_and_upload_chart(context, s3, products_df, result) -> str | None:
    """Render the NESTOR hazard-timeline chart, upload to S3, return its public URL.

    Failure to render/upload must never block the alert itself, so any error is
    logged and the alert posts text-only (image_url=None).
    """
    try:
        env_label = os.environ.get("ENV_LABEL", "").upper()
        buf = generate_forecast_hazard_chart(
            products_df, NB_SITE, variant=TRIGGER_VARIANT, env_label=env_label
        )
        data = buf.getvalue()
        run_ts = result.run_ts or pd.Timestamp.now("UTC").strftime("%Y-%m-%dT%H%MZ")
        run_path = f"{PRODUCTS_PATH}/run_ts={run_ts}/cascade_nestor.png"
        s3.putFile(data, run_path, bucket=s3.S3_BUCKET, content_type="image/png")
        s3.putFile(data, _CASCADE_CHART_LATEST, bucket=s3.S3_BUCKET, content_type="image/png")
        url = s3.publicUrl(path=run_path, bucket=s3.S3_BUCKET)
        context.log.info(f"✓ Cascade chart uploaded → {run_path}")
        return url
    except Exception as e:  # noqa: BLE001 — chart is best-effort
        context.log.warning(f"Cascade chart render/upload failed; posting text-only: {e}")
        return None


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

    # Raw-sensor backstop: fires off measured H2S so an in-progress exceedance
    # is alerted even when the model probabilities are blind (the 2026-07-12
    # 168 ppb onset passed a cascade run silently). Best-effort load — a feed
    # failure degrades to model-only triggers.
    try:
        backstop = evaluate_obs_backstop(load_realtime_h2s(s3), station=NB_SITE, now=now)
    except Exception as e:  # noqa: BLE001 — backstop must never block the cascade
        context.log.warning(f"Obs backstop unavailable (realtime feed): {e}")
        backstop = {}
    obs_fired = [t for t in TIER_ORDER if backstop and backstop[t].fired]
    base_md["obs_backstop_fired"] = ", ".join(obs_fired) or "none"
    if backstop:
        peak = backstop[TIER_ORDER[0]].peak_ppb
        base_md["obs_backstop_peak_ppb"] = peak if peak is not None else "n/a"

    state = load_state(s3)
    ops_channel = os.environ.get("SLACK_CHANNEL_OPS", slack.channel)
    dispatched_model = False

    if not fired:
        context.log.info("No cascade tiers firing this run.")
    else:
        sendable = [t for t in fired if should_send(state, t, now)]
        if not sendable:
            context.log.info(
                f"Tiers firing {fired} but all within the quiet window — suppressed."
            )
            base_md["suppressed_by_debounce"] = True
        else:
            image_url = _render_and_upload_chart(context, s3, h2s_products, result)
            text = build_cascade_message(result, h2s_products, now)
            blocks = build_cascade_blocks(result, h2s_products, now, image_url=image_url)
            slack.get_client().chat_postMessage(channel=ops_channel, text=text, blocks=blocks)
            context.log.info(f"Dispatched cascade {fired} (new: {sendable}) to {ops_channel}")
            dispatched_model = True
            # Mark every fired tier so the consolidated send debounces all of them.
            for tier in fired:
                mark_sent(state, tier, result.evaluations[tier].trigger_prob, now)

    # Dispatch the backstop only for tiers the model-driven message did not
    # just cover, each behind its own quiet window.
    obs_pending = [t for t in obs_fired if not (dispatched_model and t in fired)]
    obs_sendable = [t for t in obs_pending if should_send(state, f"{t}_obs", now)]
    dispatched_obs = False
    if obs_sendable:
        text = build_obs_backstop_message(backstop, now, obs_sendable)
        blocks = build_obs_backstop_blocks(backstop, now, obs_sendable)
        slack.get_client().chat_postMessage(channel=ops_channel, text=text, blocks=blocks)
        context.log.info(f"Dispatched obs backstop {obs_sendable} to {ops_channel}")
        dispatched_obs = True
        for tier in obs_sendable:
            mark_sent(state, f"{tier}_obs", backstop[tier].peak_ppb, now)
    elif obs_pending:
        context.log.info(f"Obs backstop firing {obs_pending} but within the quiet window.")

    save_state(s3, state)  # also round-trips any first-time `cascade` migration
    context.add_output_metadata(base_md | {
        "dispatched": dispatched_model,
        "dispatched_obs_backstop": dispatched_obs,
        "channel": ops_channel if (dispatched_model or dispatched_obs) else "n/a",
    })
