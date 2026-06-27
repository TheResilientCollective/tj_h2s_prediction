"""Daily forecast digest (Deliverable B).

An always-on situational-awareness post — unlike the cascade, which only fires
on a trigger. Once a day it renders the 24 h hazard timeline + log-axis
sparkline for every station and posts a compact Slack summary (peak H2S /
hazard per station + the NESTOR cascade tier status), so ops sees the outlook
even when nothing is firing.

  products_model_artifacts → h2s_products → forecast_digest
"""

import os

import dagster as dg
import pandas as pd

from h2s.constants import (
    H2S_THRESHOLD_HIGH,
    H2S_THRESHOLD_LOW,
    PRODUCTS_PATH,
    STATIONS,
)
from h2s.defs.cascade_alerts.cascade import NB_SITE, TIER_ORDER, TRIGGER_VARIANT, evaluate_cascade
from h2s.defs.h2s_products_pipeline import h2s_products, products_model_artifacts
from h2s.predictor.visualizations import generate_forecast_digest_chart

_KEY = lambda name: dg.AssetKey(["h2s", name])

_DIGEST_CHART_LATEST = "latest/tijuana/forecast_data/forecast_digest_latest.png"


def _hazard_word(peak: float) -> str:
    if peak >= H2S_THRESHOLD_HIGH:
        return "🔴 orange"
    if peak >= H2S_THRESHOLD_LOW:
        return "🟡 yellow"
    return "🟢 green"


def _station_summary_lines(products_df: pd.DataFrame, variant: str) -> list[str]:
    """One line per station: peak predicted H2S + worst hazard over the 24 h."""
    lines = []
    for site_name, info in STATIONS.items():
        w = products_df[
            (products_df["station"] == site_name) & (products_df["variant"] == variant)
        ]
        if w.empty or w["h2s_pred"].isna().all():
            lines.append(f"  {info['short']:<3} {site_name:<13} no forecast")
            continue
        peak = float(w["h2s_pred"].max())
        lines.append(
            f"  {info['short']:<3} {site_name:<13} peak {peak:>5.0f} ppb   {_hazard_word(peak)}"
        )
    return lines


def _cascade_status_lines(products_df: pd.DataFrame) -> list[str]:
    """NESTOR cascade tier status (fired or armed) for the digest footer."""
    result = evaluate_cascade(products_df, station=NB_SITE, variant=TRIGGER_VARIANT)
    lines = ["NESTOR cascade status:"]
    for tier in TIER_ORDER:
        ev = result.evaluations[tier]
        prob = "n/a" if ev.trigger_prob != ev.trigger_prob else f"{round(ev.trigger_prob * 100)}%"
        flag = "🔔 FIRING" if ev.fired else "armed"
        lines.append(f"  {tier} ({ev.product} P>{ev.prob_col[1:]}) {prob:<5} → {flag}")
    return lines


@dg.asset(
    key_prefix="h2s",
    group_name="forecast_digest",
    required_resource_keys={"slack", "s3"},
    kinds={"slack", "s3"},
    description="Daily all-station 24 h hazard-timeline digest posted to Slack (situational awareness)",
    ins={"h2s_products": dg.AssetIn(key=_KEY("h2s_products"))},
)
def forecast_digest(
    context: dg.AssetExecutionContext,
    h2s_products: pd.DataFrame,
) -> None:
    s3 = context.resources.s3
    slack = context.resources.slack
    env_label = os.environ.get("ENV_LABEL", "").upper()
    variant = TRIGGER_VARIANT
    run_ts = h2s_products["run_ts"].iloc[0] if "run_ts" in h2s_products.columns and len(h2s_products) else "n/a"

    # --- Chart: hazard timeline + sparkline per station --------------------
    image_url = None
    try:
        buf = generate_forecast_digest_chart(
            h2s_products, list(STATIONS.keys()), variant=variant, env_label=env_label
        )
        data = buf.getvalue()
        run_path = f"{PRODUCTS_PATH}/run_ts={run_ts}/forecast_digest.png"
        s3.putFile(data, run_path, bucket=s3.S3_BUCKET, content_type="image/png")
        s3.putFile(data, _DIGEST_CHART_LATEST, bucket=s3.S3_BUCKET, content_type="image/png")
        image_url = s3.publicUrl(path=run_path, bucket=s3.S3_BUCKET)
        context.log.info(f"✓ Digest chart uploaded → {run_path}")
    except Exception as e:  # noqa: BLE001 — chart is best-effort
        context.log.warning(f"Digest chart render/upload failed; posting text-only: {e}")

    # --- Slack summary -----------------------------------------------------
    label = f" [{env_label}]" if env_label else ""
    header = "\n".join([
        f"📊 *H2S 24 h Forecast Digest{label}* — {variant.capitalize()}",
        f"Forecast run: {run_ts}",
        "─" * 40,
        "Per-station 24 h outlook (peak predicted H2S):",
        *_station_summary_lines(h2s_products, variant),
        "",
        *_cascade_status_lines(h2s_products),
    ])
    blocks: list[dict] = [{"type": "section", "text": {"type": "mrkdwn", "text": header}}]
    if image_url:
        blocks.append({
            "type": "image",
            "image_url": image_url,
            "alt_text": "24 h H2S hazard timelines and log-scale sparklines, all stations",
        })

    channel = os.environ.get("SLACK_CHANNEL_OPS", slack.channel)
    slack.get_client().chat_postMessage(channel=channel, text=header, blocks=blocks)
    context.log.info(f"Posted forecast digest to {channel}")

    context.add_output_metadata({
        "run_ts": run_ts,
        "variant": variant,
        "stations": list(h2s_products["station"].unique()) if len(h2s_products) else [],
        "chart_posted": image_url is not None,
        "channel": channel,
    })


forecast_digest_job = dg.define_asset_job(
    name="forecast_digest_job",
    selection=dg.AssetSelection.assets(products_model_artifacts, h2s_products, forecast_digest),
    description="Refresh products, then post the daily all-station forecast digest to Slack",
    tags={"environment": "production", "pipeline": "forecast_digest"},
)

forecast_digest_schedule = dg.ScheduleDefinition(
    job=forecast_digest_job,
    cron_schedule="30 8 * * *",  # daily 08:30 UTC
    execution_timezone="America/Los_Angeles",
    default_status=dg.DefaultScheduleStatus.RUNNING,
    description="Daily: post the all-station 24 h forecast digest to the ops Slack channel",
)
