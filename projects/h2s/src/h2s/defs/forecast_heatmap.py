"""Heatmap forecast board (hourly per-station outlook).

A heatmap-first companion to the sparkline digest: posts two grids that show
*every* hazard level across all stations at a glance, not just the >30 cascade
triggers.

  • Category grid   — station × hour, cell coloured by the 4-tier category
                      (green / yellow / yellow-high / orange) and labelled with
                      the predicted H2S (ppb).
  • Probability grid — station × {P>5, P>10, P>30} × hour, shaded by the
                      exceedance probability.

  products_model_artifacts → h2s_products → forecast_heatmap_board

The board uploads both PNGs to S3 (run-scoped + a ``latest`` mirror) and posts
them to the ops Slack channel with a compact per-station category tally.
"""

import os

import dagster as dg
import pandas as pd

from h2s.constants import (
    CATEGORY_LABELS,
    CATEGORY_ORDER,
    PRODUCTS_PATH,
    STATIONS,
    h2s_category,
)
from h2s.defs.cascade_alerts.cascade import TRIGGER_VARIANT
from h2s.defs.h2s_products_pipeline import h2s_products, products_model_artifacts
from h2s.predictor.forecast_heatmap import (
    generate_category_heatmap,
    generate_probability_heatmap,
)

_KEY = lambda name: dg.AssetKey(["h2s", name])

_CATEGORY_LATEST = "latest/tijuana/forecast_data/forecast_heatmap_category_latest.png"
_PROBABILITY_LATEST = "latest/tijuana/forecast_data/forecast_heatmap_probability_latest.png"


def _category_tally_lines(products_df: pd.DataFrame, variant: str) -> list[str]:
    """One line per station: hour counts by category + peak ppb."""
    lines = []
    for site_name, info in STATIONS.items():
        w = products_df[
            (products_df["station"] == site_name) & (products_df["variant"] == variant)
        ]
        if w.empty or w["h2s_pred"].isna().all():
            lines.append(f"  {info['short']:<3} {site_name:<13} no forecast")
            continue
        cats = w["h2s_pred"].map(h2s_category)
        counts = cats.value_counts()
        parts = [
            f"{CATEGORY_LABELS[c]} {int(counts[c])}"
            for c in CATEGORY_ORDER
            if counts.get(c, 0) > 0
        ]
        peak = float(w["h2s_pred"].max())
        lines.append(
            f"  {info['short']:<3} {site_name:<13} peak {peak:>5.0f} ppb   "
            + " · ".join(parts)
        )
    return lines


def _upload(context, s3, buf, run_path, latest_path, alt) -> str | None:
    try:
        data = buf.getvalue()
        s3.putFile(data, run_path, bucket=s3.S3_BUCKET, content_type="image/png")
        s3.putFile(data, latest_path, bucket=s3.S3_BUCKET, content_type="image/png")
        url = s3.publicUrl(path=run_path, bucket=s3.S3_BUCKET)
        context.log.info(f"✓ {alt} uploaded → {run_path}")
        return url
    except Exception as e:  # noqa: BLE001 — charts are best-effort
        context.log.warning(f"{alt} render/upload failed; skipping image: {e}")
        return None


@dg.asset(
    key_prefix="h2s",
    group_name="forecast_digest",
    required_resource_keys={"slack", "s3"},
    kinds={"slack", "s3"},
    description="All-station hazard + exceedance-probability heatmaps posted to Slack",
    ins={"h2s_products": dg.AssetIn(key=_KEY("h2s_products"))},
)
def forecast_heatmap_board(
    context: dg.AssetExecutionContext,
    h2s_products: pd.DataFrame,
) -> None:
    s3 = context.resources.s3
    slack = context.resources.slack
    env_label = os.environ.get("ENV_LABEL", "").upper()
    variant = TRIGGER_VARIANT
    run_ts = (
        h2s_products["run_ts"].iloc[0]
        if "run_ts" in h2s_products.columns and len(h2s_products)
        else "n/a"
    )
    stations = list(STATIONS.keys())

    cat_url = _upload(
        context, s3,
        generate_category_heatmap(h2s_products, stations, variant=variant, env_label=env_label),
        f"{PRODUCTS_PATH}/run_ts={run_ts}/forecast_heatmap_category.png",
        _CATEGORY_LATEST, "Category heatmap",
    )
    prob_url = _upload(
        context, s3,
        generate_probability_heatmap(h2s_products, stations, variant=variant, env_label=env_label),
        f"{PRODUCTS_PATH}/run_ts={run_ts}/forecast_heatmap_probability.png",
        _PROBABILITY_LATEST, "Probability heatmap",
    )

    label = f" [{env_label}]" if env_label else ""
    header = "\n".join([
        f"🗺️ *H2S Forecast Heatmap{label}* — {variant.capitalize()}",
        f"Forecast run: {run_ts}",
        "─" * 40,
        "Per-station 24 h category tally (peak predicted H2S):",
        *_category_tally_lines(h2s_products, variant),
    ])
    blocks: list[dict] = [{"type": "section", "text": {"type": "mrkdwn", "text": header}}]
    if cat_url:
        blocks.append({"type": "image", "image_url": cat_url,
                       "alt_text": "Station × hour H2S hazard category heatmap"})
    if prob_url:
        blocks.append({"type": "image", "image_url": prob_url,
                       "alt_text": "Station × threshold exceedance-probability heatmap"})

    channel = os.environ.get("SLACK_CHANNEL_OPS", slack.channel)
    slack.get_client().chat_postMessage(channel=channel, text=header, blocks=blocks)
    context.log.info(f"Posted forecast heatmap board to {channel}")

    context.add_output_metadata({
        "run_ts": run_ts,
        "variant": variant,
        "category_chart_posted": cat_url is not None,
        "probability_chart_posted": prob_url is not None,
        "channel": channel,
    })


forecast_heatmap_job = dg.define_asset_job(
    name="forecast_heatmap_job",
    selection=dg.AssetSelection.assets(
        products_model_artifacts, h2s_products, forecast_heatmap_board
    ),
    description="Refresh products, then post the all-station forecast heatmap board to Slack",
    tags={"environment": "production", "pipeline": "forecast_heatmap"},
)

forecast_heatmap_schedule = dg.ScheduleDefinition(
    job=forecast_heatmap_job,
    cron_schedule="45 8 * * *",  # daily 08:45 UTC, just after the digest
    execution_timezone="America/Los_Angeles",
    default_status=dg.DefaultScheduleStatus.RUNNING,
    description="Daily: post the all-station forecast heatmap board to the ops Slack channel",
)
