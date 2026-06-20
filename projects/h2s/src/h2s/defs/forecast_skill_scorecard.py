"""Forecast skill scorecard (Deliverable C).

Surfaces the per-lead-hour skill curves (``forecast_skill_report``) as a
human-readable Slack post + chart, so "are the forecasts any good" is visible
without opening the parquet. Spearman / MAE / P(>30) recall vs lead hour,
Evidence vs Lean.

  forecast_validation_store → forecast_skill_report → forecast_skill_scorecard
"""

import os

import dagster as dg
import numpy as np
import pandas as pd

from h2s.defs.h2s_forecast_validation_pipeline import (
    forecast_skill_report,
    forecast_validation_store,
)
from h2s.predictor.visualizations import generate_skill_by_lead_chart

_KEY = lambda name: dg.AssetKey(["h2s", name])

_SCORECARD_CHART_LATEST = "latest/tijuana/forecast_data/forecast_skill_scorecard_latest.png"


def _wavg(series: pd.Series, weights: pd.Series) -> float | None:
    s = series.dropna()
    if s.empty:
        return None
    w = weights.loc[s.index]
    return float(np.average(s, weights=w)) if w.sum() else None


def _scorecard_lines(curves: pd.DataFrame) -> list[str]:
    """Per (product, variant): n, weighted Spearman, mean MAE, mean P(>30) recall."""
    lines = ["  product   variant   n     ρ      MAE    P>30 recall"]
    for (product, variant), g in curves.groupby(["product", "variant"]):
        rho = _wavg(g["spearman"], g["n"])
        mae = float(g["mae"].mean()) if "mae" in g else None
        rec = g["prob_recall_30"].dropna().mean() if "prob_recall_30" in g else None
        lines.append(
            f"  {product:<9} {variant:<8} {int(g['n'].sum()):<5} "
            f"{('—' if rho is None else f'{rho:+.2f}'):<6} "
            f"{('—' if mae is None else f'{mae:5.1f}'):<6} "
            f"{('—' if rec is None or np.isnan(rec) else f'{rec*100:.0f}%')}"
        )
    return lines


@dg.asset(
    key_prefix="h2s",
    group_name="forecast_validation",
    required_resource_keys={"slack", "s3"},
    kinds={"slack", "s3"},
    description="Render the per-lead-hour skill curves as a Slack scorecard + chart (Evidence vs Lean)",
    ins={"forecast_skill_report": dg.AssetIn(key=_KEY("forecast_skill_report"))},
)
def forecast_skill_scorecard(
    context: dg.AssetExecutionContext,
    forecast_skill_report: pd.DataFrame,
) -> None:
    s3 = context.resources.s3
    slack = context.resources.slack
    env_label = os.environ.get("ENV_LABEL", "").upper()
    curves = forecast_skill_report

    image_url = None
    try:
        buf = generate_skill_by_lead_chart(curves, env_label=env_label)
        data = buf.getvalue()
        s3.putFile(data, _SCORECARD_CHART_LATEST, bucket=s3.S3_BUCKET, content_type="image/png")
        image_url = s3.publicUrl(path=_SCORECARD_CHART_LATEST, bucket=s3.S3_BUCKET)
        context.log.info(f"✓ Skill scorecard chart uploaded → {_SCORECARD_CHART_LATEST}")
    except Exception as e:  # noqa: BLE001 — chart is best-effort
        context.log.warning(f"Scorecard chart render/upload failed; posting text-only: {e}")

    label = f" [{env_label}]" if env_label else ""
    if curves is None or curves.empty:
        body = (
            f"📈 *H2S Forecast Skill Scorecard{label}*\n"
            "No overlapping forecast/actual rows yet — the validation store fills "
            "in as product target hours are observed."
        )
    else:
        body = "\n".join([
            f"📈 *H2S Forecast Skill Scorecard{label}*",
            f"Skill curves: {len(curves)} (product, variant, lead) cells   "
            f"· skill_curves.parquet",
            "─" * 40,
            "```",
            *_scorecard_lines(curves),
            "```",
            "_Spearman ρ = rank skill (actual vs predicted); P>30 recall = "
            "classifier P(>30 ppb)>0.5 vs measured ≥30. Magnitude skill decays "
            "with lead hour — forecast tier is a risk ranking._",
        ])

    blocks: list[dict] = [{"type": "section", "text": {"type": "mrkdwn", "text": body}}]
    if image_url:
        blocks.append({
            "type": "image",
            "image_url": image_url,
            "alt_text": "Forecast skill (Spearman, MAE, P>30 recall) vs lead hour, Evidence vs Lean",
        })

    channel = os.environ.get("SLACK_CHANNEL_OPS", slack.channel)
    slack.get_client().chat_postMessage(channel=channel, text=body, blocks=blocks)
    context.log.info(f"Posted forecast skill scorecard to {channel}")

    context.add_output_metadata({
        "skill_cells": int(len(curves)) if curves is not None else 0,
        "chart_posted": image_url is not None,
        "channel": channel,
    })


forecast_skill_scorecard_job = dg.define_asset_job(
    name="forecast_skill_scorecard_job",
    selection=dg.AssetSelection.assets(
        forecast_validation_store, forecast_skill_report, forecast_skill_scorecard
    ),
    config={"ops": {"h2s__forecast_validation_store": {"config": {"max_age_days": None}}}},
    description="Rebuild validation store + skill curves, then post the skill scorecard to Slack",
    tags={"environment": "production", "pipeline": "forecast_validation"},
)

forecast_skill_scorecard_schedule = dg.ScheduleDefinition(
    job=forecast_skill_scorecard_job,
    cron_schedule="0 13 * * 1",  # weekly Monday 13:00 UTC
    default_status=dg.DefaultScheduleStatus.RUNNING,
    description="Weekly: rebuild forecast skill curves and post the scorecard to Slack",
)
