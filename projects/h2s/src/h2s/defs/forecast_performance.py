"""Forecast performance report + AI narrative (asymmetric, temporally-tolerant).

Reads the Phase-5 validation store (product rows joined to measured H2S) and
scores it with the analyst rubric in ``forecasting.performance_report`` —
under-predicting a hazard costs far more than over-predicting it, and an
over-prediction the actual reaches within a couple of hours is an early warning,
not a false alarm.

  forecast_validation_store → forecast_performance_report → forecast_performance_narrative

``forecast_performance_report`` renders three charts (predicted-vs-measured
scatter, skill-by-hour correlation, verdict/confusion board), stores the
structured report JSON, and posts the board to Slack. ``forecast_performance_narrative``
turns the report into plain language — via the ResilientLLM webhook when it is
configured, otherwise a deterministic local narrative — and posts it.
"""

import json
import os
from datetime import datetime, timezone

import dagster as dg
import pandas as pd

from h2s.constants import (
    PERFORMANCE_NARRATIVE_LATEST_PATH,
    PERFORMANCE_REPORT_BASE,
    PERFORMANCE_REPORT_JSON_PATH,
    PERFORMANCE_REPORT_LATEST_PATH,
)
from h2s.defs.cascade_alerts.cascade import TRIGGER_VARIANT
from h2s.defs.h2s_forecast_validation_pipeline import forecast_validation_store
from h2s.forecasting.narrative import build_llm_payload, build_narrative_text
from h2s.forecasting.performance_report import build_report
from h2s.predictor.performance_charts import (
    generate_correlation_by_hour,
    generate_pred_vs_actual_scatter,
    generate_verdict_confusion,
)

_KEY = lambda name: dg.AssetKey(["h2s", name])

_SCATTER_LATEST = "latest/tijuana/forecast_data/forecast_performance_scatter_latest.png"
_BYHOUR_LATEST = "latest/tijuana/forecast_data/forecast_performance_byhour_latest.png"
_VERDICT_LATEST = "latest/tijuana/forecast_data/forecast_performance_verdict_latest.png"


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
    group_name="forecast_validation",
    required_resource_keys={"slack", "s3"},
    kinds={"python", "slack", "s3"},
    description="Asymmetric, temporally-tolerant performance report (charts + rubric metrics) posted to Slack",
    ins={"forecast_validation_store": dg.AssetIn(key=_KEY("forecast_validation_store"))},
    config_schema={
        "variant": dg.Field(str, default_value=TRIGGER_VARIANT,
                            description="Which variant to score (evidence drives alerts)"),
    },
)
def forecast_performance_report(
    context: dg.AssetExecutionContext,
    forecast_validation_store: pd.DataFrame,
) -> dict:
    s3 = context.resources.s3
    slack = context.resources.slack
    env_label = os.environ.get("ENV_LABEL", "").upper()
    variant = context.op_config["variant"]
    run_tag = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%MZ")

    report = build_report(forecast_validation_store, variant=variant)
    summary = report["summary"]
    payload = build_llm_payload(report, variant=variant)

    # --- Persist the structured report (narrative-ready) -------------------
    report_json = json.dumps(payload, indent=2, default=str)
    s3.putFile(report_json, PERFORMANCE_REPORT_JSON_PATH.format(run_tag=run_tag),
               bucket=s3.S3_BUCKET, content_type="application/json")
    s3.putFile(report_json, PERFORMANCE_REPORT_LATEST_PATH,
               bucket=s3.S3_BUCKET, content_type="application/json")

    # --- Charts ------------------------------------------------------------
    base = f"{PERFORMANCE_REPORT_BASE}/{run_tag}"
    verdict_url = _upload(
        context, s3,
        generate_verdict_confusion(report["confusion"], summary, env_label=env_label),
        f"{base}/verdict_confusion.png", _VERDICT_LATEST, "Verdict/confusion board")
    scatter_url = _upload(
        context, s3,
        generate_pred_vs_actual_scatter(report["annotated"], env_label=env_label),
        f"{base}/pred_vs_actual.png", _SCATTER_LATEST, "Predicted-vs-measured scatter")
    byhour_url = _upload(
        context, s3,
        generate_correlation_by_hour(report["by_lead_hour"], report["by_hour_of_day"],
                                     env_label=env_label),
        f"{base}/correlation_by_hour.png", _BYHOUR_LATEST, "Correlation-by-hour")

    # --- Slack board -------------------------------------------------------
    label = f" [{env_label}]" if env_label else ""
    ta = summary.get("tolerant_accuracy")
    header = "\n".join([
        f"📈 *H2S Forecast Performance{label}* — {variant.capitalize()}",
        f"Matched forecast-hours: {summary.get('n', 0)}   "
        f"tolerant-accuracy: {'n/a' if ta is None else f'{ta:.0%}'}   "
        f"dangerous-miss: {summary.get('dangerous_miss_rate')}   "
        f"smell-miss: {summary.get('smell_miss_rate')}",
    ])
    blocks: list[dict] = [{"type": "section", "text": {"type": "mrkdwn", "text": header}}]
    for url, alt in [(verdict_url, "Verdict and category-confusion board"),
                     (byhour_url, "Skill correlation by lead hour and hour-of-day"),
                     (scatter_url, "Predicted vs measured H2S scatter")]:
        if url:
            blocks.append({"type": "image", "image_url": url, "alt_text": alt})

    channel = os.environ.get("SLACK_CHANNEL_OPS", slack.channel)
    slack.get_client().chat_postMessage(channel=channel, text=header, blocks=blocks)

    context.add_output_metadata({
        "run_tag": run_tag,
        "variant": variant,
        "n": summary.get("n", 0),
        "tolerant_accuracy": summary.get("tolerant_accuracy"),
        "dangerous_miss_rate": summary.get("dangerous_miss_rate"),
        "smell_miss_rate": summary.get("smell_miss_rate"),
        "report_path": PERFORMANCE_REPORT_LATEST_PATH,
    })
    # Compact, JSON-serialisable hand-off for the narrative asset.
    return payload


def _extract_text(obj) -> str | None:
    """Best-effort pull of narrative text from the webhook's JSON response."""
    if obj is None:
        return None
    if isinstance(obj, str):
        return obj.strip() or None
    if isinstance(obj, list) and obj:
        return _extract_text(obj[0])
    if isinstance(obj, dict):
        for key in ("text", "narrative", "output", "summary", "message", "content", "result"):
            if key in obj and obj[key]:
                inner = _extract_text(obj[key])
                if inner:
                    return inner
    return None


@dg.asset(
    key_prefix="h2s",
    group_name="forecast_validation",
    required_resource_keys={"slack", "s3", "llm"},
    kinds={"python", "slack", "s3"},
    description="Plain-language forecast-performance narrative (ResilientLLM webhook, local fallback)",
    ins={"forecast_performance_report": dg.AssetIn(key=_KEY("forecast_performance_report"))},
)
def forecast_performance_narrative(
    context: dg.AssetExecutionContext,
    forecast_performance_report: dict,
) -> None:
    s3 = context.resources.s3
    slack = context.resources.slack
    llm = context.resources.llm
    env_label = os.environ.get("ENV_LABEL", "").upper()
    payload = forecast_performance_report
    variant = payload.get("variant", TRIGGER_VARIANT)

    # Deterministic local narrative — always available; the offline fallback.
    local_text = build_narrative_text(
        {"summary": payload.get("summary", {}),
         "by_lead_hour": pd.DataFrame(payload.get("correlation_by_lead_hour", [])),
         "tolerance_hours": payload.get("tolerance_hours"),
         "smell_threshold_ppb": payload.get("smell_threshold_ppb")},
        variant=variant,
    )

    narrative, source = local_text, "local"
    token = getattr(llm, "token", "") or ""
    report_id = getattr(llm, "webhook_uuid", "") or ""
    if token and report_id:
        try:
            resp = llm.execute_with_data(report_id, payload)
            llm_text = _extract_text(resp)
            if llm_text:
                narrative, source = llm_text, "resilientllm"
                context.log.info("✓ ResilientLLM narrative received")
            else:
                context.log.warning("ResilientLLM returned no usable text; using local narrative")
        except Exception as e:  # noqa: BLE001 — webhook is best-effort
            context.log.warning(f"ResilientLLM call failed; using local narrative: {e}")
    else:
        context.log.info("ResilientLLM token/webhook unset; using local narrative")

    s3.putFile(narrative, PERFORMANCE_NARRATIVE_LATEST_PATH,
               bucket=s3.S3_BUCKET, content_type="text/plain")

    label = f" [{env_label}]" if env_label else ""
    text = f"📝 *H2S Forecast Performance Narrative{label}* _(source: {source})_\n\n{narrative}"
    channel = os.environ.get("SLACK_CHANNEL_OPS", slack.channel)
    slack.get_client().chat_postMessage(channel=channel, text=text)
    context.log.info(f"Posted performance narrative ({source}) to {channel}")

    context.add_output_metadata({
        "source": source,
        "variant": variant,
        "narrative_preview": narrative[:600],
    })


forecast_performance_job = dg.define_asset_job(
    name="forecast_performance_job",
    selection=dg.AssetSelection.assets(
        forecast_validation_store,
        forecast_performance_report,
        forecast_performance_narrative,
    ),
    description="Rebuild the validation store, score the asymmetric performance rubric, "
                "and post the charts + AI narrative to Slack",
    tags={"environment": "production", "pipeline": "forecast_performance"},
)

forecast_performance_schedule = dg.ScheduleDefinition(
    job=forecast_performance_job,
    cron_schedule="0 14 * * 1",  # weekly Monday 14:00 UTC
    default_status=dg.DefaultScheduleStatus.STOPPED,
    description="Weekly: post the forecast performance rubric report + AI narrative to Slack",
)
