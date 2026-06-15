"""Dagster glue for the observed >10 ppb "Alert Performance" machine.

  - sensor    : h2s_alert_performance_sensor   (polls obs every 5 min)
  - asset     : h2s_alert_performance_dispatcher (Slack + S3 close-out archive)
  - job       : h2s_alert_performance_job

Separate from the watch/critical machine in ``h2s_alert_system`` (which is
unchanged). On close it scores the stored product forecasts against what was
measured; if no product runs covered the event window the comparison degrades
gracefully to "unavailable".
"""

import json
import os
from datetime import datetime, timezone

import dagster as dg
import pandas as pd

from h2s.constants import (
    ALERT_PERF_ARCHIVE_PATH,
    ALERT_PERF_LATEST_PATH,
    PRODUCTS_PATH,
)
from h2s.defs.h2s_alert_system import _load_recent

from .logic import (
    evaluate_yellow,
    forecast_vs_actual,
    hours_above,
    performance_summary,
)
from .messages import build_closeout_message, build_onset_message
from .state import load_state, save_state


# ---------------------------------------------------------------------------
# Historical product loader (best-effort)
# ---------------------------------------------------------------------------

def _load_recent_products(s3, start, end, max_runs: int = 16) -> pd.DataFrame | None:
    """Concatenate stored product runs whose run_ts could cover [start, end].

    A run issued at rt forecasts rt+1..rt+24, so any run in
    [start-36h, end] may contain event hours. Returns None on any failure or
    when nothing is found — the close-out then reports the comparison as
    unavailable rather than erroring.
    """
    try:
        objs = s3.getClient().list_objects(s3.S3_BUCKET, prefix=f"{PRODUCTS_PATH}/", recursive=True)
    except Exception:
        return None

    candidates: list[tuple[pd.Timestamp, str]] = []
    for o in objs:
        name = o.object_name
        if not name.endswith("products.parquet"):
            continue
        seg = next((p for p in name.split("/") if p.startswith("run_ts=")), None)
        if seg is None:
            continue
        try:
            rt = pd.to_datetime(seg.split("=", 1)[1], format="%Y-%m-%dT%H%MZ", utc=True)
        except Exception:
            continue
        if start - pd.Timedelta(hours=36) <= rt <= end:
            candidates.append((rt, name))

    if not candidates:
        return None
    candidates.sort()
    frames = []
    for _, name in candidates[-max_runs:]:
        try:
            frames.append(pd.read_parquet(s3.get_presigned_url(name, bucket=s3.S3_BUCKET)))
        except Exception:
            continue
    return pd.concat(frames, ignore_index=True) if frames else None


# ---------------------------------------------------------------------------
# Dispatcher asset
# ---------------------------------------------------------------------------

class PerformanceConfig(dg.Config):
    send_onset: bool = False
    onset_msg: str = ""
    send_close: bool = False
    close_msg: str = ""
    archive_json: str = ""


@dg.asset(
    key_prefix="h2s",
    group_name="alert_performance",
    required_resource_keys={"slack", "s3"},
    kinds={"slack", "s3"},
    description="Dispatch yellow-tier (>10 ppb) onset + Alert-Performance close-out; archive close-outs to S3",
)
def h2s_alert_performance_dispatcher(
    context: dg.AssetExecutionContext,
    config: PerformanceConfig,
) -> None:
    slack = context.resources.slack
    s3 = context.resources.s3
    channel = os.environ.get("SLACK_CHANNEL_OPS", slack.channel)
    client = slack.get_client()

    if config.send_onset and config.onset_msg:
        context.log.info("Sending yellow onset alert.")
        client.chat_postMessage(channel=channel, text=f"```{config.onset_msg}```")

    if config.send_close and config.close_msg:
        context.log.info("Sending Alert-Performance close-out.")
        client.chat_postMessage(channel=channel, text=f"```{config.close_msg}```")
        if config.archive_json:
            ts = datetime.now(timezone.utc)
            path = f"{ALERT_PERF_ARCHIVE_PATH}/{ts.strftime('%Y-%m-%d')}/{ts.strftime('%Y-%m-%d_%H%M')}_performance.json"
            s3.putFile(config.archive_json, path=path, content_type="application/json")
            s3.putFile(config.archive_json, path=ALERT_PERF_LATEST_PATH, content_type="application/json")
            context.log.info(f"Archived performance close-out to {path}")

    if not (config.send_onset or config.send_close):
        context.log.info("No yellow-tier action this run.")


h2s_alert_performance_job = dg.define_asset_job(
    name="h2s_alert_performance_job",
    selection=dg.AssetSelection.assets(h2s_alert_performance_dispatcher),
    description="Dispatch observed >10 ppb onset + Alert-Performance close-out",
)


# ---------------------------------------------------------------------------
# Sensor
# ---------------------------------------------------------------------------

@dg.sensor(
    job=h2s_alert_performance_job,
    minimum_interval_seconds=300,
    required_resource_keys={"s3"},
    description="Poll H2S observations every 5 min; open at >10 ppb, close 2h below, and score the forecast on close",
    default_status=dg.DefaultSensorStatus.RUNNING,
)
def h2s_alert_performance_sensor(context: dg.SensorEvaluationContext):
    s3 = context.resources.s3
    try:
        df = _load_recent(s3, hours=12)
    except Exception as e:
        context.log.warning(f"Could not load H2S data: {e}")
        return

    now = pd.Timestamp.now("UTC")
    state = load_state(s3)
    verdict, cell = evaluate_yellow(df, state["yellow"], now)
    state["yellow"] = cell

    onset_msg = ""
    close_msg = ""
    archive_json = ""

    if verdict.send_onset:
        onset_msg = build_onset_message(verdict.onset_row, verdict.prev_row)

    if verdict.send_close and verdict.event_df is not None and not verdict.event_df.empty:
        event_start = pd.Timestamp(verdict.event_df["time"].iloc[0])
        event_end = pd.Timestamp(verdict.event_df["time"].iloc[-1])
        ha = hours_above(df, now)
        products = _load_recent_products(s3, event_start, event_end)
        fva = forecast_vs_actual(products, df, event_start, event_end)
        summary = performance_summary(fva)
        close_msg = build_closeout_message(verdict.event_df, ha, fva, summary, verdict.current_ppb or 0.0)
        archive_json = json.dumps({
            "generated_at": now.isoformat(),
            "event_start": str(event_start),
            "event_end": str(event_end),
            "hours_above": ha,
            "forecast_skill": summary,
            "message": close_msg,
        }, indent=2, default=str)

    save_state(s3, state)

    if not (verdict.send_onset or verdict.send_close):
        context.log.debug(
            f"No yellow action — latest {verdict.current_ppb} ppb" if not df.empty else "No data"
        )
        return

    yield dg.RunRequest(
        run_key=f"alert_perf_{now.isoformat()}",
        run_config={
            "ops": {
                "h2s__h2s_alert_performance_dispatcher": {
                    "config": {
                        "send_onset": verdict.send_onset,
                        "onset_msg": onset_msg,
                        "send_close": verdict.send_close,
                        "close_msg": close_msg,
                        "archive_json": archive_json,
                    }
                }
            }
        },
    )
