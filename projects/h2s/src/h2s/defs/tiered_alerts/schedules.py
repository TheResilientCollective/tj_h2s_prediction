"""Tiered alerts job and schedule (hourly on the half hour)."""

import dagster as dg

from .assets import (
    tier_1_scores,
    tier_2_scores,
    tier_3_scores,
    tier_alert_dispatcher,
    tiered_alert_features,
)

tiered_alerts_job = dg.define_asset_job(
    name="tiered_alerts_job",
    selection=dg.AssetSelection.assets(
        tiered_alert_features,
        tier_1_scores,
        tier_2_scores,
        tier_3_scores,
        tier_alert_dispatcher,
    ),
    description="Evaluate Tier 1–3 H2S pre-alerts against forecast horizon windows (Tiers 1–3)",
    tags={"environment": "production", "pipeline": "tiered_alerts"},
)

tiered_alerts_schedule = dg.ScheduleDefinition(
    job=tiered_alerts_job,
    cron_schedule="30 * * * *",
    default_status=dg.DefaultScheduleStatus.RUNNING,
    description="Evaluate tiered forecast alerts hourly at :30",
)
