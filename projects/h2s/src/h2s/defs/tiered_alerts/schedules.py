"""Tiered alerts job and schedule (same 6h cadence as forecast pipeline)."""

import dagster as dg

from h2s.constants import SCHEDULE_6HR
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
    cron_schedule=SCHEDULE_6HR,
    default_status=dg.DefaultScheduleStatus.RUNNING,
    description="Evaluate tiered forecast alerts every 6 hours (same cadence as h2s forecast)",
)
