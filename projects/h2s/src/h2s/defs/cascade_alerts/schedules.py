"""Forecast cascade job + schedule.

The job runs the product engine and the cascade dispatcher together so each
evaluation alerts off a freshly computed forecast (compute-on-trigger). Runs
every 6 h, aligned with the forecast/NOAA update cadence.
"""

import dagster as dg

from h2s.constants import SCHEDULE_6HR
from h2s.defs.h2s_products_pipeline import h2s_products, products_model_artifacts

from .assets import cascade_alert_dispatcher

cascade_alerts_job = dg.define_asset_job(
    name="cascade_alerts_job",
    selection=dg.AssetSelection.assets(
        products_model_artifacts,
        h2s_products,
        cascade_alert_dispatcher,
    ),
    description=(
        "Refresh nowcast/nearcast/forecast products, then evaluate the Tier 1–3 "
        "forecast cascade off NESTOR-BES probabilities and alert ops on firing."
    ),
    tags={"environment": "production", "pipeline": "cascade_alerts"},
)

cascade_alerts_schedule = dg.ScheduleDefinition(
    job=cascade_alerts_job,
    cron_schedule=SCHEDULE_6HR,
    default_status=dg.DefaultScheduleStatus.RUNNING,
    description="Every 6 h: refresh products and evaluate the Tier 1–3 forecast cascade",
)
