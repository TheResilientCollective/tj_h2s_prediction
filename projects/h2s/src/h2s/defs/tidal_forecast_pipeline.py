"""Tidal forecast pipeline using NOAA harmonic constituents.

Asset:
  tidal_forecast_asset — Hourly tidal forecast for 7 days, uploaded to S3
"""

from datetime import datetime, timezone, timedelta

import dagster as dg
import pandas as pd

from h2s.tides import generate_tidal_forecast
from h2s.constants import TIDAL_FORECAST_PATH


@dg.asset(
    key_prefix="h2s",
    group_name="h2s_tidal",
    required_resource_keys={"s3"},
    kinds={"python", "s3"},
    description="Generate tidal forecast using NOAA harmonics and upload to S3",
    config_schema={
        "forecast_hours": dg.Field(int, default_value=168, description="Hours to forecast (default 7 days)"),
    },
)
def tidal_forecast(context: dg.AssetExecutionContext) -> None:
    """Generate hourly tidal forecast using local harmonic synthesis.

    Pulls NOAA tidal harmonics (stable constants for the station location)
    and generates predictions locally, avoiding reliance on NOAA's
    prediction API which has been unstable.

    Uploads to S3 as latest/tijuana/tidal_forecast/latest.csv.
    """
    s3 = context.resources.s3
    forecast_hours = context.op_config["forecast_hours"]

    try:
        # Generate tidal forecast using local harmonic synthesis
        start_time = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
        context.log.info(f"Generating {forecast_hours}h tidal forecast from {start_time.isoformat()}")

        forecast_df = generate_tidal_forecast(start_time, hours=forecast_hours)
        context.log.info(f"✓ Generated {len(forecast_df)} tidal predictions")

        # Convert to CSV bytes
        csv_bytes = forecast_df.to_csv(index=False).encode("utf-8")

        # Upload to S3
        s3.putFile(
            path=TIDAL_FORECAST_PATH,
            data=csv_bytes,
            bucket=s3.S3_BUCKET,
        )
        context.log.info(f"✓ Uploaded tidal forecast to S3: {TIDAL_FORECAST_PATH}")

        context.add_output_metadata({
            "forecast_hours": forecast_hours,
            "rows": len(forecast_df),
            "time_range": f"{forecast_df['time'].min()} to {forecast_df['time'].max()}",
            "s3_path": TIDAL_FORECAST_PATH,
        })

    except Exception as e:
        context.log.error(f"✗ Failed to generate tidal forecast: {e}")
        raise
