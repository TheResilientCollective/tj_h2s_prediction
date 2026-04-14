"""Physics-based H2S river emission grid asset.

Runs TijuanaRiverSources + H2SGenerationModel against current forecast
conditions and uploads a GeoDemic-compatible GridData JSON to S3.

Output is consumed by Geodemic's GET /api/v1/tj-data/dispersion/river-emission-grid
endpoint.
"""

import json

import dagster as dg
import numpy as np
import pandas as pd

from h2s.constants import FLOW_COL, FORECAST_DATA_PATH, RIVER_EMISSION_GRID_LATEST_PATH
from h2s.dispersion.grid_config import GRID_BOUNDS, GRID_NROWS, GRID_NCOLS
from h2s.emissions.h2s_generation import H2SGenerationModel
from h2s.emissions.tj_river_sources import TijuanaRiverSources


@dg.asset(
    key_prefix="h2s",
    group_name="h2s_dispersion",
    required_resource_keys={"s3"},
    kinds={"python", "s3"},
    description=(
        "Physics-based H2S river emission grid (Arrhenius kinetics + Gaussian spreading). "
        "Reads current met/flow/tidal conditions from FORECAST_DATA_PATH, runs "
        "TijuanaRiverSources + H2SGenerationModel, and uploads a GeoDemic-compatible "
        "GridData JSON to S3 for consumption by Geodemic's "
        "/api/v1/tj-data/dispersion/river-emission-grid endpoint. "
        "Updated every 6 hours alongside the dispersion_forecast_job."
    ),
)
def river_emission_grid(context: dg.AssetExecutionContext) -> dg.MaterializeResult:
    log = context.log
    s3 = context.resources.s3

    # Load latest forecast met/flow/tidal conditions
    log.info(f"Loading forecast data from {FORECAST_DATA_PATH}")
    url = s3.get_presigned_url(FORECAST_DATA_PATH)
    fc_df = pd.read_parquet(url)
    fc_df["time"] = pd.to_datetime(fc_df["time"], utc=True).dt.tz_convert("America/Los_Angeles")
    log.info(f"Loaded {len(fc_df)} forecast rows; using latest row for current conditions")

    latest = fc_df.iloc[-1]
    wind_speed     = float(latest.get("wind_speed_10m",    3.0))
    wind_direction = float(latest.get("wind_direction_10m", 270.0))
    temperature    = float(latest.get("temperature_2m",    20.0))
    tide_height    = float(latest.get("tide_height",        0.5))
    # FLOW_COL = 'Flow (m^3/s)--Border'; convert m³/s → cfs (1 m³/s = 35.3147 cfs)
    raw_flow = latest.get(FLOW_COL, None)
    flow_rate_cfs = float(raw_flow) * 35.3147 if raw_flow is not None else 50.0

    log.info(
        f"Conditions: wind={wind_speed:.1f} m/s @ {wind_direction:.0f}°, "
        f"temp={temperature:.1f}°C, tide={tide_height:.2f} m, flow={flow_rate_cfs:.1f} cfs"
    )

    conditions = {
        "temperature":      temperature,
        "pH":               7.5,
        "dissolved_oxygen": 2.0,
        "tide_level":       tide_height,
        "flow_rate":        flow_rate_cfs,
        "wind_speed":       wind_speed,
        "wind_direction":   wind_direction,
    }

    # Generate gridded emission field (vectorized over 100 m grid)
    log.info("Running TijuanaRiverSources.generate_river_emission_grid()")
    sources = TijuanaRiverSources()
    emission_grid = sources.generate_river_emission_grid(
        wind_speed=wind_speed,
        wind_direction=wind_direction,
        flow_rate=flow_rate_cfs,
        tide_level=tide_height,
        temperature=temperature,
    )

    # Generate point-source emission list (for metadata / map markers)
    model = H2SGenerationModel()
    point_sources = model.get_source_emissions(conditions)

    grid_max  = float(np.max(emission_grid))
    grid_mean = float(np.mean(emission_grid[emission_grid > 0])) if np.any(emission_grid > 0) else 0.0
    log.info(f"Emission grid: max={grid_max:.1f} ppb, mean (nonzero)={grid_mean:.1f} ppb")

    # Serialize to GeoDemic GridData format
    grid_data = {
        "data":   emission_grid.tolist(),
        "bounds": GRID_BOUNDS,
        "nrows":  GRID_NROWS,
        "ncols":  GRID_NCOLS,
        "metadata": {
            "timestamp":      pd.Timestamp.utcnow().isoformat(),
            "wind_speed_ms":  wind_speed,
            "wind_direction": wind_direction,
            "temperature_c":  temperature,
            "flow_rate_cfs":  flow_rate_cfs,
            "tide_height_m":  tide_height,
            "model":          "physics_based_arrhenius",
            "point_sources": [
                {"lat": lat, "lon": lon, "ppb": round(ppb, 2)}
                for lat, lon, ppb in point_sources
            ],
        },
    }

    grid_json = json.dumps(grid_data)
    s3.putFile(grid_json.encode(), path=RIVER_EMISSION_GRID_LATEST_PATH, content_type="application/json")
    log.info(f"Uploaded river emission grid → {RIVER_EMISSION_GRID_LATEST_PATH}")

    return dg.MaterializeResult(metadata={
        "grid_max_ppb":    dg.MetadataValue.float(grid_max),
        "grid_mean_ppb":   dg.MetadataValue.float(grid_mean),
        "grid_shape":      dg.MetadataValue.text(f"{GRID_NROWS}x{GRID_NCOLS}"),
        "n_point_sources": dg.MetadataValue.int(len(point_sources)),
        "wind_speed_ms":   dg.MetadataValue.float(wind_speed),
        "wind_direction":  dg.MetadataValue.float(wind_direction),
        "flow_rate_cfs":   dg.MetadataValue.float(flow_rate_cfs),
        "s3_path":         dg.MetadataValue.text(RIVER_EMISSION_GRID_LATEST_PATH),
    })
