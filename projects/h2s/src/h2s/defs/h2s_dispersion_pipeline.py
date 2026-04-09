"""H2S Dispersion Modeling Pipeline — Backward Attribution & Forward Forecast.

Two operational modes:

BACKWARD (weekly dispersion_inversion_job):
  1. lagrangian_source_attribution  — backward particle model over inversion window
  2. emission_rate_inversion         — derive per-zone Q (g/s) from ensemble footprint
  3. hysplit_controls_generation     — generate HYSPLIT bundle, upload to S3 (no execution)

FORWARD (6h dispersion_forecast_job):
  1. emission_rate_inversion         — re-reads existing EMISSION_RATES_PATH from S3
  2. gaussian_forward_forecast       — 72h plume forecast using FORECAST_DATA_PATH met
  3. dispersion_alert_check          — threshold check, Slack alert
  4. hysplit_controls_generation     — forward CONTROL bundle, upload to S3 (no execution)

Key design decision: gaussian_forward_forecast loads forecast meteorology
(model_forecast.parquet via FORECAST_DATA_PATH) — NOT observation data.
This is the operational forecast use case.
"""

import io
import json
import zipfile

import dagster as dg
import numpy as np
import pandas as pd

from h2s.constants import (
    ALERT_TIERS,
    DISPERSION_DEFAULT_EMISSION_RATES_GS,
    DISPERSION_FORECAST_LATEST_PATH,
    DISPERSION_FORECAST_PATH,
    DISPERSION_FORECAST_DETAILED_PATH,
    DISPERSION_FORECAST_DETAILED_LATEST_PATH,
    DISPERSION_FORWARD_GRID_FRAMES_LATEST_PATH,
    DISPERSION_FORWARD_GRID_FRAMES_DETAILED_LATEST_PATH,
    DISPERSION_FORWARD_GRID_LATEST_PATH,
    DISPERSION_FORWARD_GRID_PATH,
    DISPERSION_FORWARD_GRID_DETAILED_PATH,
    DISPERSION_FORWARD_GRID_DETAILED_LATEST_PATH,
    DISPERSION_SOURCE_FOOTPRINT_GRID_LATEST_PATH,
    DISPERSION_VIZ_HEATMAP_COARSE_PATH,
    DISPERSION_VIZ_HEATMAP_DETAILED_PATH,
    DISPERSION_VIZ_SOURCE_MAP_COARSE_PATH,
    DISPERSION_VIZ_SOURCE_MAP_DETAILED_PATH,
    DISPERSION_VIZ_TIMESERIES_COARSE_PATH,
    DISPERSION_VIZ_TIMESERIES_DETAILED_PATH,
    EMISSION_RATES_PATH,
    FORECAST_DATA_PATH,
    HYSPLIT_BACKWARD_BUNDLE_LATEST,
    HYSPLIT_BACKWARD_BUNDLE_PATH,
    HYSPLIT_FORWARD_BUNDLE_LATEST,
    HYSPLIT_FORWARD_BUNDLE_PATH,
    LAGRANGIAN_ENSEMBLE_PATH,
    LAGRANGIAN_FOOTPRINT_PATH,
    OBS_DATA_PATH, LAGRANGIAN_FOOTPRINT_NAME,
)
from h2s.dispersion import (
    LagrangianConfig,
    generate_hysplit_bundle,
    run_forward_model,
    run_forward_model_gridded,
    run_forward_model_detailed,
    run_forward_model_gridded_detailed,
    footprint_to_grid_data,
    run_inversion_window,
    source_attribution,
)
from h2s.dispersion.visualizations import (
    generate_concentration_heatmap,
    generate_source_emission_map,
    generate_peak_concentration_timeseries,
)
from h2s.dispersion.gaussian import SENSORS, SOURCES, CANDIDATE_SOURCES
from h2s.dispersion.grid_config import (
    GRID_BOUNDS,
    GRID_LAT_CENTERS,
    GRID_LON_CENTERS,
    GRID_NROWS,
    GRID_NCOLS,
    GRID_RESOLUTION_METERS,
    VIZ_BOUNDS,
)
from h2s.utils import store_assets

# Zone groupings: candidate source names → east / west / south
_ZONE_MAP = {
    "east": [
        "stewarts_drain", "silva_drain", "tj_crossing_cdlp_w",
        "tj_crossing_cdlp_e", "dairy_mart_bridge", "del_sol_canyon",
    ],
    "west": [
        "oneonta_slough", "tijuana_beach_outlet", "hollister_ps",
        "sd_bay_otay_outlet", "sd_bay_fruitdale",
    ],
    "south": [
        "smugglers_gulch", "goat_canyon", "goat_canyon_ps",
        "hollister_bridge_n", "hollister_bridge_s", "saturn_blvd_bridge",
    ],
}

# Sum of calibrated defaults — used as the total Q budget when scaling zone fractions
_TOTAL_Q_GS = sum(DISPERSION_DEFAULT_EMISSION_RATES_GS.values())  # 167.0 g/s


# ==============================================================================
# Config classes
# ==============================================================================

class InversionConfig(dg.Config):
    date_start: str = "2026-02-01"
    date_end: str = "2026-04-01"
    h2s_threshold_ppb: float = 30.0
    n_particles: int = 2000
    hours_back: int = 2  # Valley-scale: sources are 1-7 km away, 37 min max travel time @ 3 m/s
    max_events: int = 0   # 0 = all events


class HysplitConfig(dg.Config):
    mode: str = "backward_traj"   # "backward_traj" | "backward_disp" | "forward_disp"
    met_dir: str = "/data/gdas"
    hours_back: int = 3  # Valley-scale trajectories: sources within 10 km
    h2s_threshold_ppb: float = 30.0
    date_start: str = "2026-02-01"
    date_end: str = "2026-04-01"


class ForwardForecastConfig(dg.Config):
    forecast_hours: int = 72


# ==============================================================================
# Asset 1: lagrangian_source_attribution
# ==============================================================================

@dg.asset(
    key_prefix="h2s",
    group_name="h2s_dispersion",
    required_resource_keys={"s3"},
    kinds={"python", "s3"},
    description=(
        "Backward Lagrangian particle model: source footprints + ensemble attribution "
        "over inversion window. Loads obs data from S3, uploads ensemble JSON and "
        "footprint array to S3."
    ),
    metadata={
        "source": "San Diego APCD H2S location data",
         "description": (
        "Backward Lagrangian particle model: source footprints + ensemble attribution "
        "over inversion window. Loads obs data from S3, uploads ensemble JSON and "
        "footprint array to S3."
    )
    },

)
def lagrangian_source_attribution(
    context: dg.AssetExecutionContext,
    config: InversionConfig,
) -> dg.MaterializeResult:
    log = context.log
    s3 = context.resources.s3
    meta = context.assets_def.metadata_by_key[context.asset_key]
    description = meta["description"]  # -> "value"
    metadata = store_assets.objectMetadata(name=str(context.asset_key.path[-1]),
                                           description=description,
                                           )

    log.info(f"Loading obs data from S3: {OBS_DATA_PATH}")
    url = s3.get_presigned_url(OBS_DATA_PATH)
    df = pd.read_parquet(url)
    df["time"] = pd.to_datetime(df["time"], utc=True).dt.tz_convert("America/Los_Angeles")
    log.info(f"Loaded {len(df)} obs rows")

    cfg = LagrangianConfig(
        n_particles=config.n_particles,
        hours_back=config.hours_back,
    )
    max_events = config.max_events if config.max_events > 0 else None

    log.info(
        f"Running backward Lagrangian: {config.date_start} → {config.date_end}, "
        f"H2S ≥ {config.h2s_threshold_ppb} ppb, {config.n_particles} particles/event"
    )
    results, ensemble_footprint = run_inversion_window(
        df=df,
        cfg=cfg,
        date_start=config.date_start,
        date_end=config.date_end,
        h2s_threshold=config.h2s_threshold_ppb,
        max_events=max_events,
    )
    n_events = len(results)
    log.info(f"Processed {n_events} events")

    if ensemble_footprint is None or n_events == 0:
        log.warning("No qualifying events found — skipping S3 upload")
        return dg.MaterializeResult(metadata={
            "n_events_processed": dg.MetadataValue.int(0),
            "warning": dg.MetadataValue.text("No qualifying events in window"),
        })

    # Compute ensemble attribution
    ensemble_attribution = source_attribution(ensemble_footprint)
    top_sources = list(ensemble_attribution.items())[:3]

    # Upload ensemble JSON to S3
    ensemble_payload = {
        "n_events": n_events,
        "date_range": f"{config.date_start} to {config.date_end}",
        "h2s_threshold_ppb": config.h2s_threshold_ppb,
        "ensemble_source_fractions": ensemble_attribution,
    }
    s3.putFile(
        json.dumps(ensemble_payload, indent=2).encode(),
        path=LAGRANGIAN_ENSEMBLE_PATH,
        content_type="application/json",
    )
    log.info(f"Uploaded ensemble JSON → {LAGRANGIAN_ENSEMBLE_PATH}")

    # Upload footprint as parquet (lat index, lon columns — inspectable as a heatmap table)
    # buf = io.BytesIO()
    # ensemble_footprint.to_parquet(buf)
    # s3.putFile(buf.getvalue(), path=LAGRANGIAN_FOOTPRINT_PATH, content_type="application/octet-stream")

    store_assets.store_dataframe_to_s3(ensemble_footprint.reset_index(), LAGRANGIAN_FOOTPRINT_PATH, LAGRANGIAN_FOOTPRINT_NAME, s3,
                                      latestdatasetpath=LAGRANGIAN_FOOTPRINT_PATH, enable_latest_path=True,
                                      formats=['csv', 'parquet','json'], metadata=metadata)
    log.info(f"Uploaded footprint parquet → {LAGRANGIAN_FOOTPRINT_PATH}")

    # Build zone lookup: source_name → zone (east/west/south)
    source_zone = {}
    for zone, sources in _ZONE_MAP.items():
        for s in sources:
            source_zone[s] = zone

    # --- GeoDemic-compatible grid output ---
    log.info("Resampling footprint to unified grid (GeoDemic GridData format)")
    footprint_grid = footprint_to_grid_data(
        ensemble_footprint,
        metadata={
            "n_events": n_events,
            "date_range": f"{config.date_start} to {config.date_end}",
            "h2s_threshold_ppb": config.h2s_threshold_ppb,
            "source_fractions": {k: round(v, 4) for k, v in ensemble_attribution.items()},
            "source_zones": source_zone,
        },
    )
    footprint_grid_json = json.dumps(footprint_grid)
    s3.putFile(
        footprint_grid_json.encode(),
        path=DISPERSION_SOURCE_FOOTPRINT_GRID_LATEST_PATH,
        content_type="application/json",
    )
    log.info(f"Uploaded footprint grid → {DISPERSION_SOURCE_FOOTPRINT_GRID_LATEST_PATH}")

    return dg.MaterializeResult(metadata={
        "n_events_processed": dg.MetadataValue.int(n_events),
        "top_source_1": dg.MetadataValue.text(top_sources[0][0] if len(top_sources) > 0 else "n/a"),
        "top_source_2": dg.MetadataValue.text(top_sources[1][0] if len(top_sources) > 1 else "n/a"),
        "top_source_3": dg.MetadataValue.text(top_sources[2][0] if len(top_sources) > 2 else "n/a"),
        "s3_ensemble": dg.MetadataValue.text(LAGRANGIAN_ENSEMBLE_PATH),
        "s3_footprint": dg.MetadataValue.text(LAGRANGIAN_FOOTPRINT_PATH),
        "s3_footprint_grid": dg.MetadataValue.text(DISPERSION_SOURCE_FOOTPRINT_GRID_LATEST_PATH),
    })


# ==============================================================================
# Asset 2: emission_rate_inversion
# ==============================================================================

@dg.asset(
    key_prefix="h2s",
    group_name="h2s_dispersion",
    required_resource_keys={"s3"},
    kinds={"python", "s3"},
    description=(
        "Derive per-zone AND per-source emission rates (g/s) from Lagrangian ensemble footprint. "
        "Groups candidate sources into east/west/south zones, scales to calibrated "
        "total Q. Falls back to DISPERSION_DEFAULT_EMISSION_RATES_GS if no ensemble "
        "is available."
    ),
    deps=[dg.AssetKey(["h2s", "lagrangian_source_attribution"])],
    metadata={
        #"source": "San Diego APCD H2S location data",
        "description": (
                "Derive per-zone AND per-source emission rates (g/s) from Lagrangian ensemble footprint. "
        "Groups candidate sources into east/west/south zones, scales to calibrated "
        "total Q. Falls back to DISPERSION_DEFAULT_EMISSION_RATES_GS if no ensemble "
        "is available."
        )
    },
)
def emission_rate_inversion(context: dg.AssetExecutionContext) -> dg.MaterializeResult:
    log = context.log
    s3 = context.resources.s3

    meta = context.assets_def.metadata_by_key[context.asset_key]
    description = meta["description"]  # -> "value"
    metadata = store_assets.objectMetadata(name=str(context.asset_key.path[-1]),
                                           description=description,
                                           )
    try:
        ensemble_bytes = s3.getFile(LAGRANGIAN_ENSEMBLE_PATH)
        ensemble = json.loads(ensemble_bytes)
        fracs = ensemble.get("ensemble_source_fractions", {})
        method = "lagrangian_ensemble_inversion"
        log.info(f"Loaded ensemble from {LAGRANGIAN_ENSEMBLE_PATH} ({len(fracs)} sources)")
    except Exception as e:
        log.warning(f"Could not load ensemble ({e}) — using calibrated defaults")
        fracs = {}
        method = "calibration_default"

    if fracs:
        # Per-zone rates (3 zones)
        zone_fracs: dict[str, float] = {}
        for zone, sources in _ZONE_MAP.items():
            zone_fracs[zone] = sum(fracs.get(s, 0.0) for s in sources)
        total = sum(zone_fracs.values()) or 1.0
        zone_fracs = {k: v / total for k, v in zone_fracs.items()}
        zone_rates = {zone: round(f * _TOTAL_Q_GS, 1) for zone, f in zone_fracs.items()}

        # Per-source rates (16 sources) — scale individual fractions by total Q
        source_total = sum(fracs.values()) or 1.0
        source_rates = {src: round((frac / source_total) * _TOTAL_Q_GS, 2) for src, frac in fracs.items()}

        log.info(f"Zone fractions: {zone_fracs}")
        log.info(f"Top 5 sources: {dict(list(sorted(source_rates.items(), key=lambda x: -x[1]))[:5])}")
    else:
        zone_rates = dict(DISPERSION_DEFAULT_EMISSION_RATES_GS)
        # Distribute zone defaults evenly across sources in each zone
        source_rates = {}
        for zone, sources in _ZONE_MAP.items():
            zone_q = zone_rates[zone]
            n_sources = len(sources)
            per_source_q = round(zone_q / n_sources, 2) if n_sources > 0 else 0.0
            for src in sources:
                source_rates[src] = per_source_q

    log.info(f"Zone emission rates: {zone_rates} g/s  (method={method})")

    payload = {
        "emission_rates_g_s": zone_rates,
        "emission_rates_per_source_g_s": source_rates,
        "timestamp": pd.Timestamp.utcnow().isoformat(),
        "method": method,
    }
    s3.putFile(
        json.dumps(payload, indent=2).encode(),
        path=EMISSION_RATES_PATH,
        content_type="application/json",
    )
    log.info(f"Uploaded emission rates → {EMISSION_RATES_PATH}")

    return dg.MaterializeResult(metadata={
        "east_g_s":  dg.MetadataValue.float(float(zone_rates["east"])),
        "west_g_s":  dg.MetadataValue.float(float(zone_rates["west"])),
        "south_g_s": dg.MetadataValue.float(float(zone_rates["south"])),
        "n_sources": dg.MetadataValue.int(len(source_rates)),
        "method":    dg.MetadataValue.text(method),
        "s3_path":   dg.MetadataValue.text(EMISSION_RATES_PATH),
    },
    value=json.dumps(payload, indent=2)
    )


# ==============================================================================
# Asset 3: hysplit_controls_generation
# ==============================================================================

@dg.asset(
    key_prefix="h2s",
    group_name="h2s_dispersion",
    required_resource_keys={"s3"},
    kinds={"python", "s3"},
    description=(
        "Generate HYSPLIT CONTROL file bundle (zip) and upload to S3. "
        "No HYSPLIT execution — download the bundle and run in a local container "
        "or submit to NOAA. Mode is set per-job via config."
    ),
    deps=[dg.AssetKey(["h2s", "emission_rate_inversion"])],
)
def hysplit_controls_generation(
    context: dg.AssetExecutionContext,
    config: HysplitConfig,
) -> dg.MaterializeResult:
    log = context.log
    s3 = context.resources.s3

    # Load emission rates
    try:
        rates_bytes = s3.getFile(EMISSION_RATES_PATH)
        rates_data = json.loads(rates_bytes)
        emission_rates = rates_data["emission_rates_g_s"]
    except Exception as e:
        log.warning(f"Could not load emission rates ({e}) — using calibrated defaults")
        emission_rates = dict(DISPERSION_DEFAULT_EMISSION_RATES_GS)

    run_tag = pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M")

    df = None
    if config.mode in ("backward_traj", "backward_disp"):
        log.info(f"Loading obs data for backward mode from {OBS_DATA_PATH}")
        url = s3.get_presigned_url(OBS_DATA_PATH)
        df = pd.read_parquet(url)
        df["time"] = pd.to_datetime(df["time"], utc=True).dt.tz_convert("America/Los_Angeles")

    start_utc = pd.Timestamp.utcnow().isoformat() if config.mode == "forward_disp" else None

    log.info(f"Generating HYSPLIT bundle: mode={config.mode}, met_dir={config.met_dir}")
    zip_bytes = generate_hysplit_bundle(
        mode=config.mode,
        df=df,
        met_dir=config.met_dir,
        emission_rates_g_s=emission_rates,
        start_utc=start_utc,
        hours_back=config.hours_back,
        h2s_threshold=config.h2s_threshold_ppb,
        date_start=config.date_start,
        date_end=config.date_end,
    )

    # Count CONTROL files in zip for metadata
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        n_control = sum(1 for n in zf.namelist() if "CONTROL" in n and not n.endswith(".sh"))

    # Upload versioned + latest
    if config.mode == "forward_disp":
        versioned_path = HYSPLIT_FORWARD_BUNDLE_PATH.format(run_tag=run_tag)
        latest_path = HYSPLIT_FORWARD_BUNDLE_LATEST
    else:
        versioned_path = HYSPLIT_BACKWARD_BUNDLE_PATH.format(run_tag=run_tag)
        latest_path = HYSPLIT_BACKWARD_BUNDLE_LATEST

    s3.putFile(zip_bytes, path=versioned_path, content_type="application/zip")
    s3.putFile(zip_bytes, path=latest_path, content_type="application/zip")
    log.info(f"Uploaded HYSPLIT bundle → {versioned_path} ({len(zip_bytes):,} bytes)")

    log.info(
        "HYSPLIT execution is NOT triggered. Download the bundle:\n"
        f"  s3://{latest_path}\n"
        "Then run in your HYSPLIT container:\n"
        "  unzip bundle.zip && bash run_hysplit_*.sh\n"
        "Or submit to NOAA READY server via email."
    )

    return dg.MaterializeResult(metadata={
        "mode":              dg.MetadataValue.text(config.mode),
        "n_control_files":   dg.MetadataValue.int(n_control),
        "zip_size_bytes":    dg.MetadataValue.int(len(zip_bytes)),
        "s3_versioned_path": dg.MetadataValue.text(versioned_path),
        "s3_latest_path":    dg.MetadataValue.text(latest_path),
    })


# ==============================================================================
# Asset 4: gaussian_forward_forecast
# ==============================================================================

@dg.asset(
    key_prefix="h2s",
    group_name="h2s_dispersion",
    required_resource_keys={"s3"},
    kinds={"python", "s3"},
    description=(
        "72h Gaussian plume forward forecast using FORECAST meteorology "
        "(FORECAST_DATA_PATH / model_forecast.parquet). Loads calibrated emission "
        "rates from S3. Outputs per-sensor ppb timeseries."
    ),
    deps=[dg.AssetKey(["h2s", "emission_rate_inversion"])],
    metadata={
        # "source": "San Diego APCD H2S location data",
        "description": (
                "72h Gaussian plume forward forecast using FORECAST meteorology "
        "(FORECAST_DATA_PATH / model_forecast.parquet). Loads calibrated emission "
        "rates from S3. Outputs per-sensor ppb timeseries."
        )
    },
)
def gaussian_forward_forecast(
    context: dg.AssetExecutionContext,
    config: ForwardForecastConfig,
) -> dg.MaterializeResult:
    log = context.log
    s3 = context.resources.s3

    # Load FORECAST meteorology (not obs data)
    log.info(f"Loading forecast met data from S3: {FORECAST_DATA_PATH}")
    url = s3.get_presigned_url(FORECAST_DATA_PATH)
    fc_df = pd.read_parquet(url)
    fc_df["time"] = pd.to_datetime(fc_df["time"], utc=True).dt.tz_convert("America/Los_Angeles")
    log.info(f"Loaded {len(fc_df)} forecast rows, time range: {fc_df['time'].min()} → {fc_df['time'].max()}")

    # Ensure is_night is present
    if "is_night" not in fc_df.columns:
        if "day_night" in fc_df.columns:
            fc_df["is_night"] = (fc_df["day_night"] == "night").astype(int)
        else:
            utc_h = fc_df["time"].dt.hour
            fc_df["is_night"] = ((utc_h < 6) | (utc_h >= 20)).astype(int)
            log.warning("is_night derived from hour (UTC < 6 or >= 20) — no day_night column found")

    # Load emission rates
    try:
        rates_bytes = s3.getFile(EMISSION_RATES_PATH)
        rates_data = json.loads(rates_bytes)
        emission_rates = rates_data["emission_rates_g_s"]
        rates_method = rates_data.get("method", "unknown")
        log.info(f"Using inverted emission rates: {emission_rates} g/s (method={rates_method})")
    except Exception as e:
        emission_rates = dict(DISPERSION_DEFAULT_EMISSION_RATES_GS)
        log.warning(f"Could not load emission rates ({e}) — using calibrated defaults: {emission_rates} g/s")

    start_time = fc_df["time"].min()
    log.info(f"Running Gaussian forward: start={start_time}, hours={config.forecast_hours}")
    result = run_forward_model(fc_df, emission_rates, start_time, config.forecast_hours)

    # Compute per-sensor peaks (ignoring NaN)
    peaks = {}
    for sensor, vals in result.concentrations.items():
        valid = [v for v in vals if v is not None and not np.isnan(v)]
        peaks[sensor] = round(max(valid, default=0.0), 1)

    run_tag = pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M")
    forecast_json = result.to_json()

    versioned_path = DISPERSION_FORECAST_PATH.format(run_tag=run_tag)
    s3.putFile(forecast_json.encode(), path=versioned_path, content_type="application/json")
    s3.putFile(forecast_json.encode(), path=DISPERSION_FORECAST_LATEST_PATH, content_type="application/json")
    log.info(f"Uploaded sensor forecast → {versioned_path}")

    # --- GeoDemic-compatible grid output ---
    log.info("Generating gridded forward forecast (GeoDemic GridData format)")
    grid_frames = run_forward_model_gridded(
        fc_df, emission_rates, start_time, config.forecast_hours,
    )

    # Upload current-hour grid (first frame)
    if grid_frames:
        first_frame_json = json.dumps(grid_frames[0])
        s3.putFile(first_frame_json.encode(), path=DISPERSION_FORWARD_GRID_LATEST_PATH, content_type="application/json")
        grid_versioned = DISPERSION_FORWARD_GRID_PATH.format(run_tag=run_tag)
        s3.putFile(first_frame_json.encode(), path=grid_versioned, content_type="application/json")
        log.info(f"Uploaded grid (current hour) → {DISPERSION_FORWARD_GRID_LATEST_PATH}")

    # Upload multi-frame (all hours) for animation — select every 6th hour to keep size manageable
    frame_indices = list(range(0, len(grid_frames), 6))
    if frame_indices[-1] != len(grid_frames) - 1:
        frame_indices.append(len(grid_frames) - 1)
    animation_frames = [grid_frames[i] for i in frame_indices]
    animation_payload = {
        "forecast_start": str(start_time),
        "n_frames": len(animation_frames),
        "frame_interval_hours": 6,
        "emission_rates_g_s": {k: float(v) for k, v in emission_rates.items()},
        "frames": animation_frames,
    }
    frames_json = json.dumps(animation_payload)
    s3.putFile(frames_json.encode(), path=DISPERSION_FORWARD_GRID_FRAMES_LATEST_PATH, content_type="application/json")
    log.info(f"Uploaded grid frames ({len(animation_frames)} frames) → {DISPERSION_FORWARD_GRID_FRAMES_LATEST_PATH}")

    # Grid peak (over all frames)
    grid_peak_ppb = max(np.array(f["data"]).max() for f in grid_frames) if grid_frames else 0.0

    # --- Generate visualizations ---
    date_str = pd.Timestamp.utcnow().strftime("%Y%m%d_%H")
    log.info("Generating visualizations (heatmap + source map + timeseries)")

    # 1. Concentration heatmap (current hour)
    heatmap_path = "n/a"
    if grid_frames:
        heatmap_buf = generate_concentration_heatmap(
            grid_frames[0],
            title="H2S Concentration Forecast (3-source model)",
            vmax=100.0,
            bounds=VIZ_BOUNDS,
        )
        heatmap_path = DISPERSION_VIZ_HEATMAP_COARSE_PATH.format(date_str=date_str)
        s3.putFile(heatmap_buf.getvalue(), path=heatmap_path, content_type="image/png")
        log.info(f"Uploaded heatmap → {heatmap_path}")

    # 2. Source emission map (3 zones)
    source_map_buf = generate_source_emission_map(
        SOURCES,
        emission_rates,
        sensors=SENSORS,
        title="H2S Source Emission Rates (3-zone model)",
        bounds=VIZ_BOUNDS,
    )
    source_map_path = DISPERSION_VIZ_SOURCE_MAP_COARSE_PATH.format(date_str=date_str)
    s3.putFile(source_map_buf.getvalue(), path=source_map_path, content_type="image/png")
    log.info(f"Uploaded source map (3-zone) → {source_map_path}")

    # 3. Peak concentration timeseries
    timeseries_buf = generate_peak_concentration_timeseries(
        result,
        title="Peak H2S Forecast (3-source model)",
    )
    timeseries_path = DISPERSION_VIZ_TIMESERIES_COARSE_PATH.format(date_str=date_str)
    s3.putFile(timeseries_buf.getvalue(), path=timeseries_path, content_type="image/png")
    log.info(f"Uploaded timeseries → {timeseries_path}")

    return dg.MaterializeResult(metadata={
        "forecast_start":    dg.MetadataValue.text(str(start_time)),
        "forecast_hours":    dg.MetadataValue.int(config.forecast_hours),
        "peak_ppb_NB":       dg.MetadataValue.float(float(peaks.get("NESTOR - BES", 0.0))),
        "peak_ppb_IB":       dg.MetadataValue.float(float(peaks.get("IB CIVIC CTR", 0.0))),
        "peak_ppb_SY":       dg.MetadataValue.float(float(peaks.get("SAN YSIDRO", 0.0))),
        "grid_peak_ppb":     dg.MetadataValue.float(float(grid_peak_ppb)),
        "grid_shape":        dg.MetadataValue.text(f"{GRID_NROWS}x{GRID_NCOLS}"),
        "grid_n_frames":     dg.MetadataValue.int(len(animation_frames)),
        "emission_rates_g_s": dg.MetadataValue.json({k: float(v) for k, v in emission_rates.items()}),
        "s3_path":           dg.MetadataValue.text(versioned_path),
        "s3_grid_latest":    dg.MetadataValue.text(DISPERSION_FORWARD_GRID_LATEST_PATH),
        "viz_heatmap":       dg.MetadataValue.text(heatmap_path),
        "viz_source_map":    dg.MetadataValue.text(source_map_path),
        "viz_timeseries":    dg.MetadataValue.text(timeseries_path),
    })


# ==============================================================================
# Asset 5: gaussian_forward_forecast_detailed
# ==============================================================================

@dg.asset(
    key_prefix="h2s",
    group_name="h2s_dispersion",
    required_resource_keys={"s3"},
    kinds={"python", "s3"},
    description=(
        "72h Gaussian plume forward forecast using 16 INDIVIDUAL candidate sources "
        "(vs 3-zone aggregate model). Provides spatially accurate concentration fields "
        "for regional hazard mapping. ~5× slower than coarse model."
    ),
    deps=[dg.AssetKey(["h2s", "emission_rate_inversion"])],
    metadata={
        "description": (
            "72h Gaussian plume forward forecast using 16 INDIVIDUAL candidate sources "
            "(vs 3-zone aggregate model). Provides spatially accurate concentration fields "
            "for regional hazard mapping. ~5× slower than coarse model."
        )
    },
)
def gaussian_forward_forecast_detailed(
    context: dg.AssetExecutionContext,
    config: ForwardForecastConfig,
) -> dg.MaterializeResult:
    log = context.log
    s3 = context.resources.s3

    # Load FORECAST meteorology (not obs data)
    log.info(f"Loading forecast met data from S3: {FORECAST_DATA_PATH}")
    url = s3.get_presigned_url(FORECAST_DATA_PATH)
    fc_df = pd.read_parquet(url)
    fc_df["time"] = pd.to_datetime(fc_df["time"], utc=True).dt.tz_convert("America/Los_Angeles")
    log.info(f"Loaded {len(fc_df)} forecast rows, time range: {fc_df['time'].min()} → {fc_df['time'].max()}")

    # Ensure is_night is present
    if "is_night" not in fc_df.columns:
        if "day_night" in fc_df.columns:
            fc_df["is_night"] = (fc_df["day_night"] == "night").astype(int)
        else:
            utc_h = fc_df["time"].dt.hour
            fc_df["is_night"] = ((utc_h < 6) | (utc_h >= 20)).astype(int)
            log.warning("is_night derived from hour (UTC < 6 or >= 20) — no day_night column found")

    # Load per-source emission rates
    try:
        rates_bytes = s3.getFile(EMISSION_RATES_PATH)
        rates_data = json.loads(rates_bytes)
        emission_rates_per_source = rates_data.get("emission_rates_per_source_g_s", {})
        rates_method = rates_data.get("method", "unknown")

        if not emission_rates_per_source:
            log.warning("No per-source rates in emission_rates.json — falling back to zone-based defaults")
            # Distribute zone rates evenly across sources
            zone_rates = rates_data.get("emission_rates_g_s", DISPERSION_DEFAULT_EMISSION_RATES_GS)
            emission_rates_per_source = {}
            for zone, sources in _ZONE_MAP.items():
                zone_q = zone_rates.get(zone, 0.0)
                per_source = zone_q / len(sources) if len(sources) > 0 else 0.0
                for src in sources:
                    emission_rates_per_source[src] = round(per_source, 2)

        log.info(f"Using {len(emission_rates_per_source)} source emission rates (method={rates_method})")
        top_5 = dict(sorted(emission_rates_per_source.items(), key=lambda x: -x[1])[:5])
        log.info(f"Top 5 sources: {top_5}")
    except Exception as e:
        log.warning(f"Could not load emission rates ({e}) — distributing defaults across 16 sources")
        emission_rates_per_source = {}
        for zone, sources in _ZONE_MAP.items():
            zone_q = DISPERSION_DEFAULT_EMISSION_RATES_GS[zone]
            per_source = zone_q / len(sources) if len(sources) > 0 else 0.0
            for src in sources:
                emission_rates_per_source[src] = round(per_source, 2)

    start_time = fc_df["time"].min()
    log.info(f"Running detailed Gaussian forward (16 sources): start={start_time}, hours={config.forecast_hours}")
    result = run_forward_model_detailed(fc_df, emission_rates_per_source, start_time, config.forecast_hours)

    # Compute per-sensor peaks (ignoring NaN)
    peaks = {}
    for sensor, vals in result.concentrations.items():
        valid = [v for v in vals if v is not None and not np.isnan(v)]
        peaks[sensor] = round(max(valid, default=0.0), 1)

    run_tag = pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M")
    forecast_json = result.to_json()

    versioned_path = DISPERSION_FORECAST_DETAILED_PATH.format(run_tag=run_tag)
    s3.putFile(forecast_json.encode(), path=versioned_path, content_type="application/json")
    s3.putFile(forecast_json.encode(), path=DISPERSION_FORECAST_DETAILED_LATEST_PATH, content_type="application/json")
    log.info(f"Uploaded detailed sensor forecast → {versioned_path}")

    # --- GeoDemic-compatible grid output (16-source version) ---
    log.info("Generating detailed gridded forward forecast (16 sources, GeoDemic GridData format)")
    grid_frames = run_forward_model_gridded_detailed(
        fc_df, emission_rates_per_source, start_time, config.forecast_hours,
    )

    # Upload current-hour grid (first frame)
    if grid_frames:
        first_frame_json = json.dumps(grid_frames[0])
        s3.putFile(first_frame_json.encode(), path=DISPERSION_FORWARD_GRID_DETAILED_LATEST_PATH, content_type="application/json")
        grid_versioned = DISPERSION_FORWARD_GRID_DETAILED_PATH.format(run_tag=run_tag)
        s3.putFile(first_frame_json.encode(), path=grid_versioned, content_type="application/json")
        log.info(f"Uploaded detailed grid (current hour) → {DISPERSION_FORWARD_GRID_DETAILED_LATEST_PATH}")

    # Upload multi-frame (all hours) for animation — select every 6th hour to keep size manageable
    frame_indices = list(range(0, len(grid_frames), 6))
    if frame_indices[-1] != len(grid_frames) - 1:
        frame_indices.append(len(grid_frames) - 1)
    animation_frames = [grid_frames[i] for i in frame_indices]
    animation_payload = {
        "forecast_start": str(start_time),
        "n_frames": len(animation_frames),
        "frame_interval_hours": 6,
        "n_sources": 16,
        "emission_rates_per_source_g_s": {k: float(v) for k, v in emission_rates_per_source.items() if v > 0},
        "frames": animation_frames,
    }
    frames_json = json.dumps(animation_payload)
    s3.putFile(frames_json.encode(), path=DISPERSION_FORWARD_GRID_FRAMES_DETAILED_LATEST_PATH, content_type="application/json")
    log.info(f"Uploaded detailed grid frames ({len(animation_frames)} frames) → {DISPERSION_FORWARD_GRID_FRAMES_DETAILED_LATEST_PATH}")

    # Grid peak (over all frames)
    grid_peak_ppb = max(np.array(f["data"]).max() for f in grid_frames) if grid_frames else 0.0

    # --- Generate visualizations ---
    date_str = pd.Timestamp.utcnow().strftime("%Y%m%d_%H")
    log.info("Generating detailed visualizations (heatmap + timeseries)")

    # 1. Concentration heatmap (current hour)
    heatmap_path = "n/a"
    if grid_frames:
        heatmap_buf = generate_concentration_heatmap(
            grid_frames[0],
            title="H2S Concentration Forecast (16-source detailed model)",
            vmax=100.0,
            bounds=VIZ_BOUNDS,
        )
        heatmap_path = DISPERSION_VIZ_HEATMAP_DETAILED_PATH.format(date_str=date_str)
        s3.putFile(heatmap_buf.getvalue(), path=heatmap_path, content_type="image/png")
        log.info(f"Uploaded detailed heatmap → {heatmap_path}")

    # 2. Source emission map (16 candidate sources)
    source_map_buf = generate_source_emission_map(
        CANDIDATE_SOURCES,
        emission_rates_per_source,
        sensors=SENSORS,
        title="H2S Source Emission Rates (16-source detailed model)",
        bounds=VIZ_BOUNDS,
    )
    source_map_path = DISPERSION_VIZ_SOURCE_MAP_DETAILED_PATH.format(date_str=date_str)
    s3.putFile(source_map_buf.getvalue(), path=source_map_path, content_type="image/png")
    log.info(f"Uploaded source map (16-source) → {source_map_path}")

    # 3. Peak concentration timeseries
    timeseries_buf = generate_peak_concentration_timeseries(
        result,
        title="Peak H2S Forecast (16-source detailed model)",
    )
    timeseries_path = DISPERSION_VIZ_TIMESERIES_DETAILED_PATH.format(date_str=date_str)
    s3.putFile(timeseries_buf.getvalue(), path=timeseries_path, content_type="image/png")
    log.info(f"Uploaded detailed timeseries → {timeseries_path}")

    return dg.MaterializeResult(metadata={
        "forecast_start":    dg.MetadataValue.text(str(start_time)),
        "forecast_hours":    dg.MetadataValue.int(config.forecast_hours),
        "n_sources":         dg.MetadataValue.int(16),
        "peak_ppb_NB":       dg.MetadataValue.float(float(peaks.get("NESTOR - BES", 0.0))),
        "peak_ppb_IB":       dg.MetadataValue.float(float(peaks.get("IB CIVIC CTR", 0.0))),
        "peak_ppb_SY":       dg.MetadataValue.float(float(peaks.get("SAN YSIDRO", 0.0))),
        "grid_peak_ppb":     dg.MetadataValue.float(float(grid_peak_ppb)),
        "grid_shape":        dg.MetadataValue.text(f"{GRID_NROWS}x{GRID_NCOLS}"),
        "grid_n_frames":     dg.MetadataValue.int(len(animation_frames)),
        "s3_path":           dg.MetadataValue.text(versioned_path),
        "s3_grid_latest":    dg.MetadataValue.text(DISPERSION_FORWARD_GRID_DETAILED_LATEST_PATH),
        "viz_heatmap":       dg.MetadataValue.text(heatmap_path),
        "viz_source_map":    dg.MetadataValue.text(source_map_path),
        "viz_timeseries":    dg.MetadataValue.text(timeseries_path),
    })


# ==============================================================================
# Asset 6: dispersion_alert_check
# ==============================================================================

@dg.asset(
    key_prefix="h2s",
    group_name="h2s_dispersion",
    required_resource_keys={"s3", "slack"},
    kinds={"python", "s3", "slack"},
    description=(
        "Check next 6h of Gaussian forward forecast against WATCH (30 ppb) and "
        "CRITICAL (100 ppb) thresholds. Sends Slack alert via SlackAlertResource "
        "if any threshold is crossed."
    ),
    deps=[dg.AssetKey(["h2s", "gaussian_forward_forecast"])],
)
def dispersion_alert_check(context: dg.AssetExecutionContext) -> dg.MaterializeResult:
    log = context.log
    s3 = context.resources.s3

    try:
        forecast_bytes = s3.getFile(DISPERSION_FORECAST_LATEST_PATH)
        forecast = json.loads(forecast_bytes)
    except Exception as e:
        log.warning(f"Could not load forecast for alert check: {e}")
        return dg.MaterializeResult(metadata={"alert": dg.MetadataValue.text("no_forecast")})

    watch_ppb    = ALERT_TIERS["watch"]["threshold"]
    critical_ppb = ALERT_TIERS["critical"]["threshold"]
    lookahead_h  = 6
    alerts_triggered = []

    for sensor_name, series in forecast.get("timeseries", {}).items():
        for entry in series[:lookahead_h]:
            ppb = entry.get("predicted_ppb") or 0
            if ppb >= critical_ppb:
                alerts_triggered.append({"tier": "CRITICAL", "sensor": sensor_name,
                                          "time": entry["time"], "predicted_ppb": ppb})
            elif ppb >= watch_ppb:
                alerts_triggered.append({"tier": "WATCH", "sensor": sensor_name,
                                          "time": entry["time"], "predicted_ppb": ppb})

    log.info(f"Dispersion alert check: {len(alerts_triggered)} threshold crossings in next {lookahead_h}h")

    if alerts_triggered:
        max_tier = "CRITICAL" if any(a["tier"] == "CRITICAL" for a in alerts_triggered) else "WATCH"
        tier_label = ALERT_TIERS["critical"]["label"] if max_tier == "CRITICAL" else ALERT_TIERS["watch"]["label"]
        emission_rates = forecast.get("emission_rates_g_s", {})

        lines = [f"• {a['sensor']}: {a['predicted_ppb']:.1f} ppb @ {a['time']}" for a in alerts_triggered[:5]]
        msg = (
            f":warning: *Dispersion model alert — {tier_label}*\n"
            f"Gaussian plume forward model predicts elevated H₂S in next {lookahead_h}h:\n"
            + "\n".join(lines)
            + f"\n_Emission rates: {emission_rates} g/s_"
        )

        try:
            slack = context.resources.slack
            client = slack.get_client()
            client.chat_postMessage(channel=slack.channel, text=msg)
            log.info(f"Slack alert sent: {tier_label}")
        except Exception as e:
            log.error(f"Slack send failed: {e}")

    return dg.MaterializeResult(metadata={
        "alert_count":      dg.MetadataValue.int(len(alerts_triggered)),
        "max_tier":         dg.MetadataValue.text(alerts_triggered[0]["tier"] if alerts_triggered else "none"),
        "sensors_affected": dg.MetadataValue.text(
            ", ".join({a["sensor"] for a in alerts_triggered}) if alerts_triggered else "none"
        ),
        "lookahead_hours":  dg.MetadataValue.int(lookahead_h),
    })


# ==============================================================================
# Exported asset list (for Definitions)
# ==============================================================================

dispersion_assets = [
    lagrangian_source_attribution,
    emission_rate_inversion,
    hysplit_controls_generation,
    gaussian_forward_forecast,
    gaussian_forward_forecast_detailed,
    dispersion_alert_check,
]
