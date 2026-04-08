"""Shared constants for H2S prediction pipeline.


S3 paths, station/source geography, hazard classification,
and model feature lists — single source of truth.
"""
# forecast schedule constant. When a forecast is updated (aka for when noaa hysplit files updated.
SCHEDULE_6HR="0 */6 * * *"
# ==============================================================================
# S3 Path Constants
# ==============================================================================

MODEL_PATH = 'tijuana/forecast/models'
TRAINING_PATH = 'tijuana/forecast/models/training'
ARCHIVE_PATH = 'tijuana/forecast/models/archive'
PREDICTIONS_PATH = 'tijuana/forecast/predictions'  # Legacy - use HOURLY_PREDICTIONS_PATH
HOURLY_PREDICTIONS_PATH = 'tijuana/forecast/hourly'  # Hive-partitioned base for pyarrow
OUTPUT_PATH = 'tijuana/forecast/output'  # Legacy - use DAILY_SUMMARY_PATH
DAILY_SUMMARY_PATH = 'tijuana/forecast/daily_summary'
VISUALIZATIONS_PATH = 'tijuana/forecast/visualizations'

LATEST_BASEPATH = 'latest/tijuana'
LATEST_FORECAST = 'tijuana/forecast'
VALIDATION_PATH = 'tijuana/forecast/validation'

STATION_MODELS_S3_BASE = 'tijuana/forecast/models/stations'
TRAINING_SNAPSHOTS_PATH = 'tijuana/forecast/training_snapshots'

# Multi-horizon forecast paths
MH_MODELS_S3_BASE = 'tijuana/forecast/models/multihorizon'
MH_OUTPUT_PATH = 'tijuana/forecast/multihorizon'

# Pre-featurized data paths
OBS_DATA_PATH = 'latest/tijuana/forecast_data/modeldata_h2s_nofill.parquet'
FORECAST_DATA_PATH = 'latest/tijuana/forecast_data/model_forecast.parquet'

# Canonical class ordering (matches XGBoost LabelEncoder: alphabetical)
H2S_CLASS_NAMES = ['green', 'orange', 'yellow']
H2S_CLASS_TO_INT = {'green': 0, 'orange': 1, 'yellow': 2}

# ==============================================================================
# Station & Source Geography
# ==============================================================================

STATIONS = {
    'SAN YSIDRO': {
        'key': 'SAN_YSIDRO', 'partition': 'san_ysidro',
        'lat': 32.552794, 'lon': -117.047286,
        'color': '#e74c3c', 'short': 'SY',
    },
    'NESTOR - BES': {
        'key': 'NESTOR__BES', 'partition': 'nestor_bes',
        'lat': 32.567097, 'lon': -117.090656,
        'color': '#2ecc71', 'short': 'NB',
    },
    'IB CIVIC CTR': {
        'key': 'IB_CIVIC_CTR', 'partition': 'ib_civic_ctr',
        'lat': 32.576139, 'lon': -117.115361,
        'color': '#3498db', 'short': 'IB',
    },
}

STATION_PARTITION_MAP = {v['partition']: k for k, v in STATIONS.items()}
STATION_KEYS = {k: v['key'] for k, v in STATIONS.items()}

SOURCES = {
    "Stewart's Drain":  {'lat': 32.54064,  'lon': -117.05801,  'color': '#ff4444'},
    "Smuggler's Gulch": {'lat': 32.5377,   'lon': -117.08623,  'color': '#ffaa00'},
    "Hollister St PS":  {'lat': 32.5476,   'lon': -117.088374, 'color': '#ff6600'},
    "Goat Canyon":      {'lat': 32.5369,   'lon': -117.09916,  'color': '#cc44cc'},
    "Goat Canyon PS":   {'lat': 32.543476, 'lon': -117.108026, 'color': '#aa44aa'},
    "Del Sol Canyon":   {'lat': 32.5393,   'lon': -117.06885,  'color': '#44aacc'},
    "Silva Drain":      {'lat': 32.539743, 'lon': -117.064269, 'color': '#88cc44'},
    "Saturn Blvd Bridge": {'lat': 32.559383,'lon': -117.092992,  'color': '#4488cc'},
    "Hollister St Bridge N":     {'lat': 32.554177,   'lon': -117.084135, 'color': '#ff6600'},
    "Hollister St Bridge S":     {'lat': 32.551466,   'lon': -117.084021,  'color': '#ff6600'},
    "Dairy Mart Bridge": {'lat': 32.548531,   'lon':  -117.064293,  'color': '#ff6600'},
    "Oneonta Slough Near IB": {'lat': 32.570082,  'lon': -117.126724, 'color': '#0000ff'},
    "Tijuana River Beach Outlet":    {'lat': 32.556206,   'lon': -117.126178,  'color': '#0000ff'},
    "Tijuana River Crossing Camino De La Plaza W":      {'lat': 32.542103,   'lon': -117.054117,   'color': '#0000ff'},
    "Tijuana River Crossing Camino De La Plaza E": {'lat': 32.542166,  'lon': -117.050325, 'color': '#0000ff'},
    "San Diego Bay ponds Otay River Outlet": {'lat': 32.594557,    'lon': -117.113542,  'color': '#0000ff'},
    "San Diego Bay Ponds near Fruitdale": {'lat': 32.595305,    'lon': -117.091869,  'color': '#0000ff'},
}

# ==============================================================================
# Column Name Constants
# ==============================================================================

FLOW_COL = 'Flow (m^3/s)--Border'
WIND_COL = 'wind_direction_10m'
SPEED_COL = 'wind_speed_10m'
ALIGNMENT_THRESHOLD_DEG = 30


# ==============================================================================
# Hazard Classification (SD County H2S Guidance)
# ==============================================================================

RISK_GREEN = 'GREEN'
RISK_YELLOW_LOW = 'YELLOW_LOW'
RISK_YELLOW_HIGH = 'YELLOW_HIGH'
RISK_ORANGE = 'ORANGE'

# Minimum H2S (ppb) for an observation to contribute to the source probability grid.
# Higher values focus the map on significant events; lower values increase coverage but
# risk contamination from background readings. Tuned in h2s_daily_pipeline._compute_source_probability_grid.
H2S_SOURCE_THRESHOLD = 10
H2S_THRESHOLD_EXTREME = 100  # ppb — extreme event trigger

H2S_THRESHOLD_LOW = 5    # ppb — green / yellow_low boundary
H2S_THRESHOLD_MED = 10   # ppb — yellow_low / yellow_high boundary
H2S_THRESHOLD_HIGH = 30  # ppb — yellow_high / orange boundary

PROB_5_CAUTION = 0.25
PROB_5_ALERT = 0.5
PROB_10_ALERT = 0.5



# S3 path for extreme event summaries
EXTREME_EVENT_PATH = 'tijuana/forecast/extreme_events'

# ==============================================================================
# Two-Tier H2S Alert System
# ==============================================================================

ALERT_TIERS = {
    "watch": {
        "label":     "WATCH",
        "threshold": 30.0,
        "audience":  "Monitoring staff",
    },
    "critical": {
        "label":     "CRITICAL",
        "threshold": 100.0,
        "audience":  "Agency decision-makers",
    },
}

ALERT_SITE_NAME          = "NESTOR"
ALERT_QUIET_HOURS        = 3      # hours below threshold before new event can open
ALERT_CLOSE_WAIT_HOURS   = 1.5    # hours after last exceedance before summary fires
ALERT_LOCAL_TZ           = "America/Los_Angeles"
ALERT_SBIWTP_BASELINE_MGD = 23.5  # long-run median flow — used for deficit display

ALERT_STATE_S3_PATH       = "tijuana/forecast/alerts/h2s_alert_state.json"
ALERT_SUMMARY_ARCHIVE_PATH = EXTREME_EVENT_PATH
ALERT_SUMMARY_LATEST_PATH  = f"{LATEST_BASEPATH}/forecast_data/extreme_event_summary.json"


def classify_risk(prob_5: float, prob_10: float, h2s_pred: float) -> str:
    """Assign risk tier from predictions (SD County guidance).

    GREEN:       H2S < 5 ppb
    YELLOW_LOW:  5 <= H2S < 10 ppb
    YELLOW_HIGH: 10 <= H2S < 30 ppb
    ORANGE:      H2S >= 30 ppb
    """
    if prob_10 > PROB_10_ALERT or h2s_pred > H2S_THRESHOLD_HIGH:
        return RISK_ORANGE
    elif prob_5 > PROB_5_ALERT or h2s_pred > H2S_THRESHOLD_MED:
        return RISK_YELLOW_HIGH
    elif prob_5 > PROB_5_CAUTION or h2s_pred > H2S_THRESHOLD_LOW:
        return RISK_YELLOW_LOW
    return RISK_GREEN

# ==============================================================================
# Model Feature Lists
# ==============================================================================

# Core features (36) — available without SBIWTP feed
CORE_FEATURES = [
    'temperature_2m', 'wind_speed_10m', 'wind_direction_sin', 'wind_direction_cos',
    'wind_gusts_10m', 'precipitation', 'relative_humidity_2m', 'surface_pressure',
    'cloud_cover', 'dewpoint_2m',
    'wind_speed_10m_avg_2h', 'wind_speed_10m_avg_3h', 'wind_speed_10m_avg_4h',
    'wind_gusts_10m_max_2h', 'wind_gusts_10m_max_3h', 'wind_gusts_10m_max_4h',
    'tide_height', 'tidal_state_encoded',
    'hour_sin', 'hour_cos', 'month_sin', 'month_cos',
    'is_night', 'source_regime',
    'flow_log', 'flow_low', 'flow_high',
    'wind_temp_interaction', 'humidity_temp_interaction',
    'stable_atm',
    'h2s_lag_1h', 'h2s_lag_3h', 'h2s_lag_6h',
    'h2s_rolling_6h', 'h2s_rolling_24h',
    'flow_lag_6h', 'flow_rolling_24h',
]

# SBIWTP effluent features (available when USIBWC feed is connected)
SBIWTP_FEATURES = [
    'sbiwtp_flow_mgd', 'sbiwtp_anomaly', 'sbiwtp_deficit',
    'sbiwtp_flow_x_temp', 'sbiwtp_hourly_mgd', 'sbiwtp_sli',
]

# Full 43-feature set used by per-station models
MODEL_FEATURES = CORE_FEATURES + SBIWTP_FEATURES

# Alias for multihorizon compatibility
BASE_FEATURES = MODEL_FEATURES

# ==============================================================================
# Dispersion Modeling S3 Paths
# ==============================================================================

DISPERSION_BASE_PATH = 'tijuana/dispersion'

# Lagrangian inversion outputs (written weekly by dispersion_inversion_job)
LAGRANGIAN_ENSEMBLE_PATH = 'tijuana/dispersion/lagrangian/ensemble.json'
LAGRANGIAN_FOOTPRINT_PATH = 'tijuana/dispersion/lagrangian/'
LAGRANGIAN_FOOTPRINT_NAME='footprint_ensemble'
# filename within the .parquet archive
# Emission rate inversion result — weekly job writes, 6h forecast job reads
EMISSION_RATES_PATH = 'tijuana/dispersion/emission_rates.json'

# HYSPLIT control bundles (zip archives). Use .format(run_tag=run_tag) to expand.
HYSPLIT_BACKWARD_BUNDLE_PATH = 'tijuana/dispersion/hysplit/backward_bundle_{run_tag}.zip'
HYSPLIT_FORWARD_BUNDLE_PATH  = 'tijuana/dispersion/hysplit/forward_bundle_{run_tag}.zip'
HYSPLIT_BACKWARD_BUNDLE_LATEST = 'tijuana/dispersion/hysplit/backward_bundle_latest.zip'
HYSPLIT_FORWARD_BUNDLE_LATEST  = 'tijuana/dispersion/hysplit/forward_bundle_latest.zip'

# Gaussian forward forecast outputs. Use .format(run_tag=run_tag) to expand.
DISPERSION_FORECAST_PATH = 'tijuana/dispersion/forward_forecast_{run_tag}.json'
DISPERSION_FORECAST_LATEST_PATH = f'{LATEST_BASEPATH}/dispersion/forward_forecast_latest.json'

# Default emission rates (g/s) — calibrated from March 13 2026 event (394 ppb @ NESTOR-BES).
# east=20, west=10, south=137 g/s. Used as fallback when inversion has not yet run.
DISPERSION_DEFAULT_EMISSION_RATES_GS: dict[str, float] = {
    "east":  20.0,   # Stewart's Drain corridor
    "west":  10.0,   # Oneonta Slough / pump station
    "south": 137.0,  # Goat Canyon / cross-border (dominant nocturnal source)
}
