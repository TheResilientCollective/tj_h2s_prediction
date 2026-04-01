"""Shared constants for H2S prediction pipeline.

S3 paths, station/source geography, hazard classification,
and model feature lists — single source of truth.
"""

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

# Multi-horizon forecast paths
MH_MODELS_S3_BASE = 'tijuana/forecast/models/multihorizon'
MH_OUTPUT_PATH = 'tijuana/forecast/multihorizon'

# Pre-featurized data paths
OBS_DATA_PATH = 'latest/tijuana/forecast_data/modeldata_h2s_nofill.parquet'
FORECAST_DATA_PATH = 'latest/tijuana/forecast_data/model_forecast.parquet'

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

H2S_THRESHOLD_LOW = 5    # ppb — green / yellow_low boundary
H2S_THRESHOLD_MED = 10   # ppb — yellow_low / yellow_high boundary
H2S_THRESHOLD_HIGH = 30  # ppb — yellow_high / orange boundary

PROB_5_CAUTION = 0.25
PROB_5_ALERT = 0.5
PROB_10_ALERT = 0.5


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
