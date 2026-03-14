"""Constants for the H2S monitoring dashboard."""

# Data URLs (public S3 bucket, no auth needed)
H2S_DATA_URL = (
    "https://oss.resilientservice.mooo.com/resilentpublic/"
    "latest/tijuana/forecast_data/modeldata_h2s_nofill.parquet"
)
LOCATIONS_URL = (
    "https://oss.resilientservice.mooo.com/resilentpublic/"
    "latest/tijuana/forecast_data/h2s_locations.csv"
)
COMPLAINTS_URL = (
    "https://oss.resilientservice.mooo.com/resilentpublic/"
    "latest/tijuana/sd_complaints/complaints.csv"
)

# H2S thresholds (ppb) per CAAQS standard
H2S_GREEN_MAX = 5
H2S_YELLOW_MAX = 30

# Colors matching the prediction system categories
COLOR_GREEN = "#2ca02c"
COLOR_YELLOW = "#FFC107"
COLOR_ORANGE = "#FF5722"
COLOR_GRAY = "#999999"

CATEGORY_COLORS = {
    "green": COLOR_GREEN,
    "yellow": COLOR_YELLOW,
    "orange": COLOR_ORANGE,
}

# Sites expected in the data
SITES = ["SAN YSIDRO", "NESTOR - BES", "IB CIVIC CTR"]
