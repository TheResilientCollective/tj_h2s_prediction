"""Unified grid specification for tj_h2s dispersion and GeoDemic visualization.

Single source of truth for the spatial grid used by both the Gaussian forward
forecast and the Lagrangian footprint. GeoDemic's HeatMapLayer renders this
grid directly via its GridData interface.

resolution_meters matches GeoDemic's ``resolution`` API parameter (meters, not
cell count).
"""

import numpy as np

# ---------------------------------------------------------------------------
# Grid specification
# ---------------------------------------------------------------------------

GRID_BOUNDS = {
    "north": 32.70,
    "south": 32.45,
    "east": -117.00,
    "west": -117.25,
}

GRID_RESOLUTION_METERS = 100  # 100 m cell size

# Derived grid dimensions (computed once at import time)
_M_PER_DEG_LAT = 111_320.0
_M_PER_DEG_LON = 111_320.0 * np.cos(np.radians(0.5 * (GRID_BOUNDS["north"] + GRID_BOUNDS["south"])))

GRID_NROWS = int(round((GRID_BOUNDS["north"] - GRID_BOUNDS["south"]) * _M_PER_DEG_LAT / GRID_RESOLUTION_METERS))
GRID_NCOLS = int(round((GRID_BOUNDS["east"] - GRID_BOUNDS["west"]) * _M_PER_DEG_LON / GRID_RESOLUTION_METERS))

# Pre-compute cell center arrays (south→north rows, west→east columns)
GRID_LAT_CENTERS = np.linspace(
    GRID_BOUNDS["south"] + 0.5 * (GRID_BOUNDS["north"] - GRID_BOUNDS["south"]) / GRID_NROWS,
    GRID_BOUNDS["north"] - 0.5 * (GRID_BOUNDS["north"] - GRID_BOUNDS["south"]) / GRID_NROWS,
    GRID_NROWS,
)
GRID_LON_CENTERS = np.linspace(
    GRID_BOUNDS["west"] + 0.5 * (GRID_BOUNDS["east"] - GRID_BOUNDS["west"]) / GRID_NCOLS,
    GRID_BOUNDS["east"] - 0.5 * (GRID_BOUNDS["east"] - GRID_BOUNDS["west"]) / GRID_NCOLS,
    GRID_NCOLS,
)
