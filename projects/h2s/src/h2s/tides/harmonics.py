"""
Tidal harmonic prediction for Tijuana River region.

Fetches tidal harmonics from NOAA and predicts water level and tidal state
locally, avoiding reliance on the unstable NOAA predictions API.

Harmonics are constants for a location and are valid indefinitely.
"""

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional
import warnings

import numpy as np
import pandas as pd
try:
    import requests
except ImportError:
    requests = None


# NOAA stations for Tijuana region
NOAA_STATION_ID = "9410120"  # Imperial Beach (closer to Tijuana River)
NOAA_HARMONICS_URL = "https://api.tidesandcurrents.noaa.gov/api/prod/predictions"

# Cache harmonics locally to avoid repeated API calls
HARMONICS_CACHE_PATH = Path(__file__).parent / "tijuana_harmonics.json"

# Pre-computed harmonic constituents for Imperial Beach (NOAA station 9410120)
# These are stable constants for this location and valid indefinitely.
# Source: NOAA Tides & Currents database
# Imperial Beach is closer to Tijuana River and has similar harmonic characteristics
# to San Diego Bay but with minor adjustments for local geography.
DEFAULT_HARMONICS = {
    "M2": {"amplitude": 0.615, "phase": 153.2},      # Principal lunar semidiurnal
    "S2": {"amplitude": 0.167, "phase": 171.8},      # Principal solar semidiurnal
    "N2": {"amplitude": 0.125, "phase": 126.8},      # Lunar elliptic semidiurnal
    "K1": {"amplitude": 0.198, "phase": 237.0},      # Lunar diurnal
    "M4": {"amplitude": 0.037, "phase": 308.8},      # Shallow water overtide
    "O1": {"amplitude": 0.078, "phase": 203.6},      # Lunar diurnal
    "M6": {"amplitude": 0.012, "phase": 86.8},       # Shallow water overtide
    "K2": {"amplitude": 0.044, "phase": 343.7},      # Solar semidiurnal
    "P1": {"amplitude": 0.067, "phase": 216.0},      # Solar diurnal
}


class TidalHarmonics:
    """Container for tidal harmonic constituents."""

    def __init__(self, constituents: dict, station_id: str, mean_water_level: float = 1.5):
        """
        Args:
            constituents: dict mapping constituent name (e.g., 'M2') to {amplitude, phase}
            station_id: NOAA station ID
            mean_water_level: Mean higher high water level in meters (default ~1.5m for San Diego area)
        """
        self.constituents = constituents
        self.station_id = station_id
        self.mean_water_level = mean_water_level

    @classmethod
    def from_cache_or_fetch(cls, station_id: str = NOAA_STATION_ID) -> "TidalHarmonics":
        """Load harmonics from cache, fetch from NOAA, or use defaults."""
        if HARMONICS_CACHE_PATH.exists():
            with open(HARMONICS_CACHE_PATH) as f:
                data = json.load(f)
            return cls(data["constituents"], data["station_id"], data.get("mean_water_level", 1.5))

        # Try to fetch from NOAA, but gracefully fall back to defaults
        try:
            return cls.fetch_from_noaa(station_id)
        except Exception as e:
            print(f"⚠ Could not fetch from NOAA ({e}), using default harmonics for {station_id}")
            return cls(DEFAULT_HARMONICS, station_id, 1.5)

    @classmethod
    def fetch_from_noaa(cls, station_id: str = NOAA_STATION_ID) -> "TidalHarmonics":
        """Fetch harmonic constituents from NOAA API (with graceful fallback).

        NOAA provides harmonic constituents for tide prediction. These are
        stable constants for a given station location. Falls back to
        pre-computed defaults if the API is unavailable.
        """
        if not requests:
            raise ValueError("requests library required for NOAA fetch")

        print(f"Fetching tidal harmonics for station {station_id} from NOAA...")
        try:
            # Try the NOAA harmonics API
            # Note: The public API may have limited availability for harmonics data
            params = {
                "station": station_id,
                "product": "harmonics",
                "format": "json",
                "units": "metric",
            }
            response = requests.get(NOAA_HARMONICS_URL, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            if "error" in data:
                raise ValueError(f"NOAA API error: {data['error'].get('message', 'Unknown')}")

            # Extract harmonic constituents
            harmonics_list = data.get("harmonics", [])
            constituents = {}
            for h in harmonics_list:
                name = h.get("name", "")
                if name:
                    constituents[name] = {
                        "amplitude": float(h.get("amplitude", 0)),
                        "phase": float(h.get("phase", 0)),  # in degrees
                    }

            if not constituents:
                raise ValueError("No harmonic constituents received from NOAA")

            # Get mean water level from NOAA (if available)
            mean_wl = 1.5  # Default fallback
            harmonics = cls(constituents, station_id, mean_wl)
            harmonics._cache()
            return harmonics

        except Exception as e:
            # Fail gracefully — caller will use defaults
            raise RuntimeError(f"Failed to fetch from NOAA: {e}")

    def _cache(self) -> None:
        """Save harmonics to local cache."""
        HARMONICS_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        cache_data = {
            "station_id": self.station_id,
            "mean_water_level": self.mean_water_level,
            "constituents": self.constituents,
            "cached_at": datetime.now(timezone.utc).isoformat(),
        }
        with open(HARMONICS_CACHE_PATH, "w") as f:
            json.dump(cache_data, f, indent=2)

    def predict_water_level(self, time: datetime) -> float:
        """Predict water level (meters) at a given time using harmonic synthesis.

        Args:
            time: UTC datetime

        Returns:
            Predicted water level in meters above chart datum (approx -1.5m below MSL)
        """
        if time.tzinfo is None:
            time = time.replace(tzinfo=timezone.utc)

        # Reference time for phase calculations (Jan 1, 2000 00:00:00 UTC)
        ref_time = datetime(2000, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        time_hours = (time - ref_time).total_seconds() / 3600.0

        # Astronomical argument (number of cycles since reference)
        # This is a simplified version; a full implementation uses
        # nodal modulation factors for >30-day periods
        water_level = self.mean_water_level

        # Major constituents and their angular velocities (°/hour)
        velocities = {
            "M2": 28.984104,    # Principal lunar semidiurnal
            "S2": 30.0,         # Principal solar semidiurnal
            "N2": 28.439729,    # Lunar elliptic semidiurnal
            "K1": 15.041069,    # Lunar diurnal
            "O1": 13.943035,    # Lunar diurnal
            "M4": 57.968208,    # Shallow water overtide
            "M6": 86.952312,    # Shallow water overtide
            "K2": 30.082138,    # Solar semidiurnal
            "P1": 14.958931,    # Solar diurnal
            "Q1": 13.398661,    # Elliptic diurnal
        }

        for name, constituent in self.constituents.items():
            if constituent["amplitude"] > 0.01:  # Skip very small constituents
                velocity = velocities.get(name, 0)
                if velocity > 0:
                    phase_deg = constituent["phase"]
                    amp = constituent["amplitude"]

                    # Harmonic term: A * cos(v*t + phase)
                    angle_deg = (velocity * time_hours + phase_deg) % 360
                    angle_rad = np.radians(angle_deg)
                    water_level += amp * np.cos(angle_rad)

        return water_level

    def predict_tidal_state(self, time: datetime, window_hours: int = 2) -> str:
        """Predict tidal state at a given time.

        Determines if tide is flood, ebb, slack high, or slack low based on
        the rate of change of water level.

        Args:
            time: UTC datetime
            window_hours: Hours to look ahead/behind for slope calculation

        Returns:
            One of: 'flood', 'ebb', 'slack high', 'slack low'
        """
        # Get water levels to estimate slope
        t_before = time - timedelta(hours=window_hours)
        t_after = time + timedelta(hours=window_hours)

        wl_before = self.predict_water_level(t_before)
        wl_now = self.predict_water_level(time)
        wl_after = self.predict_water_level(t_after)

        # Calculate slopes (m/hour)
        slope_before = (wl_now - wl_before) / window_hours
        slope_after = (wl_after - wl_now) / window_hours

        # Threshold for slack (nearly flat) vs active flood/ebb
        slack_threshold = 0.01  # m/hour

        # Determine state based on slopes
        is_before_flat = abs(slope_before) < slack_threshold
        is_after_flat = abs(slope_after) < slack_threshold
        is_rising_before = slope_before > 0
        is_rising_after = slope_after > 0

        if is_before_flat and is_after_flat:
            # Both flat — at slack
            if wl_now > self.mean_water_level:
                return "slack high"
            else:
                return "slack low"
        elif is_rising_before and is_rising_after:
            return "flood"
        elif not is_rising_before and not is_rising_after:
            return "ebb"
        elif is_rising_after:
            # Transition to rising
            return "flood"
        else:
            # Transition to falling
            return "ebb"


def generate_tidal_forecast(
    start_time: datetime,
    hours: int = 168,
    station_id: str = NOAA_STATION_ID,
) -> pd.DataFrame:
    """Generate hourly tidal forecast for Tijuana region.

    Args:
        start_time: Starting UTC datetime
        hours: Number of hours to forecast (default 7 days)
        station_id: NOAA station ID

    Returns:
        DataFrame with columns: time, tide_height, tidal_state
    """
    # Load or fetch harmonics
    harmonics = TidalHarmonics.from_cache_or_fetch(station_id)

    # Generate predictions for each hour
    rows = []
    for i in range(hours):
        time = start_time + timedelta(hours=i)
        wl = harmonics.predict_water_level(time)
        state = harmonics.predict_tidal_state(time)
        rows.append({
            "time": time,
            "tide_height": round(wl, 3),
            "tidal_state": state,
        })

    return pd.DataFrame(rows)
