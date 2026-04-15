"""
Tijuana River Odor Sources.
Defines the river channel and key odor generation points.

Moved from Geodemic (center4health/geodemic backend/app/services/tj_river_sources.py).
Vectorized numpy implementation on the tj_h2s_prediction 100 m grid.
"""

import logging
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from h2s.dispersion.grid_config import (
    GRID_LAT_MESH,
    GRID_LON_MESH,
    GRID_NCOLS,
    GRID_NROWS,
    M_PER_DEG_LAT,
    M_PER_DEG_LON,
)

logger = logging.getLogger(__name__)


@dataclass
class RiverSource:
    """Represents a continuous or point source along the river."""
    name: str
    coordinates: List[Tuple[float, float]]
    source_type: str   # 'channel', 'bridge', 'confluence', 'outfall', 'upwelling'
    base_emission: float
    flow_dependent: bool
    tide_dependent: bool


class TijuanaRiverSources:
    """Manages all odor sources along the Tijuana River system."""

    def __init__(self):
        self.sources = self._define_river_sources()

    def _define_river_sources(self) -> List[RiverSource]:
        sources = []

        sources.append(RiverSource(
            name="Tijuana River Main Channel",
            coordinates=[
                (32.5469, -117.0442),
                (32.5456, -117.0512),
                (32.5445, -117.0578),
                (32.5434, -117.0645),
                (32.5412, -117.0712),
                (32.5389, -117.0789),
                (32.5367, -117.0867),
                (32.5345, -117.0945),
                (32.5334, -117.1023),
                (32.5323, -117.1101),
                (32.5312, -117.1178),
                (32.5301, -117.1234),
            ],
            source_type="channel",
            base_emission=0.75,
            flow_dependent=True,
            tide_dependent=True,
        ))

        sources.append(RiverSource(
            name="Goat Canyon",
            coordinates=[
                (32.5456, -117.0712),
                (32.5478, -117.0689),
                (32.5500, -117.0667),
                (32.5523, -117.0645),
            ],
            source_type="confluence",
            base_emission=1.25,
            flow_dependent=True,
            tide_dependent=False,
        ))

        sources.append(RiverSource(
            name="Smuggler's Gulch",
            coordinates=[
                (32.5412, -117.0867),
                (32.5434, -117.0845),
                (32.5456, -117.0823),
            ],
            source_type="confluence",
            base_emission=1.0,
            flow_dependent=True,
            tide_dependent=False,
        ))

        sources.append(RiverSource(
            name="Dairy Mart Road Bridge",
            coordinates=[(32.5456, -117.0512)],
            source_type="bridge",
            base_emission=1.0,
            flow_dependent=True,
            tide_dependent=False,
        ))

        sources.append(RiverSource(
            name="Hollister Street Bridge",
            coordinates=[(32.5445, -117.0578)],
            source_type="bridge",
            base_emission=0.9,
            flow_dependent=True,
            tide_dependent=False,
        ))

        sources.append(RiverSource(
            name="I-5 Freeway Crossing",
            coordinates=[(32.5345, -117.0945)],
            source_type="bridge",
            base_emission=1.5,
            flow_dependent=True,
            tide_dependent=True,
        ))

        sources.append(RiverSource(
            name="Stewart's Drain",
            coordinates=[
                (32.5523, -117.0456),
                (32.5501, -117.0478),
                (32.5478, -117.0500),
            ],
            source_type="outfall",
            base_emission=1.75,
            flow_dependent=False,
            tide_dependent=False,
        ))

        sources.append(RiverSource(
            name="Yogurt Canyon",
            coordinates=[
                (32.5389, -117.0623),
                (32.5412, -117.0601),
                (32.5434, -117.0578),
            ],
            source_type="confluence",
            base_emission=1.1,
            flow_dependent=True,
            tide_dependent=False,
        ))

        sources.append(RiverSource(
            name="River Mouth/Estuary",
            coordinates=[
                (32.5301, -117.1234),
                (32.5312, -117.1212),
                (32.5323, -117.1189),
                (32.5334, -117.1167),
            ],
            source_type="upwelling",
            base_emission=2.0,
            flow_dependent=False,
            tide_dependent=True,
        ))

        sources.append(RiverSource(
            name="Border Field Wetlands",
            coordinates=[
                (32.5345, -117.1234),
                (32.5367, -117.1256),
                (32.5389, -117.1278),
            ],
            source_type="upwelling",
            base_emission=0.75,
            flow_dependent=False,
            tide_dependent=True,
        ))

        return sources

    def get_emission_strength(
        self,
        source: RiverSource,
        flow_rate: float = 50.0,
        tide_level: float = 0.5,
        temperature: float = 20.0,
    ) -> float:
        strength = source.base_emission

        if source.flow_dependent:
            strength *= min(2.0, flow_rate / 50)

        if source.tide_dependent:
            strength *= 1.5 - tide_level * 0.5

        temp_factor = 1.0 + (temperature - 20) * 0.02
        strength *= max(0.5, min(1.5, temp_factor))

        if source.source_type == "bridge":
            strength *= 1.3

        return min(50, strength)

    def generate_river_emission_grid(
        self,
        wind_speed: float,
        wind_direction: float,
        flow_rate: float = 50.0,
        tide_level: float = 0.5,
        temperature: float = 20.0,
    ) -> np.ndarray:
        """Generate emission grid from all river sources (vectorized, 100 m grid)."""
        emission_grid = np.zeros((GRID_NROWS, GRID_NCOLS))
        downwind_rad = np.radians(wind_direction + 180)

        for source in self.sources:
            emission = self.get_emission_strength(source, flow_rate, tide_level, temperature)
            if emission <= 0:
                continue

            channel_mult = 1.2 if source.source_type == "channel" else 1.0

            for src_lat, src_lon in source.coordinates:
                dx_m = (GRID_LON_MESH - src_lon) * M_PER_DEG_LON
                dy_m = (GRID_LAT_MESH - src_lat) * M_PER_DEG_LAT
                dist_km = np.sqrt(dx_m**2 + dy_m**2) / 1000.0

                mask = dist_km <= 8
                bearing_rad = np.arctan2(dx_m, dy_m)
                wind_alignment = np.cos(bearing_rad - downwind_rad)

                sigma = np.where(wind_alignment > 0, 0.5 + dist_km * 0.1, 0.4)
                eff_dist = np.where(
                    wind_alignment > 0,
                    dist_km / (1 + wind_alignment * wind_speed / 10),
                    dist_km * 3,
                )

                conc = emission * np.exp(-(eff_dist**2) / (2 * sigma**2)) * channel_mult
                emission_grid += np.where(mask, conc, 0.0)

        logger.info(
            "River emission grid: max=%.1f, affected cells=%d",
            float(np.max(emission_grid)),
            int(np.sum(emission_grid > 5)),
        )
        return emission_grid


# Singleton instance
tj_river_sources = TijuanaRiverSources()
