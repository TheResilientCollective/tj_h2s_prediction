"""H2S Generation Model based on sewage chemistry and environmental factors.

Moved from Geodemic (center4health/geodemic backend/app/services/h2s_generation.py).
Source coordinates are inlined; Geodemic app.config dependency removed.
"""

import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# Source coordinates (inlined from Geodemic app.config)
# ---------------------------------------------------------------------------
_EMISSION_SOURCES: dict[str, tuple[float, float]] = {
    "tj_river_mouth": (32.5301, -117.1234),
    "south_bay_wwtp": (32.5289, -117.1445),
    "punta_bandera":  (32.5234, -117.0987),
    "goat_canyon":    (32.5456, -117.0712),
    "stewarts_drain": (32.5527, -117.0419),
}

_COMMUNITY_LOCATIONS: dict[str, tuple[float, float]] = {
    "Imperial Beach": (32.5783, -117.1134),
    "Chula Vista":    (32.6401, -117.0842),
    "National City":  (32.6781, -117.0992),
    "Otay Mesa":      (32.5578, -116.9834),
}


class H2SGenerationModel:
    """
    Models H2S generation from sewage based on:
    - Anaerobic decomposition rates
    - Temperature dependency (Arrhenius equation)
    - pH effects on sulfate reduction
    - Dissolved oxygen levels
    - Tidal influence (sediment exposure)
    - Flow rates and residence time
    """

    def __init__(self):
        self.k0 = 0.015              # Base reaction rate constant at 20°C (hr^-1)
        self.activation_energy = 60000  # J/mol
        self.gas_constant = 8.314    # J/(mol·K)
        self.reference_temp = 293.15 # 20°C in Kelvin

        self.sulfate_conc_mgL = 250  # Typical seawater sulfate
        self.BOD_mgL = 300           # Biological Oxygen Demand in raw sewage
        self.pH_optimal = 7.2
        self.DO_threshold = 0.5      # mg/L — below this, anaerobic conditions dominate

    def calculate_h2s_production(
        self,
        flow_rate_cfs: float,
        temperature_c: float,
        pH: float,
        dissolved_oxygen: float,
        tide_level_m: float,
        location: Tuple[float, float],
    ) -> float:
        """
        Calculate H2S production rate in ppb.

        Args:
            flow_rate_cfs: Sewage flow rate in cubic feet per second
            temperature_c: Water temperature in Celsius
            pH: pH of the water/sewage mixture
            dissolved_oxygen: DO in mg/L
            tide_level_m: Tide level in meters (affects sediment exposure)
            location: (lat, lon) tuple for location-specific factors

        Returns:
            H2S concentration in ppb
        """
        residence_factor = 1.0 / (1.0 + flow_rate_cfs / 100)

        temp_k = temperature_c + 273.15
        k_temp = self.k0 * np.exp(
            -self.activation_energy / self.gas_constant * (1 / temp_k - 1 / self.reference_temp)
        )

        pH_factor = np.exp(-0.5 * ((pH - self.pH_optimal) / 1.5) ** 2)
        DO_factor = np.exp(-dissolved_oxygen / self.DO_threshold)
        tide_factor = 1.0 + 0.5 * np.exp(-tide_level_m / 0.5)
        substrate_factor = (self.sulfate_conc_mgL / 250) * (self.BOD_mgL / 300)

        base_production = 50.0

        h2s_production = (
            base_production
            * k_temp
            * pH_factor
            * DO_factor
            * tide_factor
            * substrate_factor
            * residence_factor
        )

        lat, lon = location
        if 32.53 < lat < 32.55 and -117.13 < lon < -117.11:
            h2s_production *= 1.5
        elif 32.52 < lat < 32.54 and -117.15 < lon < -117.14:
            h2s_production *= 1.3

        h2s_production += np.random.normal(0, h2s_production * 0.1)
        return max(0, h2s_production)

    def get_source_emissions(
        self,
        current_conditions: Dict,
    ) -> List[Tuple[float, float, float]]:
        """
        Get H2S emission rates for all known sources based on current conditions.

        Returns:
            List of (lat, lon, emission_rate_ppb) tuples
        """
        temp = current_conditions.get("temperature", 22.0)
        pH   = current_conditions.get("pH", 7.5)
        DO   = current_conditions.get("dissolved_oxygen", 2.0)
        tide = current_conditions.get("tide_level", 0.5)
        flow = current_conditions.get("flow_rate", 50.0)

        source_configs = [
            {"location": _EMISSION_SOURCES["tj_river_mouth"], "flow_multiplier": 1.5, "pH": pH - 0.3, "DO": DO * 0.5},
            {"location": _EMISSION_SOURCES["south_bay_wwtp"], "flow_multiplier": 1.2, "pH": pH,       "DO": DO * 0.7},
            {"location": _EMISSION_SOURCES["punta_bandera"],  "flow_multiplier": 1.0, "pH": pH + 0.2, "DO": DO},
            {"location": _EMISSION_SOURCES["goat_canyon"],    "flow_multiplier": 0.8, "pH": pH - 0.1, "DO": DO * 0.6},
            {"location": _EMISSION_SOURCES["stewarts_drain"], "flow_multiplier": 0.7, "pH": pH,       "DO": DO * 0.8},
        ]

        sources = []
        for config in source_configs:
            emission = self.calculate_h2s_production(
                flow_rate_cfs=flow * config["flow_multiplier"],
                temperature_c=temp,
                pH=config["pH"],
                dissolved_oxygen=config["DO"],
                tide_level_m=tide,
                location=config["location"],
            )
            sources.append((config["location"][0], config["location"][1], emission))

        minor_sources = [
            (*_COMMUNITY_LOCATIONS["Imperial Beach"], 15.0),
            (*_COMMUNITY_LOCATIONS["Chula Vista"],    20.0),
            (*_COMMUNITY_LOCATIONS["National City"],  18.0),
            (*_COMMUNITY_LOCATIONS["Otay Mesa"],       8.0),
        ]
        for lat, lon, base_emission in minor_sources:
            emission = base_emission * (1.0 + 0.2 * np.sin(datetime.now().hour * np.pi / 12))
            sources.append((lat, lon, emission))

        return sources

    def predict_temporal_pattern(self, hours_ahead: int = 24) -> np.ndarray:
        """
        Predict H2S generation pattern over next N hours
        considering diurnal cycles and tidal patterns.
        """
        predictions = []
        current_hour = datetime.now().hour

        for h in range(hours_ahead):
            hour = (current_hour + h) % 24
            temp = 20 + 5 * np.sin((hour - 14) * np.pi / 12)
            tide = 1.0 + 0.5 * np.sin(h * 2 * np.pi / 12.4)
            bio_factor = 1.0 + 0.3 * np.sin((hour - 15) * np.pi / 12)
            flow = 50 * (1.0 + 0.2 * np.sin((hour - 8) * np.pi / 12))

            h2s = self.calculate_h2s_production(
                flow_rate_cfs=flow,
                temperature_c=temp,
                pH=7.2,
                dissolved_oxygen=2.0,
                tide_level_m=tide,
                location=_EMISSION_SOURCES["tj_river_mouth"],
            ) * bio_factor
            predictions.append(h2s)

        return np.array(predictions)
