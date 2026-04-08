"""H2S dispersion modeling subpackage.

Provides backward Lagrangian source attribution, Gaussian plume forward
forecasting, and HYSPLIT CONTROL file generation.
"""

from h2s.dispersion.lagrangian import (
    LagrangianConfig,
    run_inversion_window,
    source_attribution,
    CANDIDATE_SOURCES,
    SENSORS as LAGRANGIAN_SENSORS,
)
from h2s.dispersion.gaussian import (
    run_forward_model,
    ForwardModelResult,
    SENSORS as DISPERSION_SENSORS,
    SOURCES as DISPERSION_SOURCE_ZONES,
)
from h2s.dispersion.hysplit_controls import generate_hysplit_bundle

__all__ = [
    "LagrangianConfig",
    "run_inversion_window",
    "source_attribution",
    "CANDIDATE_SOURCES",
    "LAGRANGIAN_SENSORS",
    "run_forward_model",
    "ForwardModelResult",
    "DISPERSION_SENSORS",
    "DISPERSION_SOURCE_ZONES",
    "generate_hysplit_bundle",
]
