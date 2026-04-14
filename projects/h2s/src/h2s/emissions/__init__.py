"""Physics-based H2S river emissions model.

Moved from Geodemic (center4health/geodemic) to serve as the canonical
implementation. Geodemic reads the gridded output from S3 via
GET /api/v1/tj-data/dispersion/river-emission-grid.

Key classes:
  H2SGenerationModel  -- Arrhenius kinetics for individual point sources
  TijuanaRiverSources -- River channel geometry + vectorized emission grid
"""
from h2s.emissions.h2s_generation import H2SGenerationModel
from h2s.emissions.tj_river_sources import TijuanaRiverSources

__all__ = ["H2SGenerationModel", "TijuanaRiverSources"]
