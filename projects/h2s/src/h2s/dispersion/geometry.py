"""
Source geometry loader and sub-point tessellation for the dispersion calibration loop.

Loads source_geometry.toml and expands point / line / polygon geometry descriptors
into discretised SubPoints.  Each SubPoint carries a weight (its arc/area fraction
of the parent source) so the forward model and sensitivity-matrix builder can sum
over sub-points without knowing the geometry type:

    Q_subpoint = Q_source * subpoint.weight     (weights sum to 1 per source)

Usage
-----
    from h2s.dispersion.geometry import load_source_geometry, SubPoint, SourceSpec

    specs = load_source_geometry()          # loads bundled source_geometry.toml
    for sid, spec in specs.items():
        for sp in spec.sub_points:
            ...                             # sp.lat, sp.lon, sp.weight

The default path is ``dispersion/source_geometry.toml`` (co-located with this file).
Pass an explicit path to ``load_source_geometry()`` for testing or alternate configs.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Metric conversions consistent with gaussian.py and lagrangian.py
M_PER_DEG_LAT: float = 111_320.0
M_PER_DEG_LON: float = 111_320.0 * np.cos(np.radians(32.55))

_DEFAULT_GEOMETRY_PATH = Path(__file__).parent / "source_geometry.toml"


# ---------------------------------------------------------------------------
# Public data types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SubPoint:
    """One discretised emission point from a named source geometry."""
    lat: float
    lon: float
    source_id: str
    zone: str
    weight: float   # fraction of parent Q this sub-point carries; sums to 1 per source


@dataclass
class SourceSpec:
    """Fully parsed descriptor for one named source."""
    source_id: str
    geometry: str           # "point" | "line" | "polygon"
    zone: str
    q_prior: float          # g/s starting guess for inversion
    tide_modulated: bool
    sub_points: list[SubPoint]

    @property
    def n_sub_points(self) -> int:
        return len(self.sub_points)


# ---------------------------------------------------------------------------
# Geometry helpers (not exported but tested directly)
# ---------------------------------------------------------------------------

def _discretize_line(
    vertices: list[tuple[float, float]],
    spacing_m: float,
    source_id: str,
    zone: str,
) -> list[SubPoint]:
    """Interpolate a polyline into evenly-spaced sub-points.

    Each sub-point's weight = (length it represents) / (total line length),
    so weights sum exactly to 1 and Q is conserved: Q_sub = Q_total * weight.

    A single-vertex degenerate line returns one sub-point with weight=1.
    """
    if len(vertices) < 2:
        la, lo = vertices[0]
        return [SubPoint(lat=la, lon=lo, source_id=source_id, zone=zone, weight=1.0)]

    # Per-segment Euclidean lengths in metres
    seg_lengths: list[float] = []
    for i in range(len(vertices) - 1):
        la0, lo0 = vertices[i]
        la1, lo1 = vertices[i + 1]
        dy = (la1 - la0) * M_PER_DEG_LAT
        dx = (lo1 - lo0) * M_PER_DEG_LON
        seg_lengths.append(float(np.sqrt(dy * dy + dx * dx)))

    total_length = sum(seg_lengths)
    if total_length == 0:
        la, lo = vertices[0]
        return [SubPoint(lat=la, lon=lo, source_id=source_id, zone=zone, weight=1.0)]

    pts: list[SubPoint] = []
    for i in range(len(vertices) - 1):
        la0, lo0 = vertices[i]
        la1, lo1 = vertices[i + 1]
        seg_len = seg_lengths[i]
        n_sub = max(1, int(seg_len / spacing_m))
        sub_len = seg_len / n_sub          # arc length each sub-point represents

        for k in range(n_sub):
            t = (k + 0.5) / n_sub
            pts.append(SubPoint(
                lat=la0 + t * (la1 - la0),
                lon=lo0 + t * (lo1 - lo0),
                source_id=source_id,
                zone=zone,
                weight=sub_len / total_length,
            ))

    return pts


def _point_in_polygon(lat: float, lon: float, ring: list[tuple[float, float]]) -> bool:
    """Ray-casting point-in-polygon test (no external dependencies)."""
    inside = False
    n = len(ring)
    j = n - 1
    for i in range(n):
        yi, xi = ring[i]
        yj, xj = ring[j]
        if ((yi > lat) != (yj > lat)) and (
            lon < (xj - xi) * (lat - yi) / (yj - yi) + xi
        ):
            inside = not inside
        j = i
    return inside


def _tessellate_polygon(
    vertices: list[tuple[float, float]],
    spacing_m: float,
    source_id: str,
    zone: str,
) -> list[SubPoint]:
    """Grid points inside a closed polygon ring, each with equal weight.

    Creates a regular lat/lon grid at metric spacing_m over the bounding box
    and retains only interior points.  Falls back to the polygon centroid if
    the polygon is smaller than one grid cell.
    """
    lats = [v[0] for v in vertices]
    lons = [v[1] for v in vertices]

    d_lat = spacing_m / M_PER_DEG_LAT
    d_lon = spacing_m / M_PER_DEG_LON

    lat_pts = np.arange(min(lats) + d_lat / 2, max(lats), d_lat)
    lon_pts = np.arange(min(lons) + d_lon / 2, max(lons), d_lon)

    interior: list[tuple[float, float]] = [
        (float(la), float(lo))
        for la in lat_pts
        for lo in lon_pts
        if _point_in_polygon(float(la), float(lo), vertices)
    ]

    if not interior:
        # Polygon too small for the grid spacing — fall back to centroid
        cen_lat = sum(lats) / len(lats)
        cen_lon = sum(lons) / len(lons)
        interior = [(cen_lat, cen_lon)]

    w = 1.0 / len(interior)
    return [
        SubPoint(lat=la, lon=lo, source_id=source_id, zone=zone, weight=w)
        for la, lo in interior
    ]


# ---------------------------------------------------------------------------
# Public loader
# ---------------------------------------------------------------------------

def load_source_geometry(
    path: Path | str | None = None,
) -> dict[str, SourceSpec]:
    """Load and expand source geometry from a TOML config file.

    Args:
        path: Path to the TOML file.  Defaults to ``source_geometry.toml``
              co-located with this module.

    Returns:
        Ordered dict mapping ``source_id`` -> :class:`SourceSpec`, each with
        ``sub_points`` already expanded from the geometry descriptor.

    Raises:
        ValueError: if a source has an unknown ``geometry`` type.
        FileNotFoundError: if ``path`` does not exist.
    """
    cfg_path = Path(path) if path is not None else _DEFAULT_GEOMETRY_PATH

    with open(cfg_path, "rb") as fh:
        cfg = tomllib.load(fh)

    result: dict[str, SourceSpec] = {}

    for source_id, spec in cfg.get("sources", {}).items():
        geom = spec["geometry"]
        zone = spec.get("zone", "unknown")
        q_prior = float(spec.get("q_prior", 0.0))
        spacing_m = float(spec.get("discretize_spacing_m", 150.0))
        tide_modulated = bool(spec.get("tide_modulated", False))

        if geom == "point":
            loc = spec["location"]
            sub_points = [
                SubPoint(
                    lat=float(loc[0]), lon=float(loc[1]),
                    source_id=source_id, zone=zone, weight=1.0,
                )
            ]

        elif geom == "line":
            verts = [(float(v[0]), float(v[1])) for v in spec["vertices"]]
            sub_points = _discretize_line(verts, spacing_m, source_id, zone)

        elif geom == "polygon":
            verts = [(float(v[0]), float(v[1])) for v in spec["vertices"]]
            sub_points = _tessellate_polygon(verts, spacing_m, source_id, zone)

        else:
            raise ValueError(
                f"Unknown geometry type '{geom}' for source '{source_id}'. "
                "Expected 'point', 'line', or 'polygon'."
            )

        result[source_id] = SourceSpec(
            source_id=source_id,
            geometry=geom,
            zone=zone,
            q_prior=q_prior,
            tide_modulated=tide_modulated,
            sub_points=sub_points,
        )

    return result


def zone_sub_points(
    specs: dict[str, SourceSpec],
    zone: str,
) -> list[SubPoint]:
    """Return all sub-points belonging to a given zone across all sources."""
    return [
        sp
        for spec in specs.values()
        if spec.zone == zone
        for sp in spec.sub_points
    ]


def all_sub_points(specs: dict[str, SourceSpec]) -> list[SubPoint]:
    """Flat list of every sub-point across all sources."""
    return [sp for spec in specs.values() for sp in spec.sub_points]
