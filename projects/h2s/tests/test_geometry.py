"""Unit tests for dispersion/geometry.py — source geometry loader and tessellation."""

import math
from pathlib import Path

import numpy as np
import pytest

from h2s.dispersion.geometry import (
    SubPoint,
    SourceSpec,
    _discretize_line,
    _point_in_polygon,
    _tessellate_polygon,
    load_source_geometry,
    all_sub_points,
    zone_sub_points,
    M_PER_DEG_LAT,
    M_PER_DEG_LON,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _seg_len_m(la0, lo0, la1, lo1) -> float:
    dy = (la1 - la0) * M_PER_DEG_LAT
    dx = (lo1 - lo0) * M_PER_DEG_LON
    return math.sqrt(dy * dy + dx * dx)


SQUARE_RING = [
    (32.560, -117.100),
    (32.570, -117.100),
    (32.570, -117.090),
    (32.560, -117.090),
]


# ---------------------------------------------------------------------------
# _discretize_line
# ---------------------------------------------------------------------------

class TestDiscretizeLine:
    def test_single_vertex_returns_one_point(self):
        pts = _discretize_line([(32.55, -117.09)], 100.0, "src", "east")
        assert len(pts) == 1
        assert pts[0].weight == pytest.approx(1.0)

    def test_weights_sum_to_one(self):
        verts = [
            (32.548531, -117.064293),
            (32.539743, -117.064269),
            (32.542103, -117.054117),
        ]
        pts = _discretize_line(verts, 120.0, "east_drain", "east")
        assert sum(p.weight for p in pts) == pytest.approx(1.0, abs=1e-9)

    def test_longer_segment_gets_more_sub_points(self):
        # Segment A: ~300 m, Segment B: ~3000 m — B should have ~10× more points.
        d_lat_300 = 300.0 / M_PER_DEG_LAT
        d_lat_3000 = 3000.0 / M_PER_DEG_LAT
        verts = [
            (32.550, -117.090),
            (32.550 + d_lat_300, -117.090),
            (32.550 + d_lat_300 + d_lat_3000, -117.090),
        ]
        pts = _discretize_line(verts, 150.0, "src", "east")
        # int(300/150)=2 + int(3000/150)=20; allow ±1 for float-precision rounding of
        # segment lengths (d_lat * M_PER_DEG_LAT may differ from 300.0 by ~1e-10 m).
        assert 18 <= len(pts) <= 24

    def test_longer_segment_carries_more_weight(self):
        # A 3× longer second segment should carry ~3× the total weight.
        # Use exact multiples of the spacing to avoid float-rounding edge cases.
        d_lat_1 = 450.0 / M_PER_DEG_LAT   # 3 sub-points at 150 m
        d_lat_3 = 1350.0 / M_PER_DEG_LAT  # 9 sub-points at 150 m
        verts = [
            (32.550, -117.090),
            (32.550 + d_lat_1, -117.090),
            (32.550 + d_lat_1 + d_lat_3, -117.090),
        ]
        pts = _discretize_line(verts, 150.0, "src", "east")

        # Weights from segment 1 (short) vs segment 2 (3× longer)
        # The first 3 sub-points belong to segment 1, the rest to segment 2.
        w_seg1 = sum(p.weight for p in pts[:3])
        w_seg2 = sum(p.weight for p in pts[3:])
        # Segment 2 is 3× longer so carries 3× the weight (within float tolerance).
        assert w_seg2 == pytest.approx(3.0 * w_seg1, rel=0.05)

    def test_source_id_and_zone_propagated(self):
        verts = [(32.55, -117.09), (32.56, -117.09)]
        pts = _discretize_line(verts, 200.0, "my_source", "south")
        for p in pts:
            assert p.source_id == "my_source"
            assert p.zone == "south"

    def test_sub_points_lie_on_the_line(self):
        # All sub-points should be interpolated between the two end vertices.
        la0, lo0 = 32.540, -117.060
        la1, lo1 = 32.550, -117.050
        verts = [(la0, lo0), (la1, lo1)]
        pts = _discretize_line(verts, 200.0, "src", "east")
        for p in pts:
            # Parametric t ∈ (0, 1)
            t_lat = (p.lat - la0) / (la1 - la0)
            t_lon = (p.lon - lo0) / (lo1 - lo0)
            assert t_lat == pytest.approx(t_lon, abs=1e-8)
            assert 0.0 < t_lat < 1.0

    def test_spacing_controls_density(self):
        verts = [(32.540, -117.060), (32.560, -117.060)]
        pts_fine = _discretize_line(verts, 50.0, "src", "east")
        pts_coarse = _discretize_line(verts, 500.0, "src", "east")
        assert len(pts_fine) > len(pts_coarse)


# ---------------------------------------------------------------------------
# _point_in_polygon
# ---------------------------------------------------------------------------

class TestPointInPolygon:
    def test_centre_of_square_is_inside(self):
        centre_lat = (32.560 + 32.570) / 2
        centre_lon = (-117.100 + -117.090) / 2
        assert _point_in_polygon(centre_lat, centre_lon, SQUARE_RING)

    def test_corners_outside(self):
        # Corners themselves are on the boundary; test clearly outside
        assert not _point_in_polygon(32.555, -117.110, SQUARE_RING)
        assert not _point_in_polygon(32.575, -117.085, SQUARE_RING)

    def test_far_outside(self):
        assert not _point_in_polygon(32.600, -117.000, SQUARE_RING)


# ---------------------------------------------------------------------------
# _tessellate_polygon
# ---------------------------------------------------------------------------

class TestTessellatePolygon:
    def test_all_points_inside_ring(self):
        pts = _tessellate_polygon(SQUARE_RING, 200.0, "ib_estuary", "west")
        for p in pts:
            assert _point_in_polygon(p.lat, p.lon, SQUARE_RING), (
                f"Sub-point ({p.lat:.6f}, {p.lon:.6f}) is outside the ring"
            )

    def test_weights_sum_to_one(self):
        pts = _tessellate_polygon(SQUARE_RING, 200.0, "ib_estuary", "west")
        assert sum(p.weight for p in pts) == pytest.approx(1.0, abs=1e-9)

    def test_equal_weights(self):
        pts = _tessellate_polygon(SQUARE_RING, 200.0, "ib_estuary", "west")
        w0 = pts[0].weight
        for p in pts:
            assert p.weight == pytest.approx(w0)

    def test_finer_spacing_more_points(self):
        pts_fine = _tessellate_polygon(SQUARE_RING, 100.0, "ib_estuary", "west")
        pts_coarse = _tessellate_polygon(SQUARE_RING, 500.0, "ib_estuary", "west")
        assert len(pts_fine) >= len(pts_coarse)

    def test_tiny_polygon_falls_back_to_centroid(self):
        # A 1 m² ring will have no interior grid points at 150 m spacing.
        d = 1e-5  # ~1 m
        tiny_ring = [
            (32.565, -117.120),
            (32.565 + d, -117.120),
            (32.565 + d, -117.120 + d),
            (32.565, -117.120 + d),
        ]
        pts = _tessellate_polygon(tiny_ring, 150.0, "tiny", "west")
        assert len(pts) == 1
        assert pts[0].weight == pytest.approx(1.0)

    def test_source_id_and_zone_propagated(self):
        pts = _tessellate_polygon(SQUARE_RING, 300.0, "estuary_x", "west")
        for p in pts:
            assert p.source_id == "estuary_x"
            assert p.zone == "west"


# ---------------------------------------------------------------------------
# load_source_geometry — bundled TOML
# ---------------------------------------------------------------------------

class TestLoadSourceGeometry:
    @pytest.fixture(scope="class")
    def specs(self):
        return load_source_geometry()

    def test_expected_sources_present(self, specs):
        expected = {
            "east_drain_corridor",
            "river_main_stem_west",
            "smugglers_goat_canyon",
            "saturn_blvd_bridge",
            "imperial_beach_estuary",
        }
        assert expected.issubset(set(specs.keys()))

    def test_geometry_types_correct(self, specs):
        assert specs["east_drain_corridor"].geometry == "line"
        assert specs["river_main_stem_west"].geometry == "line"
        assert specs["smugglers_goat_canyon"].geometry == "line"
        assert specs["saturn_blvd_bridge"].geometry == "point"
        assert specs["imperial_beach_estuary"].geometry == "polygon"

    def test_point_source_has_one_sub_point(self, specs):
        assert specs["saturn_blvd_bridge"].n_sub_points == 1
        assert specs["saturn_blvd_bridge"].sub_points[0].weight == pytest.approx(1.0)

    def test_line_sources_weight_sum(self, specs):
        for sid in ("east_drain_corridor", "river_main_stem_west", "smugglers_goat_canyon"):
            total_w = sum(sp.weight for sp in specs[sid].sub_points)
            assert total_w == pytest.approx(1.0, abs=1e-9), f"{sid} weights don't sum to 1"

    def test_polygon_source_weight_sum(self, specs):
        total_w = sum(sp.weight for sp in specs["imperial_beach_estuary"].sub_points)
        assert total_w == pytest.approx(1.0, abs=1e-9)

    def test_line_sources_have_multiple_sub_points(self, specs):
        # The east drain corridor is ~1.5 km at 120 m spacing → at least 10 sub-points
        assert specs["east_drain_corridor"].n_sub_points >= 10

    def test_polygon_interior_points_inside_boundary(self, specs):
        estuary_verts = [
            (32.566000, -117.130000),
            (32.578000, -117.130000),
            (32.578000, -117.118000),
            (32.566000, -117.118000),
        ]
        for sp in specs["imperial_beach_estuary"].sub_points:
            assert _point_in_polygon(sp.lat, sp.lon, estuary_verts), (
                f"Estuary sub-point ({sp.lat}, {sp.lon}) outside placeholder boundary"
            )

    def test_zone_assignments(self, specs):
        assert specs["east_drain_corridor"].zone == "east"
        assert specs["saturn_blvd_bridge"].zone == "west"
        assert specs["smugglers_goat_canyon"].zone == "south"
        assert specs["imperial_beach_estuary"].zone == "west"

    def test_q_prior_positive(self, specs):
        for sid, spec in specs.items():
            assert spec.q_prior >= 0.0, f"{sid} q_prior is negative"

    def test_all_sub_points_helper(self, specs):
        flat = all_sub_points(specs)
        assert len(flat) > 0
        total_per_source = {sid: sum(sp.weight for sp in spec.sub_points)
                           for sid, spec in specs.items()}
        for sid, w in total_per_source.items():
            assert w == pytest.approx(1.0, abs=1e-9), f"{sid} weight sum != 1"

    def test_zone_sub_points_helper(self, specs):
        east_pts = zone_sub_points(specs, "east")
        assert all(sp.zone == "east" for sp in east_pts)
        west_pts = zone_sub_points(specs, "west")
        assert all(sp.zone == "west" for sp in west_pts)

    def test_saturn_blvd_location(self, specs):
        sp = specs["saturn_blvd_bridge"].sub_points[0]
        assert sp.lat == pytest.approx(32.559383, abs=1e-5)
        assert sp.lon == pytest.approx(-117.092992, abs=1e-5)


# ---------------------------------------------------------------------------
# Custom TOML path
# ---------------------------------------------------------------------------

def test_load_from_explicit_path(tmp_path):
    toml_content = """\
[sources.test_point]
geometry = "point"
q_prior = 7.5
zone = "east"
location = [32.545, -117.065]

[sources.test_line]
geometry = "line"
discretize_spacing_m = 200
q_prior = 15.0
zone = "south"
vertices = [
    [32.537, -117.086],
    [32.537, -117.099],
]
"""
    p = tmp_path / "test_geo.toml"
    p.write_text(toml_content)

    specs = load_source_geometry(p)

    assert set(specs.keys()) == {"test_point", "test_line"}
    assert specs["test_point"].q_prior == pytest.approx(7.5)
    assert specs["test_point"].n_sub_points == 1
    assert sum(sp.weight for sp in specs["test_line"].sub_points) == pytest.approx(1.0)


def test_unknown_geometry_raises(tmp_path):
    toml_content = """\
[sources.bad_source]
geometry = "cylinder"
q_prior = 1.0
zone = "east"
location = [32.545, -117.065]
"""
    p = tmp_path / "bad.toml"
    p.write_text(toml_content)

    with pytest.raises(ValueError, match="Unknown geometry type"):
        load_source_geometry(p)


# ---------------------------------------------------------------------------
# Physics sanity — line source reduces SY peak sensitivity
# ---------------------------------------------------------------------------

class TestLineSensorSensitivity:
    """Verify that distributing the east drain as a line lowers SY sensitivity.

    The root-cause fix: with Q=1 g/s spread over the line, the total ppb at SY
    must be less than putting Q=1 at TJ Crossing CDLP E (the nearest legacy
    CANDIDATE_SOURCE point).  SSW wind (≈196°) puts SY nearly on-axis from TJ_E
    (~59 m crosswind at 1214 m downwind), making this the high-sensitivity scenario.
    The line spreads Q over Dairy Mart → Silva → TJ_W → TJ_E; with SSW wind, the
    Dairy Mart and Silva sub-points land far crosswind of SY (~1363 m), so they
    contribute almost nothing — all the SY signal comes from the TJ_E-vicinity
    sub-points, which carry only ~15% of total Q vs 100% for the old point.
    """
    def test_line_lowers_sy_sensitivity_vs_candidate_source_point(self):
        from h2s.dispersion.gaussian import gaussian_plume_concentration, stability_class

        SY = {"lat": 32.552794, "lon": -117.047286}
        # SSW wind puts TJ Crossing CDLP E (~SSW of SY) almost directly upwind.
        ws, wd_deg = 1.5, 196.0
        stab = stability_class(ws, is_night=True)  # class E (stable, ws=1.5 m/s night)
        wd_rad = np.radians(wd_deg)
        u = -ws * np.sin(wd_rad)
        v = -ws * np.cos(wd_rad)

        # Old geometry: full Q=1 g/s at TJ Crossing CDLP E (nearest legacy point to SY)
        tj_crossing_e = {"lat": 32.542166, "lon": -117.050325}
        c_old = gaussian_plume_concentration(
            source_lat=tj_crossing_e["lat"], source_lon=tj_crossing_e["lon"],
            emission_rate_g_s=1.0,
            receptor_lat=SY["lat"], receptor_lon=SY["lon"],
            wind_u=u, wind_v=v, stab=stab,
        )

        # New geometry: Q=1 g/s distributed over east_drain_corridor line
        specs = load_source_geometry()
        c_new = sum(
            gaussian_plume_concentration(
                source_lat=sp.lat, source_lon=sp.lon,
                emission_rate_g_s=sp.weight,  # weight * Q (Q=1)
                receptor_lat=SY["lat"], receptor_lon=SY["lon"],
                wind_u=u, wind_v=v, stab=stab,
            )
            for sp in specs["east_drain_corridor"].sub_points
        )

        assert c_old > 0.0, "TJ_E should be downwind of SY with SSW wind"
        assert c_new < c_old, (
            f"Line ({c_new:.4f} µg/m³ per g/s) should be < single point at TJ_E "
            f"({c_old:.4f} µg/m³ per g/s) at SY — geometry fix not working."
        )
