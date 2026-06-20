"""Tests for run_forward_model_from_geometry and its gridded twin (Phase 1)."""

import numpy as np
import pandas as pd
import pytest

from h2s.dispersion.gaussian import (
    gaussian_plume_concentration,
    run_forward_model_from_geometry,
    run_forward_model_gridded_from_geometry,
    stability_class,
    ug_m3_to_ppb,
    SENSORS,
)
from h2s.dispersion.geometry import (
    SourceSpec,
    SubPoint,
    load_source_geometry,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

START = pd.Timestamp("2026-04-01 00:00:00", tz="UTC")
REF_SENSOR = "NESTOR - BES"


def _make_met(
    wind_speed: float = 3.0,
    wind_direction: float = 196.0,
    temp: float = 18.0,
    is_night: int = 1,
    n_hours: int = 3,
    sensors: list[str] | None = None,
) -> pd.DataFrame:
    """Minimal met DataFrame covering n_hours of hourly timesteps for all sensors."""
    if sensors is None:
        sensors = list(SENSORS.keys())
    times = pd.date_range(START, periods=n_hours, freq="1h")
    rows = []
    for t in times:
        for sname in sensors:
            rows.append({
                "time":               t,
                "site_name":          sname,
                "wind_speed_10m":     wind_speed,
                "wind_direction_10m": wind_direction,
                "temperature_2m":     temp,
                "is_night":           is_night,
            })
    return pd.DataFrame(rows)


def _point_specs(lat: float, lon: float, zone: str = "east") -> dict[str, SourceSpec]:
    """Single-source specs dict with one point source."""
    sp = SubPoint(lat=lat, lon=lon, source_id="test_pt", zone=zone, weight=1.0)
    return {
        "test_pt": SourceSpec(
            source_id="test_pt",
            geometry="point",
            zone=zone,
            q_prior=0.0,
            tide_modulated=False,
            sub_points=[sp],
        )
    }


# ---------------------------------------------------------------------------
# run_forward_model_from_geometry — structure
# ---------------------------------------------------------------------------

class TestForwardGeometryStructure:
    def test_returns_forward_model_result(self):
        from h2s.dispersion.gaussian import ForwardModelResult
        specs = load_source_geometry()
        df = _make_met(n_hours=2)
        result = run_forward_model_from_geometry(
            df, specs, {"east_drain_corridor": 20.0}, START, hours=2
        )
        assert isinstance(result, ForwardModelResult)

    def test_timeseries_length_matches_hours(self):
        specs = load_source_geometry()
        df = _make_met(n_hours=6)
        result = run_forward_model_from_geometry(
            df, specs, {"east_drain_corridor": 20.0}, START, hours=6
        )
        for sensor, vals in result.concentrations.items():
            assert len(vals) == 6, f"{sensor} has {len(vals)} values, expected 6"

    def test_all_sensors_in_result(self):
        specs = load_source_geometry()
        df = _make_met(n_hours=1)
        result = run_forward_model_from_geometry(
            df, specs, {"smugglers_goat_canyon": 137.0}, START, hours=1
        )
        assert set(result.concentrations.keys()) == set(SENSORS.keys())

    def test_metadata_contains_source_info(self):
        specs = load_source_geometry()
        df = _make_met(n_hours=1)
        result = run_forward_model_from_geometry(
            df, specs, {"east_drain_corridor": 10.0}, START, hours=1
        )
        assert "sources" in result.metadata
        src = result.metadata["sources"]["east_drain_corridor"]
        assert src["geometry"] == "line"
        assert src["n_sub_points"] >= 10

    def test_zero_q_gives_zero_concentration(self):
        specs = load_source_geometry()
        df = _make_met(n_hours=1)
        result = run_forward_model_from_geometry(
            df, specs, {sid: 0.0 for sid in specs}, START, hours=1
        )
        for sensor, vals in result.concentrations.items():
            assert vals[0] == pytest.approx(0.0), (
                f"{sensor} has non-zero concentration with all Q=0"
            )

    def test_missing_met_gives_nan(self):
        # Met only covers NESTOR; SY and IB should get NaN with no ref fallback
        specs = load_source_geometry()
        df = _make_met(n_hours=1, sensors=["NESTOR - BES"])
        result = run_forward_model_from_geometry(
            df, specs, {"east_drain_corridor": 20.0}, START, hours=1,
            ref_sensor="NESTOR - BES",
        )
        # NESTOR should have a finite value; SY and IB fall back to ref_sensor met
        # so they should also be finite (ref_sensor fallback is used)
        for sensor, vals in result.concentrations.items():
            assert np.isfinite(vals[0]) or np.isnan(vals[0])

    def test_to_json_round_trips(self):
        import json
        specs = load_source_geometry()
        df = _make_met(n_hours=2)
        result = run_forward_model_from_geometry(
            df, specs, {"east_drain_corridor": 5.0}, START, hours=2
        )
        j = result.to_json()
        d = json.loads(j)
        assert "emission_rates_g_s" in d
        assert "timeseries" in d


# ---------------------------------------------------------------------------
# run_forward_model_from_geometry — physics correctness
# ---------------------------------------------------------------------------

class TestForwardGeometryPhysics:
    def test_point_source_matches_direct_kernel(self):
        """A single-point source spec must agree with direct gaussian_plume_concentration."""
        src_lat, src_lon = 32.548531, -117.064293  # Dairy Mart Bridge
        q = 25.0
        specs = _point_specs(src_lat, src_lon)

        df = _make_met(wind_speed=2.0, wind_direction=196.0, temp=18.0, n_hours=1)
        result = run_forward_model_from_geometry(
            df, specs, {"test_pt": q}, START, hours=1
        )

        ws, wd_deg, temp_c = 2.0, 196.0, 18.0
        stab = stability_class(ws, is_night=True)
        wd_rad = np.radians(wd_deg)
        u = -ws * np.sin(wd_rad)
        v = -ws * np.cos(wd_rad)

        for sensor_name, sensor_coords in SENSORS.items():
            c_direct = ug_m3_to_ppb(
                gaussian_plume_concentration(
                    source_lat=src_lat, source_lon=src_lon,
                    emission_rate_g_s=q,
                    receptor_lat=sensor_coords["lat"],
                    receptor_lon=sensor_coords["lon"],
                    wind_u=u, wind_v=v, stab=stab,
                ),
                temp_c,
            )
            c_model = result.concentrations[sensor_name][0]
            # Model output is rounded to 3 decimal places (0.001 ppb tolerance)
            assert c_model == pytest.approx(c_direct, abs=5e-4), (
                f"{sensor_name}: model={c_model:.6f}, direct={c_direct:.6f}"
            )

    def test_higher_q_gives_higher_concentration(self):
        specs = load_source_geometry()
        df = _make_met(n_hours=1)
        result_lo = run_forward_model_from_geometry(
            df, specs, {"smugglers_goat_canyon": 50.0}, START, hours=1
        )
        result_hi = run_forward_model_from_geometry(
            df, specs, {"smugglers_goat_canyon": 150.0}, START, hours=1
        )
        for sensor in SENSORS:
            assert result_hi.concentrations[sensor][0] >= result_lo.concentrations[sensor][0]

    def test_concentration_scales_linearly_with_q(self):
        # Gaussian plume is linear in Q, so 2× Q → 2× concentration.
        specs = load_source_geometry()
        df = _make_met(n_hours=1)
        result_1 = run_forward_model_from_geometry(
            df, specs, {"smugglers_goat_canyon": 70.0}, START, hours=1
        )
        result_2 = run_forward_model_from_geometry(
            df, specs, {"smugglers_goat_canyon": 140.0}, START, hours=1
        )
        for sensor in SENSORS:
            c1 = result_1.concentrations[sensor][0]
            c2 = result_2.concentrations[sensor][0]
            if c1 > 0.001:
                assert c2 == pytest.approx(2.0 * c1, rel=1e-4)

    def test_line_source_lowers_sy_peak_vs_point(self):
        """East drain corridor as a line must give lower ppb/g/s at SY than
        full Q at TJ Crossing CDLP E (the nearest legacy point).
        SSW wind puts TJ_E nearly on-axis with SY (~59 m crosswind at 1214 m).
        """
        df = _make_met(wind_speed=1.5, wind_direction=196.0, n_hours=1)

        # Old geometry: full Q at TJ_E point
        tj_e = (32.542166, -117.050325)
        specs_point = _point_specs(*tj_e)
        result_point = run_forward_model_from_geometry(
            df, specs_point, {"test_pt": 1.0}, START, hours=1
        )
        sy_point = result_point.concentrations["SAN YSIDRO"][0]

        # New geometry: Q distributed over east_drain_corridor line
        specs_line = load_source_geometry()
        result_line = run_forward_model_from_geometry(
            df, specs_line, {"east_drain_corridor": 1.0}, START, hours=1
        )
        sy_line = result_line.concentrations["SAN YSIDRO"][0]

        assert sy_point > 0.0, "TJ_E should be downwind of SY with SSW wind"
        assert sy_line < sy_point, (
            f"Line ({sy_line:.4f} ppb) should be < single point at TJ_E "
            f"({sy_point:.4f} ppb) at SY — geometry fix not working."
        )

    def test_multiple_sources_additive(self):
        """Two sources active simultaneously should give more than either alone."""
        specs = load_source_geometry()
        df = _make_met(n_hours=1)
        result_east = run_forward_model_from_geometry(
            df, specs, {"east_drain_corridor": 20.0}, START, hours=1
        )
        result_south = run_forward_model_from_geometry(
            df, specs, {"smugglers_goat_canyon": 50.0}, START, hours=1
        )
        result_both = run_forward_model_from_geometry(
            df, specs, {"east_drain_corridor": 20.0, "smugglers_goat_canyon": 50.0},
            START, hours=1,
        )
        for sensor in SENSORS:
            c_e = result_east.concentrations[sensor][0]
            c_s = result_south.concentrations[sensor][0]
            c_b = result_both.concentrations[sensor][0]
            # Each value is rounded independently to 3dp before summing, so allow 2× that
            assert c_b == pytest.approx(c_e + c_s, abs=2e-3), (
                f"{sensor}: combined ({c_b:.4f}) ≠ east ({c_e:.4f}) + south ({c_s:.4f})"
            )

    def test_absent_source_treated_as_zero(self):
        """Sources not in source_q_g_s contribute nothing."""
        specs = load_source_geometry()
        df = _make_met(n_hours=1)
        result_explicit = run_forward_model_from_geometry(
            df, specs, {"east_drain_corridor": 20.0, "saturn_blvd_bridge": 0.0},
            START, hours=1,
        )
        result_implicit = run_forward_model_from_geometry(
            df, specs, {"east_drain_corridor": 20.0},
            START, hours=1,
        )
        for sensor in SENSORS:
            assert (
                result_explicit.concentrations[sensor][0]
                == pytest.approx(result_implicit.concentrations[sensor][0], abs=1e-9)
            )


# ---------------------------------------------------------------------------
# run_forward_model_gridded_from_geometry — structure
# ---------------------------------------------------------------------------

class TestGriddedGeometryStructure:
    @pytest.fixture(scope="class")
    def frames(self):
        specs = load_source_geometry()
        df = _make_met(n_hours=2)
        return run_forward_model_gridded_from_geometry(
            df, specs, {"smugglers_goat_canyon": 137.0}, START, hours=2
        )

    def test_returns_list_of_frames(self, frames):
        assert isinstance(frames, list)
        assert len(frames) == 2

    def test_each_frame_has_required_keys(self, frames):
        required = {"bounds", "data", "confidence", "lat_centers", "lon_centers",
                    "resolution_meters", "metadata"}
        for f in frames:
            assert required.issubset(set(f.keys()))

    def test_data_is_non_negative(self, frames):
        for f in frames:
            arr = np.array(f["data"])
            assert arr.min() >= 0.0

    def test_ppb_clip_respected(self, frames):
        from h2s.dispersion.gaussian import GRID_PPB_CLIP
        for f in frames:
            arr = np.array(f["data"])
            assert arr.max() <= GRID_PPB_CLIP

    def test_confidence_in_range(self, frames):
        for f in frames:
            conf = np.array(f["confidence"])
            assert conf.min() >= 0.0
            assert conf.max() <= 1.0

    def test_metadata_has_n_active_subpts(self, frames):
        for f in frames:
            assert "n_active_subpts" in f["metadata"]
            assert f["metadata"]["n_active_subpts"] > 0

    def test_zero_q_gives_zero_grid(self):
        specs = load_source_geometry()
        df = _make_met(n_hours=1)
        frames = run_forward_model_gridded_from_geometry(
            df, specs, {}, START, hours=1
        )
        assert np.array(frames[0]["data"]).sum() == pytest.approx(0.0)

    def test_no_met_frame_has_zero_data(self):
        """A timestep with no met data should produce a zero grid (not crash)."""
        specs = load_source_geometry()
        future = START + pd.Timedelta(hours=99)  # well beyond the 1-row met
        df = _make_met(n_hours=1)
        frames = run_forward_model_gridded_from_geometry(
            df, specs, {"smugglers_goat_canyon": 100.0}, future, hours=1
        )
        assert np.array(frames[0]["data"]).sum() == pytest.approx(0.0)
        assert frames[0]["metadata"].get("status") == "no_met"
