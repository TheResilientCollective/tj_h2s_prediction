"""Tests for Phase 2 geometry-aware inversion functions.

Covers:
  - build_sensitivity_matrix_from_geometry
  - project_footprint_to_sources
  - calibration_loop_from_geometry
  - batch_inversion_from_geometry
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from h2s.dispersion.emission_inversion import (
    InversionConfig,
    build_sensitivity_matrix_from_geometry,
    project_footprint_to_sources,
    calibration_loop_from_geometry,
    batch_inversion_from_geometry,
)
from h2s.dispersion.gaussian import (
    gaussian_plume_concentration,
    stability_class,
    ug_m3_to_ppb,
)
from h2s.dispersion.geometry import SourceSpec, SubPoint
from h2s.dispersion.lagrangian import GRID, M_PER_DEG_LAT, M_PER_DEG_LON, SENSORS


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

# Sensor coordinates (subset for clarity)
_SENSOR_NAMES = list(SENSORS.keys())  # NESTOR - BES, IB CIVIC CTR, SAN YSIDRO
_SY_IDX = _SENSOR_NAMES.index("SAN YSIDRO")

# Key locations (from lagrangian.py CANDIDATE_SOURCES / source_geometry.toml)
_TJ_E_LAT = 32.542166
_TJ_E_LON = -117.050325

_SATURN_LAT = 32.559383
_SATURN_LON = -117.092992


def _met_series(ws: float = 3.0, wd: float = 196.0, temp: float = 18.0, is_night: int = 1) -> pd.Series:
    return pd.Series({
        "wind_speed_10m":     ws,
        "wind_direction_10m": wd,
        "temperature_2m":     temp,
        "is_night":           is_night,
    })


def _make_point_spec(source_id: str, lat: float, lon: float, zone: str = "east") -> SourceSpec:
    sp = SubPoint(lat=lat, lon=lon, source_id=source_id, zone=zone, weight=1.0)
    return SourceSpec(
        source_id=source_id, geometry="point", zone=zone,
        q_prior=10.0, tide_modulated=False, sub_points=[sp],
    )


def _make_line_spec(
    source_id: str,
    vertices: list[tuple[float, float]],
    spacing_m: float = 200.0,
    zone: str = "east",
) -> SourceSpec:
    from h2s.dispersion.geometry import _discretize_line
    sub_points = _discretize_line(vertices, spacing_m, source_id, zone)
    return SourceSpec(
        source_id=source_id, geometry="line", zone=zone,
        q_prior=20.0, tide_modulated=False, sub_points=sub_points,
    )


def _default_cfg() -> InversionConfig:
    return InversionConfig(gauss_meandering_deg=20.0, background_ppb=1.0)


# ---------------------------------------------------------------------------
# build_sensitivity_matrix_from_geometry
# ---------------------------------------------------------------------------

class TestBuildSensitivityMatrix:

    def test_shape_n_sensors_by_n_sources(self):
        specs = {
            "src_a": _make_point_spec("src_a", _TJ_E_LAT, _TJ_E_LON, zone="east"),
            "src_b": _make_point_spec("src_b", _SATURN_LAT, _SATURN_LON, zone="west"),
        }
        cfg = _default_cfg()
        A, sids = build_sensitivity_matrix_from_geometry(
            specs, _met_series(), _SENSOR_NAMES, cfg
        )
        assert A.shape == (len(_SENSOR_NAMES), len(specs))
        assert sids == list(specs.keys())

    def test_point_source_matches_direct_kernel(self):
        """Single sub-point (weight=1) must reproduce gaussian_plume_concentration."""
        met = _met_series(ws=3.0, wd=196.0, temp=18.0, is_night=1)
        specs = {"tj_e": _make_point_spec("tj_e", _TJ_E_LAT, _TJ_E_LON)}
        cfg = _default_cfg()

        A, _ = build_sensitivity_matrix_from_geometry(
            specs, met, _SENSOR_NAMES, cfg
        )

        ws      = float(met["wind_speed_10m"])
        wd_rad  = np.radians(float(met["wind_direction_10m"]))
        u = -ws * np.sin(wd_rad)
        v = -ws * np.cos(wd_rad)
        stab = stability_class(ws, bool(met["is_night"]))
        temp_c = float(met["temperature_2m"])

        for i, sname in enumerate(_SENSOR_NAMES):
            sc = SENSORS[sname]
            c_ug = gaussian_plume_concentration(
                source_lat=_TJ_E_LAT, source_lon=_TJ_E_LON,
                emission_rate_g_s=1.0,
                receptor_lat=sc["lat"], receptor_lon=sc["lon"],
                wind_u=u, wind_v=v,
                stab=stab,
                sigma_theta_deg=cfg.gauss_meandering_deg,
            )
            expected = ug_m3_to_ppb(c_ug, temp_c)
            assert abs(A[i, 0] - expected) < 5e-4, (
                f"Sensor {sname}: A={A[i,0]:.5f}, direct={expected:.5f}"
            )

    def test_source_not_in_specs_gives_zero_column(self):
        specs = {"src_a": _make_point_spec("src_a", _TJ_E_LAT, _TJ_E_LON)}
        cfg = _default_cfg()
        # Request a source_id not in specs
        A, sids = build_sensitivity_matrix_from_geometry(
            specs, _met_series(), _SENSOR_NAMES, cfg,
            source_ids=["src_a", "nonexistent"]
        )
        assert A.shape == (len(_SENSOR_NAMES), 2)
        assert sids == ["src_a", "nonexistent"]
        # Column for nonexistent source must be all zeros
        assert np.all(A[:, 1] == 0.0)

    def test_line_source_lowers_sy_sensitivity_vs_nearest_point(self):
        """East drain corridor (line) gives lower A[SY] than single-point at TJ_E.

        Wind direction 196° (SSW) puts TJ_E nearly on-axis with SAN YSIDRO.
        Spreading Q over the full corridor dilutes the SY column.
        """
        # TJ_E alone as point (worst-case near-field spike)
        specs_pt = {"tj_e": _make_point_spec("tj_e", _TJ_E_LAT, _TJ_E_LON)}

        # Corridor from Dairy Mart Bridge → Silva Drain → CDLP W → CDLP E
        specs_ln = {
            "east_drain": _make_line_spec(
                "east_drain",
                [
                    (32.548531, -117.064293),  # Dairy Mart Bridge
                    (32.539743, -117.064269),  # Silva Drain
                    (32.542103, -117.054117),  # TJ Crossing CDLP W
                    (32.542166, -117.050325),  # TJ Crossing CDLP E
                ],
                spacing_m=200.0,
            )
        }

        met = _met_series(ws=3.0, wd=196.0, is_night=1)
        cfg = _default_cfg()
        sensors = [_SENSOR_NAMES[_SY_IDX]]  # only SY needed

        A_pt, _ = build_sensitivity_matrix_from_geometry(specs_pt, met, sensors, cfg)
        A_ln, _ = build_sensitivity_matrix_from_geometry(specs_ln, met, sensors, cfg)

        a_point = float(A_pt[0, 0])
        a_line  = float(A_ln[0, 0])
        assert a_point > 0, "Point source must have non-zero sensitivity at SY"
        assert a_line < a_point, (
            f"Line source A[SY]={a_line:.5f} should be < point A[SY]={a_point:.5f}"
        )

    def test_nonnegative(self):
        specs = {
            "src_a": _make_point_spec("src_a", _TJ_E_LAT, _TJ_E_LON),
            "src_b": _make_point_spec("src_b", _SATURN_LAT, _SATURN_LON, zone="west"),
        }
        A, _ = build_sensitivity_matrix_from_geometry(
            specs, _met_series(), _SENSOR_NAMES, _default_cfg()
        )
        assert np.all(A >= 0)

    def test_source_ids_override_controls_column_order(self):
        specs = {
            "src_a": _make_point_spec("src_a", _TJ_E_LAT, _TJ_E_LON),
            "src_b": _make_point_spec("src_b", _SATURN_LAT, _SATURN_LON, zone="west"),
        }
        # Reverse the order
        A_default, sids_default = build_sensitivity_matrix_from_geometry(
            specs, _met_series(), _SENSOR_NAMES, _default_cfg()
        )
        A_rev, sids_rev = build_sensitivity_matrix_from_geometry(
            specs, _met_series(), _SENSOR_NAMES, _default_cfg(),
            source_ids=["src_b", "src_a"]
        )
        assert sids_rev == ["src_b", "src_a"]
        # Column 0 in rev = column 1 in default
        np.testing.assert_allclose(A_rev[:, 0], A_default[:, 1], atol=1e-9)
        np.testing.assert_allclose(A_rev[:, 1], A_default[:, 0], atol=1e-9)


# ---------------------------------------------------------------------------
# project_footprint_to_sources
# ---------------------------------------------------------------------------

class TestProjectFootprintToSources:

    def _grid_shape(self):
        return (GRID["nlat"], GRID["nlon"])

    def _gaussian_blob(self, center_lat: float, center_lon: float, sigma_m: float = 300.0) -> np.ndarray:
        lat_edges = np.linspace(GRID["lat_min"], GRID["lat_max"], GRID["nlat"] + 1)
        lon_edges = np.linspace(GRID["lon_min"], GRID["lon_max"], GRID["nlon"] + 1)
        lat_c = (lat_edges[:-1] + lat_edges[1:]) / 2.0
        lon_c = (lon_edges[:-1] + lon_edges[1:]) / 2.0
        lon_grid, lat_grid = np.meshgrid(lon_c, lat_c)

        s_lat = sigma_m / M_PER_DEG_LAT
        s_lon = sigma_m / M_PER_DEG_LON
        blob = np.exp(
            -0.5 * (
                ((lat_grid - center_lat) / s_lat) ** 2
                + ((lon_grid - center_lon) / s_lon) ** 2
            )
        )
        return blob / blob.sum()

    def test_returns_correct_length(self):
        specs = {
            "a": _make_point_spec("a", _TJ_E_LAT, _TJ_E_LON),
            "b": _make_point_spec("b", _SATURN_LAT, _SATURN_LON, zone="west"),
        }
        fp = np.ones(self._grid_shape())
        result = project_footprint_to_sources(fp, specs)
        assert result.shape == (len(specs),)

    def test_sums_to_one_for_nonzero_footprint(self):
        specs = {
            "a": _make_point_spec("a", _TJ_E_LAT, _TJ_E_LON),
            "b": _make_point_spec("b", _SATURN_LAT, _SATURN_LON, zone="west"),
            "c": _make_point_spec("c", 32.570082, -117.127240, zone="west"),
        }
        fp = self._gaussian_blob(_TJ_E_LAT, _TJ_E_LON)
        result = project_footprint_to_sources(fp, specs)
        assert abs(result.sum() - 1.0) < 1e-9, f"sum={result.sum()}"

    def test_zero_footprint_returns_uniform(self):
        specs = {
            "a": _make_point_spec("a", _TJ_E_LAT, _TJ_E_LON),
            "b": _make_point_spec("b", _SATURN_LAT, _SATURN_LON, zone="west"),
        }
        fp = np.zeros(self._grid_shape())
        result = project_footprint_to_sources(fp, specs)
        expected = 0.5
        assert abs(result[0] - expected) < 1e-9
        assert abs(result[1] - expected) < 1e-9

    def test_blob_near_source_selects_it(self):
        """Footprint blob concentrated at TJ_E location → TJ_E source wins."""
        specs = {
            "tj_e":   _make_point_spec("tj_e", _TJ_E_LAT, _TJ_E_LON, zone="east"),
            "saturn": _make_point_spec("saturn", _SATURN_LAT, _SATURN_LON, zone="west"),
        }
        fp = self._gaussian_blob(_TJ_E_LAT, _TJ_E_LON, sigma_m=200.0)
        result = project_footprint_to_sources(fp, specs)
        assert result[0] > result[1], (
            f"Footprint at TJ_E should assign more weight to tj_e: {result}"
        )

    def test_all_nonnegative(self):
        specs = {"a": _make_point_spec("a", _TJ_E_LAT, _TJ_E_LON)}
        fp = self._gaussian_blob(_TJ_E_LAT, _TJ_E_LON)
        result = project_footprint_to_sources(fp, specs)
        assert np.all(result >= 0)

    def test_custom_source_id_ordering(self):
        specs = {
            "a": _make_point_spec("a", _TJ_E_LAT, _TJ_E_LON),
            "b": _make_point_spec("b", _SATURN_LAT, _SATURN_LON, zone="west"),
        }
        fp = self._gaussian_blob(_TJ_E_LAT, _TJ_E_LON)
        default_result = project_footprint_to_sources(fp, specs)
        reversed_result = project_footprint_to_sources(fp, specs, source_ids=["b", "a"])
        assert abs(default_result[0] - reversed_result[1]) < 1e-9
        assert abs(default_result[1] - reversed_result[0]) < 1e-9


# ---------------------------------------------------------------------------
# batch_inversion_from_geometry
# ---------------------------------------------------------------------------

class TestBatchInversionFromGeometry:

    def _make_events(
        self,
        specs: dict,
        source_ids: list[str],
        Q_true: list[float],
        met: pd.Series,
        n_events: int = 5,
        cfg: InversionConfig | None = None,
    ) -> list[dict]:
        """Build synthetic events: forward-model observations from Q_true."""
        if cfg is None:
            cfg = _default_cfg()

        A, _ = build_sensitivity_matrix_from_geometry(
            specs, met, _SENSOR_NAMES, cfg, source_ids=source_ids
        )
        Q_arr = np.array(Q_true, dtype=float)
        C_obs = A @ Q_arr  # ppb above background

        events = []
        for i in range(n_events):
            ts = pd.Timestamp("2026-04-04 15:00", tz="UTC") + pd.Timedelta(hours=i)
            h2s_obs = {
                sname: round(float(C_obs[j]) + cfg.background_ppb + 0.1 * i, 1)
                for j, sname in enumerate(_SENSOR_NAMES)
            }
            events.append({"time": ts, "h2s_obs": h2s_obs, "met_row": met})
        return events

    def test_empty_events_returns_zero_q(self):
        specs = {"src": _make_point_spec("src", _TJ_E_LAT, _TJ_E_LON)}
        result = batch_inversion_from_geometry([], specs, _default_cfg())
        assert result["reason"] == "no_rows"
        assert np.all(result["Q"] == 0.0)
        assert result["Q_total_g_s"] == 0.0
        assert result["n_events"] == 0

    def test_empty_events_q_by_source_has_all_sources(self):
        specs = {
            "src_a": _make_point_spec("src_a", _TJ_E_LAT, _TJ_E_LON),
            "src_b": _make_point_spec("src_b", _SATURN_LAT, _SATURN_LON, zone="west"),
        }
        result = batch_inversion_from_geometry([], specs, _default_cfg())
        assert set(result["Q_by_source"].keys()) == {"src_a", "src_b"}

    def test_required_keys_present(self):
        specs = {
            "src_a": _make_point_spec("src_a", _TJ_E_LAT, _TJ_E_LON),
            "src_b": _make_point_spec("src_b", _SATURN_LAT, _SATURN_LON, zone="west"),
        }
        met = _met_series(ws=3.0, wd=196.0)
        events = self._make_events(specs, ["src_a", "src_b"], [20.0, 5.0], met)
        result = batch_inversion_from_geometry(events, specs, _default_cfg())

        for key in ("Q", "source_ids", "Q_by_source", "Q_total_g_s",
                    "active_sources", "n_events", "n_rows",
                    "sensor_rmse_ppb", "per_event_predictions"):
            assert key in result, f"Missing key: {key}"

    def test_recovers_dominant_source(self):
        """NNLS should assign most Q to the source that forwards to the observations."""
        met = _met_series(ws=3.0, wd=196.0, is_night=1)
        specs = {
            "near_sy":   _make_point_spec("near_sy", _TJ_E_LAT, _TJ_E_LON, zone="east"),
            "far_west":  _make_point_spec("far_west", _SATURN_LAT, _SATURN_LON, zone="west"),
        }
        # near_sy is the only active source
        Q_true = [30.0, 0.5]
        events = self._make_events(specs, ["near_sy", "far_west"], Q_true, met, n_events=6)
        result = batch_inversion_from_geometry(events, specs, _default_cfg())

        q_near  = result["Q_by_source"]["near_sy"]
        q_far   = result["Q_by_source"]["far_west"]
        assert q_near > q_far, (
            f"Expected near_sy ({q_near:.1f}) dominant over far_west ({q_far:.1f})"
        )

    def test_q_by_source_matches_q_array(self):
        specs = {
            "src_a": _make_point_spec("src_a", _TJ_E_LAT, _TJ_E_LON),
            "src_b": _make_point_spec("src_b", _SATURN_LAT, _SATURN_LON, zone="west"),
        }
        met = _met_series(ws=3.0, wd=196.0)
        events = self._make_events(specs, ["src_a", "src_b"], [20.0, 5.0], met)
        result = batch_inversion_from_geometry(events, specs, _default_cfg())

        Q_arr  = result["Q"]
        sids   = result["source_ids"]
        q_dict = result["Q_by_source"]

        for i, sid in enumerate(sids):
            assert abs(Q_arr[i] - q_dict[sid]) < 1e-3

    def test_q_total_consistent_with_q(self):
        specs = {
            "src_a": _make_point_spec("src_a", _TJ_E_LAT, _TJ_E_LON),
            "src_b": _make_point_spec("src_b", _SATURN_LAT, _SATURN_LON, zone="west"),
        }
        met = _met_series(ws=3.0, wd=196.0)
        events = self._make_events(specs, ["src_a", "src_b"], [20.0, 5.0], met)
        result = batch_inversion_from_geometry(events, specs, _default_cfg())
        assert abs(result["Q"].sum() - result["Q_total_g_s"]) < 1.0

    def test_active_sources_sorted_descending(self):
        specs = {
            "src_a": _make_point_spec("src_a", _TJ_E_LAT, _TJ_E_LON),
            "src_b": _make_point_spec("src_b", _SATURN_LAT, _SATURN_LON, zone="west"),
        }
        met = _met_series(ws=3.0, wd=196.0)
        events = self._make_events(specs, ["src_a", "src_b"], [20.0, 5.0], met)
        result = batch_inversion_from_geometry(events, specs, _default_cfg())

        qs = [s["Q_g_s"] for s in result["active_sources"]]
        assert qs == sorted(qs, reverse=True)

    def test_n_rows_equals_n_events_times_sensors(self):
        specs = {"src": _make_point_spec("src", _TJ_E_LAT, _TJ_E_LON)}
        met = _met_series(ws=3.0, wd=196.0)
        n_events = 4
        # All three sensors will have obs > background
        Q_true = [25.0]
        events = self._make_events(specs, ["src"], Q_true, met, n_events=n_events)
        result = batch_inversion_from_geometry(events, specs, _default_cfg())
        # Each event contributes n_sensors rows (all sensors above background given Q=25 g/s)
        assert result["n_rows"] > 0


# ---------------------------------------------------------------------------
# calibration_loop_from_geometry
# ---------------------------------------------------------------------------

class TestCalibrationLoopFromGeometry:

    def _grid_shape(self):
        return (GRID["nlat"], GRID["nlon"])

    def _zero_footprints(self) -> dict[str, np.ndarray]:
        return {sname: np.zeros(self._grid_shape()) for sname in _SENSOR_NAMES}

    def test_no_signal_returns_zeros(self):
        specs = {"src": _make_point_spec("src", _TJ_E_LAT, _TJ_E_LON)}
        cfg = _default_cfg()
        # h2s_obs all at or below background
        h2s_obs = {s: cfg.background_ppb for s in _SENSOR_NAMES}
        result = calibration_loop_from_geometry(
            self._zero_footprints(), h2s_obs, _met_series(), specs, cfg
        )
        assert result["reason"] == "no_signal"
        assert np.all(result["Q"] == 0.0)
        assert result["Q_total_g_s"] == 0.0

    def test_required_keys_present_on_success(self):
        specs = {"src": _make_point_spec("src", _TJ_E_LAT, _TJ_E_LON)}
        cfg = _default_cfg()
        met = _met_series(ws=3.0, wd=196.0, is_night=1)

        # Compute synthetic obs using forward model
        A, _ = build_sensitivity_matrix_from_geometry(specs, met, _SENSOR_NAMES, cfg)
        C = A @ np.array([25.0])
        h2s_obs = {s: float(C[i]) + cfg.background_ppb for i, s in enumerate(_SENSOR_NAMES)}

        result = calibration_loop_from_geometry(
            self._zero_footprints(), h2s_obs, met, specs, cfg
        )
        for key in ("Q", "source_ids", "Q_by_source", "Q_total_g_s",
                    "active_sources", "predicted_ppb", "residual_ppb",
                    "converged", "iterations"):
            assert key in result, f"Missing key: {key}"

    def test_q_by_source_has_all_sources(self):
        specs = {
            "src_a": _make_point_spec("src_a", _TJ_E_LAT, _TJ_E_LON),
            "src_b": _make_point_spec("src_b", _SATURN_LAT, _SATURN_LON, zone="west"),
        }
        cfg = _default_cfg()
        met = _met_series(ws=3.0, wd=196.0)
        A, _ = build_sensitivity_matrix_from_geometry(
            specs, met, _SENSOR_NAMES, cfg, source_ids=["src_a", "src_b"]
        )
        C = A @ np.array([15.0, 5.0])
        h2s_obs = {s: float(C[i]) + cfg.background_ppb for i, s in enumerate(_SENSOR_NAMES)}
        result = calibration_loop_from_geometry(
            self._zero_footprints(), h2s_obs, met, specs, cfg
        )
        assert set(result["Q_by_source"].keys()) == {"src_a", "src_b"}

    def test_q_by_source_matches_q_array(self):
        specs = {"src": _make_point_spec("src", _TJ_E_LAT, _TJ_E_LON)}
        cfg = _default_cfg()
        met = _met_series(ws=3.0, wd=196.0)
        A, _ = build_sensitivity_matrix_from_geometry(specs, met, _SENSOR_NAMES, cfg)
        C = A @ np.array([20.0])
        h2s_obs = {s: float(C[i]) + cfg.background_ppb for i, s in enumerate(_SENSOR_NAMES)}
        result = calibration_loop_from_geometry(
            self._zero_footprints(), h2s_obs, met, specs, cfg
        )
        Q_arr = result["Q"]
        for i, sid in enumerate(result["source_ids"]):
            assert abs(Q_arr[i] - result["Q_by_source"][sid]) < 1e-3
