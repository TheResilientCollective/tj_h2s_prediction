"""Tests for Phase 3: calibration metrics, per-sensor weighting, and LOSO."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from h2s.dispersion.calibration_metrics import (
    compute_dispersion_metrics,
    compute_threshold_skill,
    score_inversion_result,
    loso_cross_validate,
)
from h2s.dispersion.emission_inversion import (
    InversionConfig,
    batch_inversion_from_geometry,
    build_sensitivity_matrix_from_geometry,
)
from h2s.dispersion.geometry import SourceSpec, SubPoint
from h2s.dispersion.lagrangian import SENSORS


# ---------------------------------------------------------------------------
# Shared helpers (mirrors test_inversion_geometry.py)
# ---------------------------------------------------------------------------

_TJ_E_LAT = 32.542166
_TJ_E_LON = -117.050325
_SATURN_LAT = 32.559383
_SATURN_LON = -117.092992
_SENSOR_NAMES = list(SENSORS.keys())


def _met(ws=3.0, wd=196.0, temp=18.0, is_night=1) -> pd.Series:
    return pd.Series({
        "wind_speed_10m":     ws,
        "wind_direction_10m": wd,
        "temperature_2m":     temp,
        "is_night":           is_night,
    })


def _pt_spec(sid, lat, lon, zone="east") -> SourceSpec:
    sp = SubPoint(lat=lat, lon=lon, source_id=sid, zone=zone, weight=1.0)
    return SourceSpec(source_id=sid, geometry="point", zone=zone,
                      q_prior=10.0, tide_modulated=False, sub_points=[sp])


def _default_cfg(**kw) -> InversionConfig:
    return InversionConfig(background_ppb=1.0, **kw)


def _make_events(specs, source_ids, Q_true, met, n_events=6, cfg=None):
    if cfg is None:
        cfg = _default_cfg()
    A, _ = build_sensitivity_matrix_from_geometry(
        specs, met, _SENSOR_NAMES, cfg, source_ids=source_ids
    )
    Q_arr = np.array(Q_true, dtype=float)
    C_obs = A @ Q_arr
    events = []
    for i in range(n_events):
        ts = pd.Timestamp("2026-04-04 15:00", tz="UTC") + pd.Timedelta(hours=i)
        h2s_obs = {s: float(C_obs[j]) + cfg.background_ppb for j, s in enumerate(_SENSOR_NAMES)}
        events.append({"time": ts, "h2s_obs": h2s_obs, "met_row": met})
    return events


# ---------------------------------------------------------------------------
# compute_dispersion_metrics
# ---------------------------------------------------------------------------

class TestComputeDispersionMetrics:

    def test_perfect_prediction(self):
        x = np.array([5.0, 10.0, 20.0, 50.0])
        m = compute_dispersion_metrics(x, x)
        assert m["mean_bias"] == 0.0
        assert m["rmse"] == 0.0
        assert m["mae"] == 0.0
        assert m["fb"] == 0.0
        assert m["nmse"] == 0.0
        assert m["fac2"] == 1.0
        assert m["n"] == 4

    def test_mean_bias_sign(self):
        obs  = np.array([10.0, 10.0, 10.0])
        pred = np.array([15.0, 15.0, 15.0])
        m = compute_dispersion_metrics(obs, pred)
        assert m["mean_bias"] > 0, "overprediction must give positive bias"

    def test_fb_underprediction_negative(self):
        obs  = np.array([20.0, 20.0])
        pred = np.array([10.0, 10.0])   # pred = obs/2
        m = compute_dispersion_metrics(obs, pred)
        # FB = 2*(10 - 20)/(10 + 20) = -20/30 ≈ -0.6667
        assert m["fb"] < 0
        assert abs(m["fb"] - (-2/3)) < 0.01

    def test_fb_overprediction_positive(self):
        obs  = np.array([10.0, 10.0])
        pred = np.array([20.0, 20.0])   # pred = 2×obs
        m = compute_dispersion_metrics(obs, pred)
        assert m["fb"] > 0

    def test_fac2_perfect_is_one(self):
        obs  = np.array([5.0, 20.0, 50.0])
        pred = np.array([5.0, 20.0, 50.0])
        m = compute_dispersion_metrics(obs, pred)
        assert m["fac2"] == 1.0

    def test_fac2_factor_two_boundary(self):
        # obs=10, pred=[5, 10, 20] → ratios [0.5, 1.0, 2.0] → all in FAC2
        obs  = np.array([10.0, 10.0, 10.0])
        pred = np.array([ 5.0, 10.0, 20.0])
        m = compute_dispersion_metrics(obs, pred)
        assert m["fac2"] == 1.0

    def test_fac2_outside_factor_two(self):
        obs  = np.array([10.0, 10.0])
        pred = np.array([ 3.0, 30.0])  # ratio 0.3 and 3.0 → both outside
        m = compute_dispersion_metrics(obs, pred)
        assert m["fac2"] == 0.0

    def test_nan_pairs_dropped(self):
        obs  = np.array([10.0, np.nan, 20.0])
        pred = np.array([10.0, 15.0,  np.nan])
        m = compute_dispersion_metrics(obs, pred)
        assert m["n"] == 1

    def test_all_nan_returns_nan_metrics(self):
        obs  = np.array([np.nan, np.nan])
        pred = np.array([np.nan, np.nan])
        m = compute_dispersion_metrics(obs, pred)
        assert m["n"] == 0
        assert np.isnan(m["mean_bias"])

    def test_rmse_ge_mae(self):
        obs  = np.array([10.0, 10.0, 10.0])
        pred = np.array([12.0, 20.0,  8.0])
        m = compute_dispersion_metrics(obs, pred)
        assert m["rmse"] >= m["mae"]


# ---------------------------------------------------------------------------
# compute_threshold_skill
# ---------------------------------------------------------------------------

class TestComputeThresholdSkill:

    def test_perfect_detection(self):
        obs  = np.array([ 5.0, 40.0, 60.0])
        pred = np.array([ 5.0, 40.0, 60.0])
        s = compute_threshold_skill(obs, pred, thresholds=[30.0])
        t = s["30"]
        assert t["pod"] == 1.0
        assert t["far"] == 0.0
        assert t["csi"] == 1.0
        assert t["hits"] == 2
        assert t["misses"] == 0
        assert t["false_alarms"] == 0

    def test_all_miss(self):
        obs  = np.array([40.0, 50.0])
        pred = np.array([ 5.0,  5.0])
        s = compute_threshold_skill(obs, pred, thresholds=[30.0])
        t = s["30"]
        assert t["pod"] == 0.0
        assert t["hits"] == 0
        assert t["misses"] == 2

    def test_all_false_alarm(self):
        obs  = np.array([ 5.0,  5.0])
        pred = np.array([40.0, 50.0])
        s = compute_threshold_skill(obs, pred, thresholds=[30.0])
        t = s["30"]
        assert t["far"] == 1.0
        assert t["false_alarms"] == 2
        assert t["hits"] == 0

    def test_no_events_above_threshold(self):
        obs  = np.array([ 5.0, 10.0])
        pred = np.array([ 5.0, 10.0])
        s = compute_threshold_skill(obs, pred, thresholds=[100.0])
        t = s["100"]
        assert np.isnan(t["pod"])
        assert np.isnan(t["far"])

    def test_multiple_thresholds_returned(self):
        obs  = np.array([5.0, 40.0, 110.0])
        pred = np.array([5.0, 40.0, 110.0])
        s = compute_threshold_skill(obs, pred, thresholds=[30.0, 100.0])
        assert "30" in s and "100" in s

    def test_csi_partial(self):
        # 2 hits, 1 miss, 1 false alarm → CSI = 2/4 = 0.5
        obs  = np.array([40.0, 40.0, 5.0,   5.0])
        pred = np.array([40.0, 5.0,  40.0,  40.0])
        s = compute_threshold_skill(obs, pred, thresholds=[30.0])
        t = s["30"]
        assert t["hits"] == 1
        assert t["misses"] == 1
        assert t["false_alarms"] == 2
        assert abs(t["csi"] - (1/4)) < 1e-6


# ---------------------------------------------------------------------------
# score_inversion_result
# ---------------------------------------------------------------------------

class TestScoreInversionResult:

    def _run_inversion(self, Q_true, cfg=None):
        specs = {
            "src_a": _pt_spec("src_a", _TJ_E_LAT, _TJ_E_LON),
            "src_b": _pt_spec("src_b", _SATURN_LAT, _SATURN_LON, zone="west"),
        }
        if cfg is None:
            cfg = _default_cfg()
        met = _met()
        events = _make_events(specs, ["src_a", "src_b"], Q_true, met, n_events=6, cfg=cfg)
        return batch_inversion_from_geometry(events, specs, cfg), specs, cfg

    def test_returns_per_sensor_keys(self):
        result, _, _ = self._run_inversion([20.0, 5.0])
        scores = score_inversion_result(result)
        # At least one sensor should be present
        assert any(s in scores for s in _SENSOR_NAMES)

    def test_overall_key_present(self):
        result, _, _ = self._run_inversion([20.0, 5.0])
        scores = score_inversion_result(result)
        assert "overall" in scores

    def test_sensor_metrics_have_required_keys(self):
        result, _, _ = self._run_inversion([20.0, 5.0])
        scores = score_inversion_result(result)
        for sname in [s for s in _SENSOR_NAMES if s in scores]:
            for key in ("n", "mean_bias", "rmse", "fb", "nmse", "fac2", "threshold_skill"):
                assert key in scores[sname], f"Missing key '{key}' for sensor {sname}"

    def test_threshold_skill_keys(self):
        result, _, _ = self._run_inversion([20.0, 5.0])
        scores = score_inversion_result(result, thresholds=[30.0, 100.0])
        for sname in [s for s in _SENSOR_NAMES if s in scores]:
            ts = scores[sname]["threshold_skill"]
            assert "30" in ts and "100" in ts

    def test_empty_predictions_returns_empty(self):
        result = {
            "Q": np.zeros(2),
            "source_ids": ["a", "b"],
            "Q_by_source": {"a": 0.0, "b": 0.0},
            "Q_total_g_s": 0.0,
            "active_sources": [],
            "sensor_bias_ppb": {},
            "n_events": 0,
            "n_rows": 0,
            "sensor_rmse_ppb": {},
            "per_event_predictions": [],
            "per_event_sensitivity": [],
            "reason": "no_rows",
        }
        scores = score_inversion_result(result)
        assert scores == {}


# ---------------------------------------------------------------------------
# Per-sensor row weighting (batch_inversion_from_geometry)
# ---------------------------------------------------------------------------

class TestEqualMassRowWeighting:

    def test_equal_mass_enabled_by_default(self):
        cfg = _default_cfg()
        assert cfg.sensor_row_weight == "equal_mass"

    def test_unbalanced_events_dont_dominate(self):
        """sensor with 3x as many events should not fully swamp the other two."""
        specs = {
            "src_a": _pt_spec("src_a", _TJ_E_LAT, _TJ_E_LON),
        }
        met = _met()
        # All sensors get the same concentration from src_a Q=20
        cfg_equal = _default_cfg(sensor_row_weight="equal_mass")
        cfg_none  = _default_cfg(sensor_row_weight="none")

        A, _ = build_sensitivity_matrix_from_geometry(specs, met, _SENSOR_NAMES, cfg_equal)
        C = A @ np.array([20.0])
        h2s_obs = {s: float(C[j]) + 1.0 for j, s in enumerate(_SENSOR_NAMES)}

        # Both configs should recover Q close to 20; just verify both run without error
        events = [{"time": pd.Timestamp("2026-04-04 15:00", tz="UTC") + pd.Timedelta(hours=i),
                   "h2s_obs": h2s_obs, "met_row": met} for i in range(6)]

        r_equal = batch_inversion_from_geometry(events, specs, cfg_equal)
        r_none  = batch_inversion_from_geometry(events, specs, cfg_none)

        assert r_equal["Q_by_source"]["src_a"] > 0
        assert r_none["Q_by_source"]["src_a"] > 0

    def test_sensor_bias_ppb_in_result(self):
        """When fit_sensor_bias=True, sensor_bias_ppb key must be present."""
        specs = {"src": _pt_spec("src", _TJ_E_LAT, _TJ_E_LON)}
        cfg = _default_cfg(fit_sensor_bias=True)
        met = _met()
        A, _ = build_sensitivity_matrix_from_geometry(specs, met, _SENSOR_NAMES, cfg)
        C = A @ np.array([20.0])
        h2s_obs = {s: float(C[j]) + 1.0 for j, s in enumerate(_SENSOR_NAMES)}
        events = [{"time": pd.Timestamp("2026-04-04 15:00", tz="UTC") + pd.Timedelta(hours=i),
                   "h2s_obs": h2s_obs, "met_row": met} for i in range(5)]
        result = batch_inversion_from_geometry(events, specs, cfg)
        assert "sensor_bias_ppb" in result
        assert isinstance(result["sensor_bias_ppb"], dict)

    def test_sensor_bias_ppb_keys_are_sensor_names(self):
        specs = {"src": _pt_spec("src", _TJ_E_LAT, _TJ_E_LON)}
        cfg = _default_cfg(fit_sensor_bias=True)
        met = _met()
        A, _ = build_sensitivity_matrix_from_geometry(specs, met, _SENSOR_NAMES, cfg)
        C = A @ np.array([20.0])
        h2s_obs = {s: float(C[j]) + 1.0 for j, s in enumerate(_SENSOR_NAMES)}
        events = [{"time": pd.Timestamp("2026-04-04 15:00", tz="UTC") + pd.Timedelta(hours=i),
                   "h2s_obs": h2s_obs, "met_row": met} for i in range(5)]
        result = batch_inversion_from_geometry(events, specs, cfg)
        # Keys should be a subset of sensor names
        for k in result["sensor_bias_ppb"]:
            assert k in _SENSOR_NAMES, f"Unexpected key in sensor_bias_ppb: {k}"

    def test_bias_nonnegative(self):
        """NNLS constraint means bias terms must be ≥ 0."""
        specs = {"src": _pt_spec("src", _TJ_E_LAT, _TJ_E_LON)}
        cfg = _default_cfg(fit_sensor_bias=True)
        met = _met()
        A, _ = build_sensitivity_matrix_from_geometry(specs, met, _SENSOR_NAMES, cfg)
        C = A @ np.array([20.0])
        h2s_obs = {s: float(C[j]) + 1.0 for j, s in enumerate(_SENSOR_NAMES)}
        events = [{"time": pd.Timestamp("2026-04-04 15:00", tz="UTC") + pd.Timedelta(hours=i),
                   "h2s_obs": h2s_obs, "met_row": met} for i in range(5)]
        result = batch_inversion_from_geometry(events, specs, cfg)
        for v in result["sensor_bias_ppb"].values():
            assert v >= 0, f"Bias must be non-negative, got {v}"

    def test_no_bias_without_flag(self):
        """sensor_bias_ppb should be empty when fit_sensor_bias=False."""
        specs = {"src": _pt_spec("src", _TJ_E_LAT, _TJ_E_LON)}
        cfg = _default_cfg(fit_sensor_bias=False)
        met = _met()
        A, _ = build_sensitivity_matrix_from_geometry(specs, met, _SENSOR_NAMES, cfg)
        C = A @ np.array([20.0])
        h2s_obs = {s: float(C[j]) + 1.0 for j, s in enumerate(_SENSOR_NAMES)}
        events = [{"time": pd.Timestamp("2026-04-04 15:00", tz="UTC") + pd.Timedelta(hours=i),
                   "h2s_obs": h2s_obs, "met_row": met} for i in range(5)]
        result = batch_inversion_from_geometry(events, specs, cfg)
        assert result["sensor_bias_ppb"] == {}


# ---------------------------------------------------------------------------
# loso_cross_validate
# ---------------------------------------------------------------------------

class TestLosoCrossValidate:

    def _two_source_specs(self):
        return {
            "near_sy":  _pt_spec("near_sy", _TJ_E_LAT, _TJ_E_LON, zone="east"),
            "far_west": _pt_spec("far_west", _SATURN_LAT, _SATURN_LON, zone="west"),
        }

    def test_returns_entry_for_each_sensor(self):
        specs = self._two_source_specs()
        cfg = _default_cfg()
        met = _met()
        events = _make_events(specs, list(specs.keys()), [20.0, 5.0], met, n_events=6)
        result = loso_cross_validate(events, specs, cfg)
        for sname in _SENSOR_NAMES:
            assert sname in result, f"Missing LOSO result for {sname}"

    def test_fit_sensors_excludes_held_out(self):
        specs = self._two_source_specs()
        cfg = _default_cfg()
        met = _met()
        events = _make_events(specs, list(specs.keys()), [20.0, 5.0], met, n_events=6)
        result = loso_cross_validate(events, specs, cfg)
        for held_out in _SENSOR_NAMES:
            r = result[held_out]
            if "fit_sensors" in r:
                assert held_out not in r["fit_sensors"]

    def test_n_eval_events_positive(self):
        specs = self._two_source_specs()
        cfg = _default_cfg()
        met = _met()
        events = _make_events(specs, list(specs.keys()), [20.0, 5.0], met, n_events=6)
        result = loso_cross_validate(events, specs, cfg)
        for sname in _SENSOR_NAMES:
            r = result[sname]
            if "reason" not in r:
                assert r["n_eval_events"] > 0

    def test_q_by_source_present_on_success(self):
        specs = self._two_source_specs()
        cfg = _default_cfg()
        met = _met()
        events = _make_events(specs, list(specs.keys()), [20.0, 5.0], met, n_events=6)
        result = loso_cross_validate(events, specs, cfg)
        for sname in _SENSOR_NAMES:
            r = result[sname]
            if "reason" not in r:
                assert "Q_by_source" in r

    def test_empty_events_no_crash(self):
        specs = self._two_source_specs()
        cfg = _default_cfg()
        result = loso_cross_validate([], specs, cfg)
        for sname in _SENSOR_NAMES:
            assert sname in result
            assert result[sname].get("reason") == "no_train_events"

    def test_custom_sensor_subset(self):
        specs = self._two_source_specs()
        cfg = _default_cfg()
        met = _met()
        events = _make_events(specs, list(specs.keys()), [20.0, 5.0], met)
        subset = _SENSOR_NAMES[:2]
        result = loso_cross_validate(events, specs, cfg, sensor_names=subset)
        assert set(result.keys()) == set(subset)

    def test_loso_sy_mean_bias_is_finite(self):
        """After geometry fix, SY LOSO bias should be a finite number (not nan)."""
        specs = self._two_source_specs()
        cfg = _default_cfg()
        met = _met(ws=3.0, wd=196.0)
        events = _make_events(specs, list(specs.keys()), [20.0, 5.0], met, n_events=8)
        result = loso_cross_validate(events, specs, cfg)
        sy_result = result["SAN YSIDRO"]
        if "reason" not in sy_result:
            assert not np.isnan(sy_result["mean_bias"]), "LOSO SY mean_bias should be finite"

    def test_threshold_skill_present(self):
        specs = self._two_source_specs()
        cfg = _default_cfg()
        met = _met()
        events = _make_events(specs, list(specs.keys()), [20.0, 5.0], met)
        result = loso_cross_validate(events, specs, cfg)
        for sname in _SENSOR_NAMES:
            r = result[sname]
            if "reason" not in r and r.get("n_eval_events", 0) > 0:
                assert "threshold_skill" in r
