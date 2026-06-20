"""Dispersion-model skill metrics and LOSO cross-validation for the geometry inversion.

Implements the Chang & Hanna (2004) standard dispersion-model evaluation suite
(mean bias, FB, NMSE, FAC2) plus threshold skill (POD / FAR / CSI at 30 and
100 ppb) and a leave-one-station-out (LOSO) scorer that explicitly quantifies
the SAN YSIDRO overprediction symptom.

Typical usage
-------------
    result = batch_inversion_from_geometry(events, specs, cfg)
    scores  = score_inversion_result(result)
    loso    = loso_cross_validate(events, specs, cfg)
    # loso["SAN YSIDRO"]["mean_bias"] should approach 0 after geometry fix.
"""

from __future__ import annotations

import numpy as np

from h2s.dispersion.emission_inversion import (
    InversionConfig,
    batch_inversion_from_geometry,
    build_sensitivity_matrix_from_geometry,
)
from h2s.dispersion.lagrangian import SENSORS


__all__ = [
    "compute_dispersion_metrics",
    "compute_threshold_skill",
    "score_inversion_result",
    "loso_cross_validate",
]

# Minimum observed ppb accepted when computing ratios (FAC2, FB, NMSE).
# Below this the observation is treated as "trace / noise" and skipped.
_RATIO_MIN_PPB = 1.0


# ---------------------------------------------------------------------------
# Core metric functions
# ---------------------------------------------------------------------------

def compute_dispersion_metrics(
    obs: np.ndarray,
    pred: np.ndarray,
) -> dict:
    """Standard dispersion-model scalar metrics (Chang & Hanna 2004).

    Parameters
    ----------
    obs, pred:
        1-D arrays of observed and predicted concentrations (ppb), already
        including background (i.e. raw sensor values, not background-
        subtracted).  Must be the same length; NaNs are dropped pairwise.

    Returns
    -------
    dict with keys:
      - ``n``         : paired sample size after NaN drop
      - ``mean_bias`` : mean(pred − obs), ppb.  Positive → overprediction.
      - ``rmse``      : root-mean-square error, ppb.
      - ``mae``       : mean absolute error, ppb.
      - ``fb``        : fractional bias 2(C̄_p − C̄_o)/(C̄_p + C̄_o ); range [−2, 2];
                        0 = perfect; negative = underprediction.
      - ``nmse``      : normalised mean-square error mean((p−o)²)/(C̄_p·C̄_o);
                        0 = perfect; dimensionless.
      - ``fac2``      : fraction of pairs with 0.5 ≤ pred/obs ≤ 2.0
                        (obs filtered to > _RATIO_MIN_PPB); 1 = perfect.
    """
    o = np.asarray(obs, dtype=float)
    p = np.asarray(pred, dtype=float)
    mask = ~(np.isnan(o) | np.isnan(p))
    o, p = o[mask], p[mask]
    n = len(o)

    if n == 0:
        return {"n": 0, "mean_bias": float("nan"), "rmse": float("nan"),
                "mae": float("nan"), "fb": float("nan"), "nmse": float("nan"),
                "fac2": float("nan")}

    diff = p - o
    mean_bias = float(np.mean(diff))
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    mae  = float(np.mean(np.abs(diff)))

    mean_o = float(np.mean(o))
    mean_p = float(np.mean(p))
    denom_fb   = mean_p + mean_o
    fb   = float(2.0 * (mean_p - mean_o) / denom_fb) if abs(denom_fb) > 1e-9 else float("nan")
    denom_nmse = mean_o * mean_p
    nmse = float(np.mean(diff ** 2) / denom_nmse) if abs(denom_nmse) > 1e-9 else float("nan")

    ratio_mask = o > _RATIO_MIN_PPB
    if ratio_mask.sum() > 0:
        ratio = p[ratio_mask] / o[ratio_mask]
        fac2 = float(np.mean((ratio >= 0.5) & (ratio <= 2.0)))
    else:
        fac2 = float("nan")

    return {
        "n":         n,
        "mean_bias": round(mean_bias, 3),
        "rmse":      round(rmse, 3),
        "mae":       round(mae, 3),
        "fb":        round(fb, 4) if not np.isnan(fb) else float("nan"),
        "nmse":      round(nmse, 4) if not np.isnan(nmse) else float("nan"),
        "fac2":      round(fac2, 4) if not np.isnan(fac2) else float("nan"),
    }


def compute_threshold_skill(
    obs: np.ndarray,
    pred: np.ndarray,
    thresholds: list[float] | None = None,
) -> dict[str, dict]:
    """Binary threshold skill at each ppb cut (POD, FAR, CSI).

    Parameters
    ----------
    obs, pred:
        1-D arrays (ppb, including background).  NaN pairs are dropped.
    thresholds:
        ppb values at which to compute binary skill.  Default: [30.0, 100.0].

    Returns
    -------
    dict keyed by threshold (as int string, e.g. ``"30"``):
      - ``n``            : total paired samples
      - ``n_obs_events`` : observed-positive count (obs ≥ threshold)
      - ``n_pred_events``: predicted-positive count (pred ≥ threshold)
      - ``hits``         : both ≥ threshold
      - ``misses``       : obs ≥, pred <
      - ``false_alarms`` : obs <, pred ≥
      - ``pod``          : probability of detection (hits / (hits + misses)); NaN if 0 obs events
      - ``far``          : false alarm ratio (FA / (hits + FA)); NaN if 0 pred events
      - ``csi``          : critical success index hits / (hits + misses + FA)
    """
    if thresholds is None:
        thresholds = [30.0, 100.0]

    o = np.asarray(obs, dtype=float)
    p = np.asarray(pred, dtype=float)
    mask = ~(np.isnan(o) | np.isnan(p))
    o, p = o[mask], p[mask]

    out: dict[str, dict] = {}
    for thr in thresholds:
        o_pos = o >= thr
        p_pos = p >= thr
        hits = int(np.sum(o_pos & p_pos))
        misses = int(np.sum(o_pos & ~p_pos))
        false_alarms = int(np.sum(~o_pos & p_pos))
        n_obs_events = hits + misses
        n_pred_events = hits + false_alarms

        pod = (hits / n_obs_events)   if n_obs_events  > 0 else float("nan")
        far = (false_alarms / n_pred_events) if n_pred_events > 0 else float("nan")
        denom_csi = hits + misses + false_alarms
        csi = (hits / denom_csi) if denom_csi > 0 else float("nan")

        out[str(int(thr))] = {
            "n":             len(o),
            "n_obs_events":  n_obs_events,
            "n_pred_events": n_pred_events,
            "hits":          hits,
            "misses":        misses,
            "false_alarms":  false_alarms,
            "pod":           round(pod, 4) if not np.isnan(pod) else float("nan"),
            "far":           round(far, 4) if not np.isnan(far) else float("nan"),
            "csi":           round(csi, 4) if not np.isnan(csi) else float("nan"),
        }
    return out


# ---------------------------------------------------------------------------
# Scorer using stored per-event predictions
# ---------------------------------------------------------------------------

def score_inversion_result(
    result: dict,
    thresholds: list[float] | None = None,
) -> dict[str, dict]:
    """Compute per-sensor and overall metrics from a batch_inversion_from_geometry result.

    Reads ``result["per_event_predictions"]`` (obs_ppb / pred_ppb per row),
    so no re-computation of the forward model is needed.

    Parameters
    ----------
    result:
        Dict returned by :func:`batch_inversion_from_geometry`.
    thresholds:
        ppb thresholds for :func:`compute_threshold_skill`.  Default [30, 100].

    Returns
    -------
    dict mapping each sensor name (and ``"overall"``) to a sub-dict with
    ``compute_dispersion_metrics`` output plus ``threshold_skill``.
    """
    if thresholds is None:
        thresholds = [30.0, 100.0]

    by_sensor: dict[str, dict[str, list]] = {}
    for row in result.get("per_event_predictions", []):
        sname = row["sensor"]
        if sname not in by_sensor:
            by_sensor[sname] = {"obs": [], "pred": []}
        by_sensor[sname]["obs"].append(float(row["obs_ppb"]))
        by_sensor[sname]["pred"].append(float(row["pred_ppb"]))

    out: dict[str, dict] = {}
    all_obs:  list[float] = []
    all_pred: list[float] = []

    for sname, data in by_sensor.items():
        obs  = np.array(data["obs"])
        pred = np.array(data["pred"])
        all_obs.extend(data["obs"])
        all_pred.extend(data["pred"])
        out[sname] = {
            **compute_dispersion_metrics(obs, pred),
            "threshold_skill": compute_threshold_skill(obs, pred, thresholds=thresholds),
        }

    if all_obs:
        obs_all  = np.array(all_obs)
        pred_all = np.array(all_pred)
        out["overall"] = {
            **compute_dispersion_metrics(obs_all, pred_all),
            "threshold_skill": compute_threshold_skill(obs_all, pred_all, thresholds=thresholds),
        }

    return out


# ---------------------------------------------------------------------------
# LOSO cross-validation
# ---------------------------------------------------------------------------

def loso_cross_validate(
    events: list[dict],
    specs: dict,
    cfg: InversionConfig,
    source_ids: list[str] | None = None,
    thresholds: list[float] | None = None,
    sensor_names: list[str] | None = None,
) -> dict[str, dict]:
    """Leave-one-station-out cross-validation.

    For each sensor in *sensor_names* (default: all three monitoring stations),
    fits Q on the remaining sensors and predicts the held-out one.  The SAN
    YSIDRO held-out result is the primary SY-overprediction diagnostic: a large
    positive ``mean_bias`` means the geometry/Q is overdriving SY even when SY
    was excluded from the fit.

    Parameters
    ----------
    events:
        List of event dicts as accepted by
        :func:`batch_inversion_from_geometry`.
    specs:
        Output of :func:`load_source_geometry`.
    cfg:
        :class:`InversionConfig`.
    source_ids:
        Column ordering for sources; defaults to ``list(specs.keys())``.
    thresholds:
        ppb thresholds for threshold skill.  Default [30, 100].
    sensor_names:
        Sensors to hold out one at a time.  Default: all keys of
        ``h2s.dispersion.lagrangian.SENSORS``.

    Returns
    -------
    dict mapping each held-out sensor name to a sub-dict containing:

    - All keys from :func:`compute_dispersion_metrics` for the LOSO prediction.
    - ``"threshold_skill"`` from :func:`compute_threshold_skill`.
    - ``"n_train_events"`` : events used to fit Q.
    - ``"n_eval_events"``  : events where held-out sensor had a reading.
    - ``"fit_sensors"``    : list of sensors used to fit Q.
    - ``"Q_by_source"``    : emission rates from the held-out fit.
    - ``"reason"``         : set to ``"no_train_events"`` or ``"no_eval_events"``
                             on degenerate inputs.
    """
    if thresholds is None:
        thresholds = [30.0, 100.0]
    if sensor_names is None:
        sensor_names = list(SENSORS.keys())

    sids = list(source_ids) if source_ids is not None else list(specs.keys())
    results: dict[str, dict] = {}

    for held_out in sensor_names:
        train_sensors = [s for s in sensor_names if s != held_out]

        # Build training events: remove held-out sensor from h2s_obs
        train_events = []
        for ev in events:
            h2s_train = {k: v for k, v in ev["h2s_obs"].items() if k != held_out}
            if any(v > cfg.background_ppb for v in h2s_train.values()):
                train_events.append({**ev, "h2s_obs": h2s_train})

        if not train_events:
            results[held_out] = {
                "n_train_events": 0,
                "n_eval_events":  0,
                "fit_sensors":    train_sensors,
                "Q_by_source":    {sid: 0.0 for sid in sids},
                "reason":         "no_train_events",
            }
            continue

        # Fit Q on training sensors
        fit_result = batch_inversion_from_geometry(
            train_events, specs, cfg, source_ids=sids
        )
        Q  = fit_result["Q"]

        # Predict held-out sensor for every event that has a reading
        obs_list:  list[float] = []
        pred_list: list[float] = []

        for ev in events:
            h2s_actual = ev.get("h2s_obs", {}).get(held_out)
            if h2s_actual is None:
                continue

            A_held, _ = build_sensitivity_matrix_from_geometry(
                specs, ev["met_row"], [held_out], cfg, source_ids=sids
            )
            c_pred = float(A_held[0, :] @ Q) + cfg.background_ppb
            obs_list.append(float(h2s_actual))
            pred_list.append(c_pred)

        if not obs_list:
            results[held_out] = {
                "n_train_events": len(train_events),
                "n_eval_events":  0,
                "fit_sensors":    train_sensors,
                "Q_by_source":    fit_result["Q_by_source"],
                "reason":         "no_eval_events",
            }
            continue

        obs_arr  = np.array(obs_list)
        pred_arr = np.array(pred_list)
        metrics  = compute_dispersion_metrics(obs_arr, pred_arr)
        skill    = compute_threshold_skill(obs_arr, pred_arr, thresholds=thresholds)

        results[held_out] = {
            **metrics,
            "threshold_skill":  skill,
            "n_train_events":   len(train_events),
            "n_eval_events":    len(obs_list),
            "fit_sensors":      train_sensors,
            "Q_by_source":      fit_result["Q_by_source"],
        }

    return results
