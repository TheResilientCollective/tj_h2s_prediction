"""Fit DISPERSION_EMISSION_SCALE from stored Gaussian forward forecasts vs measured H2S.

Downloads every coarse forward-forecast run (``tijuana/dispersion/forward_forecast_*.json``)
and the H2S observation parquet (OBS_DATA_PATH), aggregates the 15-min forecast ppb to
hourly means, joins them to measured hourly H2S per sensor, and reports the candidate
uniform scale factors plus an alert-skill sweep.

Candidate scales reported
-------------------------
- ``mean_ratio``   : measured mean / forecast mean
- ``p95_ratio``    : measured p95 / forecast p95 (likewise p99)
- ``ls_scale``     : least-squares scale through the origin (dominated by the many
                     near-zero measured hours; reported for completeness only)
- exceedance match: the scale ``s`` such that frac(s·pred ≥ T) == frac(obs ≥ T) for
                     T = 5 / 10 / 30 ppb — the most operationally relevant number,
                     since the alert check fires on fixed ppb thresholds.

Usage (from projects/h2s, with S3_* set in the environment / .env):

    uv run python scripts/calibrate_emission_scale.py --cache /tmp/emission_scale_cache
    uv run python scripts/calibrate_emission_scale.py --since 2026-07-01

The script only reads from S3. Set the chosen value via DISPERSION_EMISSION_SCALE
(env) or the default in h2s/constants.py. See docs/DISPERSION_EMISSION_SCALE.md.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import pathlib
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from h2s.constants import OBS_DATA_PATH  # noqa: E402
from h2s.resources.minio import S3Resource  # noqa: E402

FORECAST_PREFIX = "tijuana/dispersion/forward_forecast_"
SCALES = [1.0, 0.5, 0.3, 0.2, 0.15, 0.12, 0.1, 0.07, 0.05]


def _s3() -> S3Resource:
    return S3Resource(
        S3_BUCKET=os.getenv("S3_BUCKET"),
        S3_ADDRESS=os.getenv("S3_ADDRESS"),
        S3_PORT=os.getenv("S3_PORT"),
        S3_USE_SSL=os.getenv("S3_USE_SSL", "true").lower() == "true",
        S3_ACCESS_KEY=os.getenv("S3_ACCESS_KEY"),
        S3_SECRET_KEY=os.getenv("S3_SECRET_KEY"),
    )


def fetch(s3: S3Resource, cache: pathlib.Path, since: str | None) -> tuple[list[pathlib.Path], pd.DataFrame]:
    cache.mkdir(parents=True, exist_ok=True)
    (cache / "forecasts").mkdir(exist_ok=True)
    names = sorted(
        o.object_name
        for o in s3.listPath(FORECAST_PREFIX, recursive=False)
        if "detailed" not in o.object_name and o.object_name.endswith(".json")
    )
    if since:
        tag_min = since.replace("-", "") + "_0000"
        names = [n for n in names if n.split("forward_forecast_")[-1][:13] >= tag_min]
    print(f"forecast runs: {len(names)}")

    def _get(n: str) -> pathlib.Path:
        dst = cache / "forecasts" / n.split("/")[-1]
        if not dst.exists():
            dst.write_bytes(s3.getFile(n))
        return dst

    with ThreadPoolExecutor(16) as ex:
        files = list(ex.map(_get, names))

    obs_path = cache / "obs.parquet"
    if not obs_path.exists():
        obs = pd.read_parquet(io.BytesIO(s3.getFile(OBS_DATA_PATH)))[["time", "site_name", "H2S"]]
        obs.to_parquet(obs_path)
    obs = pd.read_parquet(obs_path)
    return files, obs


def build_join(files: list[pathlib.Path], obs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for f in files:
        d = json.loads(f.read_text())
        tag = f.stem.replace("forward_forecast_", "")
        q_total = sum(float(v) for v in d.get("emission_rates_g_s", {}).values())
        scale_used = float(d.get("metadata", {}).get("emission_scale", 1.0))
        for sensor, pts in d["timeseries"].items():
            for p in pts:
                if p["predicted_ppb"] is None:
                    continue
                # Un-apply any scale already baked into the stored run so the fit is
                # always expressed relative to the raw calibrated rates.
                rows.append((tag, sensor, p["time"], float(p["predicted_ppb"]) / scale_used, q_total))
    pred = pd.DataFrame(rows, columns=["run_tag", "site_name", "time", "pred", "Q_total"])
    pred["time"] = pd.to_datetime(pred["time"], utc=True)
    pred["run_start"] = pred.groupby("run_tag")["time"].transform("min")
    pred["lead_h"] = ((pred["time"] - pred["run_start"]).dt.total_seconds() / 3600).astype(int)
    pred["hour"] = pred["time"].dt.floor("h")  # UTC floor: avoids DST ambiguity
    hourly = pred.groupby(["run_tag", "site_name", "hour"], as_index=False).agg(
        pred_mean=("pred", "mean"), pred_max=("pred", "max"), lead_h=("lead_h", "min"), Q_total=("Q_total", "first")
    )
    obs = obs.dropna(subset=["H2S"]).copy()
    obs["hour"] = pd.to_datetime(obs["time"], utc=True).dt.floor("h")
    j = hourly.merge(obs[["site_name", "hour", "H2S"]], on=["site_name", "hour"], how="inner")
    return j


def summarize(g: pd.DataFrame) -> pd.Series:
    p, o = g.pred_mean.values, g.H2S.values
    return pd.Series({
        "n": len(g),
        "obs_mean": o.mean(), "pred_mean": p.mean(), "mean_ratio": o.mean() / p.mean(),
        "obs_p95": np.percentile(o, 95), "pred_p95": np.percentile(p, 95),
        "p95_ratio": np.percentile(o, 95) / np.percentile(p, 95),
        "p99_ratio": np.percentile(o, 99) / np.percentile(p, 99),
        "ls_scale": (p @ o) / (p @ p) if (p @ p) > 0 else np.nan,
        "spearman": pd.Series(p).corr(pd.Series(o), method="spearman"),
        "frac_pred>=30": (p >= 30).mean(), "frac_obs>=30": (o >= 30).mean(),
    })


def exceedance_scale(j: pd.DataFrame, T: float) -> float:
    fo = (j.H2S >= T).mean()
    if fo == 0:
        return float("nan")
    thr = np.quantile(j.pred_mean, 1 - fo)
    return T / thr if thr > 0 else float("nan")


def skill(p: np.ndarray, o: np.ndarray, T: float) -> tuple[float, float, float]:
    hit = ((p >= T) & (o >= T)).sum(); miss = ((p < T) & (o >= T)).sum(); fa = ((p >= T) & (o < T)).sum()
    pod = hit / (hit + miss) if hit + miss else np.nan
    far = fa / (hit + fa) if hit + fa else np.nan
    csi = hit / (hit + miss + fa) if hit + miss + fa else np.nan
    return pod, far, csi


def report(j: pd.DataFrame) -> None:
    pd.set_option("display.width", 220)
    pd.set_option("display.float_format", lambda x: f"{x:.3g}")
    print(f"\njoined forecast-hours: {len(j)}  runs: {j.run_tag.nunique()}  "
          f"span: {j.hour.min()} → {j.hour.max()}  Q_total used: {sorted(j.Q_total.round(1).unique())}")
    print("\n=== overall ===\n" + summarize(j).to_string())
    print("\n=== by sensor ===\n" + j.groupby("site_name").apply(summarize).to_string())
    j = j.copy()
    j["lead_bucket"] = pd.cut(j.lead_h, [-1, 3, 6, 12, 24], labels=["0-3h", "4-6h", "7-12h", "13-24h"])
    print("\n=== by lead ===\n" + j.groupby("lead_bucket", observed=True).apply(summarize).to_string())
    j["month"] = j.hour.dt.strftime("%Y-%m")
    cols = ["n", "obs_mean", "pred_mean", "mean_ratio", "p95_ratio", "p99_ratio"]
    print("\n=== by month ===\n" + j.groupby("month").apply(summarize)[cols].to_string())
    print("\n=== uniform scale matching observed exceedance frequency ===")
    for T in (5, 10, 30):
        print(f"T={T:3d} ppb  obs frac={(j.H2S >= T).mean():.4f}  scale={exceedance_scale(j, T):.3f}")
    print("\n=== alert skill vs scale (hourly, all sensors) ===")
    print(f"{'scale':>6} | {'POD30':>6} {'FAR30':>6} {'CSI30':>6} | {'POD10':>6} {'FAR10':>6} {'CSI10':>6} | mean_bias")
    for s in SCALES:
        p = j.pred_mean.values * s; o = j.H2S.values
        a = skill(p, o, 30); b = skill(p, o, 10)
        print(f"{s:6.2f} | {a[0]:6.2f} {a[1]:6.2f} {a[2]:6.3f} | {b[0]:6.2f} {b[1]:6.2f} {b[2]:6.3f} | {np.mean(p - o):8.2f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", default="/tmp/emission_scale_cache", help="local cache dir for downloaded runs")
    ap.add_argument("--since", default=None, help="only runs on/after this date (YYYY-MM-DD)")
    ap.add_argument("--save-join", default=None, help="optional parquet path for the joined table")
    args = ap.parse_args()

    files, obs = fetch(_s3(), pathlib.Path(args.cache), args.since)
    j = build_join(files, obs)
    report(j)
    if args.save_join:
        j.to_parquet(args.save_join)
        print(f"\njoined table → {args.save_join}")


if __name__ == "__main__":
    main()
