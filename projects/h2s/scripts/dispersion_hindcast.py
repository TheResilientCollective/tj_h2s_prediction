"""Dispersion hindcast — compare geometry Gaussian model against observed H2S.

Loads observed met + H2S from S3, runs run_forward_model_from_geometry with
per-source Q rates, and prints/plots a comparison table at each sensor.

Usage
-----
    uv run python scripts/dispersion_hindcast.py

    # Explicit window:
    uv run python scripts/dispersion_hindcast.py \
        --start "2026-06-20 22:00" --end "2026-06-21 00:00"

    # Scale all source Q uniformly (sensitivity test):
    uv run python scripts/dispersion_hindcast.py --scale 2.0

    # Save a PNG:
    uv run python scripts/dispersion_hindcast.py --plot hindcast.png

Source Q (g/s) loaded from emission_rates_by_geometry_g_s in S3.
Falls back to q_prior values in source_geometry.toml when S3 rates are absent.
Obs bucket defaults to "resilentpublic"; override with --obs-bucket.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parents[1] / ".env")

from h2s.resources.minio import S3Resource
from h2s.constants import OBS_DATA_PATH, EMISSION_RATES_PATH
from h2s.dispersion import run_forward_model_from_geometry, load_source_geometry


SENSORS = ["NESTOR - BES", "IB CIVIC CTR", "SAN YSIDRO"]

_DEFAULT_START = "2026-06-20 22:00"
_DEFAULT_END   = "2026-06-21 00:00"


def build_s3() -> S3Resource:
    return S3Resource(
        S3_BUCKET=os.environ["S3_BUCKET"],
        S3_ADDRESS=os.environ.get("S3_ADDRESS", "oss.resilientservice.mooo.com"),
        S3_PORT=os.environ.get("S3_PORT", "443"),
        S3_USE_SSL=os.environ.get("S3_USE_SSL", "true").lower() == "true",
        S3_ACCESS_KEY=os.environ["S3_ACCESS_KEY"],
        S3_SECRET_KEY=os.environ["S3_SECRET_KEY"],
    )


def load_source_q(s3: S3Resource, specs: dict, scale: float = 1.0) -> dict[str, float]:
    """Return per-source Q (g/s), scaled by `scale`.

    Priority:
      1. emission_rates_by_geometry_g_s from S3 emission_rates.json
      2. q_prior from source_geometry.toml (calibrated design values)
    """
    try:
        data = json.loads(s3.getFile(EMISSION_RATES_PATH))
        geom = data.get("emission_rates_by_geometry_g_s", {})
        if geom:
            q = {k: float(v) * scale for k, v in geom.items()}
            print(f"Source Q from S3 geometry rates (scale={scale:.2f}):")
            for sid, qv in q.items():
                if qv > 0:
                    print(f"  {sid:35s}  {qv:.2f} g/s")
            print(f"  Q_total = {sum(q.values()):.1f} g/s")
            return q
        print("No geometry rates in S3 emission_rates.json — falling back to q_prior from TOML")
    except Exception as e:
        print(f"Could not load S3 emission rates ({e}) — falling back to q_prior from TOML")

    q = {sid: spec.q_prior * scale for sid, spec in specs.items()}
    print(f"Source Q from q_prior (scale={scale:.2f}):")
    for sid, qv in q.items():
        if qv > 0:
            print(f"  {sid:35s}  {qv:.2f} g/s")
    print(f"  Q_total = {sum(q.values()):.1f} g/s")
    return q


def run_hindcast(args):
    tz = args.tz
    t_start = pd.Timestamp(args.start).tz_localize(tz)
    t_end   = pd.Timestamp(args.end).tz_localize(tz)
    window_hours = (t_end - t_start).total_seconds() / 3600

    print(f"\nHindcast window: {t_start}  →  {t_end}  ({window_hours:.1f} h)")

    s3 = build_s3()
    specs = load_source_geometry()

    # Load obs data
    print(f"\nLoading obs data from {args.obs_bucket}/{OBS_DATA_PATH} ...")
    url = s3.publicUrl(OBS_DATA_PATH, bucket=args.obs_bucket)
    df = pd.read_parquet(url)

    raw_time = pd.to_datetime(df["time"])
    if raw_time.dt.tz is None:
        df["time"] = raw_time.dt.tz_localize("UTC").dt.tz_convert(tz)
    else:
        df["time"] = raw_time.dt.tz_convert(tz)

    t_min = df["time"].min()
    t_max = df["time"].max()
    print(f"  {len(df)} rows | range: {t_min}  →  {t_max}")

    # Auto-shift if default window is beyond data end
    if args.use_default_window and t_max < t_start:
        t_end   = t_max.ceil("h")
        t_start = t_end - pd.Timedelta(hours=2)
        window_hours = 2.0
        print(f"  Default window beyond data — shifted to: {t_start}  →  {t_end}")

    window_df = df[(df["time"] >= t_start) & (df["time"] <= t_end)].copy()
    print(f"  {len(window_df)} rows in window")

    if window_df.empty:
        print("\nNo data in the requested window.")
        print("Last 10 timestamps in the dataset:")
        print(df["time"].drop_duplicates().sort_values().tail(10).to_string())
        last = df["time"].max().floor("h")
        print(f"\nSuggested window:\n"
              f"  --start \"{(last - pd.Timedelta(hours=2)).strftime('%Y-%m-%d %H:%M')}\" "
              f"--end \"{last.strftime('%Y-%m-%d %H:%M')}\"")
        sys.exit(1)

    # Observed H2S
    print("\n--- Observed H2S (ppb) ---")
    obs_pivot = (
        window_df[window_df["site_name"].isin(SENSORS)]
        .pivot_table(index="time", columns="site_name", values="H2S", aggfunc="first")
        .reindex(columns=SENSORS)
    )
    print(obs_pivot.to_string())

    # Load per-source Q
    print()
    source_q = load_source_q(s3, specs, scale=args.scale)

    # Run geometry forward model with observed meteorology
    print(f"\nRunning geometry forward model ({len([v for v in source_q.values() if v>0])} active sources) ...")
    result = run_forward_model_from_geometry(
        df=window_df,
        specs=specs,
        source_q_g_s=source_q,
        start_time=t_start,
        hours=window_hours,
        cadence_minutes=args.cadence,
    )

    pred_df = pd.DataFrame(
        {sname: result.concentrations.get(sname, []) for sname in SENSORS},
        index=result.times,
    )
    pred_df.index.name = "time"

    print("\n--- Dispersion model predicted H2S (ppb) ---")
    print(pred_df.to_string(float_format=lambda x: f"{x:.1f}"))

    obs_aligned = obs_pivot.reindex(pred_df.index)
    resid_df = pred_df - obs_aligned
    print("\n--- Residual: predicted − observed (ppb) ---")
    print(resid_df.to_string(float_format=lambda x: f"{x:+.1f}"))

    bias = resid_df.mean()
    rmse = np.sqrt((resid_df**2).mean())
    print("\n--- Summary ---")
    for s in SENSORS:
        print(f"  {s:20s}  bias={bias[s]:+.1f} ppb   RMSE={rmse[s]:.1f} ppb")

    if args.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import matplotlib.dates

            fig, axes = plt.subplots(len(SENSORS), 1, figsize=(10, 3 * len(SENSORS)), sharex=True)
            fig.suptitle(
                f"Dispersion hindcast  {t_start.strftime('%Y-%m-%d %H:%M')} – "
                f"{t_end.strftime('%H:%M %Z')}\n"
                f"Q_total={sum(v for v in source_q.values() if v>0):.0f} g/s  "
                f"(scale={args.scale:.2f})",
                fontsize=11,
            )
            for ax, sname in zip(axes, SENSORS):
                obs_s  = obs_aligned[sname].dropna()
                pred_s = pred_df[sname].dropna()
                ax.step(obs_s.index,  obs_s.values,  where="post", label="Observed",
                        color="steelblue", lw=1.5)
                ax.step(pred_s.index, pred_s.values, where="post", label="Dispersion model",
                        color="tomato", lw=1.5, linestyle="--")
                ax.axhline(30,  color="orange", lw=0.8, linestyle=":", label="30 ppb watch")
                ax.axhline(100, color="red",    lw=0.8, linestyle=":", label="100 ppb critical")
                ax.set_ylabel("H₂S (ppb)")
                ax.set_title(sname)
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)
            axes[-1].xaxis.set_major_formatter(
                matplotlib.dates.DateFormatter("%H:%M", tz=tz))
            plt.tight_layout()
            plt.savefig(args.plot, dpi=150)
            print(f"\nPlot saved → {args.plot}")
        except Exception as e:
            print(f"Plot failed: {e}")


def main():
    p = argparse.ArgumentParser(description="Geometry dispersion hindcast vs observed H2S")
    p.add_argument("--start", default=_DEFAULT_START,
                   help="Window start (local time)")
    p.add_argument("--end",   default=_DEFAULT_END,
                   help="Window end (local time)")
    p.add_argument("--tz",    default="America/Los_Angeles")
    p.add_argument("--obs-bucket", default="resilentpublic",
                   help="S3 bucket for obs data")
    p.add_argument("--cadence", type=int, default=60,
                   help="Model timestep in minutes (default 60)")
    p.add_argument("--scale", type=float, default=1.0,
                   help="Multiply all source Q by this factor (sensitivity test)")
    p.add_argument("--plot",  default=None, help="Save comparison PNG to this path")
    args = p.parse_args()
    args.use_default_window = (args.start == _DEFAULT_START and args.end == _DEFAULT_END)
    run_hindcast(args)


if __name__ == "__main__":
    main()
