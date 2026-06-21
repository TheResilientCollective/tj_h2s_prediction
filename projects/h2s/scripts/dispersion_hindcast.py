"""Dispersion hindcast — compare Gaussian model predictions against observed H2S.

Loads observed met + H2S from S3, runs the forward model for a specified window,
and prints/plots a comparison table at each sensor.

Usage
-----
    uv run python scripts/dispersion_hindcast.py \
        --start "2026-06-21 22:00" --end "2026-06-22 00:00" --tz "America/Los_Angeles"

    # Override emission rates (g/s) if geometry rates aren't in S3 yet:
    uv run python scripts/dispersion_hindcast.py \
        --start "2026-06-21 22:00" --end "2026-06-22 00:00" \
        --east 20 --west 10 --south 137

    # Write a PNG plot:
    uv run python scripts/dispersion_hindcast.py \
        --start "2026-06-21 22:00" --end "2026-06-22 00:00" --plot hindcast.png

Reads .env for S3 credentials (S3_BUCKET, S3_ADDRESS, etc.).
Obs bucket defaults to "resilentpublic"; override with --obs-bucket.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Make sure the h2s package is importable when run from the project root
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parents[1] / ".env")

from h2s.resources.minio import S3Resource
from h2s.constants import (
    OBS_DATA_PATH,
    EMISSION_RATES_PATH,
)
from h2s.dispersion import (
    run_forward_model,
    run_forward_model_from_geometry,
    load_source_geometry,
)


SENSORS = ["NESTOR - BES", "IB CIVIC CTR", "SAN YSIDRO"]


def build_s3() -> S3Resource:
    return S3Resource(
        S3_BUCKET=os.environ["S3_BUCKET"],
        S3_ADDRESS=os.environ.get("S3_ADDRESS", "oss.resilientservice.mooo.com"),
        S3_PORT=os.environ.get("S3_PORT", "443"),
        S3_USE_SSL=os.environ.get("S3_USE_SSL", "true").lower() == "true",
        S3_ACCESS_KEY=os.environ["S3_ACCESS_KEY"],
        S3_SECRET_KEY=os.environ["S3_SECRET_KEY"],
    )


def load_emission_rates(s3: S3Resource, args) -> tuple[dict, bool]:
    """Return (rates_dict, use_geometry).

    If geometry rates are in S3 and no manual override was given, use them.
    Falls back to the 3-zone dict (east/west/south) otherwise.
    Manual --east/--west/--south always takes priority.
    """
    manual = any(v is not None for v in [args.east, args.west, args.south])
    if manual:
        rates = {
            "east":  args.east  or 20.0,
            "west":  args.west  or 10.0,
            "south": args.south or 137.0,
        }
        print(f"Using manual rates: {rates}")
        return rates, False

    try:
        data = json.loads(s3.getFile(EMISSION_RATES_PATH))
        geom = data.get("emission_rates_by_geometry_g_s", {})
        if geom:
            print(f"Loaded geometry rates from S3 ({len(geom)} sources, "
                  f"Q_total={sum(geom.values()):.1f} g/s)")
            return {k: float(v) for k, v in geom.items()}, True
        zone_rates = data.get("emission_rates_g_s", {})
        if zone_rates:
            print(f"Loaded 3-zone rates from S3: {zone_rates}")
            return {k: float(v) for k, v in zone_rates.items()}, False
    except Exception as e:
        print(f"Could not load emission rates from S3 ({e}) — using calibrated defaults")

    defaults = {"east": 20.0, "west": 10.0, "south": 137.0}
    print(f"Using calibrated defaults: {defaults}")
    return defaults, False


def run_hindcast(args):
    tz = args.tz
    t_start = pd.Timestamp(args.start).tz_localize(tz)
    t_end   = pd.Timestamp(args.end).tz_localize(tz)
    window_hours = (t_end - t_start).total_seconds() / 3600

    print(f"\nHindcast window: {t_start}  →  {t_end}  ({window_hours:.1f} h)")

    s3 = build_s3()

    # Load obs data
    print(f"Loading obs data from {args.obs_bucket}/{OBS_DATA_PATH} ...")
    url = s3.publicUrl(OBS_DATA_PATH, bucket=args.obs_bucket)
    df = pd.read_parquet(url)

    # Handle timezone: parquet may store as UTC, tz-naive, or already local.
    raw_time = pd.to_datetime(df["time"])
    if raw_time.dt.tz is None:
        # Timezone-naive — assume UTC
        df["time"] = raw_time.dt.tz_localize("UTC").dt.tz_convert(tz)
    else:
        df["time"] = raw_time.dt.tz_convert(tz)

    t_min = df["time"].min()
    t_max = df["time"].max()
    print(f"  {len(df)} rows | range: {t_min}  →  {t_max}")

    # If default window was used and lies outside the data, shift to the last 2h
    if args.use_default_window and t_max < t_start:
        t_end   = t_max.ceil("h")
        t_start = t_end - pd.Timedelta(hours=2)
        window_hours = 2.0
        print(f"  Requested window is beyond data end — shifting to: {t_start}  →  {t_end}")

    # Filter to window
    window_df = df[(df["time"] >= t_start) & (df["time"] <= t_end)].copy()
    print(f"  {len(window_df)} rows in window [{t_start}  →  {t_end}]")

    if window_df.empty:
        # Show the last available timestamps to help user pick a valid window
        print("\nNo data in the requested window.")
        print("Last 10 timestamps available in the dataset:")
        print(df["time"].drop_duplicates().sort_values().tail(10).to_string())
        print("\nRe-run with e.g.:")
        last = df["time"].max().floor("h")
        print(f"  --start \"{(last - pd.Timedelta(hours=2)).strftime('%Y-%m-%d %H:%M')}\" "
              f"--end \"{last.strftime('%Y-%m-%d %H:%M')}\"")
        sys.exit(1)

    # Show actual H2S in window
    print("\n--- Observed H2S (ppb) ---")
    obs_pivot = (
        window_df[window_df["site_name"].isin(SENSORS)]
        .pivot_table(index="time", columns="site_name", values="H2S", aggfunc="first")
        .reindex(columns=SENSORS)
    )
    print(obs_pivot.to_string())

    # Load emission rates
    rates, use_geometry = load_emission_rates(s3, args)

    # Run forward model using the obs met over the window
    print(f"\nRunning {'geometry' if use_geometry else '3-zone'} forward model ...")
    if use_geometry:
        specs = load_source_geometry()
        result = run_forward_model_from_geometry(
            df=window_df,
            specs=specs,
            source_q_g_s=rates,
            start_time=t_start,
            hours=window_hours,
            cadence_minutes=args.cadence,
        )
    else:
        result = run_forward_model(
            df=window_df,
            emission_rates_g_s=rates,
            start_time=t_start,
            hours=window_hours,
            cadence_minutes=args.cadence,
        )

    # Build comparison table
    pred_df = pd.DataFrame(
        {sname: result.concentrations.get(sname, []) for sname in SENSORS},
        index=result.times,
    )
    pred_df.index.name = "time"

    print("\n--- Dispersion model predicted H2S (ppb) ---")
    print(pred_df.to_string(float_format=lambda x: f"{x:.1f}"))

    # Align and compute residuals
    obs_aligned  = obs_pivot.reindex(pred_df.index)
    resid_df = pred_df - obs_aligned
    print("\n--- Residual: predicted − observed (ppb) ---")
    print(resid_df.to_string(float_format=lambda x: f"{x:+.1f}"))

    bias = resid_df.mean()
    rmse = np.sqrt((resid_df**2).mean())
    print("\n--- Summary ---")
    for s in SENSORS:
        print(f"  {s:20s}  bias={bias[s]:+.1f} ppb   RMSE={rmse[s]:.1f} ppb")

    # Optional plot
    if args.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(len(SENSORS), 1, figsize=(10, 3 * len(SENSORS)), sharex=True)
            fig.suptitle(f"Dispersion hindcast  {t_start.strftime('%Y-%m-%d %H:%M')} – "
                         f"{t_end.strftime('%H:%M %Z')}", fontsize=12)

            for ax, sname in zip(axes, SENSORS):
                obs_s = obs_aligned[sname].dropna()
                pred_s = pred_df[sname].dropna()
                ax.step(obs_s.index, obs_s.values, where="post", label="Observed", color="steelblue", lw=1.5)
                ax.step(pred_s.index, pred_s.values, where="post", label="Dispersion model",
                        color="tomato", lw=1.5, linestyle="--")
                ax.axhline(30, color="orange", lw=0.8, linestyle=":")
                ax.axhline(100, color="red", lw=0.8, linestyle=":")
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


_DEFAULT_START = "2026-06-20 22:00"
_DEFAULT_END   = "2026-06-21 00:00"


def main():
    p = argparse.ArgumentParser(description="Dispersion hindcast vs observed H2S")
    p.add_argument("--start", default=_DEFAULT_START,
                   help="Window start (local time, default: night of Jun 20 10pm)")
    p.add_argument("--end",   default=_DEFAULT_END,
                   help="Window end (local time, default: Jun 21 midnight)")
    p.add_argument("--tz",    default="America/Los_Angeles")
    p.add_argument("--obs-bucket", default="resilentpublic",
                   help="S3 bucket for obs data")
    p.add_argument("--cadence", type=int, default=60,
                   help="Model timestep in minutes (default: 60 = hourly)")
    p.add_argument("--east",  type=float, default=None, help="Override east Q (g/s)")
    p.add_argument("--west",  type=float, default=None, help="Override west Q (g/s)")
    p.add_argument("--south", type=float, default=None, help="Override south Q (g/s)")
    p.add_argument("--plot",  default=None, help="Save comparison PNG to this path")
    args = p.parse_args()
    args.use_default_window = (args.start == _DEFAULT_START and args.end == _DEFAULT_END)
    run_hindcast(args)


if __name__ == "__main__":
    main()
