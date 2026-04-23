#!/usr/bin/env python3
"""Test script: compare fixed vs wind-dependent diffusion in Lagrangian model."""

import pandas as pd
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
import os
import json

load_dotenv(Path('.env'))

from h2s.resources.minio import S3Resource
from h2s.constants import OBS_DATA_PATH, LAGRANGIAN_ENSEMBLE_PATH
from h2s.dispersion import LagrangianConfig, run_inversion_window, source_attribution

# Initialize S3
s3 = S3Resource(
    S3_BUCKET=os.getenv('S3_BUCKET'),
    S3_ADDRESS=os.getenv('S3_ADDRESS'),
    S3_PORT=os.getenv('S3_PORT'),
    S3_USE_SSL=os.getenv('S3_USE_SSL', 'true').lower() == 'true',
    S3_ACCESS_KEY=os.getenv('S3_ACCESS_KEY'),
    S3_SECRET_KEY=os.getenv('S3_SECRET_KEY'),
)

print("=== WIND-DEPENDENT DIFFUSION TEST ===\n")
print("Loading observation data...")
url = s3.publicUrl(OBS_DATA_PATH)
df = pd.read_parquet(url)
df["time"] = pd.to_datetime(df["time"], utc=True).dt.tz_convert("America/Los_Angeles")
print(f"Loaded {len(df):,} observations\n")

# Test parameters
date_start = "2026-02-01"
date_end = "2026-04-01"
h2s_threshold = 30.0
max_events = 10  # Small sample for quick test

print("=" * 80)
print("TEST 1: FIXED DIFFUSION (legacy, sigma_u=0.3, sigma_v=0.3)")
print("=" * 80)

cfg_fixed = LagrangianConfig(
    n_particles=1000,  # Fewer particles for speed
    hours_back=2,
    use_wind_dependent_diffusion=False,  # LEGACY MODE
    sigma_u=0.3,
    sigma_v=0.3,
)

results_fixed, footprint_fixed = run_inversion_window(
    df=df,
    cfg=cfg_fixed,
    date_start=date_start,
    date_end=date_end,
    h2s_threshold=h2s_threshold,
    max_events=max_events,
)

attribution_fixed = source_attribution(footprint_fixed)
print(f"\nProcessed {len(results_fixed)} events")
print("\nTop 5 sources (fixed diffusion):")
for i, (source, frac) in enumerate(list(attribution_fixed.items())[:5], 1):
    print(f"  {i}. {source:30s} {frac:.3f} ({frac*100:5.1f}%)")

# Zone aggregation
zone_map = {
    'east': ['stewarts_drain', 'silva_drain', 'tj_crossing_cdlp_w',
             'tj_crossing_cdlp_e', 'dairy_mart_bridge', 'del_sol_canyon'],
    'west': ['oneonta_slough', 'tijuana_beach_outlet', 'hollister_ps',
             'sd_bay_otay_outlet', 'sd_bay_fruitdale'],
    'south': ['smugglers_gulch', 'goat_canyon', 'goat_canyon_ps',
              'hollister_bridge_n', 'hollister_bridge_s', 'saturn_blvd_bridge'],
}

print("\nZone totals (fixed diffusion):")
zones_fixed = {}
for zone, sources in zone_map.items():
    total = sum(attribution_fixed.get(s, 0.0) for s in sources)
    zones_fixed[zone] = total
    print(f"  {zone:6s}: {total:.3f} ({total*100:5.1f}%)")

print("\n" + "=" * 80)
print("TEST 2: WIND-DEPENDENT DIFFUSION (sigma ~ U^0.5)")
print("=" * 80)

cfg_wind = LagrangianConfig(
    n_particles=1000,
    hours_back=2,
    use_wind_dependent_diffusion=True,  # NEW MODE
    sigma_u_coeff=0.15,
    sigma_v_coeff=0.15,
    sigma_u_exponent=0.5,
    sigma_v_exponent=0.5,
    min_wind_speed=0.5,
)

results_wind, footprint_wind = run_inversion_window(
    df=df,
    cfg=cfg_wind,
    date_start=date_start,
    date_end=date_end,
    h2s_threshold=h2s_threshold,
    max_events=max_events,
)

attribution_wind = source_attribution(footprint_wind)
print(f"\nProcessed {len(results_wind)} events")
print("\nTop 5 sources (wind-dependent diffusion):")
for i, (source, frac) in enumerate(list(attribution_wind.items())[:5], 1):
    print(f"  {i}. {source:30s} {frac:.3f} ({frac*100:5.1f}%)")

print("\nZone totals (wind-dependent diffusion):")
zones_wind = {}
for zone, sources in zone_map.items():
    total = sum(attribution_wind.get(s, 0.0) for s in sources)
    zones_wind[zone] = total
    print(f"  {zone:6s}: {total:.3f} ({total*100:5.1f}%)")

print("\n" + "=" * 80)
print("COMPARISON")
print("=" * 80)

print("\nZone attribution changes (wind-dependent - fixed):")
for zone in ['east', 'west', 'south']:
    delta = zones_wind[zone] - zones_fixed[zone]
    delta_pct = delta * 100
    symbol = "▲" if delta > 0 else "▼"
    print(f"  {zone:6s}: {delta:+.3f} ({delta_pct:+5.1f}%) {symbol}")

print("\nExpected behavior:")
print("  → Wind-dependent diffusion should give SHARPER attribution during calm events")
print("  → High-contributing sources should have HIGHER fractions (less diffusion spread)")
print("  → Low wind (< 2 m/s) → sigma ~ 0.21 m/s (vs fixed 0.30)")
print("  → High wind (> 5 m/s) → sigma ~ 0.34 m/s (vs fixed 0.30)")

print("\n" + "=" * 80)
print("NEXT STEPS")
print("=" * 80)
print("1. If results look reasonable, run full inversion:")
print("   uv run dg launch --job dispersion_inversion_job")
print("\n2. Compare emission rates with previous run:")
print("   Previous (2h fixed): east=76.1, west=33.7, south=57.2 g/s")
print("\n3. Validate forward forecasts match observations better")
print("=" * 80)
