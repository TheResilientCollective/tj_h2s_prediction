"""
HYSPLIT CONTROL file generator for H2S source attribution runs.

Given a DataFrame of high-H2S events, generates CONTROL files for:
  - Backward trajectory runs (hyts_std)
  - Backward dispersion runs (hycs_std — builds source-receptor matrix)
  - Forward dispersion runs (hycs_std — operational forecast)

The high-level generate_hysplit_bundle() function writes everything to a
temp directory and returns a zip archive as bytes for S3 upload.

HYSPLIT execution is NOT performed here. Download the zip and run in
a local HYSPLIT container or submit to NOAA via email.

Adapted from modeling_sources/generate_hysplit_controls.py:
  - SETUP.CFG embedded as SETUP_CFG_CONTENT constant (no local file dependency)
  - generate_hysplit_bundle() added as the public API
  - Internal write functions operate on a provided temp directory
"""

import io
import os
import zipfile
import tempfile
import argparse
import pandas as pd
from pathlib import Path
from typing import Optional


# Embedded SETUP.CFG content for stable nocturnal BL conditions
SETUP_CFG_CONTENT = """\
# HYSPLIT SETUP.CFG
# Shared configuration for Tijuana H2S source attribution runs
# Place in HYSPLIT working directory alongside CONTROL file
#
# Key settings for stable nocturnal boundary layer conditions
# Ref: HYSPLIT User's Guide, Appendix A

&SETUP

  ! Meteorology
  KMSL    = 0,          ! Heights: 0=AGL, 1=MSL
  NINIT   = 1,          ! Met initialization: 1=read entire met file at start
  DELT    = 0.0,        ! Integration time step (min); 0=auto from met grid

  ! Turbulence — critical for stable nocturnal BL
  KTURB   = 1,          ! Turbulence parameterization: 1=Beljaars-Holtslag (stable)
  KZMIX   = 1,          ! Vertical mixing: 1=use Kz from met, scale by TVMIX
  TVMIX   = 1.0,        ! Vertical mixing scale factor (reduce to 0.5 for very stable events)
  KHMAX   = 9999,       ! Max particle age (hr)

  ! Particle settings (dispersion runs)
  NUMPAR  = 500,        ! Number of particles per cycle (500 adequate for source attrib)
  QCYCLE  = 1.0,        ! Emission cycle interval (hr)
  INITD   = 0,          ! Initial particle distribution: 0=top-hat sphere

  ! Concentration grid
  ICHEM   = 0,          ! Chemistry: 0=none (H2S photolysis negligible at night)
  KDEF    = 0,          ! Horizontal diffusion: 0=Kx from met data
  HSCALE  = 10000.0,    ! Horizontal scale for diffusion (m); 10km for mesoscale
  VSCALE  = 100.0,      ! Vertical scale (m); 100m for shallow stable BL

  ! Output
  NCYCL   = 1,          ! Number of pollutant cycles to output
  NDUMP   = 1,          ! Dump particle positions: 1=yes (for trajectory validation)
  NSTR    = 0,          ! Restart from dump: 0=no

  ! Domain-specific: San Diego coastal stable layer
  ! Typical NBL height: 50-200m. HYSPLIT will compute from met.
  ! For very stable events (sbiwtp_deficit > 3 MGD), consider TVMIX = 0.5

/
"""

SENSORS = {
    "NESTOR - BES": {"key": "NB",  "lat": 32.567097, "lon": -117.090656},
    "IB CIVIC CTR": {"key": "IB",  "lat": 32.576139, "lon": -117.115361},
    "SAN YSIDRO":   {"key": "SY",  "lat": 32.552794, "lon": -117.047286},
}

SOURCES = {
    "east":  {"name": "Stewart's Drain corridor",  "lat": 32.541, "lon": -117.058, "agl": 2.0},
    "west":  {"name": "Oneonta Slough / PS",        "lat": 32.570, "lon": -117.127, "agl": 2.0},
    "south": {"name": "Goat Canyon / cross-border", "lat": 32.537, "lon": -117.099, "agl": 2.0},
}

GRID_CFG = {
    "sw_lat": 32.50, "sw_lon": -117.18,
    "dlat": 0.005, "dlon": 0.005,
    "nlat": 40,    "nlon": 30,
}


def _met_filename(dt_utc: pd.Timestamp) -> str:
    """Return GDAS 0.5° filename for a given UTC timestamp."""
    week = (dt_utc.day - 1) // 7 + 1
    month = dt_utc.strftime("%b").lower()
    year2 = dt_utc.strftime("%y")
    return f"gdas1.{month}{year2}.w{week}"


def _write_backward_traj(
    events: pd.DataFrame,
    run_dir: Path,
    met_dir: str,
    hours_back: int = 12,
) -> list[Path]:
    written = []
    for dt_utc_key, _grp in events.groupby("time_utc"):
        dt_utc = pd.Timestamp(str(dt_utc_key))
        tag = dt_utc.strftime("%Y%m%d_%H")
        ctrl_path = run_dir / f"CONTROL.traj_{tag}"
        lines = [
            f"{dt_utc.strftime('%y %m %d %H')}   ! Start: {dt_utc.isoformat()}",
            "3   ! sensors NB, IB, SY",
        ]
        for scfg in SENSORS.values():
            lines.append(f"{scfg['lat']:.4f}  {scfg['lon']:.4f}  10.0")
        lines += [
            f"{-hours_back}   ! backward {hours_back}h",
            "0   ! vertical motion from met",
            "10000.0   ! model top m AGL",
            "1",
            met_dir,
            _met_filename(dt_utc),
            "1",
            str(run_dir / "output") + "/",
            f"tdump_traj_{tag}",
        ]
        ctrl_path.write_text("\n".join(lines) + "\n")
        written.append(ctrl_path)
    return written


def _write_backward_disp(
    events: pd.DataFrame,
    run_dir: Path,
    met_dir: str,
    hours_back: int = 12,
) -> list[Path]:
    written = []
    g = GRID_CFG
    for dt_utc_key, grp in events.groupby("time_utc"):
        dt_utc = pd.Timestamp(str(dt_utc_key))
        for sname, scfg in SENSORS.items():
            sensor_row = grp[grp["site_name"] == sname]
            if sensor_row.empty:
                continue
            h2s_obs = float(sensor_row["H2S"].iloc[0])
            tag = f"{dt_utc.strftime('%Y%m%d_%H')}_{scfg['key']}"
            ctrl_path = run_dir / f"CONTROL.bdisp_{tag}"
            lines = [
                f"{dt_utc.strftime('%y %m %d %H')}   ! event UTC, obs H2S={h2s_obs:.1f} ppb",
                "1   ! single receptor location",
                f"{scfg['lat']:.4f}  {scfg['lon']:.4f}  10.0   ! {sname}",
                f"{-hours_back}   ! backward",
                "0", "10000.0",
                "1", met_dir, _met_filename(dt_utc),
                "1   ! species",
                "H2S",
                "1.0   ! unit emission rate g/hr (adjoint footprint)",
                f"{float(hours_back):.1f}   ! emission duration hr",
                "1   ! release at start",
                "1",
                f"{scfg['lat']:.4f}  {scfg['lon']:.4f}  10.0",
                "0.0  0.0  0.0",
                "1   ! concentration grid",
                f"{g['sw_lat']:.3f}  {g['sw_lon']:.3f}",
                f"{g['dlat']:.4f}  {g['dlon']:.4f}",
                f"{g['nlat']}  {g['nlon']}",
                str(run_dir / "output") + "/",
                f"cdump_bdisp_{tag}",
                "1", "10.0",
                f"{dt_utc.strftime('%y %m %d %H')}",
                "00 00 00 01 00",
                "0",
                "H2S",
                "0.0  0.0  0.0",
                "0.0  0.0  0.0",
                "0.0", "34.08",
            ]
            ctrl_path.write_text("\n".join(lines) + "\n")
            written.append(ctrl_path)
    return written


def _write_forward_disp(
    run_dir: Path,
    met_dir: str,
    start_utc: str,
    emission_rates_g_s: dict[str, float],
    run_hours: int = 72,
) -> Path:
    dt_utc = pd.Timestamp(start_utc, tz="UTC")
    tag = dt_utc.strftime("%Y%m%d_%H")
    ctrl_path = run_dir / f"CONTROL.fwd_{tag}"
    g = GRID_CFG
    n_sources = len(SOURCES)

    lines = [
        f"{dt_utc.strftime('%y %m %d %H')}   ! forecast start UTC",
        f"{n_sources}   ! number of sources: E/W/S",
    ]
    for zone, src in SOURCES.items():
        lines.append(f"{src['lat']:.4f}  {src['lon']:.4f}  {src['agl']:.1f}   ! {src['name']}")

    lines += [
        f"{run_hours}   ! forward run hours",
        "0", "10000.0",
        "1", met_dir, _met_filename(dt_utc),
        f"{n_sources}   ! species — one per source to track attribution",
    ]
    for zone in SOURCES:
        q = emission_rates_g_s.get(zone, 20.0)
        lines += [
            f"H2S_{zone.upper()}",
            f"{q:.1f}   ! {zone} emission rate g/s",
            "1.0   ! 1-hr cycles (continuous)",
            "0",
        ]

    lines += [
        "1   ! concentration grid",
        f"{g['sw_lat']:.3f}  {g['sw_lon']:.3f}",
        f"{g['dlat']:.4f}  {g['dlon']:.4f}",
        f"{g['nlat']}  {g['nlon']}",
        str(run_dir / "output") + "/",
        f"cdump_fwd_{tag}",
        "1", "10.0",
        f"{dt_utc.strftime('%y %m %d %H')}",
        "00 00 00 01 00",
        "1   ! time-averaged output",
    ]
    for zone in SOURCES:
        lines += [
            f"H2S_{zone.upper()}",
            "0.01  0.0  0.0",
            "0.0   0.0  0.0",
            "0.0", "34.08",
        ]

    ctrl_path.write_text("\n".join(lines) + "\n")
    return ctrl_path


def _write_run_script(run_dir: Path, mode: str, ctrl_files: list[Path]) -> Path:
    exe_map = {
        "backward_traj": "hyts_std",
        "backward_disp": "hycs_std",
        "forward_disp":  "hycs_std",
    }
    hysplit_exe = exe_map.get(mode, "hycs_std")
    script = run_dir / f"run_hysplit_{mode}.sh"

    lines = [
        "#!/bin/bash",
        f"# Auto-generated HYSPLIT batch runner for mode: {mode}",
        f"# {len(ctrl_files)} runs",
        "",
        "set -e",
        f'HYSPLIT_EXEC="${{HYSPLIT_HOME:-/opt/hysplit/exec}}/{hysplit_exe}"',
        f'SETUP_FILE="{run_dir}/SETUP.CFG"',
        "",
        "echo '=== HYSPLIT batch run ==='",
        f"echo 'Mode: {mode}'",
        f"echo 'Runs: {len(ctrl_files)}'",
        "echo ''",
        "FAILED=0",
    ]
    for i, cp in enumerate(ctrl_files):
        lines += [
            "",
            f"echo '[{i+1}/{len(ctrl_files)}] {cp.name}'",
            f"cp {cp} {cp.parent}/CONTROL",
            f"cp $SETUP_FILE {cp.parent}/SETUP.CFG",
            f"cd {cp.parent} && $HYSPLIT_EXEC && cd - > /dev/null || "
            f"{{ echo 'FAILED: {cp.name}'; FAILED=$((FAILED+1)); }}",
        ]
    lines += [
        "",
        "echo ''",
        f"echo '=== Done: {len(ctrl_files)} runs, $FAILED failed ==='",
    ]
    script.write_text("\n".join(lines) + "\n")
    script.chmod(0o755)
    return script


def generate_hysplit_bundle(
    mode: str,
    df: Optional[pd.DataFrame],
    met_dir: str,
    emission_rates_g_s: Optional[dict[str, float]] = None,
    start_utc: Optional[str] = None,
    hours_back: int = 12,
    h2s_threshold: float = 30.0,
    date_start: str = "2026-02-01",
    date_end: str = "2026-04-01",
    run_hours: int = 72,
) -> bytes:
    """
    Generate a HYSPLIT run bundle and return it as zip archive bytes.

    Args:
        mode: "backward_traj" | "backward_disp" | "forward_disp"
        df: Observation DataFrame for backward modes (None for forward).
            Must have columns: time_utc, H2S, site_name, stable_atm.
        met_dir: Path to met files directory (written into CONTROL files).
        emission_rates_g_s: {zone: Q_g_s} for forward mode.
        start_utc: ISO timestamp for forward mode (default: now).
        hours_back: Backward integration duration (hours).
        h2s_threshold: Min H2S for event selection in backward modes.
        date_start, date_end: Event selection window for backward modes.
        run_hours: Forward dispersion duration (hours).

    Returns:
        Zip archive as bytes. Upload to S3 and extract to run.
        Includes CONTROL files, SETUP.CFG, and a run script.

    Note:
        HYSPLIT execution is NOT performed. The returned bundle is intended
        to be downloaded and executed in a HYSPLIT container or submitted
        to NOAA for server-side execution.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = Path(tmpdir) / mode
        run_dir.mkdir(parents=True)
        (run_dir / "output").mkdir()

        # Write SETUP.CFG
        (run_dir / "SETUP.CFG").write_text(SETUP_CFG_CONTENT)

        ctrl_files: list[Path] = []

        if mode == "backward_traj":
            if df is None:
                raise ValueError("df required for backward_traj mode")
            events = _filter_events(df, h2s_threshold, date_start, date_end)
            ctrl_files = _write_backward_traj(events, run_dir, met_dir, hours_back)

        elif mode == "backward_disp":
            if df is None:
                raise ValueError("df required for backward_disp mode")
            events = _filter_events(df, h2s_threshold, date_start, date_end)
            ctrl_files = _write_backward_disp(events, run_dir, met_dir, hours_back)

        elif mode == "forward_disp":
            if start_utc is None:
                start_utc = pd.Timestamp.utcnow().isoformat()
            if emission_rates_g_s is None:
                emission_rates_g_s = {"east": 20.0, "west": 10.0, "south": 137.0}
            ctrl_files = [_write_forward_disp(run_dir, met_dir, start_utc, emission_rates_g_s, run_hours)]

        else:
            raise ValueError(f"Unknown mode: {mode!r}. Use 'backward_traj', 'backward_disp', or 'forward_disp'.")

        _write_run_script(run_dir, mode, ctrl_files)

        # Zip the directory
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for fpath in run_dir.rglob("*"):
                if fpath.is_file():
                    zf.write(fpath, fpath.relative_to(tmpdir))
        return buf.getvalue()


def _filter_events(
    df: pd.DataFrame,
    h2s_threshold: float,
    date_start: str,
    date_end: str,
) -> pd.DataFrame:
    mask = (
        (df["time"] >= date_start)
        & (df["time"] <= date_end)
        & (df["H2S"] >= h2s_threshold)
        & df["H2S"].notna()
        & (df["stable_atm"] == 1)
    )
    events = df[mask].copy().sort_values("H2S", ascending=False)
    if "time_utc" not in events.columns:
        events["time_utc"] = pd.to_datetime(events["time"], utc=True)
    return events


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate HYSPLIT CONTROL files for H2S attribution")
    parser.add_argument("--mode", choices=["backward_traj", "backward_disp", "forward_disp"],
                        default="backward_traj")
    parser.add_argument("--output", default="./hysplit_bundle.zip")
    parser.add_argument("--data", default=os.environ.get("H2S_DATA_PATH", "modeldata_h2s_nofill.csv"))
    parser.add_argument("--met_dir", default=os.environ.get("HYSPLIT_MET_DIR", "./met/"))
    parser.add_argument("--h2s_min", type=float, default=30.0)
    parser.add_argument("--date_start", default="2026-02-01")
    parser.add_argument("--date_end", default="2026-04-01")
    parser.add_argument("--hours_back", type=int, default=12)
    parser.add_argument("--forward_start", default=None)
    parser.add_argument("--forward_hours", type=int, default=72)
    parser.add_argument("--emission_rates", default="east=20,west=10,south=137")
    args = parser.parse_args()

    df = None
    if args.mode in ("backward_traj", "backward_disp"):
        df = pd.read_csv(args.data) if args.data.endswith(".csv") else pd.read_parquet(args.data)
        df["time"] = pd.to_datetime(df["time"], utc=True).dt.tz_convert("America/Los_Angeles")

    rates: dict[str, float] = {}
    for part in args.emission_rates.split(","):
        k, v = part.split("=")
        rates[k.strip()] = float(v.strip())

    zip_bytes = generate_hysplit_bundle(
        mode=args.mode,
        df=df,
        met_dir=args.met_dir,
        emission_rates_g_s=rates,
        start_utc=args.forward_start,
        hours_back=args.hours_back,
        h2s_threshold=args.h2s_min,
        date_start=args.date_start,
        date_end=args.date_end,
        run_hours=args.forward_hours,
    )

    Path(args.output).write_bytes(zip_bytes)
    print(f"Bundle written → {args.output} ({len(zip_bytes):,} bytes)")
