"""Shared data access and plotting style for the H2S levels / odour complaints report.

Every source below is a **public** URL — no credentials, no S3 keys. Anyone can
re-run the whole report from a clean checkout. Raw pulls are cached under
``data/`` (gitignored) so repeated runs are cheap; delete that directory to force
a refresh.

The one non-obvious source is the complaints pull. The published
``latest/tijuana/sd_complaints/complaints.csv`` on S3 carries only ``date_received``,
which is date-only (every record lands at local midnight / 07:00 UTC-offset).
Sub-daily analysis is impossible from it. The county's ArcGIS feature service also
exposes ``date_and_time_received``, which is the real clock time of the call, so we
query the service directly. See :func:`load_complaints`.
"""

from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
FIGURES = HERE / "figures"
TABLES = HERE / "tables"
for _d in (DATA, FIGURES, TABLES):
    _d.mkdir(exist_ok=True)

TZ = "America/Los_Angeles"

S3 = "https://oss.resilientservice.mooo.com/resilentpublic"
ARCGIS = (
    "https://gis-public.sandiegocounty.gov/arcgis/rest/services/Hosted/"
    "SDAPCD_Complaints/FeatureServer/0/query"
)

#: Every remote input the report reads, so the provenance appendix can be
#: generated rather than hand-maintained.
SOURCES = {
    "modeldata_h2s_nofill.parquet": (
        f"{S3}/latest/tijuana/forecast_data/modeldata_h2s_nofill.parquet",
        "Hourly H2S + meteorology + flow/tide per station. Unmeasured H2S hours "
        "are null (no gap fill), which is what makes it safe for exceedance counting.",
    ),
    "h2s_locations.csv": (
        f"{S3}/latest/tijuana/forecast_data/h2s_locations.csv",
        "Station coordinates for the San Diego APCD H2S monitors.",
    ),
    "forecast_skill_report.json": (
        f"{S3}/latest/tijuana/forecast_data/forecast_skill_report.json",
        "Operational (out-of-sample) skill of the deployed forecast products, "
        "rebuilt by station_forecast_validation_rebuild_job.",
    ),
    "NESTOR__BES_training_report.json": (
        f"{S3}/tijuana/forecast/models/stations/NESTOR__BES/training_report.json",
        "Held-out training metrics + feature importances for the deployed "
        "NESTOR-BES models (both feature variants, all four tasks).",
    ),
    "IB_CIVIC_CTR_training_report.json": (
        f"{S3}/tijuana/forecast/models/stations/IB_CIVIC_CTR/training_report.json",
        "Same, IB Civic Center.",
    ),
    "SAN_YSIDRO_training_report.json": (
        f"{S3}/tijuana/forecast/models/stations/SAN_YSIDRO/training_report.json",
        "Same, San Ysidro.",
    ),
    "NESTOR__BES_deployment_metadata.json": (
        f"{S3}/tijuana/forecast/models/stations/NESTOR__BES/deployment_metadata.json",
        "Deployed model version tag for NESTOR-BES.",
    ),
    "IB_CIVIC_CTR_deployment_metadata.json": (
        f"{S3}/tijuana/forecast/models/stations/IB_CIVIC_CTR/deployment_metadata.json",
        "Deployed model version tag for IB Civic Center.",
    ),
    "SAN_YSIDRO_deployment_metadata.json": (
        f"{S3}/tijuana/forecast/models/stations/SAN_YSIDRO/deployment_metadata.json",
        "Deployed model version tag for San Ysidro.",
    ),
    "validation.parquet": (
        f"{S3}/tijuana/forecast/validation_store/validation.parquet",
        "Every stored forecast product row joined to the H2S actually measured "
        "at its target hour — the substrate the skill report is computed from.",
    ),
    "complaints_full.json": (
        f"{ARCGIS} (paged, where=1=1)",
        "Every San Diego APCD complaint record, including "
        "date_and_time_received — the real clock time, which the published "
        "complaints.csv does not carry.",
    ),
}

# ---------------------------------------------------------------------------
# Thresholds and palette — kept identical to h2s/constants.py so the report and
# the production 4-tier view cannot drift apart.
# ---------------------------------------------------------------------------

T_YELLOW = 5.0      # odour / nuisance
T_YELLOW_HIGH = 10.0  # resident-smell level
T_ORANGE = 30.0     # hazardous

TIER_COLORS = {
    "green": "#2ca02c",
    "yellow": "#FFC107",
    "yellow-high": "#FF9800",
    "orange": "#FF5722",
}

NESTOR = "NESTOR - BES"
IB = "IB CIVIC CTR"
SAN_YSIDRO = "SAN YSIDRO"
STATIONS = [SAN_YSIDRO, NESTOR, IB]

#: The county aggregates almost every Imperial Beach odour call to this single
#: coordinate. It is a records-management artifact, not a measurement location.
COMPLAINT_POINT = (32.552044, -117.081305)


# ---------------------------------------------------------------------------
# Fetching
# ---------------------------------------------------------------------------

def _cached(name: str, fetch) -> Path:
    """Return a local path for ``name``, downloading via ``fetch`` if absent."""
    path = DATA / name
    if not path.exists():
        fetch(path)
    return path


def _download(url: str):
    def _fetch(path: Path):
        print(f"  fetching {url}")
        with urllib.request.urlopen(url, timeout=300) as r:
            path.write_bytes(r.read())
    return _fetch


def fetch(name: str) -> Path:
    """Fetch one of :data:`SOURCES` by name, using the local cache."""
    if name == "complaints_full.json":
        return _cached(name, _fetch_complaints)
    url, _ = SOURCES[name]
    return _cached(name, _download(url))


def _fetch_complaints(path: Path) -> None:
    """Page the county's ArcGIS feature service for the full complaint table.

    The service caps a response at 2000 features, so we page on ``objectid``.
    ``date_and_time_received`` is the field that carries a real time of day.
    """
    fields = (
        "objectid,nature_of_complaint,date_received,date_and_time_received,"
        "record_number,record_status,investigation_outcome,"
        "response_duration__hours_,x_coordinate,y_coordinate,"
        "cross_street___intersection,zip,city"
    )
    rows: list[dict] = []
    offset = 0
    while True:
        params = {
            "where": "1=1",
            "outFields": fields,
            "returnGeometry": "false",
            "orderByFields": "objectid ASC",
            "resultOffset": str(offset),
            "resultRecordCount": "2000",
            "f": "json",
        }
        url = ARCGIS + "?" + urllib.parse.urlencode(params)
        print(f"  fetching complaints offset={offset}")
        with urllib.request.urlopen(url, timeout=300) as r:
            payload = json.load(r)
        feats = payload.get("features", [])
        rows.extend(f["attributes"] for f in feats)
        if len(feats) < 2000:
            break
        offset += 2000
        time.sleep(0.3)
    path.write_text(json.dumps(rows))


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_h2s(station: str | None = None, measured_only: bool = True) -> pd.DataFrame:
    """Hourly H2S + met for one station (or all), tz-aware in Pacific time.

    ``measured_only`` drops gap-filled hours. Keep it on for anything that
    counts exceedances or correlates against complaints — a synthetic value must
    never be reported as an observation.
    """
    df = pd.read_parquet(fetch("modeldata_h2s_nofill.parquet"))
    df["time"] = pd.to_datetime(df["time"])
    if df["time"].dt.tz is None:
        df["time"] = df["time"].dt.tz_localize(TZ)
    else:
        df["time"] = df["time"].dt.tz_convert(TZ)
    if station is not None:
        df = df[df["site_name"] == station]
    if measured_only:
        df = df[df["h2s_measured"].fillna(False).astype(bool) & df["H2S"].notna()]
    return df.sort_values("time").reset_index(drop=True)


def load_complaints(
    odor_only: bool = True, imperial_beach_only: bool = True
) -> pd.DataFrame:
    """Complaint records with a real time of day.

    Returns columns ``time`` (tz-aware Pacific, floored to the hour in
    ``hour``), ``nature_of_complaint``, ``zip``, ``city``, coordinates.

    ``imperial_beach_only`` filters on ZIP 91932, which is how the county files
    the Tijuana River Valley odour calls. That ZIP is the analysis unit; the
    point coordinates are not (see :data:`COMPLAINT_POINT`).
    """
    rows = json.loads(fetch("complaints_full.json").read_text())
    c = pd.DataFrame(rows)
    c["time"] = (
        pd.to_datetime(c["date_and_time_received"], unit="ms", utc=True)
        .dt.tz_convert(TZ)
    )
    c["date_only"] = (
        pd.to_datetime(c["date_received"], unit="ms", utc=True).dt.tz_convert(TZ)
    )
    c = c[c["time"].notna()].copy()
    c["zip"] = c["zip"].astype("string")
    c["nature_of_complaint"] = c["nature_of_complaint"].astype("string")
    if odor_only:
        c = c[c["nature_of_complaint"].str.lower() == "odor"]
    if imperial_beach_only:
        c = c[c["zip"].str.startswith("91932", na=False)]
    c["hour"] = c["time"].dt.floor("h")
    return c.sort_values("time").reset_index(drop=True)


def hourly_panel(station: str = NESTOR) -> pd.DataFrame:
    """One row per **measured** station-hour, joined to that hour's complaint count.

    Restricted to the overlap of the two records, so hours before complaints
    carried a time of day are excluded rather than silently counted as zero.
    """
    h2s = load_h2s(station).set_index("time")
    comp = load_complaints()
    counts = comp.groupby("hour").size().rename("complaints")
    panel = h2s.join(counts, how="left")
    panel["complaints"] = panel["complaints"].fillna(0.0)
    panel = panel[panel.index >= comp["hour"].min()]
    panel["any_complaint"] = (panel["complaints"] > 0).astype(int)
    panel["hour_of_day"] = panel.index.hour
    panel["month"] = panel.index.month
    panel["log_h2s"] = np.log10(panel["H2S"].clip(lower=0.1))
    return panel


def load_validation() -> pd.DataFrame:
    """The forecast validation store: one row per (run, product, station, lead)."""
    v = pd.read_parquet(fetch("validation.parquet"))
    v["time"] = pd.to_datetime(v["time"], utc=True).dt.tz_convert(TZ)
    return v


def load_json(name: str) -> dict:
    return json.loads(fetch(name).read_text())


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def wilson(k: np.ndarray, n: np.ndarray, z: float = 1.96):
    """Wilson score interval for a binomial proportion.

    Used instead of the normal approximation because several concentration bins
    hold only a couple of hundred hours, where the normal interval runs past 0/1.
    """
    k = np.asarray(k, dtype=float)
    n = np.asarray(n, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = k / n
        denom = 1 + z**2 / n
        centre = (p + z**2 / (2 * n)) / denom
        half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return centre - half, centre + half


def h2s_category(v: float) -> str:
    """The production 4-tier view (h2s/constants.py)."""
    if not np.isfinite(v):
        return "unknown"
    if v < T_YELLOW:
        return "green"
    if v < T_YELLOW_HIGH:
        return "yellow"
    if v < T_ORANGE:
        return "yellow-high"
    return "orange"


# ---------------------------------------------------------------------------
# Plot style
# ---------------------------------------------------------------------------

def style():
    """Apply a consistent, print-friendly style. Call once per figure script."""
    import matplotlib as mpl

    mpl.use("Agg")
    mpl.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 160,
            "savefig.bbox": "tight",
            "savefig.facecolor": "white",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.5,
            "legend.frameon": False,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
        }
    )


def save(fig, name: str) -> Path:
    """Write a figure to ``figures/<name>.png`` and report the path."""
    path = FIGURES / f"{name}.png"
    fig.savefig(path)
    print(f"  wrote {path.relative_to(HERE)}")
    return path


def save_table(df: pd.DataFrame, name: str, **kw) -> Path:
    path = TABLES / f"{name}.csv"
    df.to_csv(path, **kw)
    print(f"  wrote {path.relative_to(HERE)}")
    return path
