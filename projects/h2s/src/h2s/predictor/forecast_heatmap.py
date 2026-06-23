"""Heatmap-style forecast boards for the hourly per-station outlook.

Two complementary grids over the product rows (run_products output):

  generate_category_heatmap()    station (rows) × hour (cols); each cell is
                                 coloured by the 4-tier hazard category and
                                 labelled with the predicted H2S (ppb).

  generate_probability_heatmap() station × {P>5, P>10, P>30} (rows) × hour
                                 (cols); each cell shaded by the exceedance
                                 probability and labelled with the percentage.

Both return a PNG ``BytesIO`` (matplotlib Agg) for direct S3 upload, matching
the convention in ``predictor/visualizations.py``.
"""

from io import BytesIO
from typing import List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

from h2s.constants import (  # noqa: E402
    CATEGORY_COLORS,
    CATEGORY_LABELS,
    CATEGORY_ORDER,
    CATEGORY_UNKNOWN,
    STATIONS,
    h2s_category,
)

# Category → integer code for imshow (worst = highest). Unknown maps last.
_CAT_CODE = {cat: i for i, cat in enumerate(CATEGORY_ORDER)}
_CAT_CODE[CATEGORY_UNKNOWN] = len(CATEGORY_ORDER)
_CAT_CMAP = ListedColormap(
    [CATEGORY_COLORS[c] for c in CATEGORY_ORDER] + [CATEGORY_COLORS[CATEGORY_UNKNOWN]]
)

# Sequential ramp for probabilities (white → deep red).
_PROB_CMAP = plt.get_cmap("YlOrRd")

_PROB_ROWS = [("p5", "P>5"), ("p10", "P>10"), ("p30", "P>30")]


def _short(station: str) -> str:
    return STATIONS.get(station, {}).get("short", station[:3])


def _ordered_stations(products_df: pd.DataFrame, variant: str) -> List[str]:
    """Stations that have rows for this variant, in canonical STATIONS order."""
    present = set(
        products_df.loc[products_df.get("variant") == variant, "station"].unique()
    )
    ordered = [s for s in STATIONS if s in present]
    # Include any unexpected station names so nothing is silently dropped.
    ordered += [s for s in present if s not in STATIONS]
    return ordered


def _station_frame(products_df: pd.DataFrame, station: str, variant: str) -> pd.DataFrame:
    w = products_df[
        (products_df["station"] == station) & (products_df["variant"] == variant)
    ].copy()
    if w.empty:
        return w
    w["time"] = pd.to_datetime(w["time"], utc=True, errors="coerce")
    return w.sort_values("lead_hour").drop_duplicates("lead_hour")


def _hour_axis(products_df: pd.DataFrame, variant: str):
    """Shared lead-hour columns + their hour-of-day labels + target times."""
    w = products_df[products_df["variant"] == variant]
    leads = sorted(int(x) for x in w["lead_hour"].dropna().unique())
    labels, times = [], []
    for lead in leads:
        row = w[w["lead_hour"] == lead]
        ts = pd.to_datetime(row["time"].iloc[0], utc=True, errors="coerce") if len(row) else pd.NaT
        labels.append(f"{ts.hour:02d}" if pd.notna(ts) else f"+{lead}")
        times.append(ts)
    return leads, labels, times


def _day_bands(times: list):
    """Group consecutive columns sharing a calendar date (UTC).

    Returns (start_col, end_col_inclusive, "Mon 06-23") tuples — the text-day
    labels drawn beneath the hour axis so multi-day forecasts stay readable.
    """
    bands: list = []
    start = last = None
    cur_day = None
    for c, ts in enumerate(times):
        if pd.isna(ts):
            continue
        day = ts.normalize()
        if start is None:
            start, cur_day, last = c, day, c
        elif day != cur_day:
            bands.append((start, last, times[start].strftime("%a %m-%d")))
            start, cur_day, last = c, day, c
        else:
            last = c
    if start is not None:
        bands.append((start, last, times[start].strftime("%a %m-%d")))
    return bands


def _date_caption(times: list) -> str:
    """Weekday + date span for the title, e.g. 'Fri 2026-06-23' or a range."""
    valid = [t for t in times if pd.notna(t)]
    if not valid:
        return ""
    first, last = valid[0], valid[-1]
    if first.normalize() == last.normalize():
        return first.strftime("%a %Y-%m-%d")
    return f"{first.strftime('%a %Y-%m-%d')} → {last.strftime('%a %Y-%m-%d')}"


def _draw_day_bands(ax, times: list) -> None:
    """Draw vertical day separators + centered text-day labels in a header band
    just above the grid (x in data coords, y in axes fraction)."""
    bands = _day_bands(times)
    for start, end, label in bands:
        ax.text((start + end) / 2.0, 1.015, label, transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=7.5, fontweight="bold",
                color="#34495e", clip_on=False)
    # Heavier separators where the calendar day rolls over.
    for start, _end, _label in bands[1:]:
        ax.axvline(start - 0.5, color="#34495e", linewidth=1.6)


def _empty(message: str) -> BytesIO:
    fig, ax = plt.subplots(figsize=(10, 2.2))
    ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes)
    ax.set_axis_off()
    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)
    return buf


def generate_category_heatmap(
    products_df: pd.DataFrame,
    stations: Optional[List[str]] = None,
    *,
    variant: str = "evidence",
    env_label: str = "",
) -> BytesIO:
    """Station × hour hazard grid: cell colour = category, label = ppb."""
    if products_df is None or products_df.empty:
        return _empty("No forecast rows available")

    stations = stations or _ordered_stations(products_df, variant)
    stations = [s for s in stations if not _station_frame(products_df, s, variant).empty]
    if not stations:
        return _empty(f"No forecast rows for variant '{variant}'")

    leads, hour_labels, times = _hour_axis(products_df, variant)
    n_rows, n_cols = len(stations), len(leads)

    codes = np.full((n_rows, n_cols), _CAT_CODE[CATEGORY_UNKNOWN], dtype=float)
    values = np.full((n_rows, n_cols), np.nan)
    for r, station in enumerate(stations):
        w = _station_frame(products_df, station, variant).set_index("lead_hour")
        for c, lead in enumerate(leads):
            if lead in w.index:
                v = w.loc[lead, "h2s_pred"]
                v = float(v) if pd.notna(v) else np.nan
                values[r, c] = v
                codes[r, c] = _CAT_CODE[h2s_category(v)]

    fig_w = max(8.0, 0.42 * n_cols + 2.5)
    fig_h = max(2.4, 0.85 * n_rows + 1.6)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor("#f8f9fa")
    ax.imshow(codes, aspect="auto", cmap=_CAT_CMAP, vmin=0,
              vmax=len(CATEGORY_ORDER), interpolation="nearest")

    for r in range(n_rows):
        for c in range(n_cols):
            v = values[r, c]
            txt = "·" if np.isnan(v) else (f"{v:.0f}" if v >= 1 else f"{v:.1f}")
            ax.text(c, r, txt, ha="center", va="center", fontsize=7,
                    color="#1a1a1a", fontweight="medium")

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(hour_labels, fontsize=7)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels([f"{_short(s)}  {s}" for s in stations], fontsize=9)
    ax.set_xlabel("forecast hour (UTC hour-of-day)", fontsize=8)
    ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.4)
    ax.tick_params(which="minor", length=0)
    _draw_day_bands(ax, times)

    caption = _date_caption(times)
    date_line = f"\n{caption}" if caption else ""
    label = f" [{env_label}]" if env_label else ""
    ax.set_title(
        f"H2S Hazard Heatmap — {variant.capitalize()}{label}{date_line}\n"
        "green <5  ·  yellow 5–10  ·  yellow-high 10–30 (smell)  ·  orange ≥30 ppb",
        fontsize=11, fontweight="bold", pad=18,
    )
    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)
    return buf


def generate_probability_heatmap(
    products_df: pd.DataFrame,
    stations: Optional[List[str]] = None,
    *,
    variant: str = "evidence",
    env_label: str = "",
) -> BytesIO:
    """Station × {P>5,P>10,P>30} × hour grid: cell shaded by probability."""
    if products_df is None or products_df.empty:
        return _empty("No forecast rows available")

    stations = stations or _ordered_stations(products_df, variant)
    stations = [s for s in stations if not _station_frame(products_df, s, variant).empty]
    if not stations:
        return _empty(f"No forecast rows for variant '{variant}'")

    leads, hour_labels, times = _hour_axis(products_df, variant)
    n_cols = len(leads)
    row_specs = [(s, col, lbl) for s in stations for col, lbl in _PROB_ROWS]
    n_rows = len(row_specs)

    probs = np.full((n_rows, n_cols), np.nan)
    for r, (station, col, _lbl) in enumerate(row_specs):
        w = _station_frame(products_df, station, variant).set_index("lead_hour")
        if col not in w.columns:
            continue
        for c, lead in enumerate(leads):
            if lead in w.index and pd.notna(w.loc[lead, col]):
                probs[r, c] = float(w.loc[lead, col])

    fig_w = max(8.0, 0.42 * n_cols + 3.0)
    fig_h = max(3.0, 0.42 * n_rows + 1.6)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor("#f8f9fa")
    ax.imshow(np.nan_to_num(probs, nan=0.0), aspect="auto", cmap=_PROB_CMAP,
              vmin=0.0, vmax=1.0, interpolation="nearest")

    for r in range(n_rows):
        for c in range(n_cols):
            p = probs[r, c]
            if np.isnan(p):
                ax.text(c, r, "·", ha="center", va="center", fontsize=7, color="#888")
                continue
            ax.text(c, r, f"{round(p * 100)}", ha="center", va="center", fontsize=6.5,
                    color="white" if p >= 0.55 else "#1a1a1a")

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(hour_labels, fontsize=7)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels([f"{_short(s)} {lbl}" for s, _c, lbl in row_specs], fontsize=8)
    ax.set_xlabel("forecast hour (UTC hour-of-day)", fontsize=8)

    # White gridlines between cells; heavier separators between stations.
    ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.8)
    ax.tick_params(which="minor", length=0)
    for i in range(len(_PROB_ROWS), n_rows, len(_PROB_ROWS)):
        ax.axhline(i - 0.5, color="#34495e", linewidth=1.8)
    _draw_day_bands(ax, times)

    caption = _date_caption(times)
    date_line = f"\n{caption}" if caption else ""
    label = f" [{env_label}]" if env_label else ""
    ax.set_title(
        f"H2S Exceedance-Probability Heatmap (%) — {variant.capitalize()}{label}{date_line}\n"
        "per station: P(>5) · P(>10, smell) · P(>30, hazard)",
        fontsize=11, fontweight="bold", pad=18,
    )
    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)
    return buf
