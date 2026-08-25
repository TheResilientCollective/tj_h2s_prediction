"""Figure 4 — the exceedance record, re-framed on a night that contains midnight.

The production exceedance assets (``h2sforecast/h2s_peaks`` and its astronomical
counterpart) split the record on a 6 AM / 6 PM clock or on sunset. Both leave
midnight at an edge, which cuts every event in half: an event that runs 22:00 to
06:00 is reported as two partial periods on two different dates.

This figure puts midnight in the middle. The frame is the **night of** a date:
the 24 hours from noon on that date to noon on the next, so every nocturnal
event sits whole and centred.

  (a) the entire measured record as one image — one row per night-of date, one
      column per hour from noon to noon, coloured by the production 4-tier
      scheme. Seasonal banding and the nocturnal concentration of the hazard
      are both visible without any aggregation being applied first.
  (b) the diel profile on the same noon-to-noon axis: the share of measured
      hours over each threshold at each hour of the night.
  (c) exceedance hours per night, which is the ``h2s_peaks`` quantity, counted
      on the midnight-centred frame so no event is split across two rows.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch

import common as C

STATION = C.NESTOR

#: Hours since noon. 12 == midnight, which is the centre of the axis.
NIGHT_HOURS = np.arange(24)
XTICKS = [0, 4, 8, 12, 16, 20, 24]
XLABELS = ["12:00", "16:00", "20:00", "00:00", "04:00", "08:00", "12:00"]

CMAP = ListedColormap(
    [C.TIER_COLORS["green"], C.TIER_COLORS["yellow"],
     C.TIER_COLORS["yellow-high"], C.TIER_COLORS["orange"]]
)
CMAP.set_bad("#f0f0f0")
NORM = BoundaryNorm([0, C.T_YELLOW, C.T_YELLOW_HIGH, C.T_ORANGE, 1e6], CMAP.N)


def night_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Attach the midnight-centred frame: ``night_of`` date and ``night_hour``.

    ``night_hour`` counts hours since noon, so 0 is noon, 12 is midnight and 23
    is 11:00 the following morning. ``night_of`` is the calendar date the window
    opened on, so "the night of 3 April" is 3 Apr 12:00 → 4 Apr 12:00.
    """
    out = df.copy()
    shifted = out["time"] - pd.Timedelta(hours=12)
    out["night_of"] = shifted.dt.normalize()
    out["night_hour"] = shifted.dt.hour
    return out


def main() -> None:
    C.style()
    d = night_frame(C.load_h2s(STATION))

    grid = d.pivot_table(index="night_of", columns="night_hour", values="H2S",
                         aggfunc="max")
    grid = grid.reindex(
        columns=NIGHT_HOURS,
        index=pd.date_range(grid.index.min(), grid.index.max(), freq="D",
                            tz=grid.index.tz),
    )

    fig = plt.figure(figsize=(11.5, 6.6))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.35, 1], height_ratios=[1.5, 1],
                          hspace=0.34, wspace=0.22)

    # --- (a) the whole record, midnight-centred -----------------------------
    ax = fig.add_subplot(gs[:, 0])
    y = mdates.date2num(grid.index.tz_localize(None))
    ax.imshow(
        np.ma.masked_invalid(grid.to_numpy()),
        aspect="auto", cmap=CMAP, norm=NORM, interpolation="nearest",
        extent=(0, 24, y[-1] + 1, y[0]),
    )
    ax.axvline(12, color="white", lw=1.0, alpha=0.85)
    ax.yaxis_date()
    ax.yaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.yaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.set_xticks(XTICKS)
    ax.set_xticklabels(XLABELS)
    ax.set_xlabel("hour of the night (noon → noon, midnight centred)")
    ax.set_title(f"(a) Every measured hour at {STATION}, one row per night")
    ax.grid(False)
    ax.legend(
        handles=[Patch(facecolor=C.TIER_COLORS[k], label=lbl) for k, lbl in
                 [("green", "< 5 ppb"), ("yellow", "5–10"),
                  ("yellow-high", "10–30"), ("orange", "≥ 30 ppb")]]
        + [Patch(facecolor="#f0f0f0", label="not measured")],
        loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=5, fontsize=7.5,
    )

    # --- (b) diel profile ---------------------------------------------------
    ax = fig.add_subplot(gs[0, 1])
    prof = d.groupby("night_hour")["H2S"].agg(
        n="size",
        gt5=lambda x: (x > C.T_YELLOW).mean() * 100,
        gt10=lambda x: (x > C.T_YELLOW_HIGH).mean() * 100,
        gt30=lambda x: (x > C.T_ORANGE).mean() * 100,
    ).reindex(NIGHT_HOURS)
    xs = NIGHT_HOURS + 0.5
    for col, colour, lbl in (("gt5", C.TIER_COLORS["yellow"], "> 5 ppb"),
                             ("gt10", C.TIER_COLORS["yellow-high"], "> 10 ppb"),
                             ("gt30", C.TIER_COLORS["orange"], "> 30 ppb")):
        ax.fill_between(xs, prof[col], color=colour, alpha=0.75, label=lbl)
    ax.axvline(12, color="#333", lw=1, ls="--")
    ax.set_xlim(0, 24)
    ax.set_xticks(XTICKS)
    ax.set_xticklabels(XLABELS)
    ax.set_ylabel("% of measured hours")
    ax.set_title("(b) The hazard is a single nocturnal peak,\ncentred a little after midnight")
    ax.legend(loc="upper left")

    # --- (c) exceedance hours per night -------------------------------------
    ax = fig.add_subplot(gs[1, 1])
    per_night = d.groupby("night_of")["H2S"].agg(
        hours="size",
        gt5=lambda x: int((x > C.T_YELLOW).sum()),
        gt10=lambda x: int((x > C.T_YELLOW_HIGH).sum()),
        gt30=lambda x: int((x > C.T_ORANGE).sum()),
    )
    monthly = per_night.resample("MS").mean()
    idx = monthly.index.tz_localize(None)
    ax.bar(idx, monthly["gt5"], width=24, color=C.TIER_COLORS["yellow"], label="> 5 ppb")
    ax.bar(idx, monthly["gt10"], width=24, color=C.TIER_COLORS["yellow-high"], label="> 10 ppb")
    ax.bar(idx, monthly["gt30"], width=24, color=C.TIER_COLORS["orange"], label="> 30 ppb")
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.set_ylabel("exceedance hours per night")
    ax.set_title("(c) Mean exceedance hours per night, by month")
    ax.legend(loc="upper right", ncol=3)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

    C.save(fig, "fig04_exceedance_midnight")
    plt.close(fig)

    C.save_table(prof.round(2), "tbl04_diel_profile")
    C.save_table(per_night, "tbl04_exceedance_per_night")

    peak = prof["gt5"].idxmax()
    night_mask = d["night_hour"].between(8, 20)   # 20:00 → 08:00
    print(f"peak exceedance hour: night_hour={peak} "
          f"({(12 + peak) % 24:02d}:00 local)")
    print(f"share of >5 ppb hours falling in 20:00–08:00: "
          f"{100 * (d.loc[night_mask, 'H2S'] > C.T_YELLOW).sum() / (d['H2S'] > C.T_YELLOW).sum():.1f}%")
    print(f"share of >30 ppb hours falling in 20:00–08:00: "
          f"{100 * (d.loc[night_mask, 'H2S'] > C.T_ORANGE).sum() / (d['H2S'] > C.T_ORANGE).sum():.1f}%")
    print(f"nights with ≥1 hour > 30 ppb: {int((per_night['gt30'] > 0).sum())} "
          f"of {len(per_night)}")


if __name__ == "__main__":
    main()
