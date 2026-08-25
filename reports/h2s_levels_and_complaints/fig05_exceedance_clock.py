"""Figure 5 — "when exceedances happen", re-drawn with midnight in the middle.

This is the month × hour-of-day exceedance climatology from the Nestor H2S plume
dashboard, with one change: the hour axis runs **noon to noon** instead of
midnight to midnight.

On a 00:00–23:00 axis the nocturnal hazard is cut in half and pushed to the two
opposite edges of the panel — the top rows and the bottom rows are the same
event, and the eye has to reassemble them. Centred on midnight the whole thing
is one contiguous block, and two features that the split version hides become
legible: the event window drifts later in the night through the winter and back
earlier in the summer, and the ≥30 ppb hazard sits inside a much narrower core
of the night than the ≥5 ppb one.

Three panels, one per operational threshold, on a shared colour scale within
each panel. All measured hours; gap-filled hours are excluded upstream.
"""

from __future__ import annotations

import calendar

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import common as C

STATION = C.NESTOR

#: Hours since noon: 0 = 12:00, 12 = midnight, 23 = 11:00 next day.
NIGHT_HOURS = np.arange(24)
YTICKS = [0, 4, 8, 12, 16, 20]
YLABELS = ["12:00", "16:00", "20:00", "00:00 ", "04:00", "08:00"]

#: Minimum measured hours in a (month, hour) cell before a rate is drawn.
MIN_CELL = 12


def clock_grid(d: pd.DataFrame, threshold: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """P(H2S > threshold) per (night_hour, month), plus the cell counts."""
    rate = d.pivot_table(index="night_hour", columns="month", values="H2S",
                         aggfunc=lambda x: 100 * (x > threshold).mean())
    n = d.pivot_table(index="night_hour", columns="month", values="H2S",
                      aggfunc="size")
    rate = rate.reindex(index=NIGHT_HOURS, columns=range(1, 13))
    n = n.reindex(index=NIGHT_HOURS, columns=range(1, 13))
    return rate.where(n >= MIN_CELL), n


def main() -> None:
    C.style()
    d = C.load_h2s(STATION)
    shifted = d["time"] - pd.Timedelta(hours=12)
    d["night_hour"] = shifted.dt.hour
    # Month of the night the hour belongs to, so a 02:00 hour is filed under the
    # evening it started in rather than under the following calendar day.
    d["month"] = shifted.dt.month

    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.6))
    panels = [
        (C.T_YELLOW, "P(H₂S > 5 ppb)", "(a) odour / nuisance level"),
        (C.T_YELLOW_HIGH, "P(H₂S > 10 ppb)", "(b) resident-smell level"),
        (C.T_ORANGE, "P(H₂S > 30 ppb)", "(c) hazardous level"),
    ]
    tables = []
    for ax, (thr, cbar_label, title) in zip(axes, panels):
        rate, n = clock_grid(d, thr)
        im = ax.imshow(np.ma.masked_invalid(rate.to_numpy()), aspect="auto",
                       cmap="magma_r", interpolation="nearest",
                       extent=(0.5, 12.5, 23.5, -0.5), vmin=0)
        ax.axhline(12, color="#2196f3", lw=1.2, ls="--")
        ax.text(0.6, 11.6, "midnight", color="#2196f3", fontsize=7.5,
                ha="left", va="bottom")
        ax.set_xticks(range(1, 13))
        ax.set_xticklabels([calendar.month_abbr[m][0] for m in range(1, 13)])
        ax.set_yticks(YTICKS)
        ax.set_yticklabels(YLABELS)
        ax.set_xlabel("month of the night")
        ax.set_title(title, fontsize=10)
        ax.grid(False)
        cb = fig.colorbar(im, ax=ax, pad=0.03, fraction=0.05)
        cb.set_label(f"% of measured hours\n{cbar_label}", fontsize=7.5)
        cb.ax.tick_params(labelsize=7)
        t = rate.stack().rename("pct_exceeding").reset_index()
        t.columns = ["night_hour", "month", "pct_exceeding"]
        t.insert(0, "threshold_ppb", thr)
        t["local_hour"] = (t["night_hour"] + 12) % 24
        tables.append(t)
    axes[0].set_ylabel("hour of the night (noon → noon)")

    fig.suptitle(
        f"When exceedances happen at {STATION} — midnight-centred "
        f"({d['time'].min():%b %Y} – {d['time'].max():%b %Y}, measured hours only)",
        fontsize=11, y=1.02,
    )
    fig.tight_layout()
    C.save(fig, "fig05_exceedance_clock")
    plt.close(fig)

    out = pd.concat(tables, ignore_index=True)
    C.save_table(out, "tbl05_exceedance_clock", index=False)

    # Where in the night the hazard sits, month by month — the drift the
    # split-axis version cannot show.
    rate5, _ = clock_grid(d, C.T_YELLOW)
    rate30, _ = clock_grid(d, C.T_ORANGE)
    centroid = pd.DataFrame({
        "month": range(1, 13),
        "peak_hour_gt5": [(rate5[m].idxmax() + 12) % 24 if rate5[m].notna().any()
                          else np.nan for m in range(1, 13)],
        "peak_hour_gt30": [(rate30[m].idxmax() + 12) % 24 if rate30[m].notna().any()
                           else np.nan for m in range(1, 13)],
        "hours_over_5_pct": [rate5[m].mean() for m in range(1, 13)],
        "width_gt5_hours": [(rate5[m] > 0.5 * rate5[m].max()).sum()
                            for m in range(1, 13)],
        "width_gt30_hours": [(rate30[m] > 0.5 * rate30[m].max()).sum()
                             for m in range(1, 13)],
    })
    C.save_table(centroid.round(2), "tbl05_night_window", index=False)
    print(centroid.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
