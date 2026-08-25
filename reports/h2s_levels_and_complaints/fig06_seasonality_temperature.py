"""Figure 6 — the seasonal cycle, and whether temperature explains it.

The working hypothesis this figure was built to test: *H2S concentrates more on
cool nights, which is why summer nights are quiet and winter and spring nights
are not.*

The first half of that is right and large. The second half — that temperature is
the mechanism — does not survive. Three tests are run:

  (a) **Is the cycle real?** The two years of record are plotted separately. A
      seasonal claim from a single spring would be one event, not a season.
  (b) **Does temperature act between months or within them?** Between calendar
      months, cooler is dramatically dirtier. Within a month, night-to-night
      temperature deviations carry almost nothing. A variable that only works
      between months is a marker for the season, not a lever on the hazard.
  (c) **Do the local dispersion drivers explain it?** If cool nights trapped
      more, then calm and stable nights should be the dirty ones and the
      seasonal cycle should follow the seasonal cycle in calm/stable nights.
      It runs the other way: August has the calmest and most stable nights of
      the year and the cleanest air.

Everything is computed on night hours (20:00–08:00 local), which is where
essentially the whole hazard lives — see Figure 4.
"""

from __future__ import annotations

import calendar

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

import common as C

STATION = C.NESTOR
NIGHT_START, NIGHT_END = 20, 8   # local hours; night is [20:00, 08:00)


def night_hours(df: pd.DataFrame) -> pd.DataFrame:
    h = df["time"].dt.hour
    out = df[(h >= NIGHT_START) | (h < NIGHT_END)].copy()
    out["month"] = out["time"].dt.month
    out["year_month"] = out["time"].dt.strftime("%Y-%m")
    # Nights are labelled by the date they opened on, so a single night is not
    # split across two calendar months at the boundary.
    out["season_year"] = (out["time"] - pd.Timedelta(hours=12)).dt.year
    return out


def main() -> None:
    C.style()
    d = night_hours(C.load_h2s(STATION))

    monthly = d.groupby("year_month").agg(
        n_hours=("H2S", "size"),
        pct_gt5=("H2S", lambda x: 100 * (x > C.T_YELLOW).mean()),
        pct_gt30=("H2S", lambda x: 100 * (x > C.T_ORANGE).mean()),
        median_h2s=("H2S", "median"),
        p95_h2s=("H2S", lambda x: x.quantile(0.95)),
        mean_temp=("temperature_2m", "mean"),
        mean_wind=("wind_speed_10m", "mean"),
        mean_humidity=("relative_humidity_2m", "mean"),
        frac_stable=("stable_atm", "mean"),
    )
    monthly["month"] = [int(k[5:]) for k in monthly.index]
    monthly["year"] = [int(k[:4]) for k in monthly.index]
    C.save_table(monthly.round(3), "tbl06_monthly_night_climatology")

    fig = plt.figure(figsize=(12.4, 7.2))
    gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.32)

    # --- (a) the cycle, one line per year -----------------------------------
    ax = fig.add_subplot(gs[0, :2])
    for year, colour in zip(sorted(monthly["year"].unique()),
                            ["#1f77b4", "#d62728", "#2ca02c"]):
        sub = monthly[monthly["year"] == year].sort_values("month")
        ax.plot(sub["month"], sub["pct_gt5"], "-o", ms=4, color=colour,
                label=f"{year}  (>5 ppb)")
        ax.plot(sub["month"], sub["pct_gt30"], "--s", ms=3.5, color=colour,
                alpha=0.6, label=f"{year}  (>30 ppb)")
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels([calendar.month_abbr[m] for m in range(1, 13)])
    ax.set_ylabel("% of night hours over threshold")
    ax.set_ylim(0, 92)
    ax.set_title("(a) The seasonal cycle repeats across both years of record — "
                 "spring peak, late-summer minimum")
    ax.legend(ncol=3, fontsize=7, loc="upper left")

    tax = ax.twinx()
    clim = monthly.groupby("month")["mean_temp"].mean()
    tax.plot(clim.index, clim.values, ":", color="#8c564b", lw=2)
    tax.set_ylabel("mean night temperature (°C)", color="#8c564b")
    tax.tick_params(axis="y", colors="#8c564b")
    tax.grid(False)

    # --- (b) between-month vs within-month ----------------------------------
    ax = fig.add_subplot(gs[0, 2])
    ax.scatter(monthly["mean_temp"], monthly["pct_gt5"], s=26, c=monthly["month"],
               cmap="twilight", edgecolor="k", linewidth=0.4, zorder=3)
    r_between = stats.spearmanr(monthly["mean_temp"], monthly["pct_gt5"])
    fitx = np.linspace(monthly["mean_temp"].min(), monthly["mean_temp"].max(), 10)
    b = np.polyfit(monthly["mean_temp"], monthly["pct_gt5"], 1)
    ax.plot(fitx, np.polyval(b, fitx), "-", color="#666", lw=1.2)
    ax.set_xlabel("month mean night temperature (°C)")
    ax.set_ylabel("% of night hours > 5 ppb")
    ax.set_title(f"(b) Between months: ρ = {r_between.statistic:+.2f}\n"
                 f"(p = {r_between.pvalue:.1e}, n = {len(monthly)} months)")

    # --- (c) within-month partial associations ------------------------------
    ax = fig.add_subplot(gs[1, 0])
    w = d.dropna(subset=["temperature_2m", "wind_speed_10m", "H2S"]).copy()
    w["h2s_rank"] = w.groupby("month")["H2S"].rank(pct=True)
    within = {}
    for col, label in (("temperature_2m", "temperature"),
                       ("wind_speed_10m", "wind speed"),
                       ("relative_humidity_2m", "humidity"),
                       ("stable_atm", "stability flag")):
        dev = w[col] - w.groupby("month")[col].transform("mean")
        within[label] = stats.spearmanr(dev, w["h2s_rank"], nan_policy="omit").statistic
    between = {}
    for col, label in (("mean_temp", "temperature"), ("mean_wind", "wind speed"),
                       ("mean_humidity", "humidity"), ("frac_stable", "stability flag")):
        between[label] = stats.spearmanr(monthly[col], monthly["pct_gt5"]).statistic
    labels = list(within)
    xpos = np.arange(len(labels))
    ax.bar(xpos - 0.2, [between.get(k, np.nan) for k in labels], width=0.38,
           color="#1f77b4", label="between months")
    ax.bar(xpos + 0.2, [within[k] for k in labels], width=0.38,
           color="#ff7f0e", label="within month")
    ax.axhline(0, color="#333", lw=0.8)
    ax.set_xticks(xpos)
    ax.set_xticklabels(labels, rotation=20)
    ax.set_ylabel("Spearman ρ")
    ax.set_title("(c) Temperature works only between\nmonths — it marks the season")
    ax.legend(fontsize=7.5)

    # --- (d) dispersion drivers run the wrong way ---------------------------
    ax = fig.add_subplot(gs[1, 1])
    order = monthly.groupby("month")[["mean_wind", "frac_stable", "pct_gt5"]].mean()
    ax.plot(order.index, order["mean_wind"], "-o", ms=4, color="#1f77b4",
            label="mean night wind (m/s)")
    ax.plot(order.index, 10 * order["frac_stable"], "-s", ms=4, color="#9467bd",
            label="stable-atmosphere fraction ×10")
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels([calendar.month_abbr[m][0] for m in range(1, 13)])
    ax.set_ylabel("m/s  /  fraction ×10")
    dax = ax.twinx()
    dax.plot(order.index, order["pct_gt5"], "-", color=C.TIER_COLORS["orange"],
             lw=2.4, label="% night hours > 5 ppb")
    dax.set_ylabel("% night hours > 5 ppb", color=C.TIER_COLORS["orange"])
    dax.tick_params(axis="y", colors=C.TIER_COLORS["orange"])
    dax.grid(False)
    ax.set_ylim(2.6, 8.6)
    ax.set_title("(d) August has the calmest, most stable\nnights — and the cleanest air")
    ax.legend(fontsize=7, loc="upper left")

    # --- (e) the cycle survives conditioning on wind direction --------------
    ax = fig.add_subplot(gs[1, 2])
    sector = (d["wind_direction_10m"] % 360)
    d["drainage"] = ((sector >= 22.5) & (sector < 157.5))  # NE–SE, down-valley
    for flag, colour, lbl in ((True, "#8c564b", "down-valley wind (NE–SE)"),
                              (False, "#17becf", "all other directions")):
        sub = d[d["drainage"] == flag]
        g = sub.groupby("month")["H2S"].agg(
            pct=lambda x: 100 * (x > C.T_YELLOW).mean(), n="size")
        g = g[g["n"] >= 30]
        ax.plot(g.index, g["pct"], "-o", ms=4, color=colour, label=lbl)
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels([calendar.month_abbr[m][0] for m in range(1, 13)])
    ax.set_ylabel("% of night hours > 5 ppb")
    ax.set_title("(e) The cycle is still there inside\neach wind regime")
    ax.legend(fontsize=7.5)

    C.save(fig, "fig06_seasonality_temperature")
    plt.close(fig)

    stats_tbl = pd.DataFrame(
        [{"association": k, "between_months": between.get(k, np.nan),
          "within_month": within[k]} for k in labels]
    )
    C.save_table(stats_tbl, "tbl06_temperature_association", index=False)

    print(stats_tbl.round(3).to_string(index=False))
    print("\nmonthly night climatology (pooled over years):")
    pooled = d.groupby("month").agg(
        n_hours=("H2S", "size"),
        pct_gt5=("H2S", lambda x: round(100 * (x > C.T_YELLOW).mean(), 1)),
        pct_gt30=("H2S", lambda x: round(100 * (x > C.T_ORANGE).mean(), 1)),
        median=("H2S", "median"),
        p95=("H2S", lambda x: round(x.quantile(0.95), 1)),
        temp=("temperature_2m", "mean"),
        wind=("wind_speed_10m", "mean"),
        stable=("stable_atm", "mean"),
    ).round(2)
    print(pooled.to_string())
    C.save_table(pooled, "tbl06_pooled_month_climatology")


if __name__ == "__main__":
    main()
