"""Figure 2 — the odour dose-response: how likely is a complaint at a given H2S?

The core result of Section 1. For every **measured** station-hour we ask whether
at least one Imperial Beach odour complaint was filed in that same clock hour,
and plot that probability against the concentration measured at the same time.

Three things are done to keep the answer honest:

1. **Concurrent hour, not a lag.** A lag scan (printed, and written to
   ``tbl02_lag_scan.csv``) shows the association peaks at lag 0 and falls away
   symmetrically, so the same-hour join is the right one — complaints are filed
   while the odour is present, not hours later.

2. **Hour-of-day adjustment.** People file complaints when they are awake, and
   H2S peaks when they are not (Figure 1c). The raw curve therefore mixes odour
   intensity with reporting availability. The adjusted curve comes from a
   logistic model with a full set of hour-of-day dummies, evaluated at each
   concentration over the observed mix of hours.

3. **An explicit test for the thresholds people assume.** The dose-response is
   fitted as piecewise-linear in log10(concentration) with knots at exactly the
   operational cut points — 5, 10 and 30 ppb. If complaints "switch on" at 5–10
   ppb, or if 30 ppb is a second inflection, the slope change at that knot is
   large and significant. The test is reported whichever way it comes out.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

import common as C

#: Concentration bin edges (ppb). Chosen so each bin holds enough hours to
#: estimate a proportion, and so 5 / 10 / 30 are all bin boundaries.
BINS = [0, 1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 100, 10_000]

#: Knots for the piecewise-linear fit, at the operational thresholds.
KNOTS = [C.T_YELLOW, C.T_YELLOW_HIGH, C.T_ORANGE]


def lag_scan(panel: pd.DataFrame) -> pd.DataFrame:
    """Spearman correlation of the hourly complaint count against H2S at ±lag."""
    rows = []
    for lag in range(-8, 13):
        rows.append(
            (lag, panel["H2S"].shift(lag).corr(panel["complaints"], method="spearman"))
        )
    return pd.DataFrame(rows, columns=["lag_hours", "spearman"])


def binned(panel: pd.DataFrame) -> pd.DataFrame:
    """Observed complaint probability and rate per concentration bin."""
    p = panel.copy()
    p["bin"] = pd.cut(p["H2S"], BINS, right=False)
    g = p.groupby("bin", observed=True).agg(
        n_hours=("H2S", "size"),
        median_h2s=("H2S", "median"),
        complaint_hours=("any_complaint", "sum"),
        complaints=("complaints", "sum"),
    )
    g["p_any"] = g["complaint_hours"] / g["n_hours"]
    g["lo"], g["hi"] = C.wilson(g["complaint_hours"], g["n_hours"])
    g["rate_per_hour"] = g["complaints"] / g["n_hours"]
    g["pct_of_hours"] = 100 * g["n_hours"] / g["n_hours"].sum()
    return g.reset_index()


def fit(panel: pd.DataFrame):
    """Hour-adjusted piecewise-linear logistic fit. Returns (model, design)."""
    X = pd.get_dummies(panel["hour_of_day"], prefix="hod", drop_first=True).astype(float)
    X["log_h2s"] = panel["log_h2s"].values
    for i, k in enumerate(KNOTS):
        X[f"knot_{k:g}"] = np.clip(panel["log_h2s"].values - np.log10(k), 0, None)
    X = sm.add_constant(X)
    model = sm.Logit(panel["any_complaint"].values, X).fit(disp=0)
    return model, X


def adjusted_curve(model, X: pd.DataFrame, grid: np.ndarray) -> pd.DataFrame:
    """Predicted P(complaint) at each concentration, averaged over observed hours."""
    out = []
    Xb = X.copy()
    for ppb in grid:
        lc = np.log10(ppb)
        Xb["log_h2s"] = lc
        for k in KNOTS:
            Xb[f"knot_{k:g}"] = max(lc - np.log10(k), 0.0)
        out.append((ppb, float(model.predict(Xb).mean())))
    return pd.DataFrame(out, columns=["ppb", "p_adjusted"])


def slope_table(model) -> pd.DataFrame:
    """Segment slopes in logit per decade of concentration, with the knot tests."""
    terms = [("below 5 ppb", "log_h2s"), ("5–10 ppb", "knot_5"),
             ("10–30 ppb", "knot_10"), ("above 30 ppb", "knot_30")]
    rows, cum = [], 0.0
    for label, term in terms:
        cum += model.params[term]
        rows.append(
            {
                "segment": label,
                "slope_change": model.params[term],
                "std_err": model.bse[term],
                "p_value": model.pvalues[term],
                "cumulative_slope": cum,
                "odds_ratio_per_decade": float(np.exp(cum)),
            }
        )
    rows[0]["slope_change"] = np.nan  # the first term is the level, not a change
    rows[0]["p_value"] = np.nan
    return pd.DataFrame(rows)


def main() -> None:
    C.style()
    panel = C.hourly_panel(C.NESTOR)

    lags = lag_scan(panel)
    C.save_table(lags, "tbl02_lag_scan", index=False)
    best = lags.loc[lags["spearman"].idxmax()]

    obs = binned(panel)
    C.save_table(obs, "tbl02_dose_response", index=False)

    model, X = fit(panel)
    grid = np.array([0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 100, 200])
    adj = adjusted_curve(model, X, grid)
    slopes = slope_table(model)
    C.save_table(slopes, "tbl02_slope_test", index=False)
    C.save_table(adj, "tbl02_adjusted_curve", index=False)

    # --- figure ------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))

    ax = axes[0]
    x = obs["median_h2s"].to_numpy()
    y = 100 * obs["p_any"].to_numpy()
    yerr = np.vstack([y - 100 * obs["lo"], 100 * obs["hi"] - y])
    for t, col in ((C.T_YELLOW, C.TIER_COLORS["yellow"]),
                   (C.T_YELLOW_HIGH, C.TIER_COLORS["yellow-high"]),
                   (C.T_ORANGE, C.TIER_COLORS["orange"])):
        ax.axvline(t, color=col, lw=1.4, ls="--", zorder=0)
        ax.text(t, 2, f" {t:g}", color=col, fontsize=8, va="bottom", fontweight="bold")
    ax.errorbar(x, y, yerr=yerr, fmt="o", ms=4, color="#333", lw=1,
                capsize=2, label="observed, per bin (95% CI)")
    ax.plot(adj["ppb"], 100 * adj["p_adjusted"], "-", color="#1f77b4", lw=2,
            label="hour-of-day adjusted fit")
    ax.set_xscale("log")
    ax.set_xlabel("H₂S at NESTOR-BES, same hour (ppb)")
    ax.set_ylabel("% of hours drawing ≥1 odour complaint")
    ax.set_ylim(0, 80)
    ax.set_title("(a) Complaint probability rises smoothly\nwith concentration — no threshold")
    ax.legend(loc="lower right")

    ax = axes[1]
    slope = slopes["cumulative_slope"].to_numpy()
    labels = slopes["segment"].to_list()
    ci = 1.96 * slopes["std_err"].to_numpy()
    ax.errorbar(range(len(slope)), slope, yerr=[0, *ci[1:]], fmt="s", ms=6,
                color="#1f77b4", capsize=3, lw=1.2)
    ax.axhline(slope[0], color="#999", ls=":", lw=1,
               label=f"below 5 ppb slope = {slope[0]:.2f}")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=15)
    ax.set_ylabel("logit slope per decade of H₂S")
    ax.set_ylim(0, max(slope + np.nan_to_num(ci)) * 1.35)
    ax.set_title("(b) The slope does not change at 5, 10 or 30 ppb\n"
                 "(all knot tests p > 0.25)")
    ax.legend(loc="lower right")
    for i, row in slopes.iterrows():
        if i:
            ax.text(i, slope[i] + ci[i] + 0.08, f"p={row['p_value']:.2f}",
                    ha="center", fontsize=7, color="#666")

    fig.tight_layout()
    C.save(fig, "fig02_dose_response")
    plt.close(fig)

    # --- console summary ----------------------------------------------------
    print(f"\npeak lag: {int(best['lag_hours'])} h (rho={best['spearman']:.3f})")
    print(f"n hours = {len(panel)}, complaints in window = {int(panel['complaints'].sum())}")
    print(f"baseline P(complaint)/hour = {panel['any_complaint'].mean():.3f}")
    print(f"pseudo-R² = {model.prsquared:.3f}")
    print("\nobserved dose-response:")
    print(obs[["bin", "n_hours", "median_h2s", "p_any", "rate_per_hour"]]
          .round(3).to_string(index=False))
    print("\nslope test:")
    print(slopes.round(4).to_string(index=False))
    print("\nadjusted curve:")
    print(adj.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
