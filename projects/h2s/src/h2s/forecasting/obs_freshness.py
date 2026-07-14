"""Near-real-time H2S seed top-up for the recursive forecast engines.

Root cause (2026-07 red-tier recall investigation, see
docs/RED_TIER_RECALL_DIAGNOSIS.md): the products and daily pipelines seed the
autoregressive recursion from ``modeldata_h2s_nofill.parquet``, whose newest
rows end at the last *complete day* — 7 to 36+ hours behind real time. The
recursion therefore never saw an in-progress event: during the 2026-06-27
218.8 ppb NESTOR-BES peak the "nowcast" predicted 1.3–3.7 ppb with p30 < 3%,
while the same deployed models score p30 ≈ 0.9 when given the true lags.

The APCD feed (``tijuana/sd_apcd_air/output/hs2_lastday.csv``) refreshes every
few minutes with ~1–2 h data latency. This module extends the historical H2S
series with those rows so the recursion is seeded at (or near) the current
hour. Only the five autoregressive H2S features need this — the exogenous
features still come from the forecast met frame.
"""

from __future__ import annotations

import pandas as pd

from h2s.constants import (
    APCD_H2S_PARAMETER,
    APCD_HS2_LASTDAY_PATH,
    APCD_PUBLIC_BUCKET,
)

# Same QC cut the modeldata loaders apply (drop >500 ppb, clip negatives to 0).
H2S_MAX_VALID_PPB = 500.0


def load_realtime_h2s(s3) -> pd.DataFrame:
    """Load the near-real-time APCD H2S feed → [site_name, time, H2S].

    Reads ``hs2_lastday.csv`` from the public bucket, filters to the H2S
    parameter, applies the same QC as the modeldata loaders, and floors
    timestamps to the hour (the feed is hourly; flooring makes dedup exact).
    Site names in the feed match the modeldata ``site_name`` values.
    """
    url = s3.publicUrl(path=APCD_HS2_LASTDAY_PATH, bucket=APCD_PUBLIC_BUCKET)
    df = pd.read_csv(url)

    if "Parameter" in df.columns:
        df = df[df["Parameter"] == APCD_H2S_PARAMETER]

    df = df.rename(columns={
        "Site Name": "site_name",
        "Date with time": "time",
        "Result": "H2S",
    })
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    df["H2S"] = pd.to_numeric(df["H2S"], errors="coerce")
    df = df.dropna(subset=["site_name", "time", "H2S"])
    df = df[df["H2S"] <= H2S_MAX_VALID_PPB]
    df["H2S"] = df["H2S"].clip(lower=0)
    df["time"] = df["time"].dt.floor("h")

    return (
        df.sort_values(["site_name", "time"])
        .drop_duplicates(["site_name", "time"], keep="last")
        .loc[:, ["site_name", "time", "H2S"]]
        .reset_index(drop=True)
    )


def merge_h2s_series(
    hist_df: pd.DataFrame,
    realtime_df: pd.DataFrame | None,
    site_name: str,
) -> pd.DataFrame:
    """One site's hourly H2S series: historical rows topped up with realtime.

    Realtime rows only *extend* the series (times strictly after the last
    historical row) — the QC'd historical feed stays authoritative where the
    two overlap. Returns [time, H2S] sorted ascending; empty frame if the site
    has no rows in either source.
    """
    cols = ["time", "H2S"]

    if "site_name" in hist_df.columns:
        out = hist_df.loc[hist_df["site_name"] == site_name, cols].copy()
    else:
        out = hist_df.loc[:, cols].copy()
    out = out.sort_values("time")

    if realtime_df is not None and len(realtime_df):
        rt = realtime_df.loc[realtime_df["site_name"] == site_name, cols]
        if len(out):
            rt = rt[rt["time"] > out["time"].max()]
        if len(rt):
            out = pd.concat([out, rt], ignore_index=True).sort_values("time")

    return out.drop_duplicates("time", keep="last").reset_index(drop=True)


def seed_gap_hours(series: pd.DataFrame, now: pd.Timestamp | None = None) -> float | None:
    """Hours between the newest observation in ``series`` and ``now``.

    The recursion treats ``series[-1]`` as one hour before lead 1, so this is
    the effective staleness of the autoregressive seed. None if empty.
    """
    if series is None or len(series) == 0:
        return None
    now = now if now is not None else pd.Timestamp.now("UTC")
    return float((now - series["time"].iloc[-1]).total_seconds() / 3600.0)
