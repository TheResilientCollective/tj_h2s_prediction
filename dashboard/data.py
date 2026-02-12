"""Data loading and filtering for the H2S dashboard."""

import pandas as pd
import panel as pn

from .constants import (
    COMPLAINTS_URL,
    H2S_DATA_URL,
    H2S_GREEN_MAX,
    H2S_YELLOW_MAX,
    LOCATIONS_URL,
)


@pn.cache(ttl=300)
def load_h2s_data() -> pd.DataFrame:
    """Load H2S parquet data from public S3 URL."""
    df = pd.read_parquet(H2S_DATA_URL)
    df["time"] = pd.to_datetime(df["time"])
    if df["time"].dt.tz is None:
        df["time"] = df["time"].dt.tz_localize("America/Los_Angeles")
    else:
        df["time"] = df["time"].dt.tz_convert("America/Los_Angeles")

    # Assign H2S category
    df["h2s_category"] = pd.cut(
        df["H2S"],
        bins=[-float("inf"), H2S_GREEN_MAX, H2S_YELLOW_MAX, float("inf")],
        labels=["green", "yellow", "orange"],
    )

    # Derived time columns
    df["year"] = df["time"].dt.year
    df["week"] = df["time"].dt.isocalendar().week.astype(int)
    df["date"] = df["time"].dt.date

    return df


@pn.cache(ttl=300)
def load_locations() -> pd.DataFrame:
    """Load site locations CSV from public S3 URL."""
    return pd.read_csv(LOCATIONS_URL)


@pn.cache(ttl=300)
def load_complaints() -> pd.DataFrame:
    """Load complaints CSV from public S3 URL."""
    df = pd.read_csv(COMPLAINTS_URL)
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    df["week_start"] = df["date"].dt.to_period("W").apply(lambda p: p.start_time)
    return df


def filter_data(
    df: pd.DataFrame,
    year: int,
    sites: list[str],
) -> pd.DataFrame:
    """Filter H2S data by year and selected sites."""
    mask = (df["year"] == year) & (df["site_name"].isin(sites))
    return df.loc[mask].copy()
