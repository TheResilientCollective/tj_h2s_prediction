"""H2S Threshold Categorization Logic.

Applies client-specified thresholds to H2S measurements:
- Green: H2S < 5 ppb (safe)
- Yellow: 5 ≤ H2S < 30 ppb (caution)
- Orange: H2S ≥ 30 ppb (alert)

These thresholds were updated in January 2026 from the previous specification:
- Old Yellow: 5-15 ppb
- Old Orange: ≥15 ppb
"""

import pandas as pd
from typing import Optional


def categorize_h2s(value: float) -> Optional[str]:
    """Categorize H2S value using current client-specified thresholds.

    Args:
        value: H2S measurement in ppb

    Returns:
        Category string ('green', 'yellow', 'orange') or None if value is NaN

    Examples:
        >>> categorize_h2s(3.5)
        'green'
        >>> categorize_h2s(12.0)
        'yellow'
        >>> categorize_h2s(45.0)
        'orange'
    """
    if pd.isna(value):
        return None

    if value < 5:
        return 'green'
    elif value < 30:
        return 'yellow'
    else:
        return 'orange'


def apply_categorization(df: pd.DataFrame, h2s_column: str = 'H2S') -> pd.DataFrame:
    """Apply H2S categorization to a DataFrame.

    Args:
        df: DataFrame with H2S measurements
        h2s_column: Name of column containing H2S values (default: 'H2S')

    Returns:
        DataFrame with added 'h2s_category' column

    Raises:
        KeyError: If h2s_column not found in DataFrame
    """
    if h2s_column not in df.columns:
        raise KeyError(f"Column '{h2s_column}' not found in DataFrame")

    df = df.copy()
    df['h2s_category'] = df[h2s_column].apply(categorize_h2s)

    return df


def get_threshold_info() -> dict:
    """Return current H2S threshold configuration.

    Returns:
        Dict with threshold information for documentation and logging
    """
    return {
        'green_max': 5,
        'yellow_min': 5,
        'yellow_max': 30,
        'orange_min': 30,
        'version': '2.0',
        'effective_date': '2026-01',
        'description': 'Client-specified thresholds (updated Jan 2026)'
    }
