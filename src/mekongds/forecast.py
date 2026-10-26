"""Feature engineering shared by training (nb 04) and inference (nb 05).

One implementation so the two cannot drift apart.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

OWN_LAGS = {"KOM": [0, 1, 2, 3, 4, 5, 6], "PRE": [0, 1, 2, 3, 5, 7, 14], "KOH": [0, 1, 2, 3, 5, 7, 14]}
ROLLS = {"KOM": [3, 7, 15], "PRE": [3, 7, 15, 30], "KOH": [3, 7, 15, 30]}
RISES = [1, 3]
UP = {
    "KOM": {"KRA": [0, 1, 2], "STR": [1, 2, 3], "PKS": [3, 4, 5], "KHO": [3, 4, 5],
            "MUK": [4, 5, 6], "SAV": [4, 5, 6], "NAK": [5, 6, 7], "PAK": [6, 7, 8]},
    "PRE": {"KOM": [0, 5, 10, 15], "KOH": [0, 1, 2], "PPP": [0, 1, 2], "KRA": [0, 1, 2], "NEA": [0, 1, 2]},
    "KOH": {"PPP": [0, 1, 2], "PPB": [0, 1, 2], "KOM": [0, 1, 2], "NEA": [0, 1, 2], "PRE": [0, 1, 2]},
}


def build_features(w: pd.DataFrame, station: str, include_upstream: bool = True) -> pd.DataFrame:
    """Build the feature matrix for `station` from the wide daily stage matrix."""
    s = w[station]
    cols = {}
    for lag in OWN_LAGS[station]:
        cols[f"{station}_lag{lag}"] = s.shift(lag)
    for win in ROLLS[station]:
        cols[f"{station}_mean{win}"] = s.rolling(win).mean()
    for r in RISES:
        cols[f"{station}_rise{r}"] = s - s.shift(r)
    if include_upstream:
        for up, lags in UP[station].items():
            for lag in lags:
                cols[f"{up}_lag{lag}"] = w[up].shift(lag)
    doy = w.index.dayofyear.values
    for k in (1, 2, 3):
        cols[f"sin{k}"] = np.sin(2 * np.pi * k * doy / 365.25)
        cols[f"cos{k}"] = np.cos(2 * np.pi * k * doy / 365.25)
    return pd.DataFrame(cols, index=w.index)
