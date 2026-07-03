import numpy as np
import pandas as pd
from omegaconf import DictConfig

_ORDER_ROLES = ("high", "low", "open", "close", "bid", "ask")


def _invalid_ordering(panel: pd.DataFrame, ordering: DictConfig) -> pd.Series:
    """Mark rows violating configured price-bar or quote ordering relationships."""
    high, low, open_, close, bid, ask = (ordering.get(role) for role in _ORDER_ROLES)
    columns = panel.columns
    invalid = pd.Series(False, index=panel.index)

    if high in columns and low in columns:
        invalid |= panel[high] < panel[low]
    if high in columns and open_ in columns:
        invalid |= panel[high] < panel[open_]
    if high in columns and close in columns:
        invalid |= panel[high] < panel[close]
    if low in columns and open_ in columns:
        invalid |= panel[low] > panel[open_]
    if low in columns and close in columns:
        invalid |= panel[low] > panel[close]
    if bid in columns and ask in columns:
        invalid |= panel[bid] > panel[ask]

    return invalid


def _out_of_bounds(series: pd.Series, bounds: DictConfig) -> pd.Series:
    """Mark values violating any configured strict or inclusive scalar bound."""
    invalid = pd.Series(False, index=series.index)
    gt, ge, lt, le = (bounds.get(bound) for bound in ("gt", "ge", "lt", "le"))
    if gt is not None:
        invalid |= series <= gt
    if ge is not None:
        invalid |= series < ge
    if lt is not None:
        invalid |= series >= lt
    if le is not None:
        invalid |= series > le
    return invalid


def consistency_check(
    panel: pd.DataFrame,
    positive: list[str],
    non_negative: list[str],
    ordering: DictConfig,
    bounded: DictConfig,
) -> pd.DataFrame:
    """Replace economically inconsistent cells with missing values.

    Positivity and scalar bounds are evaluated per cell. Relational constraints such as
    high/low or bid/ask invalidate all participating configured columns on an affected row,
    preserving uncertainty rather than retaining a partially inconsistent tuple.
    """
    panel = panel.copy()
    ordering_columns = [column for column in ordering.values() if column is not None]

    for column in positive:
        panel.loc[panel[column] <= 0, column] = np.nan
    for column in non_negative:
        panel.loc[panel[column] < 0, column] = np.nan

    if ordering_columns:
        panel.loc[_invalid_ordering(panel, ordering), ordering_columns] = np.nan

    for name, bounds in bounded.items():
        column = str(name)
        panel.loc[_out_of_bounds(panel[column], bounds), column] = np.nan

    return panel
