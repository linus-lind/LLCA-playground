from __future__ import annotations

import logging
from typing import cast

import pandas as pd

from llca.preprocessing.modules.subgroup import subgroup_key

logger = logging.getLogger(__name__)


def _drop_sparse_columns(
    panel: pd.DataFrame, check_columns: list[str], threshold: float
) -> pd.DataFrame:
    """Drop globally sparse checked columns when no subgroup partition is configured."""
    rates = panel[check_columns].isna().mean()
    dropped = [str(column) for column in rates[rates > threshold].index]
    if dropped:
        logger.warning(
            "missing_threshold_filter: dropping columns with NaN rate > %.4f: %s",
            threshold,
            dropped,
        )
    return panel.drop(columns=dropped)


def _row_group(keys: list[pd.Index | pd.Series]) -> pd.Index:
    """Return one per-row group label aligned to ``pandas`` groupby-by-list semantics.

    ``groupby`` on a single-element key list yields a flat index of scalars, while two or
    more keys yield a ``MultiIndex`` of tuples. The membership test below must compare
    against the same structure, so a single key becomes a flat ``Index`` and multiple keys
    a ``MultiIndex``. ``pd.MultiIndex.from_arrays`` on one key would instead produce a
    one-level index of 1-tuples, which cannot be matched against the scalar group labels.
    """
    if len(keys) == 1:
        return cast(pd.Index, pd.Index(keys[0]))
    return pd.MultiIndex.from_arrays(keys)


def _drop_sparse_groups(
    panel: pd.DataFrame,
    check_columns: list[str],
    threshold: float,
    keys: list[pd.Index | pd.Series],
) -> pd.DataFrame:
    """Remove whole subgroups whose worst checked column exceeds the missing-rate threshold.

    Each row is assigned its composite group label; a group is kept only when every checked
    column's within-group NaN rate is at most ``threshold``. Works identically for a single
    grouping key (for example per-entity mode) and composite keys.
    """
    missing_rate = panel[check_columns].isna().groupby(keys).mean()
    worst_column_rate = missing_rate.max(axis=1)
    valid_groups = worst_column_rate[worst_column_rate <= threshold].index

    keep = _row_group(keys).isin(valid_groups)
    dropped = [str(group) for group in worst_column_rate[worst_column_rate > threshold].index]
    if dropped:
        logger.warning(
            "missing_threshold_filter: dropping %d/%d subgroups with "
            "worst-column NaN rate > %.4f: %s",
            len(dropped),
            len(worst_column_rate),
            threshold,
            dropped,
        )
    return panel[keep]


def missing_threshold_filter(
    panel: pd.DataFrame,
    threshold: float,
    subgroup_keys: list[str],
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Filter sparse data globally by column or locally by composite subgroup.

    Without subgroup keys, sparsity removes columns. With entity or configured subgroup
    keys, it removes all rows of a group whose worst checked column exceeds ``threshold``.
    Unchecked grouping columns are excluded from the default checked set.
    """
    check_columns = (
        list(columns) if columns else [c for c in panel.columns if c not in subgroup_keys]
    )
    keys = subgroup_key(panel, subgroup_keys)

    if not keys:
        return _drop_sparse_columns(panel, check_columns, threshold)
    return _drop_sparse_groups(panel, check_columns, threshold, keys)
