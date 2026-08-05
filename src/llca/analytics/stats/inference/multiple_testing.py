"""Family-wise and false-discovery multiple-testing corrections.

The adjustments operate on plain arrays of p-values (ignoring NaN entries) and on the unique
off-diagonal family of a symmetric comparison matrix.

References
----------
Holm (1979); Benjamini & Hochberg (1995).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def holm_adjust(p_values: np.ndarray) -> np.ndarray:
    """Control the family-wise error rate across ``p_values`` with Holm's (1979) step-down method.

    Returns adjusted p-values in the input's positions. NaN entries are excluded from the
    family and preserved as NaN; the finite entries are scaled by their descending rank and
    made monotone so each is at least as large as any more significant one, capped at one.
    """
    array = np.asarray(p_values, dtype=float)
    finite = np.isfinite(array)
    result = np.full(array.shape, np.nan)
    values = array[finite]
    m = values.shape[0]
    if m == 0:
        return result
    order = np.argsort(values)
    running = 0.0
    adjusted = np.empty(m)
    for rank, position in enumerate(order):
        candidate = (m - rank) * values[position]
        running = max(running, candidate)
        adjusted[position] = min(running, 1.0)
    result[finite] = adjusted
    return result


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    """Control the false-discovery rate across ``p_values`` with Benjamini-Hochberg (1995).

    Returns adjusted p-values in the input's positions. NaN entries are excluded from the
    family and preserved as NaN; the finite entries are scaled by the total count over their
    ascending rank and made monotone from the largest downward, capped at one.
    """
    array = np.asarray(p_values, dtype=float)
    finite = np.isfinite(array)
    result = np.full(array.shape, np.nan)
    values = array[finite]
    m = values.shape[0]
    if m == 0:
        return result
    order = np.argsort(values)
    adjusted = np.empty(m)
    running = 1.0
    for rank in range(m - 1, -1, -1):
        position = order[rank]
        candidate = values[position] * m / (rank + 1)
        running = min(running, candidate)
        adjusted[position] = min(running, 1.0)
    result[finite] = adjusted
    return result


def adjust_pairwise(matrix: pd.DataFrame, method: str) -> pd.DataFrame:
    """Correct the p-values of a symmetric comparison matrix for multiple testing.

    Treats the upper-triangular entries as the family of distinct pairwise comparisons,
    adjusts them by ``method`` (``"holm"`` or ``"bh"``), and mirrors the results back into both
    triangles. ``"none"`` returns the matrix unchanged; any other value raises. The diagonal is
    left untouched.
    """
    if method == "none":
        return matrix
    corrector = {"holm": holm_adjust, "bh": benjamini_hochberg}.get(method)
    if corrector is None:
        raise ValueError(f"unknown multiple-testing correction {method!r}")
    result = matrix.copy()
    rows, columns = np.triu_indices(len(matrix), k=1)
    family = matrix.to_numpy(dtype=float)[rows, columns]
    adjusted = corrector(family)
    values = result.to_numpy(dtype=float)
    values[rows, columns] = adjusted
    values[columns, rows] = adjusted
    return pd.DataFrame(values, index=matrix.index, columns=matrix.columns)
