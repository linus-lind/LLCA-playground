from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import numpy as np


def _as_float(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=float)


def _require_same_shape(left: np.ndarray, right: np.ndarray) -> None:
    if left.shape != right.shape:
        raise ValueError(f"shape mismatch: {left.shape} != {right.shape}")


def _require_positive(values: np.ndarray, name: str) -> None:
    if bool((np.isfinite(values) & (values <= 0)).any()):
        raise ValueError(f"{name} must be strictly positive")


def _safe_log(values: np.ndarray) -> np.ndarray:
    values = _as_float(values)
    out = np.full(values.shape, np.nan, dtype=float)
    positive = values > 0
    out[positive] = np.log(values[positive])
    return out


def log_change(values: np.ndarray, *, horizon: int = 1) -> np.ndarray:
    """Return ``log(x[t]) - log(x[t-horizon])`` with unavailable leading rows as NaN."""
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    log_values = _safe_log(values)
    out = np.full(log_values.shape, np.nan, dtype=float)
    if horizon < log_values.shape[0]:
        out[horizon:] = log_values[horizon:] - log_values[:-horizon]
    return out


def simple_change(values: np.ndarray, *, horizon: int = 1) -> np.ndarray:
    """Return ``x[t] / x[t-horizon] - 1`` with invalid comparisons represented as NaN."""
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    values = _as_float(values)
    out = np.full(values.shape, np.nan, dtype=float)
    if horizon < values.shape[0]:
        current = values[horizon:]
        previous = values[:-horizon]
        valid = np.isfinite(current) & np.isfinite(previous) & (previous != 0.0)
        changed = np.full(current.shape, np.nan, dtype=float)
        changed[valid] = current[valid] / previous[valid] - 1.0
        out[horizon:] = changed
    return out


def log_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """Return element-wise log ratios, propagating non-positive inputs as NaN."""
    log_num = _safe_log(numerator)
    log_den = _safe_log(denominator)
    _require_same_shape(log_num, log_den)
    return cast(np.ndarray, log_num - log_den)


def ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """Return an element-wise ratio, representing zero denominators as NaN."""
    numerator = _as_float(numerator)
    denominator = _as_float(denominator)
    _require_same_shape(numerator, denominator)
    out = np.full(numerator.shape, np.nan, dtype=float)
    nonzero = denominator != 0
    out[nonzero] = numerator[nonzero] / denominator[nonzero]
    return out


def net_ratio(
    add: Sequence[np.ndarray],
    subtract: Sequence[np.ndarray],
    denominator: np.ndarray,
) -> np.ndarray:
    """Return ``(sum(add) - sum(subtract)) / denominator`` element-wise.

    Generalises :func:`ratio` to a signed linear combination in the numerator, so
    scaled aggregates such as gross profit ``(sales - cogs) / assets`` or cash-flow
    accruals ``(earnings - operating_cash_flow) / assets`` are expressible without a
    bespoke transform. A zero denominator yields NaN, matching :func:`ratio`, and any
    missing (NaN) term propagates into the affected rows so an incomplete numerator is
    never silently treated as zero. Every array must share the denominator's shape.
    """
    denominator = _as_float(denominator)
    numerator = np.zeros(denominator.shape, dtype=float)
    for values in add:
        term = _as_float(values)
        _require_same_shape(numerator, term)
        numerator = numerator + term
    for values in subtract:
        term = _as_float(values)
        _require_same_shape(numerator, term)
        numerator = numerator - term
    out = np.full(denominator.shape, np.nan, dtype=float)
    nonzero = denominator != 0
    out[nonzero] = numerator[nonzero] / denominator[nonzero]
    return out


def log_difference(current: np.ndarray, previous: np.ndarray) -> np.ndarray:
    """Return ``log(current[t]) - log(previous[t-1])`` with the first row unavailable."""
    log_current = _safe_log(current)
    log_previous = _safe_log(previous)
    _require_same_shape(log_current, log_previous)
    out = np.full(log_current.shape, np.nan, dtype=float)
    out[1:] = log_current[1:] - log_previous[:-1]
    return out


def range_location(high: np.ndarray, low: np.ndarray, value: np.ndarray) -> np.ndarray:
    """Map a value's location within ``[low, high]`` to the centered scale ``[-1, 1]``.

    Values outside the interval may exceed that range; zero-width intervals become NaN.
    All arrays must have identical shapes and are transformed element-wise.
    """
    high = _as_float(high)
    low = _as_float(low)
    value = _as_float(value)
    _require_same_shape(high, low)
    _require_same_shape(high, value)
    span = high - low
    out = np.full(high.shape, np.nan, dtype=float)
    nonzero = span != 0
    out[nonzero] = (2 * value[nonzero] - high[nonzero] - low[nonzero]) / span[nonzero]
    return out


def relative_spread(upper: np.ndarray, lower: np.ndarray) -> np.ndarray:
    """Return ``(upper - lower) / midpoint`` with zero midpoints represented as NaN."""
    upper = _as_float(upper)
    lower = _as_float(lower)
    _require_same_shape(upper, lower)
    mid = (upper + lower) / 2
    out = np.full(upper.shape, np.nan, dtype=float)
    nonzero = mid != 0
    out[nonzero] = (upper[nonzero] - lower[nonzero]) / mid[nonzero]
    return out
