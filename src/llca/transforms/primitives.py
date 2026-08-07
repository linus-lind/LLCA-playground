from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence
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


def positive_indicator(values: np.ndarray, *, horizon: int = 1) -> np.ndarray:
    """Return ``1`` where the horizon simple change is positive, ``0`` where non-positive.

    The label is the direction of the same return :func:`simple_change` produces, so shifting
    it identically yields a classification target aligned with a shifted forward return. Rows
    whose change is undefined (leading horizon, non-finite, zero denominator) stay NaN so they
    are dropped from supervision rather than counted as a downward move.
    """
    change = simple_change(values, horizon=horizon)
    out = np.full(change.shape, np.nan, dtype=float)
    observed = np.isfinite(change)
    out[observed] = (change[observed] > 0.0).astype(float)
    return out


def log_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """Return element-wise log ratios, propagating non-positive inputs as NaN."""
    log_num = _safe_log(numerator)
    log_den = _safe_log(denominator)
    _require_same_shape(log_num, log_den)
    return cast(np.ndarray, log_num - log_den)


def simple_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """Return element-wise ``numerator / denominator - 1`` with zero denominators as NaN."""
    numerator = _as_float(numerator)
    denominator = _as_float(denominator)
    _require_same_shape(numerator, denominator)
    out = np.full(numerator.shape, np.nan, dtype=float)
    nonzero = denominator != 0
    out[nonzero] = numerator[nonzero] / denominator[nonzero] - 1.0
    return out


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


def simple_difference(current: np.ndarray, previous: np.ndarray) -> np.ndarray:
    """Return ``current[t] / previous[t-1] - 1`` with the first row unavailable."""
    current = _as_float(current)
    previous = _as_float(previous)
    _require_same_shape(current, previous)
    out = np.full(current.shape, np.nan, dtype=float)
    curr = current[1:]
    prev = previous[:-1]
    valid = np.isfinite(curr) & np.isfinite(prev) & (prev != 0.0)
    changed = np.full(curr.shape, np.nan, dtype=float)
    changed[valid] = curr[valid] / prev[valid] - 1.0
    out[1:] = changed
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


def _trailing_windows(values: np.ndarray, window: int) -> np.ndarray:
    """Return a ``[len(values), window]`` array whose row ``t`` holds ``values[t-window+1 .. t]``.

    The leading ``window - 1`` positions are padded with NaN so the result preserves the
    input length; callers reduce each row with a NaN-aware statistic. The padding is local
    to one array, so applying this per entity never mixes an entity's history with another's.
    """
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    values = _as_float(values)
    if values.size == 0:
        return np.empty((0, window), dtype=float)
    padded = np.concatenate([np.full(window - 1, np.nan, dtype=float), values])
    return np.lib.stride_tricks.sliding_window_view(padded, window)


def _resolve_min_periods(window: int, min_periods: int | None) -> int:
    """Return the required finite-observation count for a valid window, defaulting to full."""
    if min_periods is None:
        return window
    if min_periods < 1:
        raise ValueError(f"min_periods must be >= 1, got {min_periods}")
    return min(min_periods, window)


def _rolling_reduce(
    values: np.ndarray,
    *,
    window: int,
    statistic: Callable[[np.ndarray], np.ndarray],
    min_periods: int | None,
) -> np.ndarray:
    """Apply a NaN-aware row ``statistic`` to trailing windows, blanking sparse windows.

    Rows observing fewer than ``min_periods`` finite values (default: the full ``window``)
    become NaN, so a statistic is only emitted once enough history has accrued. Empty-slice
    and degenerate-variance warnings from the underlying reductions are suppressed because
    those rows are masked out regardless.
    """
    windows = _trailing_windows(values, window)
    required = _resolve_min_periods(window, min_periods)
    counts = np.isfinite(windows).sum(axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        reduced = statistic(windows)
    return np.where(counts >= required, reduced, np.nan)


def rolling_volatility(
    values: np.ndarray, *, window: int, min_periods: int | None = None
) -> np.ndarray:
    """Return the trailing sample standard deviation of one-step log returns.

    Realized volatility over ``window`` return observations ending at each row; the leading
    price row yields no return, so the first usable window starts one row later.
    """
    returns = log_change(_as_float(values), horizon=1)
    return _rolling_reduce(
        returns,
        window=window,
        statistic=lambda w: np.nanstd(w, axis=1, ddof=1),
        min_periods=min_periods,
    )


def downside_deviation(
    values: np.ndarray, *, window: int, min_periods: int | None = None
) -> np.ndarray:
    """Return the trailing root-mean-square of negative one-step log returns.

    The Sortino-style downside deviation with a zero minimum-acceptable-return: positive
    returns contribute nothing but still count toward the window, so the measure scales with
    both the frequency and severity of losses.
    """
    returns = log_change(_as_float(values), horizon=1)

    def statistic(w: np.ndarray) -> np.ndarray:
        losses = np.where(np.isfinite(w), np.minimum(w, 0.0) ** 2, np.nan)
        return cast(np.ndarray, np.sqrt(np.nanmean(losses, axis=1)))

    return _rolling_reduce(returns, window=window, statistic=statistic, min_periods=min_periods)


def rolling_skewness(
    values: np.ndarray, *, window: int, min_periods: int | None = None
) -> np.ndarray:
    """Return the trailing population skewness of one-step log returns.

    Uses the biased moment estimator ``m3 / m2**1.5``; windows with zero return dispersion
    have undefined shape and become NaN.
    """
    returns = log_change(_as_float(values), horizon=1)

    def statistic(w: np.ndarray) -> np.ndarray:
        mean = np.nanmean(w, axis=1, keepdims=True)
        deviation = w - mean
        m2 = np.nanmean(deviation**2, axis=1)
        m3 = np.nanmean(deviation**3, axis=1)
        variance = np.where(m2 > 0, m2, np.nan)
        return cast(np.ndarray, m3 / variance**1.5)

    return _rolling_reduce(returns, window=window, statistic=statistic, min_periods=min_periods)


def high_proximity(
    value: np.ndarray, high: np.ndarray, *, window: int, min_periods: int | None = None
) -> np.ndarray:
    """Return ``value`` relative to its trailing maximum ``high`` over ``window`` rows.

    A proximity of one places the value at its rolling peak; non-positive or unavailable
    rolling maxima yield NaN. All arrays must share a shape.
    """
    value = _as_float(value)
    high = _as_float(high)
    _require_same_shape(value, high)
    rolling_max = _rolling_reduce(
        high,
        window=window,
        statistic=lambda w: np.nanmax(w, axis=1),
        min_periods=min_periods,
    )
    out = np.full(value.shape, np.nan, dtype=float)
    valid = np.isfinite(value) & np.isfinite(rolling_max) & (rolling_max > 0)
    out[valid] = value[valid] / rolling_max[valid]
    return out


def amihud_illiquidity(
    price: np.ndarray,
    volume: np.ndarray,
    *,
    window: int,
    min_periods: int | None = None,
    log: bool = False,
) -> np.ndarray:
    """Return the trailing mean absolute-return-to-dollar-volume ratio (Amihud 2002).

    Each day's price impact ``|log return| / (price * volume)`` is averaged over ``window``
    rows. Days with non-positive dollar volume carry no observation and are skipped rather
    than treated as perfectly liquid. Both inputs must share a shape.

    The raw ratio spans several orders of magnitude, so ``log=True`` returns its natural
    logarithm, which is far better conditioned for a downstream global standardizer that
    rescales but does not de-skew; non-positive averages then map to NaN.
    """
    price = _as_float(price)
    volume = _as_float(volume)
    _require_same_shape(price, volume)
    returns = log_change(price, horizon=1)
    dollar_volume = price * volume
    daily = np.full(price.shape, np.nan, dtype=float)
    tradable = np.isfinite(returns) & np.isfinite(dollar_volume) & (dollar_volume > 0)
    daily[tradable] = np.abs(returns[tradable]) / dollar_volume[tradable]
    illiquidity = _rolling_reduce(
        daily,
        window=window,
        statistic=lambda w: np.nanmean(w, axis=1),
        min_periods=min_periods,
    )
    if not log:
        return illiquidity
    out = np.full(illiquidity.shape, np.nan, dtype=float)
    positive = np.isfinite(illiquidity) & (illiquidity > 0)
    out[positive] = np.log(illiquidity[positive])
    return out
