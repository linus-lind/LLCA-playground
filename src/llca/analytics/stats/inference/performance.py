"""Single-model risk-adjusted performance tests.

The Sharpe ratio is a monotone transform of the mean excess return, so its one-sided
significance reuses the HAC mean test while a stationary bootstrap brackets the ratio itself.

References
----------
Lo (2002).
"""

from __future__ import annotations

import numpy as np

from llca.analytics.stats.inference.foundation import (
    _finite,
    hac_mean_test,
    stationary_bootstrap_indices,
)
from llca.analytics.stats.statistics import EPS


def sharpe_ratio(daily_excess: np.ndarray, annualization_periods: int) -> float:
    """Return the annualized Sharpe ratio of a daily excess-return series.

    Divides the mean by the sample standard deviation and scales by
    ``sqrt(annualization_periods)``. Yields ``nan`` when there are fewer than two finite
    observations or the returns have essentially no dispersion.
    """
    array = _finite(daily_excess)
    if array.shape[0] < 2:
        return float("nan")
    std = float(array.std(ddof=1))
    if std <= EPS:
        return float("nan")
    return float(array.mean() / std * np.sqrt(annualization_periods))


def sharpe_significance(
    daily_excess: np.ndarray,
    *,
    annualization_periods: int,
    lag: int | None = None,
    n_boot: int = 2000,
    block_length: float = 10.0,
    seed: int = 0,
    confidence: float = 0.95,
) -> dict[str, float]:
    """Assess whether the annualized Sharpe ratio is positive and interval-estimate it.

    The significance test reduces to a one-sided HAC mean test on the excess returns, since a
    positive Sharpe ratio is equivalent to a positive mean. A stationary bootstrap
    (``n_boot`` resamples, mean block length ``block_length``, seeded by ``seed``) then gives a
    ``confidence``-level percentile interval for the ratio that tolerates serial dependence, per
    Lo (2002). Returns the point estimate, its t-statistic and p-value, and the interval bounds.
    """
    array = _finite(daily_excess)
    point = sharpe_ratio(array, annualization_periods)
    test = hac_mean_test(array, null=0.0, lag=lag, alternative="greater")
    low = high = float("nan")
    if array.shape[0] >= 2 and n_boot > 0:
        rng = np.random.default_rng(seed)
        indices = stationary_bootstrap_indices(array.shape[0], block_length, n_boot, rng)
        samples = array[indices]
        means = samples.mean(axis=1)
        stds = samples.std(axis=1, ddof=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            ratios = np.where(stds > EPS, means / stds * np.sqrt(annualization_periods), np.nan)
        ratios = ratios[np.isfinite(ratios)]
        if ratios.size:
            tail = (1.0 - confidence) / 2.0
            low, high = (float(value) for value in np.quantile(ratios, [tail, 1.0 - tail]))
    return {
        "annualized_sharpe": point,
        "sharpe_t_statistic": test.t_statistic,
        "sharpe_p_value": test.p_value,
        "sharpe_ci_low": low,
        "sharpe_ci_high": high,
    }
