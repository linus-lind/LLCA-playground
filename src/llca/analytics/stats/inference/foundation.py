"""HAC and stationary-bootstrap infrastructure shared by the inference tests.

Every function takes plain arrays and returns finite floats. Autocorrelation in daily
financial series is handled with Newey-West heteroskedasticity-and-autocorrelation-consistent
(HAC) variances, and the tests that lack a trustworthy closed form use the stationary bootstrap
resampler defined here so the reported p-values stay valid under serial dependence.

References
----------
Newey & West (1987, 1994); Politis & Romano (1994).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
import statsmodels.api as sm
from scipy.stats import norm
from scipy.stats import t as student_t

from llca.analytics.stats.statistics import EPS

type Alternative = str  # "greater", "less", or "two-sided"


def newey_west_bandwidth(n: int) -> int:
    """Return the data-driven HAC truncation lag for ``n`` observations.

    Implements the Newey-West (1994) rule ``floor(4 * (n / 100) ** (2/9))``, falling back to
    ``0`` when the sample is too small to admit any lag.
    """
    if n < 2:
        return 0
    return int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))


def _finite(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    return cast(np.ndarray, array[np.isfinite(array)])


def long_run_variance(values: np.ndarray, lag: int) -> float:
    """Estimate the long-run variance of ``values`` with a Bartlett kernel truncated at ``lag``.

    Sums the sample variance and the Bartlett-weighted autocovariances out to ``lag``, giving
    the spectral density at frequency zero that underlies the HAC variance of the sample mean.
    The result is clamped to be non-negative, and is ``nan`` for fewer than two observations.
    """
    array = np.asarray(values, dtype=float)
    n = array.shape[0]
    if n < 2:
        return float("nan")
    centered = array - array.mean()
    total = float(np.dot(centered, centered) / n)
    for k in range(1, min(lag, n - 1) + 1):
        weight = 1.0 - k / (lag + 1.0)
        covariance = float(np.dot(centered[k:], centered[:-k]) / n)
        total += 2.0 * weight * covariance
    return max(total, 0.0)


def _p_value(statistic: float, alternative: Alternative, df: float | None) -> float:
    if not np.isfinite(statistic):
        return float("nan")
    survival = student_t.sf if df is not None else norm.sf
    args = (df,) if df is not None else ()
    if alternative == "greater":
        return float(survival(statistic, *args))
    if alternative == "less":
        return float(survival(-statistic, *args))
    return float(2.0 * survival(abs(statistic), *args))


@dataclass(frozen=True, slots=True)
class MeanTest:
    """One HAC mean test: point estimate, standard error, t-statistic and p-value."""

    mean: float
    std_error: float
    t_statistic: float
    p_value: float
    observations: int


def hac_mean_test(
    values: np.ndarray,
    *,
    null: float = 0.0,
    lag: int | None = None,
    alternative: Alternative = "greater",
) -> MeanTest:
    """Test the mean of an autocorrelated series against ``null`` with HAC standard errors.

    Regresses the centred series on a constant using a Newey-West covariance (truncation lag
    ``lag``, or the automatic bandwidth when ``lag`` is ``None``), so the standard error
    survives serial correlation. The t-statistic is graded against a Student-t distribution
    with ``n - 1`` degrees of freedom under the requested ``alternative``. Series shorter than
    two finite points return an all-``nan`` result.
    """
    array = _finite(values)
    n = array.shape[0]
    if n < 2:
        return MeanTest(float("nan"), float("nan"), float("nan"), float("nan"), n)
    resolved_lag = newey_west_bandwidth(n) if lag is None else lag
    fit = sm.OLS(array - null, np.ones(n)).fit(cov_type="HAC", cov_kwds={"maxlags": resolved_lag})
    mean = float(fit.params[0]) + null
    std_error = float(fit.bse[0])
    statistic = float(fit.tvalues[0]) if std_error > EPS else float("nan")
    p_value = _p_value(statistic, alternative, df=n - 1)
    return MeanTest(mean, std_error, statistic, p_value, n)


def stationary_bootstrap_indices(
    n: int,
    block_length: float,
    n_boot: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw ``n_boot`` stationary-bootstrap resamples of ``n`` positions as an index matrix.

    Follows Politis-Romano (1994): each resample walks forward one position at a time, and at
    every step restarts at a uniformly random position with probability ``1 / block_length``,
    giving geometrically distributed blocks with that mean length. Walks wrap past the end of
    the sample so no position is favoured. The returned array has shape ``[n_boot, n]``.
    """
    if n <= 0:
        return np.empty((n_boot, 0), dtype=np.int64)
    restart_probability = 1.0 / max(block_length, 1.0)
    indices = np.empty((n_boot, n), dtype=np.int64)
    indices[:, 0] = rng.integers(0, n, size=n_boot)
    restart = rng.random((n_boot, n)) < restart_probability
    fresh = rng.integers(0, n, size=(n_boot, n))
    for step in range(1, n):
        continued = (indices[:, step - 1] + 1) % n
        indices[:, step] = np.where(restart[:, step], fresh[:, step], continued)
    return indices
