"""Single-model directional and predictive-content hypothesis tests.

Each test operates on already-aligned daily series so serial dependence is corrected once the
same-day observations have been collapsed into one number.

References
----------
Pesaran & Timmermann (1992); Anatolyev & Gerko (2005).
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from llca.analytics.stats.inference.foundation import _finite, _p_value, hac_mean_test
from llca.analytics.stats.statistics import EPS


def pesaran_timmermann(predicted_up: np.ndarray, actual_up: np.ndarray) -> dict[str, float]:
    """Test whether predicted directions track realized directions (Pesaran-Timmermann, 1992).

    Contrasts the observed fraction of correct sign calls against the fraction expected if the
    two direction series were independent, standardising the gap into an asymptotically normal
    statistic. The reported p-value is one-sided (better than chance). Returns the statistic,
    that p-value, and the observed and expected hit rates; a sample with no directional
    variation, or fewer than two points, yields ``nan``.
    """
    predicted = np.asarray(predicted_up, dtype=bool)
    actual = np.asarray(actual_up, dtype=bool)
    n = predicted.shape[0]
    if n < 2:
        return {"pt_statistic": float("nan"), "pt_p_value": float("nan"), "hit_rate": float("nan")}
    hit_rate = float((predicted == actual).mean())
    p_actual = float(actual.mean())
    p_predicted = float(predicted.mean())
    expected = p_actual * p_predicted + (1.0 - p_actual) * (1.0 - p_predicted)
    var_hit = expected * (1.0 - expected) / n
    var_expected = (
        ((2.0 * p_actual - 1.0) ** 2) * p_predicted * (1.0 - p_predicted) / n
        + ((2.0 * p_predicted - 1.0) ** 2) * p_actual * (1.0 - p_actual) / n
        + 4.0 * p_actual * p_predicted * (1.0 - p_actual) * (1.0 - p_predicted) / n**2
    )
    denominator = var_hit - var_expected
    statistic = (
        float((hit_rate - expected) / np.sqrt(denominator)) if denominator > EPS else float("nan")
    )
    return {
        "pt_statistic": float(statistic),
        "pt_p_value": _p_value(float(statistic), "greater", df=None),
        "hit_rate": hit_rate,
        "expected_hit_rate": expected,
    }


def directional_accuracy_test(
    daily_hit_rate: np.ndarray,
    *,
    baseline: float = 0.5,
    lag: int | None = None,
) -> dict[str, float]:
    """Test whether the mean daily hit rate beats ``baseline`` (0.5 by default).

    Takes a series of per-date hit rates — one number per day, so same-day observations are
    already pooled — and applies a one-sided HAC mean test. Returns the mean hit rate with its
    t-statistic and p-value.
    """
    test = hac_mean_test(daily_hit_rate, null=baseline, lag=lag, alternative="greater")
    return {
        "mean_hit_rate": test.mean,
        "hit_rate_t_statistic": test.t_statistic,
        "hit_rate_p_value": test.p_value,
    }


def excess_profitability_test(
    daily_excess_profit: np.ndarray,
    *,
    lag: int | None = None,
) -> dict[str, float]:
    """Test whether the signal's directional trading earns a positive excess profit.

    Consumes the daily excess-profit series of Anatolyev-Gerko (2005) — each value is a day's
    covariance between the taken direction and the realized return — and runs a one-sided HAC
    mean test against zero. A significant positive mean means acting on the signal's sign beats
    a direction-agnostic allocation. Returns the mean with its t-statistic and p-value.
    """
    test = hac_mean_test(daily_excess_profit, null=0.0, lag=lag, alternative="greater")
    return {
        "excess_profitability": test.mean,
        "excess_profitability_t_statistic": test.t_statistic,
        "excess_profitability_p_value": test.p_value,
    }


def information_coefficient_test(
    daily_ic: np.ndarray,
    *,
    annualization_periods: int,
    lag: int | None = None,
    confidence: float = 0.95,
) -> dict[str, float]:
    """Summarise the significance and information ratio of a daily information-coefficient series.

    Runs a one-sided HAC mean test (IC greater than zero) and derives the information ratio as
    the mean IC over its per-day standard deviation, together with its scaling by
    ``sqrt(annualization_periods)``. Also returns a two-sided ``confidence``-level interval for
    the mean IC formed from the HAC standard error.
    """
    array = _finite(daily_ic)
    test = hac_mean_test(array, null=0.0, lag=lag, alternative="greater")
    std = float(array.std(ddof=1)) if array.shape[0] > 1 else float("nan")
    information_ratio = test.mean / std if std and std > EPS else float("nan")
    critical = float(norm.ppf(0.5 + confidence / 2.0))
    half_width = critical * test.std_error if np.isfinite(test.std_error) else float("nan")
    return {
        "mean_ic": test.mean,
        "ic_t_statistic": test.t_statistic,
        "ic_p_value": test.p_value,
        "information_ratio": information_ratio,
        "annualized_information_ratio": float(information_ratio * np.sqrt(annualization_periods)),
        "ic_ci_low": test.mean - half_width,
        "ic_ci_high": test.mean + half_width,
    }
