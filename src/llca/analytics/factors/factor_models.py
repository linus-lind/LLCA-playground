"""Factor-model alpha, timing, and spanning tests for realized portfolio returns.

Everything here operates on realized daily *return series* (one value per date), so a model
that trades a single asset and one that trades a large long-short book are treated identically
— the number of underlying positions never enters. Three factor models are supported:

* **Fama-French 6** — the tradable FF5 factors plus momentum; the headline unconditional alpha.
* **Conditional timing** — Ferson-Schadt conditional *market* beta on lagged, demeaned macro
  instruments, a Treynor-Mazuy market-convexity term, and a Christopherson-Ferson-Glassman
  conditional alpha. The non-market FF6 betas stay unconditional. The reported alpha is the
  intercept at the average conditioning state (instruments are demeaned).
* **IPCA** — the estimated latent factors (see :mod:`llca.analytics.factors`) fed in here like
  any other tradable factor set.

Regressions use ordinary least squares with a Newey-West (HAC) covariance so that
autocorrelated daily errors do not inflate the alpha's significance. Cross-model superiority is
a HAC test on the intercept of the *return-difference* regression, and the joint zero-alpha
test uses :mod:`linearmodels` with a kernel (HAC) covariance. Spanning follows Huberman-Kandel:
the two
restrictions ``alpha = 0`` and ``sum(beta) = 1`` are tested jointly with a HAC Wald statistic.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from linearmodels.asset_pricing import TradedFactorModel

from llca.analytics.stats.inference import newey_west_bandwidth


def _resolved_lag(n: int, lag: int | None) -> int:
    return newey_west_bandwidth(n) if lag is None else lag


def _hac_ols(
    y: np.ndarray, x: np.ndarray, lag: int | None
) -> sm.regression.linear_model.RegressionResultsWrapper:
    """Regress ``y`` on ``x`` and an intercept using a Newey-West HAC covariance.

    The truncation lag is ``lag`` when given, otherwise the automatic bandwidth for the sample
    size, so the reported standard errors tolerate autocorrelated residuals.
    """
    design = sm.add_constant(x, has_constant="add")
    n = design.shape[0]
    return sm.OLS(y, design).fit(cov_type="HAC", cov_kwds={"maxlags": _resolved_lag(n, lag)})


@dataclass(frozen=True, slots=True)
class FactorAlpha:
    """One factor-model regression's alpha, its HAC significance, loadings, and fit."""

    alpha: float
    alpha_std_error: float
    alpha_t_statistic: float
    alpha_p_value: float
    annualized_alpha: float
    betas: dict[str, float]
    beta_p_values: dict[str, float]
    r_squared: float
    observations: int


def _align(excess: pd.Series, factors: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    joined = pd.concat([excess.rename("excess"), factors], axis=1, join="inner").dropna()
    return joined["excess"], joined[list(factors.columns)]


def factor_alpha(
    excess_returns: pd.Series,
    factors: pd.DataFrame,
    *,
    annualization_periods: int,
    lag: int | None = None,
) -> FactorAlpha | None:
    """Estimate a portfolio's factor-model alpha and loadings by HAC regression.

    Regresses excess returns on the ``factors``; the intercept is Jensen's alpha, reported with
    its HAC standard error, t-statistic, two-sided p-value, and annualized value, alongside each
    factor beta and its p-value and the fit's R-squared. Returns ``None`` when too few aligned
    observations remain to identify the coefficients.
    """
    excess, aligned = _align(excess_returns, factors)
    n, k = aligned.shape
    if n <= k + 1:
        return None
    result = _hac_ols(excess.to_numpy(dtype=float), aligned.to_numpy(dtype=float), lag)
    betas = {str(name): float(result.params[i + 1]) for i, name in enumerate(aligned.columns)}
    beta_p = {str(name): float(result.pvalues[i + 1]) for i, name in enumerate(aligned.columns)}
    return FactorAlpha(
        alpha=float(result.params[0]),
        alpha_std_error=float(result.bse[0]),
        alpha_t_statistic=float(result.tvalues[0]),
        alpha_p_value=float(result.pvalues[0]),
        annualized_alpha=float(result.params[0]) * annualization_periods,
        betas=betas,
        beta_p_values=beta_p,
        r_squared=float(result.rsquared),
        observations=int(n),
    )


@dataclass(frozen=True, slots=True)
class TimingModel:
    """Conditional timing regression: intercept alpha, market beta, timing gamma, and fit."""

    alpha: float
    alpha_p_value: float
    annualized_alpha: float
    market_beta: float
    market_beta_p_value: float
    timing_gamma: float
    timing_p_value: float
    r_squared: float
    observations: int
    coefficients: dict[str, float]
    coefficient_p_values: dict[str, float]


def timing_model(
    excess_returns: pd.Series,
    factors: pd.DataFrame,
    market_column: str,
    instruments: pd.DataFrame,
    *,
    annualization_periods: int,
    instrument_lag: int = 1,
    market_squared: bool = True,
    conditional_alpha: bool = True,
    lag: int | None = None,
) -> TimingModel | None:
    """Estimate a conditional market-timing regression for one portfolio.

    Builds a design of a constant, the market factor interacted with lagged demeaned
    instruments (Ferson-Schadt conditional beta), optional standalone instrument terms
    (Christopherson-Ferson-Glassman conditional alpha), the remaining factors unconditionally,
    and optionally the squared market factor (Treynor-Mazuy convexity), then fits it with a HAC
    covariance. The intercept is the average-state alpha and the squared-market coefficient the
    timing gamma. Instruments are lagged by ``instrument_lag`` and demeaned over the regression
    sample. Raises if ``market_column`` is absent; returns ``None`` when observations are too few.
    """
    if market_column not in factors.columns:
        raise ValueError(f"market column '{market_column}' absent from factors")
    excess, aligned = _align(excess_returns, factors)
    z = instruments.shift(instrument_lag) if instrument_lag else instruments
    z = z.reindex(aligned.index)
    complete = z.notna().all(axis=1)
    excess = excess.loc[complete]
    aligned = aligned.loc[complete]
    z = z.loc[complete]
    # Demean on the actual joint complete-case regression sample. Column-specific missing
    # rows would otherwise shift the interpretation of the conditional-alpha intercept.
    z = z - z.mean()
    other = [column for column in aligned.columns if column != market_column]
    pieces: dict[str, pd.Series] = {}
    if conditional_alpha:
        for column in z.columns:
            pieces[f"alpha_{column}"] = z[column]
    pieces[market_column] = aligned[market_column]
    for column in z.columns:
        pieces[f"{market_column}_x_{column}"] = aligned[market_column] * z[column]
    for column in other:
        pieces[column] = aligned[column]
    gamma_name = f"{market_column}_squared"
    if market_squared:
        pieces[gamma_name] = aligned[market_column] ** 2
    design = pd.concat([excess.rename("excess"), pd.DataFrame(pieces)], axis=1).dropna()
    n = len(design)
    predictors = [name for name in pieces]
    if n <= len(predictors) + 1:
        return None
    result = _hac_ols(
        design["excess"].to_numpy(dtype=float),
        design[predictors].to_numpy(dtype=float),
        lag,
    )
    order = ["const", *predictors]
    gamma_index = order.index(gamma_name) if market_squared else None
    market_index = order.index(market_column)
    coefficients = {name: float(result.params[index + 1]) for index, name in enumerate(predictors)}
    coefficient_p_values = {
        name: float(result.pvalues[index + 1]) for index, name in enumerate(predictors)
    }
    return TimingModel(
        alpha=float(result.params[0]),
        alpha_p_value=float(result.pvalues[0]),
        annualized_alpha=float(result.params[0]) * annualization_periods,
        market_beta=float(result.params[market_index]),
        market_beta_p_value=float(result.pvalues[market_index]),
        timing_gamma=float(result.params[gamma_index]) if gamma_index is not None else float("nan"),
        timing_p_value=(
            float(result.pvalues[gamma_index]) if gamma_index is not None else float("nan")
        ),
        r_squared=float(result.rsquared),
        observations=int(n),
        coefficients=coefficients,
        coefficient_p_values=coefficient_p_values,
    )


def alpha_difference(
    returns_a: pd.Series,
    returns_b: pd.Series,
    factors: pd.DataFrame,
    *,
    lag: int | None = None,
) -> dict[str, float]:
    """Test whether two portfolios have the same factor-model alpha.

    Regresses the return difference ``returns_a - returns_b`` on the factors so the intercept is
    the alpha gap; its two-sided HAC p-value tests equality and its sign shows which portfolio
    leads. Differencing removes the shared factor exposure and cross-correlation. Returns the
    intercept and p-value, or ``nan`` when too few aligned observations remain.
    """
    difference = (returns_a - returns_b).rename("difference")
    excess, aligned = _align(difference, factors)
    n, k = aligned.shape
    if n <= k + 1:
        return {"alpha_difference": float("nan"), "alpha_difference_p_value": float("nan")}
    result = _hac_ols(excess.to_numpy(dtype=float), aligned.to_numpy(dtype=float), lag)
    return {
        "alpha_difference": float(result.params[0]),
        "alpha_difference_p_value": float(result.pvalues[0]),
    }


def joint_alpha_test(
    portfolios: pd.DataFrame,
    factors: pd.DataFrame,
    *,
    lag: int | None = None,
) -> dict[str, float]:
    """Jointly test that all portfolios' factor-model alphas are simultaneously zero.

    Treats the portfolios as test assets priced by the factors and fits the system with a
    Bartlett-kernel (HAC) covariance, returning the J-statistic and its p-value. Collinear
    portfolios are dropped down to a full-rank subset first. Degenerate cases short-circuit: no
    independent portfolio returns a passing result, and too few observations or an estimation
    failure return ``nan``.
    """
    joined = pd.concat([portfolios, factors], axis=1, join="inner").dropna()
    independent: list[object] = []
    rank = 0
    for column in portfolios.columns:
        candidate_columns = [*independent, column]
        candidate = joined[candidate_columns].to_numpy(dtype=float)
        candidate_rank = int(np.linalg.matrix_rank(candidate))
        if candidate_rank > rank:
            independent.append(column)
            rank = candidate_rank
    if not independent:
        return {"joint_alpha_statistic": 0.0, "joint_alpha_p_value": 1.0}
    if len(joined) <= len(independent) + factors.shape[1] + 1:
        return {
            "joint_alpha_statistic": float("nan"),
            "joint_alpha_p_value": float("nan"),
        }
    n = len(joined)
    try:
        model = TradedFactorModel(joined[independent], joined[list(factors.columns)])
        result = model.fit(cov_type="kernel", kernel="bartlett", bandwidth=_resolved_lag(n, lag))
    except (ValueError, np.linalg.LinAlgError):
        return {
            "joint_alpha_statistic": float("nan"),
            "joint_alpha_p_value": float("nan"),
        }
    j = result.j_statistic
    return {
        "joint_alpha_statistic": float(j.stat),
        "joint_alpha_p_value": float(j.pval),
    }


def spanning_test(
    portfolio: pd.Series,
    benchmark: pd.DataFrame,
    *,
    lag: int | None = None,
) -> dict[str, float]:
    """Test whether a portfolio expands the mean-variance frontier of benchmark assets.

    Regresses the portfolio's excess return on the benchmark returns and jointly tests the
    Huberman-Kandel spanning restrictions ``alpha = 0`` and ``sum(beta) = 1`` with a HAC Wald
    statistic on two degrees of freedom. A small p-value means the portfolio adds investment
    opportunity beyond the benchmark. Returns the statistic and p-value, or ``nan`` when too few
    aligned observations remain.
    """
    joined = pd.concat([portfolio.rename("p"), benchmark], axis=1, join="inner").dropna()
    n, k = len(joined), benchmark.shape[1]
    if n <= k + 2:
        return {"spanning_statistic": float("nan"), "spanning_p_value": float("nan")}
    result = _hac_ols(
        joined["p"].to_numpy(dtype=float),
        joined[list(benchmark.columns)].to_numpy(dtype=float),
        lag,
    )
    # Restriction matrix R @ params = q with params = [const, beta_1..beta_k]:
    #   row 1: const = 0 ; row 2: sum(beta) = 1.
    restriction = np.zeros((2, k + 1))
    restriction[0, 0] = 1.0
    restriction[1, 1:] = 1.0
    q = np.array([0.0, 1.0])
    wald = result.wald_test((restriction, q), scalar=True, use_f=False)
    return {
        "spanning_statistic": float(np.asarray(wald.statistic).item()),
        "spanning_p_value": float(np.asarray(wald.pvalue).item()),
    }


def rolling_betas(
    excess_returns: pd.Series,
    factors: pd.DataFrame,
    *,
    window: int,
) -> pd.DataFrame:
    """Trace a portfolio's factor betas over trailing windows, one column per factor.

    Each date holds the OLS loadings estimated on the preceding ``window`` observations, showing
    how factor exposures move through time. Raises ``ValueError`` if ``window`` is not large
    enough to identify the intercept and factor coefficients.
    """
    excess, aligned = _align(excess_returns, factors)
    columns = list(aligned.columns)
    if window <= len(columns) + 1:
        raise ValueError(
            "rolling factor-beta window must exceed the number of regression "
            "coefficients (intercept plus factors)"
        )
    n = len(aligned)
    records: dict[pd.Timestamp, list[float]] = {}
    values = aligned.to_numpy(dtype=float)
    response = excess.to_numpy(dtype=float)
    ones = np.ones((n, 1))
    design = np.hstack([ones, values])
    for end in range(window, n + 1):
        block = design[end - window : end]
        target = response[end - window : end]
        beta, *_ = np.linalg.lstsq(block, target, rcond=None)
        records[aligned.index[end - 1]] = [float(value) for value in beta[1:]]
    if not records:
        return pd.DataFrame(columns=columns)
    frame = pd.DataFrame.from_dict(records, orient="index", columns=columns).sort_index()
    frame.index.name = aligned.index.name
    return frame


def cumulative_alpha(
    excess_returns: pd.Series,
    factors: pd.DataFrame,
) -> pd.Series:
    """Accumulate a portfolio's abnormal return net of its factor exposure over time.

    Estimates full-sample betas once by OLS, subtracts the factor-explained part from each
    date's excess return to get the abnormal return, and returns its running sum — the
    portfolio's alpha accrual. Returns an empty series when too few aligned observations remain.
    """
    excess, aligned = _align(excess_returns, factors)
    if len(aligned) < aligned.shape[1] + 2:
        return pd.Series(dtype=float)
    design = sm.add_constant(aligned.to_numpy(dtype=float), has_constant="add")
    beta, *_ = np.linalg.lstsq(design, excess.to_numpy(dtype=float), rcond=None)
    factor_part = design[:, 1:] @ beta[1:]
    abnormal = excess.to_numpy(dtype=float) - factor_part
    return pd.Series(np.cumsum(abnormal), index=aligned.index, name="cumulative_alpha")
