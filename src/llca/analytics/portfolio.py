from __future__ import annotations

from collections.abc import Callable
from typing import cast

import numpy as np
import pandas as pd
import torch
from scipy.stats import kurtosis, skew
from torch import Tensor

from llca.analytics.modules.portfolio_evaluation import PortfolioEvaluation
from llca.core.returns import ReturnType
from llca.data.index_spec import entity_level

_EPS = 1e-12


def _dense(series: pd.Series, name: str) -> pd.DataFrame:
    entity = entity_level(series)
    if entity is None:
        return series.rename(name).to_frame()
    return series.unstack(level=entity).sort_index()


def _normalised_weights(
    scores: pd.DataFrame,
    valid: pd.DataFrame,
    normalize: Callable[[Tensor, Tensor], Tensor],
) -> pd.DataFrame:
    """Apply the objective's allocation rule to dense scores and restore panel axes.

    ``scores`` and ``valid`` share shape ``[D, N]``. Invalid cells are zeroed before the
    model-independent normalization callback and remain exactly zero in the result.
    """
    score_tensor = torch.from_numpy(scores.fillna(0.0).to_numpy(dtype=np.float32))
    mask_tensor = torch.from_numpy(valid.to_numpy(dtype=bool))
    with torch.inference_mode():
        weights = normalize(score_tensor, mask_tensor).detach().cpu().numpy()
    result = pd.DataFrame(weights, index=scores.index, columns=scores.columns)
    result = result.where(valid, 0.0)
    if not np.isfinite(result.to_numpy()).all():
        raise ValueError("portfolio normalisation produced non-finite weights")
    return result


def _simple_returns(returns: pd.DataFrame, return_type: ReturnType) -> pd.DataFrame:
    """Convert configured asset returns to simple returns required for portfolio accounting."""
    if return_type == "simple":
        simple = returns.copy()
    else:
        simple = pd.DataFrame(
            np.expm1(returns.to_numpy(dtype=float)),
            index=returns.index,
            columns=returns.columns,
        )
    if (simple <= -1.0).any(axis=None):
        raise ValueError("asset simple returns must be greater than -100%")
    return simple


def _turnover_and_costs(
    weights: pd.DataFrame,
    asset_returns: pd.DataFrame,
    *,
    execution_fee: float,
    bid_ask_spread: float,
    slippage: float,
    borrow_cost: float,
    include_initial_trade: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Derive drift-adjusted trades, side turnover, and linear trading costs.

    Target weights ``[D, N]`` are compared with prior holdings after asset-return and NAV
    drift. L1 turnover is the absolute weight change; one-way turnover is half of it.
    Transaction costs scale L1 turnover, while borrow cost scales current short exposure.
    The first trade is included only when explicitly requested.
    """
    values = weights.to_numpy(dtype=float)
    returns = asset_returns.to_numpy(dtype=float)
    changes = np.zeros_like(values)
    if include_initial_trade:
        changes[0] = values[0]
    for position in range(1, len(weights)):
        previous = values[position - 1]
        previous_returns = returns[position - 1]
        nav_growth = 1.0 + float(np.dot(previous, previous_returns))
        if nav_growth <= _EPS:
            raise ValueError("portfolio NAV was exhausted before drifted turnover calculation")
        drifted = previous * (1.0 + previous_returns) / nav_growth
        changes[position] = values[position] - drifted

    long_changes = np.abs(np.maximum(values, 0.0) - np.maximum(values - changes, 0.0)).sum(axis=1)
    short_changes = np.abs(np.maximum(-values, 0.0) - np.maximum(-(values - changes), 0.0)).sum(
        axis=1
    )
    l1_turnover = np.abs(changes).sum(axis=1)
    turnover = pd.DataFrame(
        {
            "l1_turnover": l1_turnover,
            "one_way_turnover": 0.5 * l1_turnover,
            "long_turnover": long_changes,
            "short_turnover": short_changes,
        },
        index=weights.index,
    )
    transaction_rate = execution_fee + bid_ask_spread + slippage
    short_exposure = np.maximum(-values, 0.0).sum(axis=1)
    costs = pd.DataFrame(
        {
            "execution_fee": execution_fee * l1_turnover,
            "bid_ask": bid_ask_spread * l1_turnover,
            "slippage": slippage * l1_turnover,
            "borrow": borrow_cost * short_exposure,
        },
        index=weights.index,
    )
    costs["transaction"] = transaction_rate * l1_turnover
    costs["total"] = costs["transaction"] + costs["borrow"]
    return turnover, costs


def _drawdown_path(returns: pd.Series) -> pd.DataFrame:
    """Build compounded wealth, high-water mark, drawdown, and underwater duration paths."""
    wealth = (1.0 + returns).cumprod()
    high_water_mark = wealth.cummax()
    drawdown = wealth / high_water_mark - 1.0
    duration = np.zeros(len(drawdown), dtype=int)
    current = 0
    for index, value in enumerate(drawdown.to_numpy()):
        current = current + 1 if value < 0.0 else 0
        duration[index] = current
    return pd.DataFrame(
        {
            "wealth": wealth,
            "high_water_mark": high_water_mark,
            "drawdown": drawdown,
            "duration": duration,
        },
        index=returns.index,
    )


def _drawdown_episode_lengths(drawdown: pd.Series) -> list[int]:
    """Return lengths of completed and still-open periods below the high-water mark."""
    lengths: list[int] = []
    current = 0
    for value in drawdown.to_numpy():
        if value < 0.0:
            current += 1
        elif current:
            lengths.append(current)
            current = 0
    if current:
        lengths.append(current)
    return lengths


def _maximum_consecutive(condition: pd.Series) -> int:
    """Return the longest contiguous run of true period-level observations."""
    maximum = 0
    current = 0
    for value in condition.to_numpy(dtype=bool):
        current = current + 1 if value else 0
        maximum = max(maximum, current)
    return maximum


def _shape_statistics(returns: pd.Series) -> tuple[float, float]:
    """Return bias-corrected skewness and excess kurtosis when statistically defined."""
    if len(returns) < 4 or float(returns.std(ddof=1)) <= _EPS:
        return float("nan"), float("nan")
    values = returns.to_numpy(dtype=float)
    return (
        float(skew(values, bias=False)),
        float(kurtosis(values, fisher=True, bias=False)),
    )


def _autocorrelation(returns: pd.Series, lag: int) -> float:
    """Correlate returns with their positional lag when both slices have variation."""
    if len(returns) <= lag:
        return float("nan")
    left = returns.iloc[:-lag]
    right = returns.iloc[lag:]
    if len(left) < 2:
        return float("nan")
    left_std = float(left.std(ddof=1))
    right_std = float(right.std(ddof=1))
    if not np.isfinite(left_std) or not np.isfinite(right_std):
        return float("nan")
    if left_std <= _EPS or right_std <= _EPS:
        return float("nan")
    return float(left.corr(pd.Series(right.to_numpy(), index=left.index)))


def _rolling_compound(returns: pd.Series, window: int) -> pd.Series:
    return (1.0 + returns).rolling(window).apply(np.prod, raw=True) - 1.0


def _rolling_tail_risk(
    returns: pd.Series,
    window: int,
    levels: tuple[float, ...],
) -> pd.DataFrame:
    """Historical rolling loss quantiles and conditional tail means on net returns."""
    losses = -returns
    result = pd.DataFrame(index=returns.index)
    for level in levels:
        label = str(int(round(level * 100)))
        result[f"var_{label}"] = losses.rolling(window, min_periods=window).quantile(level)
        result[f"expected_shortfall_{label}"] = losses.rolling(window, min_periods=window).apply(
            lambda values, quantile=level: float(
                np.mean(values[values >= np.quantile(values, quantile)])
            ),
            raw=True,
        )
    return result


def _performance_metrics(
    returns: pd.Series,
    *,
    prefix: str,
    annualization_periods: int,
    risk_free_rate: float,
    minimum_acceptable_return: float,
    var_levels: tuple[float, ...],
) -> tuple[dict[str, float], pd.DataFrame]:
    """Compute compounded, annualized, downside, tail, and path-dependent performance.

    Inputs are simple periodic returns. Risk-free and minimum acceptable rates are annual
    rates converted to the configured period basis. VaR and expected shortfall are positive
    loss magnitudes from the empirical return distribution; drawdowns are negative wealth
    deviations. ``prefix`` keeps gross and net definitions separate in one report.
    """
    if returns.empty:
        raise ValueError("portfolio return series must not be empty")
    if (returns <= -1.0).any():
        raise ValueError("portfolio returns must be greater than -100% to compound")
    drawdowns = _drawdown_path(returns)
    total_return = float(drawdowns["wealth"].iloc[-1] - 1.0)
    years = len(returns) / annualization_periods
    cagr = float(drawdowns["wealth"].iloc[-1] ** (1.0 / years) - 1.0)
    mean = float(returns.mean())
    std = float(returns.std(ddof=1))
    daily_rf = (1.0 + risk_free_rate) ** (1.0 / annualization_periods) - 1.0
    daily_mar = (1.0 + minimum_acceptable_return) ** (1.0 / annualization_periods) - 1.0
    downside = np.minimum(returns.to_numpy(dtype=float) - daily_mar, 0.0)
    downside_deviation = float(np.sqrt(np.mean(np.square(downside))))
    maximum_drawdown = float(drawdowns["drawdown"].min())
    negative_drawdowns = drawdowns.loc[drawdowns["drawdown"] < 0.0, "drawdown"]
    episodes = _drawdown_episode_lengths(drawdowns["drawdown"])
    gains = returns[returns > daily_mar] - daily_mar
    losses = daily_mar - returns[returns < daily_mar]
    wins = returns[returns > 0.0]
    losing = returns[returns < 0.0]
    return_skewness, return_kurtosis = _shape_statistics(returns)

    metrics = {
        f"{prefix}_total_return": total_return,
        f"{prefix}_cagr": cagr,
        f"{prefix}_annualized_arithmetic_return": mean * annualization_periods,
        f"{prefix}_annualized_volatility": std * np.sqrt(annualization_periods),
        f"{prefix}_downside_deviation": downside_deviation * np.sqrt(annualization_periods),
        f"{prefix}_sharpe_ratio": (
            (mean - daily_rf) / std * np.sqrt(annualization_periods) if std > _EPS else float("nan")
        ),
        f"{prefix}_sortino_ratio": (
            (mean - daily_mar) / downside_deviation * np.sqrt(annualization_periods)
            if downside_deviation > _EPS
            else float("nan")
        ),
        f"{prefix}_calmar_ratio": (
            cagr / abs(maximum_drawdown) if maximum_drawdown < -_EPS else float("nan")
        ),
        f"{prefix}_omega_ratio": _safe_sum_ratio(gains, losses),
        f"{prefix}_maximum_drawdown": maximum_drawdown,
        f"{prefix}_average_underwater_drawdown": (
            float(negative_drawdowns.mean()) if not negative_drawdowns.empty else 0.0
        ),
        f"{prefix}_maximum_drawdown_duration": float(max(episodes, default=0)),
        f"{prefix}_average_drawdown_duration": (float(np.mean(episodes)) if episodes else 0.0),
        f"{prefix}_time_under_water": float((drawdowns["drawdown"] < 0.0).mean()),
        f"{prefix}_ulcer_index": float(
            np.sqrt(np.mean(np.square(drawdowns["drawdown"].to_numpy())))
        ),
        f"{prefix}_skewness": return_skewness,
        f"{prefix}_excess_kurtosis": return_kurtosis,
        f"{prefix}_positive_period_rate": float((returns > 0.0).mean()),
        f"{prefix}_average_win": float(wins.mean()) if not wins.empty else float("nan"),
        f"{prefix}_average_loss": (float(losing.mean()) if not losing.empty else float("nan")),
        f"{prefix}_payoff_ratio": (
            float(wins.mean() / abs(losing.mean()))
            if not wins.empty and not losing.empty and abs(losing.mean()) > _EPS
            else float("nan")
        ),
        f"{prefix}_profit_factor": _safe_sum_ratio(wins, -losing),
        f"{prefix}_expected_return_per_period": mean,
        f"{prefix}_largest_win": float(returns.max()),
        f"{prefix}_largest_loss": float(returns.min()),
        f"{prefix}_maximum_consecutive_wins": float(_maximum_consecutive(returns > 0.0)),
        f"{prefix}_maximum_consecutive_losses": float(_maximum_consecutive(returns < 0.0)),
        f"{prefix}_tail_ratio": _tail_ratio(returns),
    }
    for lag in (1, 5, 21):
        metrics[f"{prefix}_autocorrelation_{lag}"] = _autocorrelation(returns, lag)
    for window in (5, 21, 63):
        rolling = _rolling_compound(returns, window).dropna()
        metrics[f"{prefix}_worst_rolling_{window}_period_return"] = (
            float(rolling.min()) if not rolling.empty else float("nan")
        )
    losses_series = -returns
    for level in var_levels:
        label = str(int(round(level * 100)))
        value_at_risk = float(losses_series.quantile(level))
        tail = losses_series[losses_series >= value_at_risk]
        metrics[f"{prefix}_var_{label}"] = value_at_risk
        metrics[f"{prefix}_expected_shortfall_{label}"] = float(tail.mean())
    return metrics, drawdowns


def _safe_sum_ratio(numerator: pd.Series, denominator: pd.Series) -> float:
    denominator_sum = float(denominator.sum())
    return float(numerator.sum()) / denominator_sum if denominator_sum > _EPS else float("nan")


def _tail_ratio(returns: pd.Series) -> float:
    lower = abs(float(returns.quantile(0.05)))
    return float(returns.quantile(0.95)) / lower if lower > _EPS else float("nan")


def _period_returns(daily: pd.DataFrame, frequency: str) -> pd.DataFrame:
    columns = ["gross_return", "net_return"]
    return (1.0 + daily[columns]).resample(frequency).prod() - 1.0


def _position_holding_period(weights: pd.DataFrame, active_threshold: float) -> float:
    """Average contiguous active-position length across all entity columns."""
    durations: list[int] = []
    for column in weights:
        active = weights[column].abs().to_numpy() > active_threshold
        current = 0
        for value in active:
            if value:
                current += 1
            elif current:
                durations.append(current)
                current = 0
        if current:
            durations.append(current)
    return float(np.mean(durations)) if durations else float("nan")


def _asset_attribution(
    contributions: pd.DataFrame,
    weights: pd.DataFrame,
) -> pd.DataFrame:
    """Attribute additive gross return and covariance-based portfolio risk by entity.

    Contributions are ``weight * simple_return`` on the date-by-entity grid. Variance
    shares use each entity contribution's covariance with total portfolio return; shares
    therefore sum to one apart from numerical error when portfolio variance is positive.
    """
    positive_weights = weights.clip(lower=0.0)
    negative_weights = weights.clip(upper=0.0)
    portfolio_return = contributions.sum(axis=1)
    portfolio_variance = float(portfolio_return.var(ddof=1))
    portfolio_volatility = float(portfolio_return.std(ddof=1))
    covariance = pd.Series(
        {column: float(contributions[column].cov(portfolio_return)) for column in contributions},
        dtype=float,
    )
    result = pd.DataFrame(
        {
            "gross_return_contribution": contributions.sum(axis=0),
            "mean_daily_contribution": contributions.mean(axis=0),
            "long_contribution": contributions.where(positive_weights > 0.0, 0.0).sum(axis=0),
            "short_contribution": contributions.where(negative_weights < 0.0, 0.0).sum(axis=0),
            "mean_absolute_weight": weights.abs().mean(axis=0),
            "maximum_absolute_weight": weights.abs().max(axis=0),
            "active_periods": (weights.abs() > _EPS).sum(axis=0),
            "variance_contribution_share": (
                covariance / portfolio_variance if portfolio_variance > _EPS else np.nan
            ),
            "volatility_contribution": (
                covariance / portfolio_volatility if portfolio_volatility > _EPS else np.nan
            ),
        }
    )
    return result.sort_values("gross_return_contribution", ascending=False)


def _signal_attribution(
    scores: pd.DataFrame,
    asset_returns: pd.DataFrame,
    weights: pd.DataFrame,
    bucket_count: int,
) -> pd.DataFrame:
    """Attribute realized contribution to within-date score-rank buckets."""
    score_series = cast(pd.Series, scores.stack(future_stack=True))
    return_series = cast(pd.Series, asset_returns.stack(future_stack=True))
    weight_series = cast(pd.Series, weights.stack(future_stack=True))
    valid = score_series.notna() & return_series.notna() & (weight_series.abs() > _EPS)
    frame = pd.DataFrame(
        {
            "score": score_series[valid],
            "asset_return": return_series[valid],
            "weight": weight_series[valid],
        }
    )
    time = str(frame.index.names[0])
    percentiles = frame["score"].groupby(level=time).rank(method="first", pct=True)
    frame["bucket"] = np.ceil(percentiles * bucket_count).clip(1, bucket_count).astype(int)
    frame["contribution"] = frame["weight"] * frame["asset_return"]
    return frame.groupby("bucket", observed=True).agg(
        observations=("contribution", "size"),
        mean_score=("score", "mean"),
        mean_asset_return=("asset_return", "mean"),
        mean_weight=("weight", "mean"),
        total_return_contribution=("contribution", "sum"),
        mean_daily_row_contribution=("contribution", "mean"),
    )


def _maximum_drawdown_attribution(
    drawdowns: pd.DataFrame,
    contributions: pd.DataFrame,
    costs: pd.DataFrame,
) -> pd.DataFrame:
    """Attribute additive returns and costs from the preceding wealth peak to the worst trough."""
    trough = drawdowns["drawdown"].idxmin()
    history = drawdowns.loc[:trough, "wealth"]
    peak = history.idxmax()
    period = contributions.loc[peak:trough]
    result = pd.DataFrame(
        {
            "return_contribution": period.sum(axis=0),
            "mean_daily_contribution": period.mean(axis=0),
        }
    )
    cost_rows = pd.DataFrame(
        {
            "return_contribution": [
                -costs.loc[peak:trough, "transaction"].sum(),
                -costs.loc[peak:trough, "borrow"].sum(),
            ],
            "mean_daily_contribution": [
                -costs.loc[peak:trough, "transaction"].mean(),
                -costs.loc[peak:trough, "borrow"].mean(),
            ],
        },
        index=pd.Index(["transaction_costs", "borrow_costs"], name=result.index.name),
    )
    result = pd.concat([result, cost_rows])
    result["peak"] = peak
    result["trough"] = trough
    return result.sort_values("return_contribution")


def build_portfolio_evaluation(
    scores: pd.Series,
    target_returns: pd.Series,
    *,
    normalize: Callable[[Tensor, Tensor], Tensor],
    return_type: ReturnType,
    annualization_periods: int,
    risk_free_rate: float,
    minimum_acceptable_return: float,
    var_levels: tuple[float, ...],
    rolling_window: int,
    signal_buckets: int,
    active_weight_threshold: float,
    include_initial_trade: bool,
    execution_fee: float,
    bid_ask_spread: float,
    slippage: float,
    borrow_cost: float,
) -> PortfolioEvaluation:
    """Construct one realized portfolio path and derive all metrics consistently.

    Aligned scalar scores and target returns are reshaped to ``[D, N]``. The objective's
    normalization callback creates weights once; all gross/net returns, exposures,
    turnover, costs, drawdowns, rolling statistics, and attributions reuse those weights.
    Log targets are converted to simple asset returns before accounting. Final
    reconciliation verifies that long plus short contributions equal gross return and
    gross return minus costs equals net return on every date.
    """
    score_frame = _dense(scores, "score")
    raw_return_frame = _dense(target_returns, "target").reindex_like(score_frame)
    valid = score_frame.notna() & raw_return_frame.notna()
    invalid_dates = valid.index[~valid.any(axis=1)]
    if len(invalid_dates):
        raise ValueError(f"portfolio dates without valid observations: {invalid_dates[0]}")
    weights = _normalised_weights(score_frame, valid, normalize)
    asset_returns = _simple_returns(raw_return_frame.fillna(0.0), return_type).where(valid, 0.0)
    contributions = weights * asset_returns
    gross_returns = contributions.sum(axis=1).rename("gross_return")

    turnover, costs = _turnover_and_costs(
        weights,
        asset_returns,
        execution_fee=execution_fee,
        bid_ask_spread=bid_ask_spread,
        slippage=slippage,
        borrow_cost=borrow_cost,
        include_initial_trade=include_initial_trade,
    )
    net_returns = (gross_returns - costs["total"]).rename("net_return")
    long_weights = weights.clip(lower=0.0)
    short_weights = weights.clip(upper=0.0)
    exposures = pd.DataFrame(
        {
            "gross": weights.abs().sum(axis=1),
            "net": weights.sum(axis=1),
            "long": long_weights.sum(axis=1),
            "short": -short_weights.sum(axis=1),
        }
    )
    active = weights.abs() > active_weight_threshold
    composition = pd.DataFrame(
        {
            "active_positions": active.sum(axis=1),
            "long_positions": (weights > active_weight_threshold).sum(axis=1),
            "short_positions": (weights < -active_weight_threshold).sum(axis=1),
            "concentration_hhi": weights.pow(2).sum(axis=1),
            "effective_positions": 1.0 / weights.pow(2).sum(axis=1).clip(lower=_EPS),
            "largest_absolute_weight": weights.abs().max(axis=1),
            "top_5_weight_share": weights.abs().apply(
                lambda row: float(row.nlargest(min(5, len(row))).sum()), axis=1
            ),
            "top_10_weight_share": weights.abs().apply(
                lambda row: float(row.nlargest(min(10, len(row))).sum()), axis=1
            ),
        }
    )
    daily = pd.DataFrame(
        {
            "gross_return": gross_returns,
            "net_return": net_returns,
            "long_return_contribution": (long_weights * asset_returns).sum(axis=1),
            "short_return_contribution": (short_weights * asset_returns).sum(axis=1),
            "long_leg_return": (long_weights * asset_returns).sum(axis=1)
            / exposures["long"].replace(0.0, np.nan),
            "short_leg_return": (short_weights * asset_returns).sum(axis=1)
            / exposures["short"].replace(0.0, np.nan),
        }
    )

    gross_metrics, _ = _performance_metrics(
        gross_returns,
        prefix="gross",
        annualization_periods=annualization_periods,
        risk_free_rate=risk_free_rate,
        minimum_acceptable_return=minimum_acceptable_return,
        var_levels=var_levels,
    )
    net_metrics, drawdowns = _performance_metrics(
        net_returns,
        prefix="net",
        annualization_periods=annualization_periods,
        risk_free_rate=risk_free_rate,
        minimum_acceptable_return=minimum_acceptable_return,
        var_levels=var_levels,
    )
    rolling_mean = net_returns.rolling(rolling_window).mean()
    rolling_std = net_returns.rolling(rolling_window).std(ddof=1)
    rolling = pd.DataFrame(
        {
            "net_return": _rolling_compound(net_returns, rolling_window),
            "annualized_volatility": rolling_std * np.sqrt(annualization_periods),
            "sharpe_ratio": rolling_mean
            / rolling_std.replace(0.0, np.nan)
            * np.sqrt(annualization_periods),
        }
    )
    rolling["one_way_turnover"] = (
        turnover["one_way_turnover"].rolling(rolling_window, min_periods=rolling_window).mean()
    )
    rolling["net_exposure"] = (
        exposures["net"].rolling(rolling_window, min_periods=rolling_window).mean()
    )

    monthly_returns = _period_returns(daily, "ME")
    yearly_returns = _period_returns(daily, "YE")

    metrics = gross_metrics | net_metrics
    metrics |= {
        "observations": float(len(daily)),
        "gross_to_net_total_return_drag": (
            gross_metrics["gross_total_return"] - net_metrics["net_total_return"]
        ),
        "total_cost": float(costs["total"].sum()),
        "total_transaction_cost": float(costs["transaction"].sum()),
        "total_borrow_cost": float(costs["borrow"].sum()),
        "annualized_cost_drag": float(costs["total"].mean() * annualization_periods),
        "total_l1_turnover": float(turnover["l1_turnover"].sum()),
        "total_one_way_turnover": float(turnover["one_way_turnover"].sum()),
        "total_long_turnover": float(turnover["long_turnover"].sum()),
        "total_short_turnover": float(turnover["short_turnover"].sum()),
        "mean_daily_l1_turnover": float(turnover["l1_turnover"].mean()),
        "annualized_l1_turnover": float(turnover["l1_turnover"].mean() * annualization_periods),
        "mean_daily_one_way_turnover": float(turnover["one_way_turnover"].mean()),
        "mean_daily_long_turnover": float(turnover["long_turnover"].mean()),
        "mean_daily_short_turnover": float(turnover["short_turnover"].mean()),
        "mean_gross_exposure": float(exposures["gross"].mean()),
        "mean_net_exposure": float(exposures["net"].mean()),
        "mean_long_exposure": float(exposures["long"].mean()),
        "mean_short_exposure": float(exposures["short"].mean()),
        "minimum_net_exposure": float(exposures["net"].min()),
        "maximum_net_exposure": float(exposures["net"].max()),
        "mean_active_positions": float(composition["active_positions"].mean()),
        "mean_long_positions": float(composition["long_positions"].mean()),
        "mean_short_positions": float(composition["short_positions"].mean()),
        "mean_concentration_hhi": float(composition["concentration_hhi"].mean()),
        "mean_effective_positions": float(composition["effective_positions"].mean()),
        "maximum_absolute_weight": float(composition["largest_absolute_weight"].max()),
        "average_position_holding_period": _position_holding_period(
            weights, active_weight_threshold
        ),
        "annualized_long_return_contribution": float(
            daily["long_return_contribution"].mean() * annualization_periods
        ),
        "annualized_short_return_contribution": float(
            daily["short_return_contribution"].mean() * annualization_periods
        ),
        "total_long_return_contribution": float(daily["long_return_contribution"].sum()),
        "total_short_return_contribution": float(daily["short_return_contribution"].sum()),
    }
    for side in ("long", "short"):
        leg_returns = daily[f"{side}_leg_return"].fillna(0.0).to_numpy(dtype=float)
        metrics |= {
            f"{side}_leg_total_return": float(np.prod(1.0 + leg_returns) - 1.0),
            f"{side}_leg_annualized_arithmetic_return": float(
                np.mean(leg_returns) * annualization_periods
            ),
            f"{side}_leg_annualized_volatility": float(
                np.std(leg_returns, ddof=1) * np.sqrt(annualization_periods)
            ),
        }
    for exposure in ("gross", "net", "long", "short"):
        values = exposures[exposure]
        metrics |= {
            f"median_{exposure}_exposure": float(values.median()),
            f"minimum_{exposure}_exposure": float(values.min()),
            f"maximum_{exposure}_exposure": float(values.max()),
            f"p01_{exposure}_exposure": float(values.quantile(0.01)),
            f"p99_{exposure}_exposure": float(values.quantile(0.99)),
        }
    for frequency, period_returns in (("month", monthly_returns), ("year", yearly_returns)):
        for return_kind in ("gross", "net"):
            values = period_returns[f"{return_kind}_return"]
            metrics |= {
                f"{return_kind}_positive_{frequency}_rate": float((values > 0.0).mean()),
                f"{return_kind}_best_{frequency}_return": float(values.max()),
                f"{return_kind}_worst_{frequency}_return": float(values.min()),
            }
    side_attribution = pd.DataFrame(
        {
            "total_contribution": [
                daily["long_return_contribution"].sum(),
                daily["short_return_contribution"].sum(),
                -costs["transaction"].sum(),
                -costs["borrow"].sum(),
            ],
            "mean_daily_contribution": [
                daily["long_return_contribution"].mean(),
                daily["short_return_contribution"].mean(),
                -costs["transaction"].mean(),
                -costs["borrow"].mean(),
            ],
        },
        index=pd.Index(["long", "short", "transaction_costs", "borrow_costs"], name="source"),
    )
    asset_attribution = _asset_attribution(contributions, weights)
    absolute_return_contribution = asset_attribution["gross_return_contribution"].abs()
    return_contribution_total = float(absolute_return_contribution.sum())
    absolute_risk_contribution = asset_attribution["variance_contribution_share"].abs()
    risk_contribution_total = float(absolute_risk_contribution.sum())
    if return_contribution_total > _EPS:
        return_shares = absolute_return_contribution / return_contribution_total
        metrics["return_contribution_hhi"] = float(np.square(return_shares).sum())
        for count in (1, 5, 10):
            metrics[f"top_{count}_absolute_return_contribution_share"] = float(
                return_shares.nlargest(min(count, len(return_shares))).sum()
            )
    if risk_contribution_total > _EPS:
        risk_shares = absolute_risk_contribution / risk_contribution_total
        metrics["risk_contribution_hhi"] = float(np.square(risk_shares).sum())
        for count in (1, 5, 10):
            metrics[f"top_{count}_absolute_risk_contribution_share"] = float(
                risk_shares.nlargest(min(count, len(risk_shares))).sum()
            )
    signal_attribution = _signal_attribution(
        score_frame.where(valid), asset_returns.where(valid), weights, signal_buckets
    )
    drawdown_attribution = _maximum_drawdown_attribution(drawdowns, contributions, costs)

    gross_reconciliation = float(
        (daily["long_return_contribution"] + daily["short_return_contribution"] - gross_returns)
        .abs()
        .max()
    )
    net_reconciliation = float((gross_returns - costs["total"] - net_returns).abs().max())
    if gross_reconciliation > 1e-10 or net_reconciliation > 1e-10:
        raise RuntimeError("portfolio contribution reconciliation failed")

    return PortfolioEvaluation(
        metrics=metrics,
        daily=daily,
        weights=weights,
        asset_returns=asset_returns,
        asset_contributions=contributions,
        exposures=exposures,
        turnover=turnover,
        costs=costs,
        composition=composition,
        drawdowns=drawdowns,
        rolling=rolling,
        tail_risk=_rolling_tail_risk(net_returns, rolling_window, var_levels),
        monthly_returns=monthly_returns,
        yearly_returns=yearly_returns,
        asset_attribution=asset_attribution,
        side_attribution=side_attribution,
        signal_attribution=signal_attribution,
        maximum_drawdown_attribution=drawdown_attribution,
    )
