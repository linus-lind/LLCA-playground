"""Return, risk, signal-bucket, and worst-drawdown attribution of a realized portfolio path.

These functions decompose an already-constructed portfolio (dense weights, asset returns,
contributions, cash, and costs) into per-entity, per-side, per-bucket, and drawdown-episode
attributions. They own no portfolio construction; :func:`llca.analytics.evaluation.portfolio.
build_portfolio_evaluation` produces the inputs and calls them.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd

from llca.analytics.stats.statistics import EPS, rank_buckets


def attribute_assets(
    contributions: pd.DataFrame,
    weights: pd.DataFrame,
    cash_contribution: pd.Series,
    cash_weight: pd.Series,
) -> pd.DataFrame:
    """Break the portfolio's gross return and risk down by individual asset and cash.

    Each asset (plus a cash row) reports its summed and mean return contribution, its long- and
    short-side contribution, weight statistics and active-period count, and its risk share —
    the covariance of its contribution with total portfolio return over portfolio variance,
    which sums to one up to rounding when portfolio variance is positive. Rows are ordered by
    descending return contribution.
    """
    cash_label = "Cash"
    while cash_label in contributions.columns:
        cash_label = f"_{cash_label}"
    all_contributions = contributions.assign(**{cash_label: cash_contribution})
    all_weights = weights.assign(**{cash_label: cash_weight})
    positive_weights = weights.clip(lower=0.0)
    negative_weights = weights.clip(upper=0.0)
    portfolio_return = all_contributions.sum(axis=1)
    portfolio_variance = float(portfolio_return.var(ddof=1))
    portfolio_volatility = float(portfolio_return.std(ddof=1))
    # Every column's covariance with the portfolio return in one centered matrix product rather
    # than a per-asset ``Series.cov``; contributions are dense, so a shared observation count and
    # ddof=1 reproduce the pairwise result exactly (and keep the shares summing to one).
    observations = len(portfolio_return)
    if observations > 1:
        centered = all_contributions.sub(all_contributions.mean(axis=0), axis=1)
        covariance = centered.mul(portfolio_return - portfolio_return.mean(), axis=0).sum(
            axis=0
        ) / (observations - 1)
    else:
        covariance = pd.Series(np.nan, index=all_contributions.columns, dtype=float)
    long_contribution = contributions.where(positive_weights > 0.0, 0.0).sum(axis=0)
    short_contribution = contributions.where(negative_weights < 0.0, 0.0).sum(axis=0)
    long_contribution.loc[cash_label] = 0.0
    short_contribution.loc[cash_label] = 0.0
    result = pd.DataFrame(
        {
            "gross_return_contribution": all_contributions.sum(axis=0),
            "mean_daily_contribution": all_contributions.mean(axis=0),
            "long_contribution": long_contribution,
            "short_contribution": short_contribution,
            "mean_absolute_weight": all_weights.abs().mean(axis=0),
            "maximum_absolute_weight": all_weights.abs().max(axis=0),
            "active_periods": (all_weights.abs() > EPS).sum(axis=0),
            "variance_contribution_share": (
                covariance / portfolio_variance if portfolio_variance > EPS else np.nan
            ),
            "volatility_contribution": (
                covariance / portfolio_volatility if portfolio_volatility > EPS else np.nan
            ),
        }
    )
    return result.sort_values("gross_return_contribution", ascending=False)


def attribute_signal_buckets(
    scores: pd.DataFrame,
    asset_returns: pd.DataFrame,
    weights: pd.DataFrame,
    bucket_count: int,
    target_threshold: float,
    active_weight_threshold: float,
) -> pd.DataFrame:
    """Group every held position into score buckets and attribute return to each.

    Scores are ranked within each date when a usable cross-section exists and pooled across
    time otherwise. Each bucket reports its score bounds, coverage, mean score and asset return,
    active-position hit rate, mean weight, and both total and mean-daily return contribution,
    revealing where in the score distribution the portfolio earns its return.
    """
    score_series = cast(pd.Series, scores.stack(future_stack=True))
    return_series = cast(pd.Series, asset_returns.stack(future_stack=True))
    weight_series = cast(pd.Series, weights.stack(future_stack=True))
    valid = score_series.notna() & return_series.notna()
    frame = pd.DataFrame(
        {
            "score": score_series[valid],
            "asset_return": return_series[valid],
            "weight": weight_series[valid].fillna(0.0),
        }
    )
    time = str(frame.index.names[0])
    usable_cross_section = any(
        group["score"].nunique() > 1 and group["asset_return"].nunique() > 1
        for _, group in frame.groupby(level=time, sort=False)
    )
    frame["bucket"] = rank_buckets(frame["score"], bucket_count, pooled=not usable_cross_section)
    frame["contribution"] = frame["weight"] * frame["asset_return"]
    active = frame["weight"].abs() > active_weight_threshold
    frame["correct_direction"] = (
        ((frame["weight"] > 0.0) == (frame["asset_return"] > target_threshold))
        .astype(float)
        .where(active)
    )
    summary = frame.groupby("bucket", observed=True).agg(
        score_low=("score", "min"),
        score_high=("score", "max"),
        observations=("contribution", "size"),
        directional_observations=("correct_direction", "count"),
        mean_score=("score", "mean"),
        mean_asset_return=("asset_return", "mean"),
        hit_rate=("correct_direction", "mean"),
        mean_weight=("weight", "mean"),
        total_return_contribution=("contribution", "sum"),
    )
    daily = (
        frame.groupby([pd.Grouper(level=time), "bucket"], observed=True)["contribution"]
        .sum()
        .unstack("bucket")
        .reindex(pd.Index(frame.index.get_level_values(time).unique()))
        .fillna(0.0)
    )
    summary["mean_daily_contribution"] = daily.mean(axis=0).reindex(summary.index)
    return summary


def attribute_maximum_drawdown(
    drawdowns: pd.DataFrame,
    contributions: pd.DataFrame,
    cash_contribution: pd.Series,
    costs: pd.DataFrame,
) -> pd.DataFrame:
    """Attribute the worst drawdown to per-asset returns, cash, and costs.

    Locates the deepest trough and the high-water-mark peak preceding it (a virtual pre-sample
    peak when the drawdown opens at the very start), then sums each asset's contribution, the
    cash return, and the transaction and borrow costs over the decline — starting the period
    after the peak date. Returns those contributions with the peak and trough timestamps,
    ordered from most negative upward.
    """
    trough = drawdowns["drawdown"].idxmin()
    history = drawdowns.loc[:trough, "wealth"]
    trough_position = int(drawdowns.index.get_indexer(pd.Index([trough]))[0])
    recovered = history.index[
        np.isclose(
            drawdowns.loc[history.index, "drawdown"].to_numpy(dtype=float),
            0.0,
            rtol=0.0,
            atol=EPS,
        )
    ]
    if len(recovered):
        peak_timestamp: pd.Timestamp | None = pd.Timestamp(recovered[-1])
        peak_position = int(drawdowns.index.get_indexer(pd.Index([peak_timestamp]))[0])
    else:
        # The worst episode began at the virtual pre-sample NAV of one.
        peak_timestamp = None
        peak_position = -1
    # The return on the high-water-mark date creates the peak and is not part of the
    # subsequent drawdown. Attribution starts with the next realized portfolio return.
    period = contributions.iloc[peak_position + 1 : trough_position + 1]
    period_cash = cash_contribution.iloc[peak_position + 1 : trough_position + 1]
    period_costs = costs.iloc[peak_position + 1 : trough_position + 1]
    result = pd.DataFrame(
        {
            "return_contribution": period.sum(axis=0),
            "mean_daily_contribution": period.mean(axis=0),
        }
    )
    non_asset_rows = pd.DataFrame(
        {
            "return_contribution": [
                period_cash.sum(),
                -period_costs["transaction"].sum(),
                -period_costs["borrow"].sum(),
            ],
            "mean_daily_contribution": [
                period_cash.mean(),
                -period_costs["transaction"].mean(),
                -period_costs["borrow"].mean(),
            ],
        },
        index=pd.Index(["cash", "transaction_costs", "borrow_costs"], name=result.index.name),
    )
    result = pd.concat([result, non_asset_rows])
    peak_value = pd.NaT if peak_timestamp is None else peak_timestamp
    result["peak"] = peak_value
    result["trough"] = trough
    return result.sort_values("return_contribution")
