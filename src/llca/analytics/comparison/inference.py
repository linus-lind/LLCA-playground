"""Cross-model and per-model statistical inference over one comparison.

This module is the pandas plumbing that applies the array-level hypothesis tests in
:mod:`llca.analytics.stats.inference` to the analytics domain objects. It pulls scores,
weights, and returns off each :class:`~llca.analytics.comparison.ModelEvaluationResult`,
aligns them, and produces three kinds of computed evidence:

* per-model significance statistics (directional content, information coefficient,
  risk-adjusted return) as a model-by-key frame, consumed both as inline stars on the metric
  tables and as the standalone significance table;
* symmetric model-by-model comparison matrices (Diebold-Mariano accuracy, Sharpe-ratio
  differences, and return/signal/position similarity); and
* the model confidence set.

Nothing here renders: the outputs are plain frames and small dataclasses that the reporting
layer turns into the significance table and the combined comparison figure. Running the
inference once, up front, keeps the tests that decide the report's conclusions explicit in
the run flow rather than hidden inside the exporters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
import pandas as pd

from llca.analytics.comparison.aggregation import ComparisonEvaluation, ModelEvaluationResult
from llca.analytics.modules.analytics_config import ModelEvaluationConfig
from llca.analytics.stats import inference
from llca.analytics.stats.statistics import EPS
from llca.data.index_spec import entity_level, time_level


@dataclass(frozen=True, slots=True)
class ComparisonMatrix:
    """One symmetric model-by-model matrix plus the metadata a heatmap panel needs.

    ``is_pvalue`` marks matrices whose cells are p-values so the figure can add significance
    stars. ``value_range`` is the statistic's *theoretical* span (p-values 0..1, correlations
    -1..1): colouring maps that fixed range red->blue rather than the observed min/max, so a
    cluster of tiny p-values still reads as uniformly red instead of being spread across the
    palette. Only the strict lower triangle is ever displayed; the rest is redundant.
    """

    name: str
    title: str
    frame: pd.DataFrame
    is_pvalue: bool
    value_range: tuple[float, float]
    caption: str


@dataclass(frozen=True, slots=True)
class ModelConfidenceSummary:
    """Per-model model-confidence-set result rendered as a small table in the figure."""

    title: str
    frame: pd.DataFrame
    caption: str


def _scores(result: ModelEvaluationResult) -> pd.Series:
    """Extract a model's per-item portfolio scores, enforcing the portfolio output contract.

    Raises ``RuntimeError`` if the result is not a portfolio prediction carrying a score series.
    """
    predictions = result.evaluation.predictions
    if predictions.kind != "portfolio" or not isinstance(predictions.values, pd.Series):
        raise RuntimeError("portfolio analytics received a non-portfolio prediction result")
    return predictions.values.astype(float)


def _allocation_decisions(result: ModelEvaluationResult) -> pd.Series:
    """Align the realized portfolio weights back onto the score index, one weight per item.

    Cross-sectional weights are stacked to the panel shape and single-instrument weights taken
    from the lone column. Raises ``RuntimeError`` if any scored item lacks a weight.
    """
    scores = _scores(result)
    weights = result.evaluation.portfolio.weights
    if entity_level(scores) is None:
        values = weights.iloc[:, 0]
    else:
        values = cast(pd.Series, weights.stack(future_stack=True))
    decisions = values.reindex(scores.index).astype(float)
    if decisions.isna().any():
        raise RuntimeError("portfolio weights do not align to prediction items")
    return decisions


def _net_returns(result: ModelEvaluationResult) -> pd.Series:
    return result.evaluation.portfolio.daily["net_return"]


def _excess_returns(result: ModelEvaluationResult) -> pd.Series:
    return result.evaluation.portfolio.daily["excess_net_return"]


def _collapse_to_time(series: pd.Series) -> pd.Series:
    """Average a panel series within each date, yielding one value per day.

    Collapsing same-day observations first lets the through-time tests treat the result as an
    ordinary time series.
    """
    return series.groupby(level=time_level(series)).mean()


def _excess_profitability_series(scores: pd.Series, target: pd.Series) -> pd.Series:
    """Build the excess-profit series feeding the Anatolyev-Gerko test.

    For a genuine panel this is each date's within-cross-section covariance between the traded
    sign and the realized return. When there is no usable cross-section — a single instrument
    per date — it degrades to the through-time covariance of sign and return on the
    date-collapsed series. In both cases a positive mean signals profitable direction-taking.
    """
    if entity_level(scores) is not None:
        time = time_level(scores)
        sign = np.sign(scores)
        sign_demeaned = sign - sign.groupby(level=time).transform("mean")
        target_demeaned = target - target.groupby(level=time).transform("mean")
        product = sign_demeaned * target_demeaned
        # Each date's within-cross-section covariance is the mean demeaned product; single-item
        # dates carry no cross-section and are dropped rather than counted as a zero covariance.
        daily = product.groupby(level=time).mean()
        daily = daily.where(scores.groupby(level=time).size() > 1).dropna()
        if len(daily) >= 2:
            return cast(pd.Series, daily)
        scores, target = _collapse_to_time(scores), _collapse_to_time(target)
    sign = np.sign(scores)
    return cast(pd.Series, ((sign - sign.mean()) * (target - target.mean())).dropna())


def _model_significance(
    result: ModelEvaluationResult, config: ModelEvaluationConfig
) -> dict[str, float]:
    """Run every applicable single-model significance test, keyed by statistic name.

    Applies the Pesaran-Timmermann test (time-series signals only), the excess-profitability
    test on economically active positions, the information-coefficient and directional-accuracy
    tests where their per-date inputs exist, and Sharpe significance on excess returns. The
    merged dictionary is the per-model row later split into inline stars and the significance
    table. HAC lag and thresholds come from ``config``.
    """
    values: dict[str, float] = {}
    scores = _scores(result)
    decisions = _allocation_decisions(result)
    target = result.evaluation.target.astype(float)
    lag = config.hac_lag
    aligned_target = target.reindex(scores.index)
    active = decisions.abs() > config.active_weight_threshold
    economic_decisions = decisions.where(active, 0.0)
    if result.evaluation.signal.ic_basis == "rolling_time_series":
        dated_scores = _collapse_to_time(decisions)
        dated_target = _collapse_to_time(aligned_target)
        active_dates = dated_scores.abs() > config.active_weight_threshold
        values |= inference.pesaran_timmermann(
            (dated_scores[active_dates] > 0.0).to_numpy(),
            (dated_target[active_dates] > config.target_threshold).to_numpy(),
        )
    values |= inference.excess_profitability_test(
        _excess_profitability_series(economic_decisions, aligned_target).to_numpy(), lag=lag
    )
    per_date = result.evaluation.signal.per_date
    if "rank_ic" in per_date.columns:
        rank_ic = per_date["rank_ic"].dropna()
        ic_lag = lag
        if result.evaluation.signal.ic_basis == "rolling_time_series" and len(rank_ic) > 1:
            automatic = inference.newey_west_bandwidth(len(rank_ic))
            configured = automatic if lag is None else lag
            ic_lag = min(len(rank_ic) - 1, max(config.rolling_window - 1, configured))
        values |= inference.information_coefficient_test(
            rank_ic.to_numpy(),
            annualization_periods=config.annualization_periods,
            lag=ic_lag,
            confidence=1.0 - config.test_significance_level,
        )
    if "hit_rate" in per_date.columns:
        values |= inference.directional_accuracy_test(
            per_date["hit_rate"].to_numpy(), baseline=0.5, lag=lag
        )
    excess = result.evaluation.portfolio.daily["excess_net_return"].to_numpy()
    values |= inference.sharpe_significance(
        excess,
        annualization_periods=config.annualization_periods,
        lag=lag,
        n_boot=config.bootstrap_resamples,
        block_length=config.bootstrap_block_length,
        seed=config.bootstrap_seed,
        confidence=1.0 - config.test_significance_level,
    )
    return values


def _loss_series(result: ModelEvaluationResult) -> pd.Series:
    """Return the model's daily loss — the negated net return — labelled by model.

    This is the common loss definition consumed by the Diebold-Mariano matrix and the model
    confidence set, where a smaller value is better.
    """
    return (-_net_returns(result)).rename(result.label)


def pairwise_matrix(
    labels: list[str],
    value: object,
    *,
    diagonal: float,
) -> pd.DataFrame:
    """Assemble a symmetric matrix over ``labels`` by evaluating ``value`` on each pair.

    ``value`` is called once per unordered pair and its result mirrored across the diagonal,
    which is filled with ``diagonal``.
    """
    matrix = pd.DataFrame(np.nan, index=labels, columns=labels, dtype=float)
    for i, left in enumerate(labels):
        matrix.iat[i, i] = diagonal
        for j in range(i + 1, len(labels)):
            result = value(left, labels[j])  # type: ignore[operator]
            matrix.iat[i, j] = result
            matrix.iat[j, i] = result
    return matrix


def correction_label(config: ModelEvaluationConfig) -> str:
    if config.multiple_testing_correction == "none":
        return "no multiple-testing correction"
    return f"{config.multiple_testing_correction.upper()}-adjusted"


def _shared_losses(
    results: list[ModelEvaluationResult],
) -> tuple[list[str], dict[str, pd.Series]] | None:
    """Gather each model's loss series and label, or ``None`` when fewer than two models.

    Pairwise loss comparisons are undefined with a single model, so callers skip them on
    ``None``.
    """
    if len(results) < 2:
        return None
    labels = [result.label for result in results]
    losses = {result.label: _loss_series(result) for result in results}
    return labels, losses


def _diebold_mariano_matrix(
    results: list[ModelEvaluationResult], config: ModelEvaluationConfig
) -> ComparisonMatrix | None:
    shared = _shared_losses(results)
    if shared is None:
        return None
    labels, losses = shared

    def dm_p(left: str, right: str) -> float:
        joint = pd.concat([losses[left], losses[right]], axis=1, join="inner").dropna()
        outcome = inference.diebold_mariano(
            joint.iloc[:, 0].to_numpy(), joint.iloc[:, 1].to_numpy(), lag=config.hac_lag
        )
        return outcome["dm_p_value"]

    matrix = pairwise_matrix(labels, dm_p, diagonal=float("nan"))
    matrix = inference.adjust_pairwise(matrix, config.multiple_testing_correction)
    return ComparisonMatrix(
        name="diebold_mariano_pvalues",
        title="Diebold-Mariano p-values",
        frame=matrix,
        is_pvalue=True,
        value_range=(0.0, 1.0),
        caption=(f"Equal daily economic loss (-net return); HLN/HAC, {correction_label(config)}."),
    )


def _model_confidence_summary(
    results: list[ModelEvaluationResult], config: ModelEvaluationConfig
) -> ModelConfidenceSummary | None:
    shared = _shared_losses(results)
    if shared is None:
        return None
    labels, losses = shared
    frame = pd.concat([losses[label] for label in labels], axis=1, join="inner").dropna()
    frame.columns = labels
    mcs = inference.model_confidence_set(
        frame,
        alpha=config.test_significance_level,
        n_boot=config.bootstrap_resamples,
        block_length=config.bootstrap_block_length,
        seed=config.bootstrap_seed,
    )
    return ModelConfidenceSummary(
        title="Model Confidence Set",
        frame=mcs,
        caption=(
            f"Hansen-Lunde-Nason {1.0 - config.test_significance_level:.0%} confidence set "
            f"({config.test_significance_level:.0%} significance) via the stationary bootstrap; "
            "members are indistinguishable from the best model."
        ),
    )


def _sharpe_difference_matrix(
    results: list[ModelEvaluationResult], config: ModelEvaluationConfig
) -> ComparisonMatrix | None:
    returns = {result.label: _excess_returns(result) for result in results}
    if len(returns) < 2:
        return None
    labels = [result.label for result in results]

    def sharpe_p(left: str, right: str) -> float:
        joint = pd.concat([returns[left], returns[right]], axis=1, join="inner").dropna()
        outcome = inference.sharpe_difference(
            joint.iloc[:, 0].to_numpy(),
            joint.iloc[:, 1].to_numpy(),
            annualization_periods=config.annualization_periods,
            n_boot=config.bootstrap_resamples,
            block_length=config.bootstrap_block_length,
            seed=config.bootstrap_seed,
        )
        return outcome["bootstrap_p_value"]

    matrix = pairwise_matrix(labels, sharpe_p, diagonal=float("nan"))
    matrix = inference.adjust_pairwise(matrix, config.multiple_testing_correction)
    return ComparisonMatrix(
        name="sharpe_difference_pvalues",
        title="Sharpe-Difference Bootstrap p-values",
        frame=matrix,
        is_pvalue=True,
        value_range=(0.0, 1.0),
        caption=(
            "Equal net excess-return Sharpe (paired stationary bootstrap); "
            f"{correction_label(config)}."
        ),
    )


def _return_correlation_matrix(results: list[ModelEvaluationResult]) -> ComparisonMatrix | None:
    returns = {result.label: _net_returns(result) for result in results}
    if len(returns) < 2:
        return None
    wide = pd.concat(returns, axis=1, join="inner").dropna()
    return ComparisonMatrix(
        name="portfolio_return_correlation",
        title="Net Return Correlation",
        frame=wide.corr(method="pearson"),
        is_pvalue=False,
        value_range=(-1.0, 1.0),
        caption="Pearson correlation of daily net returns.",
    )


def _signal_correlation_matrix(
    results: list[ModelEvaluationResult], common_index: pd.Index
) -> ComparisonMatrix | None:
    scores = {result.label: _scores(result).reindex(common_index) for result in results}
    if len(scores) < 2:
        return None
    wide = pd.concat(scores, axis=1, join="inner").dropna()
    return ComparisonMatrix(
        name="signal_correlation",
        title="Signal Rank Correlation",
        frame=wide.corr(method="spearman"),
        is_pvalue=False,
        value_range=(-1.0, 1.0),
        caption="Spearman correlation of model scores.",
    )


def _position_overlap_matrix(results: list[ModelEvaluationResult]) -> ComparisonMatrix | None:
    weights = {result.label: result.evaluation.portfolio.weights for result in results}
    if len(weights) < 2:
        return None
    labels = [result.label for result in results]

    def overlap(left: str, right: str) -> float:
        dates = weights[left].index.intersection(weights[right].index)
        columns = weights[left].columns.union(weights[right].columns)
        a = weights[left].reindex(index=dates, columns=columns).fillna(0.0).to_numpy(dtype=float)
        b = weights[right].reindex(index=dates, columns=columns).fillna(0.0).to_numpy(dtype=float)
        numerator = (a * b).sum(axis=1)
        denominator = np.sqrt((a * a).sum(axis=1) * (b * b).sum(axis=1))
        with np.errstate(invalid="ignore", divide="ignore"):
            cosine = np.where(denominator > EPS, numerator / denominator, np.nan)
        return float(np.nanmean(cosine)) if np.isfinite(cosine).any() else float("nan")

    matrix = pairwise_matrix(labels, overlap, diagonal=1.0)
    return ComparisonMatrix(
        name="position_overlap",
        title="Portfolio Position Overlap",
        frame=matrix,
        is_pvalue=False,
        value_range=(-1.0, 1.0),
        caption="Mean daily cosine similarity of portfolio weights.",
    )


def build_model_significance_frame(
    comparison: ComparisonEvaluation, config: ModelEvaluationConfig
) -> pd.DataFrame:
    """Per-model significance statistics as a model-by-key frame.

    Consumed by the metric tables to attach rank-IC / Sharpe / directional stars to their
    estimates, and by the significance table for the tests that stay standalone.
    """
    data = {result.label: _model_significance(result, config) for result in comparison.results}
    return pd.DataFrame(data).transpose()


def build_comparison_matrices(
    comparison: ComparisonEvaluation, config: ModelEvaluationConfig, common_index: pd.Index
) -> list[ComparisonMatrix]:
    """Build every applicable symmetric model-comparison matrix for the combined figure.

    ``common_index`` is the cross-model item overlap consumed only by the signal-correlation
    matrix; it is supplied directly rather than carried on the comparison object.
    """
    results = list(comparison.results)
    candidates = (
        _diebold_mariano_matrix(results, config),
        _sharpe_difference_matrix(results, config),
        _return_correlation_matrix(results),
        _signal_correlation_matrix(results, common_index),
        _position_overlap_matrix(results),
    )
    return [matrix for matrix in candidates if matrix is not None]


def build_model_confidence_summary(
    comparison: ComparisonEvaluation, config: ModelEvaluationConfig
) -> ModelConfidenceSummary | None:
    """Build the per-model model-confidence-set summary for the combined figure, if defined."""
    return _model_confidence_summary(list(comparison.results), config)


@dataclass(frozen=True, slots=True)
class ComparisonInference:
    """Every inferential statistic for one comparison, computed once before any rendering.

    ``model_significance`` holds the per-model significance statistics (attached as stars to
    the metric tables and rendered as the significance table); ``comparison_matrices`` and
    ``model_confidence`` feed the combined cross-model comparison figure.
    """

    model_significance: pd.DataFrame
    comparison_matrices: tuple[ComparisonMatrix, ...]
    model_confidence: ModelConfidenceSummary | None


def evaluate_comparison_inference(
    comparison: ComparisonEvaluation, config: ModelEvaluationConfig, common_index: pd.Index
) -> ComparisonInference:
    """Run every statistical significance test once, up front, before report rendering.

    Keeping the inference here — rather than hidden inside the table/figure exporters — makes
    the tests that decide the report's conclusions explicit in the run flow, and computes each
    per-model significance statistic exactly once for both the metric-table stars and the
    standalone significance table.
    """
    return ComparisonInference(
        model_significance=build_model_significance_frame(comparison, config),
        comparison_matrices=tuple(build_comparison_matrices(comparison, config, common_index)),
        model_confidence=build_model_confidence_summary(comparison, config),
    )
