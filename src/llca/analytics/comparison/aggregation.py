from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from llca.analytics.modules.registered_model import RegisteredModelMetadata
from llca.analytics.modules.test_evaluation import TestEvaluation


@dataclass(frozen=True, slots=True)
class ModelEvaluationResult:
    """One model's metadata and report within a common comparison universe."""

    metadata: RegisteredModelMetadata
    evaluation: TestEvaluation

    @property
    def label(self) -> str:
        return self.metadata.config.label


@dataclass(frozen=True, slots=True)
class ComparisonEvaluation:
    """Per-model portfolio reports, each on its own native universe, plus shared metric tables.

    The cross-model item overlap used to validate agreeing targets is not carried here: it is
    consumed only by the signal-correlation matrix, which receives it directly from the
    inference step, so it does not travel through this object.
    """

    results: tuple[ModelEvaluationResult, ...]
    start: pd.Timestamp
    end: pd.Timestamp
    loss_metrics: pd.DataFrame
    signal_metrics: pd.DataFrame
    portfolio_metrics: pd.DataFrame


def _metric_table(metrics: dict[str, dict[str, float]]) -> pd.DataFrame:
    table = pd.DataFrame.from_dict(metrics, orient="index", dtype=float)
    table.index.name = "model"
    return table.sort_index(axis=1)


def build_comparison(
    results: tuple[ModelEvaluationResult, ...],
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> ComparisonEvaluation:
    """Collect the evaluated models into a comparison over the ``start``-``end`` window.

    Requires at least one result and rejects duplicate model labels. Stacks each model's
    objective, signal, and portfolio metrics into three model-indexed tables and returns them
    with the results and window bounds. Raises ``ValueError`` if there are no results or the
    labels are not unique.
    """
    if not results:
        raise ValueError("model comparison requires at least one evaluation result")
    labels = [result.label for result in results]
    if len(labels) != len(set(labels)):
        raise ValueError("model comparison labels must be unique")
    loss_metrics = {result.label: result.evaluation.objective_metrics for result in results}
    signal_metrics = {result.label: result.evaluation.signal.metrics for result in results}
    portfolio_metrics = {result.label: result.evaluation.portfolio.metrics for result in results}
    return ComparisonEvaluation(
        results=results,
        start=start,
        end=end,
        loss_metrics=_metric_table(loss_metrics),
        signal_metrics=_metric_table(signal_metrics),
        portfolio_metrics=_metric_table(portfolio_metrics),
    )
