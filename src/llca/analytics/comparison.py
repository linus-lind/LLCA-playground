from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from llca.analytics.modules.test_evaluation import TestEvaluation
from llca.analytics.utils.registered_model_metadata import RegisteredModelMetadata


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
    """Aligned reports and rectangular metric tables for all configured models."""

    results: tuple[ModelEvaluationResult, ...]
    start: pd.Timestamp
    end: pd.Timestamp
    common_index: pd.Index
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
    common_index: pd.Index,
) -> ComparisonEvaluation:
    """Create outer-union tables so task-specific metrics remain comparable when valid."""
    if not results:
        raise ValueError("model comparison requires at least one evaluation result")
    labels = [result.label for result in results]
    if len(labels) != len(set(labels)):
        raise ValueError("model comparison labels must be unique")
    loss_metrics = {result.label: result.evaluation.objective_metrics for result in results}
    signal_metrics = {result.label: result.evaluation.signal.metrics for result in results}
    portfolio_metrics = {
        result.label: (
            result.evaluation.portfolio.metrics if result.evaluation.portfolio is not None else {}
        )
        for result in results
    }
    return ComparisonEvaluation(
        results=results,
        start=start,
        end=end,
        common_index=common_index,
        loss_metrics=_metric_table(loss_metrics),
        signal_metrics=_metric_table(signal_metrics),
        portfolio_metrics=_metric_table(portfolio_metrics),
    )
