"""Dispatch registered prediction contracts to their analytics implementation."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, cast

import numpy as np
import pandas as pd
import torch
from pandas.api.types import is_numeric_dtype
from torch import Tensor, nn

from llca.analytics.evaluation.portfolio import build_portfolio_evaluation
from llca.analytics.evaluation.signals import evaluate_signal
from llca.analytics.inputs.risk_free import align_risk_free
from llca.analytics.modules.analytics_config import ModelEvaluationConfig
from llca.analytics.modules.portfolio_evaluation import PortfolioEvaluation
from llca.analytics.modules.test_evaluation import TestEvaluation
from llca.core.returns import ReturnType
from llca.data.index_spec import entity_level, time_level
from llca.data.modules.masked_panel import MaskedPanel
from llca.models.estimators.evaluation_spec import ObjectiveLayout, ObjectiveTensorAdapter
from llca.models.estimators.objective_output import (
    DiagnosticObjectiveOutput,
    DiagnosticValue,
    objective_loss,
)
from llca.models.estimators.prediction import PredictionKind, PredictionOutput


class _EvaluationHandler(Protocol):
    """One prediction-kind adapter registered with the analytics dispatcher."""

    def __call__(
        self,
        predictions: PredictionOutput,
        target: pd.Series,
        objective: nn.Module | None,
        config: ModelEvaluationConfig,
        risk_free: pd.Series,
        *,
        objective_layout: ObjectiveLayout,
        objective_adapter: ObjectiveTensorAdapter | None,
    ) -> TestEvaluation: ...


def valid_supervision(
    supervision: MaskedPanel,
    column: str,
    index: pd.Index,
) -> pd.Series:
    """Boolean mask over ``index`` of rows whose supervision label is usable.

    A row is valid when its observation flag is set and its label is present — finite for a
    numeric label, non-null otherwise. Returns a bool Series indexed by ``index``.
    """
    target = supervision.values[column].reindex(index)
    observed = (
        supervision.observed[column].reindex(index).astype("boolean").fillna(False).astype(bool)
    )
    finite = (
        pd.Series(np.isfinite(target.to_numpy(dtype=float)), index=target.index, dtype=bool)
        if is_numeric_dtype(target.dtype)
        else target.notna()
    )
    return observed & finite


def _aligned_target(
    predictions: PredictionOutput,
    supervision: MaskedPanel,
    supervision_column: str,
) -> tuple[PredictionOutput, pd.Series]:
    """Pair predictions with their supervision targets, keeping only rows with a usable label.

    A row survives when its supervision is both flagged observed and finite (non-null for
    non-numeric labels). Both the predictions and the target series are filtered to those rows
    and returned. Raises if no row qualifies or if the two indices fail to match afterwards.
    """
    target = supervision.values[supervision_column].reindex(predictions.index)
    valid = valid_supervision(supervision, supervision_column, predictions.index)
    if not valid.any():
        raise ValueError("test predictions have no valid aligned supervision")
    aligned = predictions.select(valid.to_numpy(dtype=bool))
    aligned_target = target[valid]
    if not aligned.index.equals(aligned_target.index):
        raise RuntimeError("prediction/target alignment failed")
    return aligned, aligned_target


def _objective_matrices(
    predictions: pd.Series,
    target: pd.Series,
) -> tuple[Tensor, Tensor, Tensor, pd.Index]:
    """Reshape aligned scores and targets into dense date-by-entity tensors for the objective.

    A panel is pivoted so rows are dates and columns entities; a date-only series becomes a
    single column. Missing cells are zero-filled and reported through a companion boolean mask
    marking where both score and target are present. Returns ``(scores, targets, valid, dates)``
    where ``dates`` is the pivoted row axis, used to align a per-date risk-free rate.
    """
    entity = entity_level(predictions)
    if entity is None:
        score_frame = predictions.to_frame("value").sort_index()
        target_frame = target.to_frame("value").reindex_like(score_frame)
    else:
        score_frame = predictions.unstack(level=entity).sort_index()
        target_frame = target.unstack(level=entity).reindex_like(score_frame)
    valid = score_frame.notna() & target_frame.notna()
    return (
        torch.from_numpy(score_frame.fillna(0.0).to_numpy(dtype=np.float32)),
        torch.from_numpy(target_frame.fillna(0.0).to_numpy(dtype=np.float32)),
        torch.from_numpy(valid.to_numpy(dtype=bool)),
        score_frame.index,
    )


def _scalar_value(name: str, value: DiagnosticValue) -> float:
    """Coerce a diagnostic value to a Python float, requiring tensors to be single elements.

    Raises ``TypeError`` if ``value`` is a tensor with more than one element.
    """
    if isinstance(value, Tensor):
        if value.numel() != 1:
            raise TypeError(f"objective metric '{name}' must be a scalar tensor")
        return float(value.detach().cpu().item())
    return float(value)


def scalar_objective_metrics(output: object) -> dict[str, float]:
    """Turn an objective output into a flat dict of reporting scalars.

    Always includes ``loss``; a diagnostic-carrying output contributes its extra metrics too.
    Raises ``ValueError`` if a diagnostic tries to shadow the ``loss`` key.
    """
    metrics = {"loss": _scalar_value("loss", objective_loss(output))}
    if isinstance(output, DiagnosticObjectiveOutput):
        for name, value in output.diagnostic_metrics().items():
            if name == "loss":
                raise ValueError("objective diagnostics must not redefine 'loss'")
            metrics[name] = _scalar_value(name, value)
    return metrics


def _portfolio_risk_free_tensor(
    objective: nn.Module, risk_free: pd.Series, dates: pd.Index
) -> Tensor | None:
    """Align the per-date risk-free rate for a portfolio objective's recomputation.

    Returns ``None`` for non-portfolio objectives (a pointwise loss such as MSE takes no
    risk-free rate), so the same aligned rate that funds residual cash in the portfolio
    accounting also drives the objective metrics, applied exactly once per date.
    """
    if not callable(getattr(objective, "normalize_weights", None)):
        return None
    aligned = align_risk_free(risk_free, dates)
    return torch.from_numpy(aligned.to_numpy(dtype=np.float32))


def _evaluate_objective_metrics(
    predictions: PredictionOutput,
    target: pd.Series,
    objective: nn.Module,
    risk_free: pd.Series,
    *,
    layout: ObjectiveLayout,
    adapter: ObjectiveTensorAdapter | None,
) -> dict[str, float]:
    """Recompute the training objective on the test predictions and return its metrics.

    The score/target/valid tensors and their per-date row axis come from an explicit
    ``adapter`` when supplied, otherwise from the dense panel packing for the ``"panel"``
    layout. Because both paths yield ``dates``, a portfolio objective is rerun with the same
    per-date risk-free rate used by the portfolio accounting -- aligned once, regardless of
    layout -- so its metrics and the reported returns share one funding convention. The
    objective is run under inference mode and its output reduced to scalar metrics. A
    ``"row"`` layout without an adapter is unsupported and raises.
    """
    if adapter is not None:
        score_tensor, target_tensor, valid_tensor, dates = adapter(predictions, target)
    elif layout == "panel":
        if not isinstance(predictions.values, pd.Series) or not is_numeric_dtype(target.dtype):
            raise TypeError("panel objective evaluation requires scalar numerical outputs")
        score_tensor, target_tensor, valid_tensor, dates = _objective_matrices(
            predictions.values.astype(float),
            target.astype(float),
        )
    else:
        raise NotImplementedError(
            "row-layout objective analytics is not implemented; provide an "
            "EvaluationSpec.objective_adapter to attach that tensor contract"
        )
    risk_free_tensor = _portfolio_risk_free_tensor(objective, risk_free, dates)
    with torch.inference_mode():
        if risk_free_tensor is None:
            output = objective(score_tensor, target_tensor, valid_tensor)
        else:
            output = objective(
                score_tensor, target_tensor, valid_tensor, risk_free=risk_free_tensor
            )
    return scalar_objective_metrics(output)


def _build_portfolio(
    predictions: PredictionOutput,
    target: pd.Series,
    objective: nn.Module | None,
    config: ModelEvaluationConfig,
    risk_free: pd.Series,
) -> PortfolioEvaluation:
    """Turn portfolio scores into a full portfolio evaluation using the objective's contract.

    Reads the weight-normalization callable, return type, and cost parameters off the
    registered objective (falling back to config defaults) and hands them, with the scores and
    realized returns, to the portfolio builder. Raises ``TypeError`` if the scores are not a
    scalar series or the objective exposes no ``normalize_weights``.
    """
    if not isinstance(predictions.values, pd.Series):
        raise TypeError("portfolio construction requires one scalar score per date/instrument")
    normalize: object = getattr(objective, "normalize_weights", None)
    if not callable(normalize):
        raise TypeError(
            "portfolio analytics requires the registered portfolio objective's callable "
            "normalize_weights contract; prediction values have no implicit weight semantics"
        )
    typed_normalize = cast(Callable[[Tensor, Tensor], Tensor], normalize)
    return_type = cast(ReturnType, getattr(objective, "return_type", config.return_type))
    return build_portfolio_evaluation(
        predictions.values.astype(float),
        target.astype(float),
        normalize=typed_normalize,
        return_type=return_type,
        annualization_periods=config.annualization_periods,
        risk_free=risk_free,
        minimum_acceptable_return=config.minimum_acceptable_return,
        var_levels=config.var_levels,
        autocorrelation_lags=config.autocorrelation_lags,
        worst_rolling_windows=config.worst_rolling_windows,
        rolling_window=config.rolling_window,
        signal_buckets=config.signal_buckets,
        active_weight_threshold=config.active_weight_threshold,
        include_initial_trade=config.include_initial_trade,
        execution_fee=float(getattr(objective, "execution_fee", 0.0)),
        bid_ask_spread=float(getattr(objective, "bid_ask_spread", 0.0)),
        slippage=float(getattr(objective, "slippage", 0.0)),
        borrow_cost=float(getattr(objective, "borrow_cost", 0.0)),
        target_threshold=config.target_threshold,
    )


def _portfolio_item_series(
    frame: pd.DataFrame,
    predictions: pd.Series,
    name: str,
) -> pd.Series:
    """Flatten a dense date-by-entity portfolio field back onto the prediction-item index.

    A single-column frame maps a date-only series; a wider frame is stacked to the panel shape.
    The result is reindexed to the predictions and renamed to ``name``. Raises if the field
    cannot cover every prediction item.
    """
    entity = entity_level(predictions)
    if entity is None:
        if frame.shape[1] != 1:
            raise RuntimeError(f"date-only portfolio field '{name}' must have one column")
        values = frame.iloc[:, 0]
    else:
        values = cast(pd.Series, frame.stack(future_stack=True))
    result = values.reindex(predictions.index).astype(float).rename(name)
    if result.isna().any():
        raise RuntimeError(f"portfolio field '{name}' could not be restored to prediction items")
    return result


def _evaluate_portfolio_predictions(
    predictions: PredictionOutput,
    target: pd.Series,
    objective: nn.Module | None,
    config: ModelEvaluationConfig,
    risk_free: pd.Series,
    *,
    objective_layout: ObjectiveLayout,
    objective_adapter: ObjectiveTensorAdapter | None,
) -> TestEvaluation:
    if not isinstance(predictions.values, pd.Series) or not is_numeric_dtype(target.dtype):
        raise TypeError("portfolio analytics requires scalar numerical scores and returns")
    objective_metrics = (
        _evaluate_objective_metrics(
            predictions,
            target,
            objective,
            risk_free,
            layout=objective_layout,
            adapter=objective_adapter,
        )
        if objective is not None
        else {}
    )
    portfolio = _build_portfolio(predictions, target, objective, config, risk_free)
    realized_target = _portfolio_item_series(
        portfolio.asset_returns,
        predictions.values,
        "realized_simple_return",
    )
    allocation = _portfolio_item_series(
        portfolio.weights,
        predictions.values,
        "allocation_weight",
    )
    signal = evaluate_signal(
        predictions,
        realized_target,
        decisions=allocation,
        bucket_count=config.signal_buckets,
        target_threshold=config.target_threshold,
        active_weight_threshold=config.active_weight_threshold,
        annualization_periods=config.annualization_periods,
        rolling_window=config.rolling_window,
        signal_decay_periods=config.signal_decay_periods,
    )
    dates = pd.Index(predictions.index.get_level_values(time_level(predictions.values))).nunique()
    return TestEvaluation(
        predictions=predictions,
        target=realized_target,
        signal=signal,
        objective_metrics=objective_metrics,
        portfolio=portfolio,
        valid_observations=len(target),
        dates=int(dates),
    )


_EVALUATORS: dict[PredictionKind, _EvaluationHandler] = {
    "portfolio": _evaluate_portfolio_predictions,
}


def require_supported_prediction_kind(kind: PredictionKind) -> None:
    """Raise ``NotImplementedError`` unless ``kind`` has a registered analytics evaluator."""
    if kind not in _EVALUATORS:
        raise NotImplementedError(
            f"analytics for prediction kind '{kind}' is not implemented; "
            "register a dedicated evaluator before enabling this prediction contract"
        )


def evaluate_predictions(
    predictions: PredictionOutput,
    supervision: MaskedPanel,
    supervision_column: str,
    objective: nn.Module | None,
    config: ModelEvaluationConfig,
    risk_free: pd.Series,
    *,
    objective_layout: ObjectiveLayout = "panel",
    objective_adapter: ObjectiveTensorAdapter | None = None,
) -> TestEvaluation:
    """Evaluate a model's predictions against its supervision and return the full report.

    Verifies the prediction kind is supported, aligns predictions to their valid labels, and
    delegates to the evaluator registered for that kind. ``objective_layout`` and
    ``objective_adapter`` describe how the training objective's tensors are reconstructed.
    """
    require_supported_prediction_kind(predictions.kind)
    aligned, target = _aligned_target(predictions, supervision, supervision_column)
    evaluator = _EVALUATORS[predictions.kind]
    return evaluator(
        aligned,
        target,
        objective,
        config,
        risk_free,
        objective_layout=objective_layout,
        objective_adapter=objective_adapter,
    )
