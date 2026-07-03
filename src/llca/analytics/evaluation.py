from __future__ import annotations

from collections.abc import Callable
from typing import cast

import numpy as np
import pandas as pd
import torch
from pandas.api.types import is_numeric_dtype
from torch import Tensor, nn

from llca.analytics.modules.portfolio_evaluation import PortfolioEvaluation
from llca.analytics.modules.test_evaluation import TestEvaluation
from llca.analytics.portfolio import build_portfolio_evaluation
from llca.analytics.signals import evaluate_signal
from llca.analytics.utils.config import ModelEvaluationConfig
from llca.core.returns import ReturnType
from llca.data.index_spec import entity_level
from llca.data.modules.masked_panel import MaskedPanel
from llca.models.estimators.evaluation_spec import ObjectiveLayout, ObjectiveTensorAdapter
from llca.models.estimators.prediction import PredictionOutput
from llca.training.modules.training_diagnostics import (
    DiagnosticObjectiveOutput,
    DiagnosticValue,
    objective_loss,
)


def _aligned_target(
    predictions: PredictionOutput,
    supervision: MaskedPanel,
    supervision_column: str,
) -> tuple[PredictionOutput, pd.Series]:
    """Restrict predictions to finite targets observed at the same indexed rows.

    Availability uses the ``MaskedPanel.observed`` contract rather than merely accepting
    carried-forward target values. Every optional prediction component is filtered by the
    same positional mask, preserving exact index alignment.
    """
    target = supervision.values[supervision_column].reindex(predictions.index)
    observed = (
        supervision.observed[supervision_column]
        .reindex(predictions.index)
        .astype("boolean")
        .fillna(False)
        .astype(bool)
    )
    finite = (
        pd.Series(np.isfinite(target.to_numpy(dtype=float)), index=target.index, dtype=bool)
        if is_numeric_dtype(target.dtype)
        else target.notna()
    )
    valid = observed & finite
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
) -> tuple[Tensor, Tensor, Tensor]:
    """Pack aligned scalar outputs into dense date-by-entity objective tensors.

    Cross-sectional data become ``[D, N]`` via unstacking; date-only data become ``[D, 1]``.
    Missing entity/date combinations are zero-filled and identified by a same-shaped
    boolean validity mask.
    """
    entity = entity_level(predictions)
    if entity is None:
        score_frame = predictions.to_frame("prediction")
        target_frame = target.to_frame("target")
    else:
        score_frame = predictions.unstack(level=entity).sort_index()
        target_frame = target.unstack(level=entity).reindex_like(score_frame)
    valid = score_frame.notna() & target_frame.notna()
    return (
        torch.from_numpy(score_frame.fillna(0.0).to_numpy(dtype=np.float32)),
        torch.from_numpy(target_frame.fillna(0.0).to_numpy(dtype=np.float32)),
        torch.from_numpy(valid.to_numpy(dtype=bool)),
    )


def _objective_rows(
    predictions: PredictionOutput,
    target: pd.Series,
) -> tuple[Tensor, Tensor, Tensor]:
    """Convert independent prediction rows to tensors without imposing panel structure.

    Scalar outputs become ``[M]`` and multiclass scores become ``[M, C]``. Multiclass
    targets are encoded against the score-column labels and returned as integer class
    indices; all other numerical targets retain floating-point semantics.
    """
    score_values = predictions.values.to_numpy(dtype=np.float32)
    scores = torch.from_numpy(score_values)
    if predictions.kind == "classification" and isinstance(predictions.values, pd.DataFrame):
        classes = {label: index for index, label in enumerate(predictions.values.columns)}
        encoded = target.map(classes)
        if encoded.isna().any():
            unknown = target[encoded.isna()].iloc[0]
            raise ValueError(f"classification target contains unknown class {unknown!r}")
        targets = torch.from_numpy(encoded.to_numpy(dtype=np.int64))
    else:
        if not is_numeric_dtype(target.dtype):
            raise TypeError("row objective evaluation requires numerical targets")
        targets = torch.from_numpy(target.to_numpy(dtype=np.float32))
    return scores, targets, torch.ones(len(target), dtype=torch.bool)


def _evaluate_objective(
    predictions: PredictionOutput,
    target: pd.Series,
    objective: nn.Module,
    *,
    layout: ObjectiveLayout,
    adapter: ObjectiveTensorAdapter | None,
) -> object:
    """Re-evaluate an objective through a built-in or estimator-provided tensor adapter."""
    if adapter is not None:
        score_tensor, target_tensor, valid_tensor = adapter(predictions, target)
    elif layout == "rows":
        score_tensor, target_tensor, valid_tensor = _objective_rows(predictions, target)
    else:
        if not isinstance(predictions.values, pd.Series) or not is_numeric_dtype(target.dtype):
            raise TypeError("panel objective evaluation requires scalar numerical outputs")
        score_tensor, target_tensor, valid_tensor = _objective_matrices(
            predictions.values.astype(float), target.astype(float)
        )
    with torch.inference_mode():
        output = objective(score_tensor, target_tensor, valid_tensor)
    scalar_objective_metrics(output)
    return output


def _scalar_value(name: str, value: DiagnosticValue) -> float:
    """Convert one objective diagnostic to a finite-reporting scalar."""
    if isinstance(value, Tensor):
        if value.numel() != 1:
            raise TypeError(f"objective metric '{name}' must be a scalar tensor")
        return float(value.detach().cpu().item())
    return float(value)


def scalar_objective_metrics(output: object) -> dict[str, float]:
    """Extract a scalar loss and optional structural diagnostics from any objective result."""
    metrics = {"loss": _scalar_value("loss", objective_loss(output))}
    if isinstance(output, DiagnosticObjectiveOutput):
        for name, value in output.diagnostic_metrics().items():
            if name == "loss":
                raise ValueError("objective diagnostics must not redefine 'loss'")
            metrics[name] = _scalar_value(name, value)
    return metrics


def _direct_allocations(scores: Tensor, mask: Tensor) -> Tensor:
    """Use model-emitted allocation weights directly while zeroing unavailable items."""
    return torch.where(mask, scores, torch.zeros_like(scores))


def _portfolio(
    predictions: PredictionOutput,
    target: pd.Series,
    objective: nn.Module | None,
    config: ModelEvaluationConfig,
) -> PortfolioEvaluation | None:
    """Build portfolio analytics when the objective exposes score normalization.

    The callable normalization contract keeps analytics independent of a concrete loss
    class while ensuring weights are constructed exactly as they were during training.
    """
    normalize: object
    if predictions.kind == "allocation":
        normalize = _direct_allocations
    elif objective is not None:
        normalize = getattr(objective, "normalize_weights", None)
    else:
        normalize = None
    if not callable(normalize):
        return None
    if not isinstance(predictions.values, pd.Series):
        raise TypeError("portfolio construction requires one scalar score per date/instrument")
    typed_normalize = cast(Callable[[Tensor, Tensor], Tensor], normalize)
    return_type = cast(ReturnType, getattr(objective, "return_type", config.return_type))
    return build_portfolio_evaluation(
        predictions.values.astype(float),
        target.astype(float),
        normalize=typed_normalize,
        return_type=return_type,
        annualization_periods=config.annualization_periods,
        risk_free_rate=config.risk_free_rate,
        minimum_acceptable_return=config.minimum_acceptable_return,
        var_levels=config.var_levels,
        rolling_window=config.rolling_window,
        signal_buckets=config.signal_buckets,
        active_weight_threshold=config.active_weight_threshold,
        include_initial_trade=config.include_initial_trade,
        execution_fee=float(getattr(objective, "execution_fee", 0.0)),
        bid_ask_spread=float(getattr(objective, "bid_ask_spread", 0.0)),
        slippage=float(getattr(objective, "slippage", 0.0)),
        borrow_cost=float(getattr(objective, "borrow_cost", 0.0)),
    )


def evaluate_predictions(
    predictions: PredictionOutput,
    supervision: MaskedPanel,
    supervision_column: str,
    objective: nn.Module | None,
    config: ModelEvaluationConfig,
    *,
    objective_layout: ObjectiveLayout = "panel",
    objective_adapter: ObjectiveTensorAdapter | None = None,
) -> TestEvaluation:
    """Build task-aware signal, objective, and optional portfolio analytics.

    Only target rows that are both explicitly observed and valid enter evaluation;
    lookback-only predictions and unavailable labels are excluded. Every report section
    therefore uses an identical item universe, with prediction coverage retained as a
    separate diagnostic.
    """
    aligned, target = _aligned_target(predictions, supervision, supervision_column)
    signal = evaluate_signal(
        aligned,
        target,
        bucket_count=config.signal_buckets,
        probability_bins=config.probability_bins,
        classification_threshold=config.classification_threshold,
        target_threshold=config.target_threshold,
        annualization_periods=config.annualization_periods,
        rolling_window=config.rolling_window,
        signal_decay_periods=config.signal_decay_periods,
    )
    signal.metrics["prediction_coverage"] = len(target) / len(predictions.values)
    objective_output = (
        _evaluate_objective(
            aligned,
            target,
            objective,
            layout=objective_layout,
            adapter=objective_adapter,
        )
        if objective is not None
        else None
    )
    portfolio = _portfolio(aligned, target, objective, config)
    dates = pd.Index(aligned.index.get_level_values(0)).nunique()
    return TestEvaluation(
        predictions=aligned,
        signal=signal,
        objective_metrics=(
            scalar_objective_metrics(objective_output) if objective_output is not None else {}
        ),
        portfolio=portfolio,
        valid_observations=len(target),
        dates=int(dates),
    )
