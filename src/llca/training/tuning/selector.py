"""Model-agnostic orchestration of inner-CV hyperparameter selection.

The selector never imports a concrete estimator or objective. It is handed a factory that
builds a fresh estimator from a parameter mapping, an accessor for the realized returns an
objective consumes, and a callable that reduces one fold's aligned scores and returns to a
scalar loss. It fits a fresh estimator for every fold of every candidate, averages the per-fold
losses, and applies the conservative paired standard-error rule against the baseline.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from typing import Protocol

import pandas as pd

from llca.data.index_spec import time_level
from llca.data.modules.masked_panel import MaskedPanels
from llca.models.estimators.prediction import PredictionOutput
from llca.training.modules.training_policy import TrainingPolicy
from llca.training.tuning.inner_cv import InnerFold, build_inner_folds
from llca.training.tuning.result import HyperparameterSelectionResult
from llca.training.tuning.search import generate_candidates
from llca.training.tuning.search_space import ParameterValue
from llca.training.tuning.selection import adopt_candidate, paired_improvement
from llca.training.tuning.settings import HyperparameterSelection

Parameters = Mapping[str, ParameterValue]
FoldObjective = Callable[[pd.Series, pd.Series], float]
RealizedReturns = Callable[[MaskedPanels], pd.Series]


class FoldModel(Protocol):
    """The minimal estimator surface the selector drives for each inner fold."""

    def fit(self, train: MaskedPanels, *, training: TrainingPolicy) -> None: ...

    def predict(self, test: MaskedPanels) -> PredictionOutput: ...


CandidateFactory = Callable[[Parameters], FoldModel]


def _restrict_dates(series: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    """Return the rows whose date lies in the inclusive ``[start, end]`` scoring window."""
    dates = series.index.get_level_values(time_level(series))
    return series[(dates >= start) & (dates <= end)]


def _score_fold(
    parameters: Parameters,
    fold: InnerFold,
    candidate_factory: CandidateFactory,
    realized_returns: RealizedReturns,
    fold_objective: FoldObjective,
    training: TrainingPolicy,
) -> tuple[float, pd.Index]:
    """Fit a fresh estimator on the fold's train slice and score its validation predictions."""
    model = candidate_factory(parameters)
    model.fit(fold.train, training=training)
    prediction = model.predict(fold.validation)
    values = prediction.values
    if not isinstance(values, pd.Series):
        raise TypeError("inner-CV objective requires one scalar prediction per observation")
    scores = _restrict_dates(values, fold.val_start, fold.val_end)
    returns = _restrict_dates(realized_returns(fold.validation), fold.val_start, fold.val_end)
    loss = fold_objective(scores, returns)
    if not math.isfinite(loss):
        raise ValueError(f"inner-CV objective produced a non-finite loss for {dict(parameters)}")
    return loss, scores.index


def _evaluate(
    parameters: Parameters,
    folds: Sequence[InnerFold],
    candidate_factory: CandidateFactory,
    realized_returns: RealizedReturns,
    fold_objective: FoldObjective,
    training: TrainingPolicy,
) -> tuple[list[float], list[pd.Index]]:
    losses: list[float] = []
    indices: list[pd.Index] = []
    for fold in folds:
        loss, index = _score_fold(
            parameters, fold, candidate_factory, realized_returns, fold_objective, training
        )
        losses.append(loss)
        indices.append(index)
    return losses, indices


def _assert_same_observations(reference: Sequence[pd.Index], candidate: Sequence[pd.Index]) -> None:
    for fold_index, (expected, actual) in enumerate(zip(reference, candidate, strict=True)):
        if not expected.equals(actual):
            raise ValueError(
                f"a candidate scored different validation observations than the baseline on "
                f"inner fold {fold_index}; candidate and baseline coverage must be identical"
            )


def select_hyperparameters(
    *,
    train: MaskedPanels,
    primary: str,
    selection: HyperparameterSelection,
    candidate_factory: CandidateFactory,
    realized_returns: RealizedReturns,
    fold_objective: FoldObjective,
    training: TrainingPolicy,
) -> HyperparameterSelectionResult:
    """Select hyperparameters by inner walk-forward CV, retaining the baseline unless beaten.

    The baseline and every generated candidate are scored on the same materialized folds; the
    candidate with the smallest mean fold loss is adopted only if it beats the baseline under
    the configured paired standard-error rule, otherwise the baseline is kept. Raises if fewer
    than ``min_folds`` folds fit inside the training window.
    """
    folds = build_inner_folds(train, primary, selection.inner_cv)
    if len(folds) < selection.inner_cv.min_folds:
        raise ValueError(
            f"inner cross-validation produced {len(folds)} fold(s) but at least "
            f"{selection.inner_cv.min_folds} are required; reduce the inner train/validation/"
            "step sizes or provide a longer training window"
        )

    baseline = dict(selection.baseline)
    baseline_losses, baseline_indices = _evaluate(
        baseline, folds, candidate_factory, realized_returns, fold_objective, training
    )
    candidates = generate_candidates(selection.search, selection.search_space, baseline)

    best_parameters: dict[str, ParameterValue] | None = None
    best_losses: list[float] | None = None
    best_mean = math.inf
    for parameters in candidates:
        losses, indices = _evaluate(
            parameters, folds, candidate_factory, realized_returns, fold_objective, training
        )
        _assert_same_observations(baseline_indices, indices)
        mean_loss = math.fsum(losses) / len(losses)
        if mean_loss < best_mean:
            best_mean = mean_loss
            best_parameters = parameters
            best_losses = losses
    assert best_parameters is not None and best_losses is not None

    improvement_mean, improvement_se = paired_improvement(baseline_losses, best_losses)
    adopt = adopt_candidate(
        baseline_losses, best_losses, standard_error_margin=selection.standard_error_margin
    )
    return HyperparameterSelectionResult(
        enabled=True,
        search_method=selection.search.method,
        fold_count=len(folds),
        evaluated_candidates=len(candidates),
        baseline_parameters=baseline,
        selected_parameters=best_parameters if adopt else baseline,
        selected_is_baseline=not adopt,
        baseline_mean_loss=math.fsum(baseline_losses) / len(baseline_losses),
        best_candidate_mean_loss=best_mean,
        baseline_fold_losses=tuple(baseline_losses),
        best_candidate_fold_losses=tuple(best_losses),
        standard_error_margin=selection.standard_error_margin,
        improvement_mean=improvement_mean,
        improvement_standard_error=improvement_se,
    )
