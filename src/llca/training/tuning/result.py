"""Typed, serializable provenance of one hyperparameter-selection decision."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from llca.training.tuning.search_space import ParameterValue


@dataclass(frozen=True, slots=True)
class HyperparameterSelectionResult:
    """The complete outcome of an inner-CV selection, sufficient for logging and reproduction.

    Losses are the per-fold objective values (lower is better) that the baseline and the
    winning candidate obtained on the identical inner folds. ``selected_is_baseline`` is true
    whenever the conservative rule retained the baseline, either because no candidate improved
    the mean loss or because the improvement did not exceed the standard-error margin.
    """

    enabled: bool
    search_method: str
    fold_count: int
    evaluated_candidates: int
    baseline_parameters: Mapping[str, ParameterValue]
    selected_parameters: Mapping[str, ParameterValue]
    selected_is_baseline: bool
    baseline_mean_loss: float
    best_candidate_mean_loss: float
    baseline_fold_losses: tuple[float, ...]
    best_candidate_fold_losses: tuple[float, ...]
    standard_error_margin: float
    improvement_mean: float
    improvement_standard_error: float

    def summary_metrics(self) -> dict[str, float]:
        """Return the scalar metrics worth surfacing on the training run."""
        return {
            "hyperparameter_selection/fold_count": float(self.fold_count),
            "hyperparameter_selection/evaluated_candidates": float(self.evaluated_candidates),
            "hyperparameter_selection/baseline_mean_loss": self.baseline_mean_loss,
            "hyperparameter_selection/best_candidate_mean_loss": self.best_candidate_mean_loss,
            "hyperparameter_selection/improvement_mean": self.improvement_mean,
            "hyperparameter_selection/improvement_standard_error": self.improvement_standard_error,
            "hyperparameter_selection/selected_is_baseline": float(self.selected_is_baseline),
        }

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable record for a detailed run artifact."""
        return {
            "enabled": self.enabled,
            "search_method": self.search_method,
            "fold_count": self.fold_count,
            "evaluated_candidates": self.evaluated_candidates,
            "baseline_parameters": dict(self.baseline_parameters),
            "selected_parameters": dict(self.selected_parameters),
            "selected_is_baseline": self.selected_is_baseline,
            "baseline_mean_loss": self.baseline_mean_loss,
            "best_candidate_mean_loss": self.best_candidate_mean_loss,
            "baseline_fold_losses": list(self.baseline_fold_losses),
            "best_candidate_fold_losses": list(self.best_candidate_fold_losses),
            "standard_error_margin": self.standard_error_margin,
            "improvement_mean": self.improvement_mean,
            "improvement_standard_error": self.improvement_standard_error,
        }
