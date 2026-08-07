"""Runtime settings for inner-cross-validation hyperparameter selection.

These immutable structures are produced from Hydra configuration by ``llca.mappers`` and
consumed by the model-agnostic selection algorithm. They never reference a concrete estimator
or objective: the deployment objective and the candidate estimators are supplied as callables
so the same machinery serves any statistical model family.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from llca.training.tuning.search_space import SearchSpace


@dataclass(frozen=True, slots=True)
class InnerCvSettings:
    """Temporal walk-forward geometry for the inner cross-validation folds.

    Sizes are counts of trading dates on the outer training calendar. ``purge`` drops dates
    between each fold's train and validation windows to prevent the forward-looking label
    horizon from leaking across the boundary; ``lookback`` prepends warmup dates to each slice
    for models that rebuild history-dependent inputs (zero for precomputed-feature estimators).
    ``min_folds`` is the minimum number of complete folds required for a valid selection.
    """

    train_size: int
    val_size: int
    step_size: int
    purge: int
    lookback: int
    min_folds: int


@dataclass(frozen=True, slots=True)
class SearchSettings:
    """Candidate-generation policy: the search method, trial budget, and reproducibility seed.

    ``n_trials`` and ``seed`` are only consulted by stochastic methods such as ``random``;
    deterministic ``grid`` enumeration ignores them.
    """

    method: str
    n_trials: int
    seed: int


@dataclass(frozen=True, slots=True)
class HyperparameterSelection:
    """Fully resolved configuration for one model's inner-CV hyperparameter selection.

    The deployment objective is intentionally absent: selection scores candidates through the
    same objective the model is finally evaluated with, which the estimator supplies at fit
    time. ``baseline`` is the complete default hyperparameter mapping used both when selection
    is disabled and as the reference every searched candidate must beat.
    """

    enabled: bool
    inner_cv: InnerCvSettings
    search: SearchSettings
    search_space: SearchSpace
    baseline: Mapping[str, Any]
    standard_error_margin: float
