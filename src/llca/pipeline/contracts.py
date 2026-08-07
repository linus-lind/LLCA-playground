"""Stable contracts shared by data, model, objective, and execution plugins."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum

from omegaconf import DictConfig

type EntityId = int | str


class EntityScope(StrEnum):
    """Describe how much of an entity-indexed dataset one model consumes."""

    UNIVERSE = "universe"
    TARGET = "target"


class ObjectiveKind(StrEnum):
    """Semantic contract between native model outputs and an optimization objective."""

    PORTFOLIO = "portfolio"
    REGRESSION = "regression"
    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"
    CUSTOM = "custom"


class TrainingEngine(StrEnum):
    """Execution family used to fit an estimator implementation."""

    TORCH = "torch"
    SKLEARN = "sklearn"
    CUSTOM = "custom"


@dataclass(frozen=True, slots=True)
class DatasetRequirement:
    """Bind one logical dataset to its model-specific entity scope."""

    name: str
    entity_scope: EntityScope = EntityScope.UNIVERSE


@dataclass(frozen=True, slots=True)
class DataRequirements:
    """Declare all logical datasets required to construct one estimator input."""

    primary_dataset: str
    datasets: tuple[DatasetRequirement, ...]
    target_entity: EntityId | None = None

    def __post_init__(self) -> None:
        names = [requirement.name for requirement in self.datasets]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"duplicate model dataset requirements: {duplicates}")
        if self.primary_dataset not in names:
            raise ValueError(
                f"primary dataset '{self.primary_dataset}' is not part of model requirements"
            )
        if (
            any(requirement.entity_scope is EntityScope.TARGET for requirement in self.datasets)
            and self.target_entity is None
        ):
            raise ValueError("target-scoped datasets require a configured target entity")


@dataclass(frozen=True, slots=True)
class DatasetQuery:
    """Selection pushed into one logical dataset before preprocessing and features."""

    entity_ids: tuple[EntityId, ...] | None = None


@dataclass(frozen=True, slots=True)
class DataPlan:
    """Resolved model-aware load plan for one pipeline invocation."""

    primary_dataset: str
    datasets: dict[str, DatasetQuery]
    csv_chunk_size: int


type DataRequirementsResolver = Callable[[DictConfig], DataRequirements]


@dataclass(frozen=True, slots=True)
class ModelCapabilities:
    """Cross-cutting requirements owned by a model plugin rather than core orchestration."""

    resolve_data: DataRequirementsResolver
    objective_kinds: frozenset[ObjectiveKind]
    training_engines: frozenset[TrainingEngine]
    data_view: str = "aligned_panel"
    tunable_parameters: frozenset[str] = frozenset()
    """Hyperparameter names this model may select through inner cross-validation."""
