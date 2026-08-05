from __future__ import annotations

from omegaconf import DictConfig
from torch import nn

from llca.mappers.modules.registry import Registry
from llca.models.estimators.prediction import PredictionKind
from llca.pipeline.contracts import ObjectiveKind

loss_registry: Registry[nn.Module] = Registry("loss")
_objective_kinds: dict[str, ObjectiveKind] = {}


def register_objective_kind(name: str, kind: ObjectiveKind) -> None:
    """Attach output semantics to an objective plugin for compatibility validation."""
    if name in _objective_kinds:
        raise ValueError(f"objective kind for '{name}' is already registered")
    _objective_kinds[name] = kind


def objective_kind(name: str) -> ObjectiveKind:
    """Return the semantic output contract of a configured objective."""
    try:
        return _objective_kinds[name]
    except KeyError as exc:
        raise KeyError(f"loss '{name}' has no registered objective kind") from exc


def prediction_kind(name: str) -> PredictionKind:
    """Map one registered objective to the model-output contract used by analytics."""
    kind = objective_kind(name)
    mapping: dict[ObjectiveKind, PredictionKind] = {
        ObjectiveKind.PORTFOLIO: "portfolio",
        ObjectiveKind.REGRESSION: "regression",
        ObjectiveKind.BINARY_CLASSIFICATION: "binary",
        ObjectiveKind.MULTICLASS_CLASSIFICATION: "multiclass",
    }
    try:
        return mapping[kind]
    except KeyError as exc:
        raise ValueError(
            f"objective kind '{kind}' requires an explicit prediction-kind adapter"
        ) from exc


def build_loss(cfg: DictConfig) -> nn.Module:
    """Construct a registered differentiable objective from validated Hydra settings."""
    return loss_registry.build(cfg.name, cfg)
