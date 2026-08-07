from __future__ import annotations

from collections.abc import Callable
from typing import Any

from omegaconf import DictConfig
from torch import nn

from llca.mappers.modules.registry import Registry
from llca.models.estimators.estimator import Estimator
from llca.pipeline.contracts import DataRequirements, ModelCapabilities

EstimatorFactory = Callable[[], Estimator[Any]]
model_registry: Registry[EstimatorFactory] = Registry("model")
_capabilities: dict[str, ModelCapabilities] = {}


def register_model_capabilities(name: str, capabilities: ModelCapabilities) -> None:
    """Attach data, objective, and training-engine contracts to one model plugin."""
    if name in _capabilities:
        raise ValueError(f"model capabilities for '{name}' are already registered")
    _capabilities[name] = capabilities


def model_capabilities(name: str) -> ModelCapabilities:
    """Return canonical model capabilities."""
    try:
        return _capabilities[name]
    except KeyError as exc:
        raise KeyError(f"model '{name}' has no registered capabilities") from exc


def model_data_requirements(cfg: DictConfig) -> DataRequirements:
    """Resolve the logical datasets and entity scopes needed by a model configuration."""
    return model_capabilities(str(cfg.name)).resolve_data(cfg)


def build_model(
    cfg: DictConfig,
    *,
    loss: nn.Module | None = None,
    loss_config: DictConfig | None = None,
    hyperparameter_selection: DictConfig | None = None,
) -> EstimatorFactory:
    """Build a factory while exposing objective runtime, config, and tuning policy to its plugin."""
    return model_registry.build(
        cfg.name,
        cfg,
        loss=loss,
        loss_config=loss_config,
        hyperparameter_selection=hyperparameter_selection,
    )
