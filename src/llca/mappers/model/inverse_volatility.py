"""Registry binding for the inverse-volatility long-only benchmark."""

from __future__ import annotations

from functools import partial

from omegaconf import DictConfig
from torch import nn

from llca.mappers.config_validation import ConfigField, check_fields
from llca.mappers.loss.mapper import prediction_kind
from llca.mappers.model.mapper import (
    EstimatorFactory,
    model_registry,
    register_model_capabilities,
)
from llca.mappers.model.tabular import tabular_data_requirements, validate_tabular_io
from llca.models.estimators.baseline import InverseVolatilityEstimator
from llca.pipeline.contracts import ModelCapabilities, ObjectiveKind, TrainingEngine

_VOLATILITY_FIELDS = [
    ConfigField("dataset", "str"),
    ConfigField("column", "str"),
    ConfigField("floor", "number", positive=True, required=False),
]


@model_registry.register_validator("inverse-volatility")
def validate(cfg: DictConfig) -> list[str]:
    """Validate the panel roles and the volatility feature the benchmark weights by."""
    errors = validate_tabular_io(cfg)
    volatility = cfg.model.get("volatility")
    if not isinstance(volatility, DictConfig):
        errors.append("model.volatility must be a mapping with 'dataset' and 'column'")
        return errors
    errors.extend(check_fields(volatility, "model.volatility", _VOLATILITY_FIELDS))
    datasets = cfg.data.get("datasets") if cfg.get("data") is not None else None
    available = set(datasets.keys()) if isinstance(datasets, DictConfig) else set()
    name = str(volatility.get("dataset"))
    if name and name != "None" and name not in available:
        errors.append(
            f"model.volatility references dataset '{name}' which is not configured in data.datasets"
        )
    return errors


@model_registry.register("inverse-volatility")
def build(
    cfg: DictConfig,
    *,
    loss: nn.Module | None = None,
    loss_config: DictConfig | None = None,
    **_: object,
) -> EstimatorFactory:
    """Bind the model configuration and the objective's output contract to the baseline."""
    del loss
    kind = prediction_kind(str(loss_config.name)) if loss_config is not None else "portfolio"
    return partial(InverseVolatilityEstimator, config=cfg, prediction_kind=kind)


register_model_capabilities(
    "inverse-volatility",
    ModelCapabilities(
        resolve_data=tabular_data_requirements,
        objective_kinds=frozenset({ObjectiveKind.PORTFOLIO}),
        training_engines=frozenset({TrainingEngine.SKLEARN}),
    ),
)
