"""Registry binding for the equal-weight long-only sanity baseline."""

from __future__ import annotations

from functools import partial

from omegaconf import DictConfig
from torch import nn

from llca.mappers.loss.mapper import prediction_kind
from llca.mappers.model.mapper import (
    EstimatorFactory,
    model_registry,
    register_model_capabilities,
)
from llca.mappers.model.tabular import tabular_data_requirements, validate_tabular_io
from llca.models.estimators.baseline import EqualWeightEstimator
from llca.pipeline.contracts import ModelCapabilities, ObjectiveKind, TrainingEngine


@model_registry.register_validator("equal-weight")
def validate(cfg: DictConfig) -> list[str]:
    """Validate the panel roles the equal-weight baseline reads."""
    return validate_tabular_io(cfg)


@model_registry.register("equal-weight")
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
    return partial(EqualWeightEstimator, config=cfg, prediction_kind=kind)


register_model_capabilities(
    "equal-weight",
    ModelCapabilities(
        resolve_data=tabular_data_requirements,
        objective_kinds=frozenset({ObjectiveKind.PORTFOLIO}),
        training_engines=frozenset({TrainingEngine.SKLEARN}),
    ),
)
