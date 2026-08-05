"""Registry binding for the full cross-sectional FMG-CTCT-2 model."""

from __future__ import annotations

from functools import partial

from omegaconf import DictConfig
from torch import nn

from llca.mappers.loss.mapper import prediction_kind
from llca.mappers.model.fmg.validation import fmg_data_requirements, validate_transformer_fmg
from llca.mappers.model.mapper import (
    EstimatorFactory,
    model_registry,
    register_model_capabilities,
)
from llca.models.estimators.fmg import FmgCtct2Estimator
from llca.pipeline.contracts import EntityScope, ModelCapabilities, ObjectiveKind, TrainingEngine


@model_registry.register_validator("fmg-ctct-2")
def validate(cfg: DictConfig) -> list[str]:
    """Validate the full cross-sectional FMG-CTCT-2 model."""
    return validate_transformer_fmg(cfg)


@model_registry.register("fmg-ctct-2")
def build(
    cfg: DictConfig,
    *,
    loss: nn.Module | None = None,
    loss_config: DictConfig | None = None,
    **_: object,
) -> EstimatorFactory:
    """Bind the model configuration and objective to a fresh estimator."""
    if loss is None:
        raise ValueError(f"{cfg.name} requires a loss function")
    if loss_config is None:
        raise ValueError(f"{cfg.name} requires the configured loss metadata")
    return partial(
        FmgCtct2Estimator,
        config=cfg,
        loss=loss,
        prediction_kind=prediction_kind(str(loss_config.name)),
    )


register_model_capabilities(
    "fmg-ctct-2",
    ModelCapabilities(
        resolve_data=lambda cfg: fmg_data_requirements(
            cfg,
            input_scope=EntityScope.UNIVERSE,
            supervision_scope=EntityScope.UNIVERSE,
        ),
        objective_kinds=frozenset(
            {
                ObjectiveKind.PORTFOLIO,
                ObjectiveKind.REGRESSION,
                ObjectiveKind.BINARY_CLASSIFICATION,
            }
        ),
        training_engines=frozenset({TrainingEngine.TORCH}),
    ),
)
