"""Registry binding for the target-only transformer FMG-CTT model."""

from __future__ import annotations

from functools import partial

from omegaconf import DictConfig
from torch import nn

from llca.mappers.loss.mapper import prediction_kind
from llca.mappers.model.fmg.validation import (
    fmg_data_requirements,
    validate_single_asset_allocation,
    validate_single_asset_objective,
)
from llca.mappers.model.mapper import (
    EstimatorFactory,
    model_registry,
    register_model_capabilities,
)
from llca.models.estimators.fmg import FmgCttEstimator
from llca.pipeline.contracts import EntityScope, ModelCapabilities, ObjectiveKind, TrainingEngine


@model_registry.register_validator("fmg-ctt")
def validate(cfg: DictConfig) -> list[str]:
    return validate_single_asset_allocation(cfg, "fmg-ctt")


@model_registry.register("fmg-ctt")
def build(
    cfg: DictConfig,
    *,
    loss: nn.Module | None = None,
    loss_config: DictConfig | None = None,
    **_: object,
) -> EstimatorFactory:
    validate_single_asset_objective("fmg-ctt", loss)
    if loss_config is None:
        raise ValueError(f"{cfg.name} requires the configured loss metadata")
    return partial(
        FmgCttEstimator,
        config=cfg,
        loss=loss,
        prediction_kind=prediction_kind(str(loss_config.name)),
    )


register_model_capabilities(
    "fmg-ctt",
    ModelCapabilities(
        resolve_data=lambda cfg: fmg_data_requirements(
            cfg,
            input_scope=EntityScope.TARGET,
            supervision_scope=EntityScope.TARGET,
        ),
        objective_kinds=frozenset({ObjectiveKind.PORTFOLIO}),
        training_engines=frozenset({TrainingEngine.TORCH}),
    ),
)
