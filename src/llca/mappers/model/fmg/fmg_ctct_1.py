"""Registry binding for the target-query FMG-CTCT-1 model."""

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
from llca.models.estimators.fmg import FmgCtct1Estimator
from llca.pipeline.contracts import EntityScope, ModelCapabilities, ObjectiveKind, TrainingEngine


@model_registry.register_validator("fmg-ctct-1")
def validate(cfg: DictConfig) -> list[str]:
    return validate_single_asset_allocation(cfg, "fmg-ctct-1")


@model_registry.register("fmg-ctct-1")
def build(
    cfg: DictConfig,
    *,
    loss: nn.Module | None = None,
    loss_config: DictConfig | None = None,
    **_: object,
) -> EstimatorFactory:
    validate_single_asset_objective("fmg-ctct-1", loss)
    if loss_config is None:
        raise ValueError(f"{cfg.name} requires the configured loss metadata")
    return partial(
        FmgCtct1Estimator,
        config=cfg,
        loss=loss,
        prediction_kind=prediction_kind(str(loss_config.name)),
    )


register_model_capabilities(
    "fmg-ctct-1",
    ModelCapabilities(
        resolve_data=lambda cfg: fmg_data_requirements(
            cfg,
            input_scope=EntityScope.UNIVERSE,
            supervision_scope=EntityScope.TARGET,
        ),
        objective_kinds=frozenset({ObjectiveKind.PORTFOLIO}),
        training_engines=frozenset({TrainingEngine.TORCH}),
    ),
)
