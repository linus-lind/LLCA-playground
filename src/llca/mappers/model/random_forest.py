"""Registry binding for the cross-sectional random-forest baseline model."""

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
from llca.models.estimators.random_forest import RandomForestEstimator
from llca.pipeline.contracts import ModelCapabilities, ObjectiveKind, TrainingEngine

_RF_FIELDS = [
    ConfigField("n_estimators", "int", positive=True),
    ConfigField("min_samples_leaf", "int", positive=True, required=False),
    ConfigField("max_depth", "int", positive=True, required=False),
    ConfigField("bootstrap", "bool", required=False),
]
_MAX_FEATURES_CHOICES = ("sqrt", "log2")


@model_registry.register_validator("rf")
def validate(cfg: DictConfig) -> list[str]:
    """Validate the random-forest inputs and its scikit-learn hyperparameters."""
    errors = validate_tabular_io(cfg)
    errors.extend(check_fields(cfg.model, "model", _RF_FIELDS))
    max_features = cfg.model.get("max_features")
    if max_features is not None and not (
        (isinstance(max_features, int | float) and not isinstance(max_features, bool))
        or max_features in _MAX_FEATURES_CHOICES
    ):
        errors.append(
            f"model.max_features must be a positive number or one of {list(_MAX_FEATURES_CHOICES)}"
        )
    return errors


@model_registry.register("rf")
def build(
    cfg: DictConfig,
    *,
    loss: nn.Module | None = None,
    loss_config: DictConfig | None = None,
    **_: object,
) -> EstimatorFactory:
    """Bind the model configuration and the objective's output contract to the estimator."""
    del loss
    kind = prediction_kind(str(loss_config.name)) if loss_config is not None else "portfolio"
    return partial(RandomForestEstimator, config=cfg, prediction_kind=kind)


register_model_capabilities(
    "rf",
    ModelCapabilities(
        resolve_data=tabular_data_requirements,
        objective_kinds=frozenset({ObjectiveKind.PORTFOLIO, ObjectiveKind.REGRESSION}),
        training_engines=frozenset({TrainingEngine.SKLEARN}),
    ),
)
