"""Registry binding for the single-asset elastic-net logistic classifier baseline."""

from __future__ import annotations

from functools import partial

from omegaconf import DictConfig
from torch import nn

from llca.mappers.config_validation import ConfigField, check_fields, is_number
from llca.mappers.loss.mapper import prediction_kind
from llca.mappers.model.mapper import (
    EstimatorFactory,
    model_registry,
    register_model_capabilities,
)
from llca.mappers.model.tabular import (
    single_asset_tabular_data_requirements,
    validate_single_asset_tabular_io,
)
from llca.mappers.model.tuning import build_hyperparameter_selection
from llca.models.estimators.logistic_net import LogisticNetEstimator
from llca.pipeline.contracts import ModelCapabilities, ObjectiveKind, TrainingEngine

# C, l1_ratio, and class weighting shape the fitted model and may be searched; the solver and
# penalty are fixed in code (saga + elasticnet), and max_iter/tol are convergence controls set
# robustly rather than tuned.
_TUNABLE_PARAMETERS = frozenset({"C", "l1_ratio", "class_weight"})
_CLASS_WEIGHTS = ("balanced",)

_LINEAR_FIELDS = [
    ConfigField("l1_ratio", "number"),
    ConfigField("C", "number", positive=True, required=False),
    ConfigField("max_iter", "int", positive=True, required=False),
    ConfigField("tol", "number", positive=True, required=False),
    ConfigField("fit_intercept", "bool", required=False),
]


@model_registry.register_validator("elastic-net")
def validate(cfg: DictConfig) -> list[str]:
    """Validate the logistic classifier inputs and its scikit-learn hyperparameters."""
    errors = validate_single_asset_tabular_io(cfg)
    errors.extend(check_fields(cfg.model, "model", _LINEAR_FIELDS))
    l1_ratio = cfg.model.get("l1_ratio")
    if is_number(l1_ratio) and not 0.0 <= float(l1_ratio) <= 1.0:
        errors.append("model.l1_ratio must be between 0.0 (ridge) and 1.0 (lasso)")
    class_weight = cfg.model.get("class_weight")
    if not (class_weight is None or class_weight in _CLASS_WEIGHTS):
        errors.append(f"model.class_weight must be null or one of {list(_CLASS_WEIGHTS)}")
    return errors


@model_registry.register("elastic-net")
def build(
    cfg: DictConfig,
    *,
    loss: nn.Module | None = None,
    loss_config: DictConfig | None = None,
    hyperparameter_selection: DictConfig | None = None,
    **_: object,
) -> EstimatorFactory:
    """Bind the configuration, deployment objective, and selection policy to the estimator."""
    kind = prediction_kind(str(loss_config.name)) if loss_config is not None else "portfolio"
    selection = build_hyperparameter_selection(hyperparameter_selection, cfg)
    return partial(
        LogisticNetEstimator,
        config=cfg,
        prediction_kind=kind,
        cv_objective=loss,
        selection=selection,
    )


register_model_capabilities(
    "elastic-net",
    ModelCapabilities(
        resolve_data=single_asset_tabular_data_requirements,
        objective_kinds=frozenset({ObjectiveKind.PORTFOLIO}),
        training_engines=frozenset({TrainingEngine.SKLEARN}),
        tunable_parameters=_TUNABLE_PARAMETERS,
    ),
)
