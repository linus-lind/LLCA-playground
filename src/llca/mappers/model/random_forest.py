"""Registry binding for the single-asset random-forest classifier baseline model."""

from __future__ import annotations

from functools import partial
from typing import cast

from omegaconf import DictConfig, ListConfig
from torch import nn

from llca.mappers.config_validation import ConfigField, check_fields, is_int, is_number
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
from llca.models.estimators.random_forest import RandomForestClassifierEstimator
from llca.pipeline.contracts import ModelCapabilities, ObjectiveKind, TrainingEngine

# Statistical hyperparameters that materially shape the fitted trees' bias/variance and may
# therefore appear in the search grid. The ensemble size (n_estimators), structural switches
# (bootstrap, oob_score, monotonic_cst), and runtime controls (verbose, warm_start, and the
# seed / n_jobs from the training policy) are configurable but intentionally excluded.
_TUNABLE_PARAMETERS = frozenset(
    {
        "criterion",
        "max_depth",
        "min_samples_split",
        "min_samples_leaf",
        "min_weight_fraction_leaf",
        "max_features",
        "max_leaf_nodes",
        "min_impurity_decrease",
        "class_weight",
        "ccp_alpha",
        "max_samples",
    }
)

_CRITERIA = ("gini", "entropy", "log_loss")
_MAX_FEATURES_CHOICES = ("sqrt", "log2")
_CLASS_WEIGHTS = ("balanced", "balanced_subsample")

_RF_FIELDS = [
    ConfigField("n_estimators", "int", positive=True),
    ConfigField("max_depth", "int", positive=True, required=False),
    ConfigField("min_samples_split", "number", positive=True, required=False),
    ConfigField("min_samples_leaf", "number", positive=True, required=False),
    ConfigField("min_weight_fraction_leaf", "number", minimum=0, required=False),
    ConfigField("max_leaf_nodes", "int", required=False),
    ConfigField("min_impurity_decrease", "number", minimum=0, required=False),
    ConfigField("ccp_alpha", "number", minimum=0, required=False),
    ConfigField("bootstrap", "bool", required=False),
    ConfigField("oob_score", "bool", required=False),
    ConfigField("warm_start", "bool", required=False),
    ConfigField("verbose", "int", minimum=0, required=False),
]


def _valid_max_features(value: object) -> bool:
    if value is None or value in _MAX_FEATURES_CHOICES:
        return True
    return is_number(value) and cast(float, value) > 0.0


def _valid_class_weight(value: object) -> bool:
    return value is None or value in _CLASS_WEIGHTS or isinstance(value, DictConfig)


def _rf_semantic_errors(model: DictConfig) -> list[str]:
    """Check scikit-learn choice sets and cross-parameter constraints before any fitting."""
    errors: list[str] = []
    criterion = model.get("criterion")
    if criterion is not None and criterion not in _CRITERIA:
        errors.append(f"model.criterion must be one of {list(_CRITERIA)}")
    if not _valid_max_features(model.get("max_features")):
        errors.append(
            f"model.max_features must be a positive number, null, or one of "
            f"{list(_MAX_FEATURES_CHOICES)}"
        )
    if not _valid_class_weight(model.get("class_weight")):
        errors.append(f"model.class_weight must be null or one of {list(_CLASS_WEIGHTS)}")

    fraction = model.get("min_weight_fraction_leaf")
    if is_number(fraction) and not 0.0 <= float(fraction) <= 0.5:
        errors.append("model.min_weight_fraction_leaf must be in [0.0, 0.5]")
    max_leaf_nodes = model.get("max_leaf_nodes")
    if is_int(max_leaf_nodes) and int(max_leaf_nodes) < 2:
        errors.append("model.max_leaf_nodes must be >= 2 (or null for unlimited)")

    max_samples = model.get("max_samples")
    if max_samples is not None:
        if is_int(max_samples) and int(max_samples) <= 0:
            errors.append("model.max_samples integer must be positive")
        elif isinstance(max_samples, float) and not 0.0 < max_samples <= 1.0:
            errors.append("model.max_samples float must be in (0.0, 1.0]")
        elif not is_number(max_samples):
            errors.append("model.max_samples must be a number in (0, 1], a positive int, or null")

    bootstrap = bool(model.get("bootstrap", True))
    if not bootstrap and max_samples is not None:
        errors.append("model.max_samples requires model.bootstrap: true")
    if not bootstrap and bool(model.get("oob_score", False)):
        errors.append("model.oob_score requires model.bootstrap: true")

    monotonic = model.get("monotonic_cst")
    if monotonic is not None and not isinstance(monotonic, list | ListConfig):
        errors.append("model.monotonic_cst must be a list of per-feature {-1, 0, 1} or null")
    return errors


@model_registry.register_validator("rf")
def validate(cfg: DictConfig) -> list[str]:
    """Validate the random-forest classifier inputs and its scikit-learn hyperparameters."""
    errors = validate_single_asset_tabular_io(cfg)
    errors.extend(check_fields(cfg.model, "model", _RF_FIELDS))
    errors.extend(_rf_semantic_errors(cfg.model))
    return errors


@model_registry.register("rf")
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
        RandomForestClassifierEstimator,
        config=cfg,
        prediction_kind=kind,
        cv_objective=loss,
        selection=selection,
    )


register_model_capabilities(
    "rf",
    ModelCapabilities(
        resolve_data=single_asset_tabular_data_requirements,
        objective_kinds=frozenset({ObjectiveKind.PORTFOLIO}),
        training_engines=frozenset({TrainingEngine.SKLEARN}),
        tunable_parameters=_TUNABLE_PARAMETERS,
    ),
)
