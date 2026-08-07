"""Shared data-requirements and input validation for cross-sectional tabular models."""

from __future__ import annotations

from omegaconf import DictConfig

from llca.mappers.config_validation import ConfigField, as_list, check_fields
from llca.mappers.model.objective_binding import loss_is_portfolio, validate_risk_free_binding
from llca.pipeline.contracts import DataRequirements, DatasetRequirement, EntityScope


def _context_names(inputs: DictConfig) -> list[str]:
    context = inputs.get("context")
    return [] if context is None else [str(name) for name in as_list(context)]


def tabular_data_requirements(model: DictConfig) -> DataRequirements:
    """Resolve the feature, optional context, and supervision datasets a tabular model reads."""
    inputs = model.inputs
    feature_name = str(inputs.features)
    supervision_name = str(model.supervision.dataset)
    scopes: dict[str, EntityScope] = {feature_name: EntityScope.UNIVERSE}
    for name in _context_names(inputs):
        scopes.setdefault(name, EntityScope.UNIVERSE)
    scopes[supervision_name] = EntityScope.UNIVERSE
    return DataRequirements(
        primary_dataset=feature_name,
        datasets=tuple(
            DatasetRequirement(name=name, entity_scope=scope) for name, scope in scopes.items()
        ),
    )


_TARGET_FIELDS = [ConfigField("entity_id", "int", positive=True)]
_CLASSIFICATION_FIELDS = [ConfigField("dataset", "str"), ConfigField("column", "str")]


def single_asset_tabular_data_requirements(model: DictConfig) -> DataRequirements:
    """Resolve datasets for a single-target-entity tabular model, scoping panels to the target."""
    inputs = model.inputs
    feature_name = str(inputs.features)
    supervision_name = str(model.supervision.dataset)
    classification_name = str(model.classification.dataset)
    scopes: dict[str, EntityScope] = {feature_name: EntityScope.TARGET}
    for name in _context_names(inputs):
        scopes.setdefault(name, EntityScope.UNIVERSE)
    scopes[supervision_name] = EntityScope.TARGET
    scopes.setdefault(classification_name, EntityScope.TARGET)
    risk_free = model.get("risk_free")
    if isinstance(risk_free, DictConfig):
        # A date-only series shared across the universe; entity scope is immaterial for it.
        scopes.setdefault(str(risk_free.dataset), EntityScope.UNIVERSE)
    target = model.get("target")
    target_entity = target.get("entity_id") if isinstance(target, DictConfig) else None
    return DataRequirements(
        primary_dataset=feature_name,
        datasets=tuple(
            DatasetRequirement(name=name, entity_scope=scope) for name, scope in scopes.items()
        ),
        target_entity=target_entity,
    )


def validate_single_asset_tabular_io(cfg: DictConfig) -> list[str]:
    """Validate a single-asset classifier's tabular I/O plus its target and label bindings."""
    errors = validate_tabular_io(cfg)
    model = cfg.model
    block_errors = check_fields(
        model,
        "model",
        [ConfigField("target", "mapping"), ConfigField("classification", "mapping")],
    )
    errors.extend(block_errors)
    if block_errors:
        return errors
    errors.extend(check_fields(model.target, "model.target", _TARGET_FIELDS))
    errors.extend(
        check_fields(model.classification, "model.classification", _CLASSIFICATION_FIELDS)
    )
    datasets = cfg.data.get("datasets") if cfg.get("data") is not None else None
    available = set(datasets.keys()) if isinstance(datasets, DictConfig) else set()
    label_dataset = str(model.classification.get("dataset"))
    if label_dataset and label_dataset != "None" and label_dataset not in available:
        errors.append(
            f"model references dataset '{label_dataset}' which is not configured in data.datasets"
        )
    if loss_is_portfolio(cfg):
        errors.extend(validate_risk_free_binding(cfg, str(model.get("name", "model"))))
    return errors


def validate_tabular_io(cfg: DictConfig) -> list[str]:
    """Validate the named feature/context/supervision roles against configured datasets."""
    model = cfg.model
    errors = check_fields(
        model, "model", [ConfigField("inputs", "mapping"), ConfigField("supervision", "mapping")]
    )
    if errors:
        return errors
    inputs = model.inputs
    errors = check_fields(
        inputs,
        "model.inputs",
        [
            ConfigField("features", "str"),
            ConfigField("context", "list", required=False, allow_scalar=True),
        ],
    )
    errors.extend(
        check_fields(
            model.supervision,
            "model.supervision",
            [ConfigField("dataset", "str"), ConfigField("column", "str")],
        )
    )
    datasets = cfg.data.get("datasets") if cfg.get("data") is not None else None
    available = set(datasets.keys()) if isinstance(datasets, DictConfig) else set()
    referenced = [str(inputs.get("features")), *_context_names(inputs)]
    referenced.append(str(model.supervision.get("dataset")))
    for name in referenced:
        if name and name != "None" and name not in available:
            errors.append(
                f"model references dataset '{name}' which is not configured in data.datasets"
            )
    return errors
