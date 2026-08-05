"""Shared data-requirements and input validation for cross-sectional tabular models."""

from __future__ import annotations

from omegaconf import DictConfig

from llca.mappers.config_validation import ConfigField, as_list, check_fields
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
