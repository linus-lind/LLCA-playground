from omegaconf import DictConfig, ListConfig

from llca.mappers.config_validation import (
    ConfigField,
    check_fields,
    check_required_columns,
    register_validator,
)
from llca.mappers.features.mapper import feature_registry

_HORIZON_FIELDS = [ConfigField("horizon", "int", required=False, positive=True)]
_SHIFT_FIELDS = [ConfigField("shift", "int", required=False)]


@feature_registry.register_validator("log_change")
def _validate_log_change(spec: DictConfig) -> list[str]:
    return check_fields(spec, "features.log_change", _HORIZON_FIELDS)


@feature_registry.register_validator("simple_change")
def _validate_simple_change(spec: DictConfig) -> list[str]:
    return check_fields(spec, "features.simple_change", _HORIZON_FIELDS)


@feature_registry.register_validator("cross_sectional_median")
def _validate_cross_sectional_median(spec: DictConfig) -> list[str]:
    return check_fields(spec, "features.cross_sectional_median", _HORIZON_FIELDS)


@register_validator
def _validate_features(cfg: DictConfig) -> list[str]:
    """Validate dataset bindings, registered transforms, fields, and declared columns."""
    features = cfg.get("features")
    if features is None:
        return []

    datasets = cfg.data.get("datasets")
    dataset_names = list(datasets.keys()) if isinstance(datasets, DictConfig) else []

    errors = []
    for dataset in features:
        if dataset not in dataset_names:
            errors.append(f"features.{dataset} does not match any dataset {dataset_names}")
            continue
        specs = features.get(dataset)
        if not isinstance(specs, ListConfig | list):
            errors.append(f"features.{dataset} must be a list of feature specs")
            continue

        for spec in specs:
            if not feature_registry.is_registered(spec.name):
                errors.append(f"features.{dataset} feature '{spec.name}' is not registered")
                continue
            prefix = f"features.{dataset} feature '{spec.name}'"
            errors.extend(check_fields(spec, prefix, _SHIFT_FIELDS))
            errors.extend(feature_registry.validate(spec.name, spec))
            errors.extend(
                check_required_columns(spec, prefix, feature_registry.column_refs(spec.name))
            )

        errors.extend(_duplicate_output_names(dataset, specs))

    return errors


def _duplicate_output_names(dataset: str, specs: ListConfig | list[DictConfig]) -> list[str]:
    """Detect explicit aliases that would overwrite another feature in the same panel."""
    named = [str(spec.get("as")) for spec in specs if spec.get("as") is not None]
    duplicates = sorted({name for name in named if named.count(name) > 1})
    return [
        f"features.{dataset} defines multiple features with the same output name '{name}'"
        for name in duplicates
    ]
