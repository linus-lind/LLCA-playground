from __future__ import annotations

from omegaconf import DictConfig, ListConfig

from llca.mappers.config_validation import (
    ConfigField,
    check_fields,
    check_required_columns,
    is_int,
    register_validator,
)
from llca.mappers.features.mapper import feature_registry

_HORIZON_FIELDS = [ConfigField("horizon", "int", required=False, positive=True)]
_LOG_CHANGE_FIELDS = [
    *_HORIZON_FIELDS,
    ConfigField("skip_missing", "bool", required=False),
]
_SHIFT_FIELDS = [ConfigField("shift", "int", required=False)]
_WINDOW_FIELDS = [
    ConfigField("window", "int", required=True, positive=True),
    ConfigField("min_periods", "int", required=False, positive=True),
]


def _validate_rolling(spec: DictConfig, name: str) -> list[str]:
    """Validate a trailing-window transform's window length and optional minimum count."""
    errors = check_fields(spec, f"features.{name}", _WINDOW_FIELDS)
    window = spec.get("window")
    min_periods = spec.get("min_periods")
    if is_int(window) and is_int(min_periods) and min_periods > window:
        errors.append(f"features.{name}.min_periods must be <= window")
    return errors


@feature_registry.register_validator("log_change")
def _validate_log_change(spec: DictConfig) -> list[str]:
    return check_fields(spec, "features.log_change", _LOG_CHANGE_FIELDS)


@feature_registry.register_validator("simple_change")
def _validate_simple_change(spec: DictConfig) -> list[str]:
    return check_fields(spec, "features.simple_change", _HORIZON_FIELDS)


@feature_registry.register_validator("positive_indicator")
def _validate_positive_indicator(spec: DictConfig) -> list[str]:
    return check_fields(spec, "features.positive_indicator", _HORIZON_FIELDS)


@feature_registry.register_validator("net_ratio")
def _validate_net_ratio(spec: DictConfig) -> list[str]:
    """Require a non-empty ``add`` list and, when present, a list-valued ``subtract``."""
    errors: list[str] = []
    add = spec.get("add")
    if not isinstance(add, list | ListConfig) or len(add) == 0:
        errors.append("features.net_ratio.add must be a non-empty list of columns")
    subtract = spec.get("subtract")
    if subtract is not None and not isinstance(subtract, list | ListConfig):
        errors.append("features.net_ratio.subtract must be a list of columns")
    return errors


@feature_registry.register_validator("cross_sectional_median")
def _validate_cross_sectional_median(spec: DictConfig) -> list[str]:
    return check_fields(spec, "features.cross_sectional_median", _HORIZON_FIELDS)


@feature_registry.register_validator("rolling_volatility")
def _validate_rolling_volatility(spec: DictConfig) -> list[str]:
    return _validate_rolling(spec, "rolling_volatility")


@feature_registry.register_validator("downside_deviation")
def _validate_downside_deviation(spec: DictConfig) -> list[str]:
    return _validate_rolling(spec, "downside_deviation")


@feature_registry.register_validator("rolling_skewness")
def _validate_rolling_skewness(spec: DictConfig) -> list[str]:
    return _validate_rolling(spec, "rolling_skewness")


@feature_registry.register_validator("high_proximity")
def _validate_high_proximity(spec: DictConfig) -> list[str]:
    return _validate_rolling(spec, "high_proximity")


@feature_registry.register_validator("amihud_illiquidity")
def _validate_amihud_illiquidity(spec: DictConfig) -> list[str]:
    errors = _validate_rolling(spec, "amihud_illiquidity")
    errors.extend(
        check_fields(
            spec,
            "features.amihud_illiquidity",
            [ConfigField("log", "bool", required=False)],
        )
    )
    return errors


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
