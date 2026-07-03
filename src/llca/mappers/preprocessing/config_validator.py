import exchange_calendars as xcals
from omegaconf import DictConfig, ListConfig

from llca.data.modules.column_selection import ALL_COLUMNS, is_all_columns
from llca.mappers.config_validation import (
    check_fields,
    check_required_columns,
    is_number,
    register_validator,
)
from llca.mappers.modules.config_field import ConfigField
from llca.mappers.preprocessing.mapper import preprocessing_registry

_IMPUTE_FIELDS = [
    ConfigField("ffill", "list", required=False),
    ConfigField("fill_zero", "list", required=False),
    ConfigField("subgroup_keys", "list", required=False),
]
_MISSING_THRESHOLD_FIELDS = [
    ConfigField("threshold", "number", minimum=0.0, maximum=1.0),
    ConfigField("subgroup_keys", "list", required=False),
]
_TRADING_CALENDAR_FIELDS = [ConfigField("calendar", "str")]
_COLUMNS_FIELD = [ConfigField("columns", "list", required=False)]


@preprocessing_registry.register_validator("corporate_adjustment")
def _validate_corporate_adjustment(spec: DictConfig) -> list[str]:
    """Require factor columns for each requested corporate-action adjustment family."""
    has_price = bool(spec.get("price_columns"))
    has_share = spec.get("volume") is not None or spec.get("shares_outstanding") is not None

    errors = []
    if not (has_price or has_share):
        errors.append(
            "preprocessing.corporate_adjustment must configure price adjustment (price_columns + "
            "price_factor) and/or share adjustment (volume/shares_outstanding + share_factor)"
        )
    if has_price and not isinstance(spec.get("price_factor"), str):
        errors.append(
            "preprocessing.corporate_adjustment.price_factor is required when price_columns is set"
        )
    if has_share and not isinstance(spec.get("share_factor"), str):
        errors.append(
            "preprocessing.corporate_adjustment.share_factor is required when volume/shares_outstanding is set"
        )
    return errors


@preprocessing_registry.register_validator("impute")
def _validate_impute(spec: DictConfig) -> list[str]:
    return check_fields(spec, "preprocessing.impute", _IMPUTE_FIELDS)


@preprocessing_registry.register_validator("missing_threshold_filter")
def _validate_missing_threshold_filter(spec: DictConfig) -> list[str]:
    """Validate sparsity threshold and explicit or wildcard checked columns."""
    errors = check_fields(spec, "preprocessing.missing_threshold_filter", _MISSING_THRESHOLD_FIELDS)
    columns = spec.get("columns")
    if columns is None:
        errors.append(
            f"preprocessing.missing_threshold_filter.columns is required: a list of columns or '{ALL_COLUMNS}'"
        )
    elif not is_all_columns(columns) and not isinstance(columns, list | ListConfig):
        errors.append(
            f"preprocessing.missing_threshold_filter.columns must be a list of columns or '{ALL_COLUMNS}'"
        )
    return errors


@preprocessing_registry.register_validator("trading_calendar_filter")
def _validate_trading_calendar_filter(spec: DictConfig) -> list[str]:
    """Validate an exchange calendar against the installed calendar registry."""
    errors = check_fields(spec, "preprocessing.trading_calendar_filter", _TRADING_CALENDAR_FIELDS)
    if errors:
        return errors

    calendar = spec.get("calendar")
    if calendar not in xcals.get_calendar_names():
        errors.append(
            f"preprocessing.trading_calendar_filter.calendar '{calendar}' is not a known exchange calendar"
        )
    return errors


_BOUNDS = ("gt", "ge", "lt", "le")


def _validate_bounds(bounds: DictConfig, prefix: str) -> list[str]:
    """Validate consistent strict/inclusive lower and upper scalar bounds."""
    errors = []

    unknown = [str(key) for key in bounds if str(key) not in _BOUNDS]
    if unknown:
        errors.append(f"{prefix} has unsupported bound(s) {unknown}; allowed: {list(_BOUNDS)}")

    present = {bound: bounds.get(bound) is not None for bound in _BOUNDS}
    for bound in _BOUNDS:
        value = bounds.get(bound)
        if value is not None and not is_number(value):
            errors.append(f"{prefix}.{bound} must be a number")

    if not any(present.values()):
        errors.append(f"{prefix} must specify at least one of {list(_BOUNDS)}")
    if present["lt"] and present["le"]:
        errors.append(f"{prefix} cannot set both 'lt' and 'le'")
    if present["gt"] and present["ge"]:
        errors.append(f"{prefix} cannot set both 'gt' and 'ge'")

    lower = bounds.get("gt") if present["gt"] else bounds.get("ge")
    upper = bounds.get("lt") if present["lt"] else bounds.get("le")
    if is_number(lower) and is_number(upper) and lower >= upper:
        lower_name = "gt" if present["gt"] else "ge"
        upper_name = "lt" if present["lt"] else "le"
        errors.append(
            f"{prefix} lower bound '{lower_name}' ({lower}) must be less than "
            f"upper bound '{upper_name}' ({upper})"
        )

    return errors


@preprocessing_registry.register_validator("consistency_check")
def _validate_consistency_check(spec: DictConfig) -> list[str]:
    """Validate per-column bound mappings for the consistency transform."""
    bounded = spec.get("bounded")
    if bounded is None:
        return []

    prefix = "preprocessing.consistency_check.bounded"
    if not isinstance(bounded, DictConfig):
        return [f"{prefix} must be a mapping of column -> bounds"]

    errors = []
    for name, bounds in bounded.items():
        column_prefix = f"{prefix}.{str(name)}"
        if not isinstance(bounds, DictConfig):
            errors.append(
                f"{column_prefix} must be a mapping of bound operators ({'/'.join(_BOUNDS)})"
            )
            continue
        errors.extend(_validate_bounds(bounds, column_prefix))
    return errors


@preprocessing_registry.register_validator("non_negative_check")
def _validate_non_negative_check(spec: DictConfig) -> list[str]:
    return check_fields(spec, "preprocessing.non_negative_check", _COLUMNS_FIELD)


@preprocessing_registry.register_validator("forward_fill")
def _validate_forward_fill(spec: DictConfig) -> list[str]:
    return check_fields(spec, "preprocessing.forward_fill", _COLUMNS_FIELD)


def validate_preprocessing(steps: ListConfig | None, prefix: str) -> list[str]:
    """Validate an ordered transform list through the shared preprocessing registry."""
    if steps is None:
        return []
    if not isinstance(steps, ListConfig | list):
        return [f"{prefix} must be a list of preprocessing steps"]

    errors = []
    for index, step in enumerate(steps):
        if not preprocessing_registry.is_registered(step.name):
            errors.append(f"{prefix}[{index}] step '{step.name}' is not registered")
            continue
        step_prefix = f"{prefix}[{index}] step '{step.name}'"
        errors.extend(preprocessing_registry.validate(step.name, step))
        errors.extend(
            check_required_columns(step, step_prefix, preprocessing_registry.column_refs(step.name))
        )
    return errors


@register_validator
def _validate_preprocessing_group(cfg: DictConfig) -> list[str]:
    """Validate flat single-dataset or keyed multi-dataset preprocessing layouts."""
    preprocessing = cfg.get("preprocessing")
    if preprocessing is None:
        return []

    datasets = cfg.data.get("datasets")
    if not isinstance(datasets, DictConfig) or not datasets:
        return []
    names = [str(name) for name in datasets.keys()]

    if isinstance(preprocessing, ListConfig):
        if len(names) != 1:
            return [
                "preprocessing is a flat list but multiple datasets exist; key it by dataset name"
            ]
        return validate_preprocessing(preprocessing, "preprocessing")

    errors = [
        f"preprocessing.{str(key)} does not match any dataset {names}"
        for key in preprocessing
        if str(key) not in names
    ]
    for name in names:
        errors.extend(validate_preprocessing(preprocessing.get(name), f"preprocessing.{name}"))
    return errors
