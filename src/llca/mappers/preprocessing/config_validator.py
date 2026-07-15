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


_COMPARISON_OPERATORS = ("gt", "ge", "lt", "le", "eq", "ne")
_INVALIDATION_ACTIONS = ("left", "operands", "raise")


def _validate_column_operand(value: object, prefix: str) -> list[str]:
    """Validate a scalar column name or a non-empty list of unique column names."""
    if isinstance(value, str):
        return [] if value else [f"{prefix} must not be empty"]
    if not isinstance(value, list | ListConfig):
        return [f"{prefix} must be a column name or a list of column names"]
    if not value:
        return [f"{prefix} must not be empty"]

    errors = [
        f"{prefix}[{index}] must be a non-empty column name"
        for index, column in enumerate(value)
        if not isinstance(column, str) or not column
    ]
    valid = [column for column in value if isinstance(column, str) and column]
    if len(valid) != len(set(valid)):
        errors.append(f"{prefix} column names must be unique")
    return errors


def _validate_expression(expression: object, prefix: str) -> list[str]:
    """Validate one generic left/operator/right comparison expression."""
    if not isinstance(expression, DictConfig):
        return [f"{prefix} must be a mapping with left, op, and right"]

    errors = []
    allowed = {"left", "op", "right"}
    unknown = sorted(str(key) for key in expression if str(key) not in allowed)
    if unknown:
        errors.append(f"{prefix} has unsupported field(s) {unknown}")

    if "left" not in expression:
        errors.append(f"{prefix}.left is required")
    else:
        errors.extend(_validate_column_operand(expression.left, f"{prefix}.left"))

    operator = expression.get("op")
    if operator not in _COMPARISON_OPERATORS:
        errors.append(f"{prefix}.op must be one of {list(_COMPARISON_OPERATORS)}, got {operator!r}")

    if "right" not in expression:
        errors.append(f"{prefix}.right is required")
    else:
        right = expression.right
        if not is_number(right) and not (isinstance(right, str) and right):
            errors.append(f"{prefix}.right must be a column name or a number")
    return errors


def _validate_constraint_rule(rule: object, prefix: str) -> list[str]:
    """Validate a named group of expressions and its atomic invalidation policy."""
    if not isinstance(rule, DictConfig):
        return [f"{prefix} must be a mapping"]

    errors = []
    allowed = {"name", "expressions", "invalidate"}
    unknown = sorted(str(key) for key in rule if str(key) not in allowed)
    if unknown:
        errors.append(f"{prefix} has unsupported field(s) {unknown}")

    name = rule.get("name")
    if not isinstance(name, str) or not name:
        errors.append(f"{prefix}.name must be a non-empty string")

    expressions = rule.get("expressions")
    if not isinstance(expressions, list | ListConfig) or not expressions:
        errors.append(f"{prefix}.expressions must be a non-empty list")
    else:
        for index, expression in enumerate(expressions):
            errors.extend(_validate_expression(expression, f"{prefix}.expressions[{index}]"))

    invalidate = rule.get("invalidate", "left")
    if isinstance(invalidate, str):
        if invalidate not in _INVALIDATION_ACTIONS:
            errors.append(
                f"{prefix}.invalidate must be one of {list(_INVALIDATION_ACTIONS)} "
                "or a list of column names"
            )
    else:
        errors.extend(_validate_column_operand(invalidate, f"{prefix}.invalidate"))
    return errors


@preprocessing_registry.register_validator("consistency_check")
def _validate_consistency_check(spec: DictConfig) -> list[str]:
    """Validate generic, auditable scalar and cross-column consistency rules."""
    prefix = "preprocessing.consistency_check"
    errors = []
    allowed = {"name", "constraints"}
    unknown = sorted(str(key) for key in spec if str(key) not in allowed)
    if unknown:
        errors.append(f"{prefix} has unsupported field(s) {unknown}")

    constraints = spec.get("constraints")
    if not isinstance(constraints, list | ListConfig) or not constraints:
        errors.append(f"{prefix}.constraints must be a non-empty list")
        return errors

    names = []
    for index, rule in enumerate(constraints):
        errors.extend(_validate_constraint_rule(rule, f"{prefix}.constraints[{index}]"))
        if isinstance(rule, DictConfig) and isinstance(rule.get("name"), str):
            names.append(str(rule.name))
    if len(names) != len(set(names)):
        errors.append(f"{prefix}.constraints rule names must be unique")
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
