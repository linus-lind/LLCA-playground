from collections.abc import Callable, Iterable, Sequence
from numbers import Integral, Real

from omegaconf import DictConfig, ListConfig

from llca.mappers.modules.column_ref import ColumnRef, referenced_columns
from llca.mappers.modules.config_field import ConfigField
from llca.mappers.modules.config_validation_error import ConfigValidationError

__all__ = [
    "ConfigField",
    "as_list",
    "check_required_columns",
    "check_fields",
    "is_int",
    "is_number",
    "register_validator",
    "validate_config",
]

ConfigValidator = Callable[[DictConfig], list[str]]
_validators: list[ConfigValidator] = []


def register_validator(fn: ConfigValidator) -> ConfigValidator:
    """Register one package-level validator executed by ``validate_config``."""
    _validators.append(fn)
    return fn


def validate_config(cfg: DictConfig) -> None:
    """Run all registered validators and report their errors together.

    Unexpected validator exceptions are converted into validation messages so one faulty
    subsystem does not hide independent configuration problems in other pipeline stages.
    """
    errors: list[str] = []
    for validator in _validators:
        try:
            errors.extend(validator(cfg))
        except Exception as exc:
            errors.append(f"{validator.__name__} raised an unexpected error: {exc}")

    if errors:
        raise ConfigValidationError(errors)


def is_int(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, Integral)


def is_number(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, Real)


_KIND_CHECKS: dict[str, Callable[[object], bool]] = {
    "int": is_int,
    "number": is_number,
    "str": lambda value: isinstance(value, str),
    "bool": lambda value: isinstance(value, bool),
    "list": lambda value: isinstance(value, list | ListConfig),
    "mapping": lambda value: isinstance(value, DictConfig),
}
_KIND_LABELS = {
    "int": "an integer",
    "number": "a number",
    "str": "a string",
    "bool": "a boolean",
    "list": "a list",
    "mapping": "a mapping",
}


def _check_field(cfg: DictConfig, prefix: str, field: ConfigField) -> list[str]:
    """Evaluate one declarative field contract and return fully qualified errors."""
    full = f"{prefix}.{field.name}"

    if field.name not in cfg or cfg[field.name] is None:
        return [f"{full} is required"] if field.required else []

    value = cfg[field.name]
    if field.kind == "list" and field.allow_scalar and not isinstance(value, list | ListConfig):
        return []

    if not _KIND_CHECKS[field.kind](value):
        return [f"{full} must be {_KIND_LABELS[field.kind]}"]

    errors = []
    if field.non_empty and len(value) == 0:
        errors.append(f"{full} must not be empty")
    if field.positive and value <= 0:
        errors.append(f"{full} must be positive")
    if field.minimum is not None and value < field.minimum:
        errors.append(f"{full} must be >= {field.minimum}")
    if field.maximum is not None and value >= field.maximum:
        errors.append(f"{full} must be < {field.maximum}")
    return errors


def check_fields(cfg: DictConfig, prefix: str, fields: Iterable[ConfigField]) -> list[str]:
    """Apply reusable field contracts without raising on user configuration errors."""
    errors = []
    for field in fields:
        errors.extend(_check_field(cfg, prefix, field))
    return errors


def as_list(value: object) -> list[object]:
    if value is None:
        return []
    if isinstance(value, list | ListConfig):
        return list(value)
    return [value]


def check_required_columns(spec: DictConfig, prefix: str, refs: Sequence[ColumnRef]) -> list[str]:
    """Require every mandatory declarative column reference to resolve at least once."""
    return [
        f"{prefix} must bind column field '{ref.field}'"
        for ref in refs
        if ref.required and not referenced_columns(spec, [ref])
    ]
