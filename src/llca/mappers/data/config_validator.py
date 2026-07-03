from typing import Any

from omegaconf import DictConfig

from llca.data.modules.column_selection import ALL_COLUMNS, is_all_columns
from llca.mappers.config_validation import check_fields, register_validator
from llca.mappers.modules.config_field import ConfigField

_DATASET_FIELDS = [ConfigField("path", "str"), ConfigField("date_format", "str", required=False)]


def _valid_entry(entry: Any) -> bool:
    if isinstance(entry, str):
        return bool(entry)
    return isinstance(entry, DictConfig) and isinstance(entry.get("raw"), str)


def _validate_mapping(name: str, spec: DictConfig, group: str) -> list[str]:
    """Validate canonical-to-raw mappings used by ingestion column groups."""
    mapping = spec.get(group)
    if mapping is None:
        return []
    if not isinstance(mapping, DictConfig):
        return [f"data.datasets.{name}.{group} must map canonical names to raw columns"]
    return [
        f"data.datasets.{name}.{group}.{str(canonical)} must be a raw column name or {{raw: ..., dtype: ...}}"
        for canonical, entry in mapping.items()
        if not _valid_entry(entry)
    ]


def _validate_columns(name: str, spec: DictConfig) -> list[str]:
    """Require explicit value mappings or the all-columns selection sentinel."""
    columns = spec.get("columns")
    if columns is None:
        return [
            f"data.datasets.{name}.columns is required: map canonical names to raw columns, "
            f"or '{ALL_COLUMNS}' to load every column"
        ]
    if is_all_columns(columns):
        return []
    return _validate_mapping(name, spec, "columns")


def _validate_rename(name: str, spec: DictConfig) -> list[str]:
    """Restrict rename overrides to wildcard column loading."""
    if spec.get("rename") is None:
        return []
    errors = _validate_mapping(name, spec, "rename")
    if not is_all_columns(spec.get("columns")):
        errors.append(
            f"data.datasets.{name}.rename is only allowed together with columns: '{ALL_COLUMNS}'"
        )
    return errors


def _validate_dataset(name: str, spec: DictConfig, time_role: str | None) -> list[str]:
    """Validate one dataset's path, index mapping, value selection, and aliases."""
    errors = check_fields(spec, f"data.datasets.{name}", _DATASET_FIELDS)

    index = spec.get("index")
    if not isinstance(index, DictConfig):
        errors.append(f"data.datasets.{name}.index must map the index roles to raw columns")
    elif time_role is not None and time_role not in index:
        errors.append(
            f"data.datasets.{name}.index must map the time role '{time_role}' to a raw column"
        )

    for group in ("index", "auxiliary"):
        errors.extend(_validate_mapping(name, spec, group))
    errors.extend(_validate_columns(name, spec))
    errors.extend(_validate_rename(name, spec))
    return errors


@register_validator
def _validate_data(cfg: DictConfig) -> list[str]:
    """Validate the global index contract and every independently configured dataset."""
    data = cfg.data
    index = data.get("index")
    time_role = index.get("time") if isinstance(index, DictConfig) else None

    errors = []
    if time_role is None:
        errors.append("data.index must define a 'time' axis")

    datasets_errors = check_fields(
        data, "data", [ConfigField("datasets", "mapping", non_empty=True)]
    )
    if datasets_errors:
        return errors + datasets_errors
    datasets = data.datasets

    for name, spec in datasets.items():
        errors.extend(_validate_dataset(str(name), spec, time_role))
    return errors
