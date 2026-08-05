from __future__ import annotations

from typing import Any

from omegaconf import DictConfig, ListConfig

from llca.data.modules.column_selection import ALL_COLUMNS, is_all_columns
from llca.mappers.config_validation import check_fields, register_validator
from llca.mappers.modules.config_field import ConfigField

_DATASET_FIELDS = [
    ConfigField("path", "str"),
    ConfigField("kind", "str", required=False),
    ConfigField("frequency", "str", required=False),
    ConfigField("date_format", "str", required=False),
]
_DATASET_KINDS = ("panel", "context", "events", "table")


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
    kind = spec.get("kind")
    if isinstance(kind, str) and kind not in _DATASET_KINDS:
        errors.append(f"data.datasets.{name}.kind '{kind}' must be one of {list(_DATASET_KINDS)}")

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
    selection = data.get("selection")
    if selection is not None:
        if not isinstance(selection, DictConfig):
            errors.append("data.selection must be a mapping")
        else:
            errors.extend(
                check_fields(
                    selection,
                    "data.selection",
                    [
                        ConfigField("entity_ids", "list", required=False),
                        ConfigField("csv_chunk_size", "int", required=False, positive=True),
                    ],
                )
            )
            entity_ids = selection.get("entity_ids")
            if isinstance(entity_ids, list | ListConfig):
                invalid = [value for value in entity_ids if not isinstance(value, int | str)]
                if invalid:
                    errors.append("data.selection.entity_ids values must be integers or strings")
                if len(set(entity_ids)) != len(entity_ids):
                    errors.append("data.selection.entity_ids must not contain duplicates")
    cache = data.get("cache")
    if cache is not None:
        if not isinstance(cache, DictConfig):
            errors.append("data.cache must be a mapping")
        else:
            errors.extend(
                check_fields(
                    cache,
                    "data.cache",
                    [
                        ConfigField("enabled", "bool"),
                        ConfigField("directory", "str"),
                    ],
                )
            )
    return errors
