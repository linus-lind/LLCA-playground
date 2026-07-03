from dataclasses import dataclass
from typing import Any, cast

import pandas as pd
from omegaconf import DictConfig

from llca.core.paths import DATA_DIR
from llca.data.index_spec import IndexSpec
from llca.data.modules.column_selection import is_all_columns

_DATE = "date"
_NUMERIC = "numeric"


@dataclass(frozen=True, slots=True)
class _Column:
    """Bind a canonical pipeline name to its raw CSV name and parsing strategy."""

    canonical: str
    raw: str
    dtype: str


def _field(canonical: str, entry: Any, default_dtype: str) -> _Column:
    if isinstance(entry, str):
        return _Column(canonical, entry, default_dtype)
    return _Column(canonical, str(entry["raw"]), str(entry.get("dtype", default_dtype)))


def _index_fields(spec: DictConfig, index: IndexSpec) -> list[_Column]:
    mapping = spec.index
    fields = [_field(index.time, mapping[index.time], _DATE)]
    if index.entity is not None and index.entity in mapping:
        fields.append(_field(index.entity, mapping[index.entity], "Int64"))
    return fields


def _value_fields(mapping: DictConfig | None) -> list[_Column]:
    if mapping is None:
        return []
    return [_field(str(canonical), entry, _NUMERIC) for canonical, entry in mapping.items()]


def _all_column_fields(
    header: pd.Index, spec: DictConfig, index_fields: list[_Column]
) -> list[_Column]:
    """Select wildcard value columns after excluding index and configured omissions."""
    excluded = {field.raw for field in index_fields} | set(spec.get("exclude", []))
    return [
        _Column(str(column), str(column), _NUMERIC) for column in header if column not in excluded
    ]


def _column_fields(
    spec: DictConfig, header: pd.Index, index_fields: list[_Column], aux_fields: list[_Column]
) -> list[_Column]:
    """Resolve explicit or wildcard value columns without duplicating renamed auxiliaries."""
    if not is_all_columns(spec.get("columns")):
        return _value_fields(spec.get("columns"))

    rename_fields = _value_fields(spec.get("rename"))
    overrides = rename_fields + aux_fields
    override_raw = {field.raw for field in overrides}
    override_canonical = {field.canonical for field in overrides}
    base_fields = [
        field
        for field in _all_column_fields(header, spec, index_fields)
        if field.raw not in override_raw and field.canonical not in override_canonical
    ]
    return base_fields + rename_fields


def _assert_columns_present(spec: DictConfig, header: pd.Index, fields: list[_Column]) -> None:
    """Fail before loading when configured raw columns are absent from the CSV header."""
    available = set(header)
    missing = sorted({field.raw for field in fields if field.raw not in available})
    if missing:
        raise ValueError(
            f"dataset '{spec.path}' does not contain the requested raw column(s): {missing}"
        )


def load_dataset(spec: DictConfig, index: IndexSpec) -> pd.DataFrame:
    """Load one configured CSV into the canonical indexed panel representation.

    Only requested columns are read. Dates and explicitly typed index fields retain their
    configured types; value columns are coerced to numeric with invalid cells represented
    as missing. Raw names are mapped to canonical names before sorting by the global index
    contract.
    """
    header = pd.read_csv(DATA_DIR / str(spec.path), nrows=0).columns
    index_fields = _index_fields(spec, index)
    aux_fields = _value_fields(spec.get("auxiliary"))
    column_fields = _column_fields(spec, header, index_fields, aux_fields)

    fields = index_fields + column_fields + aux_fields
    _assert_columns_present(spec, header, fields)

    date_raw = [field.raw for field in fields if field.dtype == _DATE]
    read_dtype = {
        field.raw: field.dtype for field in fields if field.dtype not in (_DATE, _NUMERIC)
    }

    raw = pd.read_csv(
        DATA_DIR / spec.path,
        usecols=list(dict.fromkeys(field.raw for field in fields)),
        parse_dates=date_raw or None,
        date_format=spec.get("date_format"),
        dtype=cast(Any, read_dtype or None),
    )
    for field in fields:
        if field.dtype == _NUMERIC:
            raw[field.raw] = pd.to_numeric(raw[field.raw], errors="coerce")

    renamed = raw.rename(columns={field.raw: field.canonical for field in fields})
    return renamed.set_index([field.canonical for field in index_fields]).sort_index()
