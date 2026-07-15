from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pandas as pd
from omegaconf import DictConfig

from llca.core.paths import DATA_DIR
from llca.data.index_spec import IndexSpec
from llca.data.modules.column_selection import is_all_columns
from llca.data.modules.panels import Panels
from llca.pipeline.contracts import DatasetQuery

_DATE = "date"
_NUMERIC = "numeric"


@dataclass(frozen=True, slots=True)
class _Column:
    """Bind a canonical pipeline name to its raw CSV name and parsing strategy."""

    canonical: str
    raw: str
    dtype: str


@dataclass(frozen=True, slots=True)
class _Layout:
    """Resolved projection and canonical schema for one logical dataset."""

    spec: DictConfig
    index_fields: tuple[_Column, ...]
    fields: tuple[_Column, ...]
    entity_name: str | None

    @property
    def entity_field(self) -> _Column | None:
        return next(
            (field for field in self.index_fields if field.canonical == self.entity_name),
            None,
        )


def _field(canonical: str, entry: Any, default_dtype: str) -> _Column:
    if isinstance(entry, str):
        return _Column(canonical, entry, default_dtype)
    return _Column(canonical, str(entry["raw"]), str(entry.get("dtype", default_dtype)))


def _index_fields(spec: DictConfig, index: IndexSpec) -> list[_Column]:
    mapping = spec.index
    fields = [_field(index.time, mapping[index.time], _DATE)]
    if index.entity is not None and index.entity in mapping:
        fields.append(_field(index.entity, mapping[index.entity], "Int64"))
    known = {field.canonical for field in fields}
    fields.extend(
        _field(str(canonical), entry, "string")
        for canonical, entry in mapping.items()
        if str(canonical) not in known
    )
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


def _layout(spec: DictConfig, index: IndexSpec, header: pd.Index) -> _Layout:
    index_fields = _index_fields(spec, index)
    aux_fields = _value_fields(spec.get("auxiliary"))
    column_fields = _column_fields(spec, header, index_fields, aux_fields)
    fields = index_fields + column_fields + aux_fields
    _assert_columns_present(spec, header, fields)
    return _Layout(
        spec=spec,
        index_fields=tuple(index_fields),
        fields=tuple(fields),
        entity_name=index.entity,
    )


def _read_dtypes(layouts: Mapping[str, _Layout]) -> dict[str, str]:
    """Merge explicit CSV dtypes and reject conflicting views of one raw column."""
    result: dict[str, str] = {}
    for layout in layouts.values():
        for field in layout.fields:
            if field.dtype in (_DATE, _NUMERIC):
                continue
            previous = result.get(field.raw)
            if previous is not None and previous != field.dtype:
                raise ValueError(
                    f"raw column '{field.raw}' has conflicting dtypes '{previous}' and "
                    f"'{field.dtype}' across logical datasets"
                )
            result[field.raw] = field.dtype
    return result


def _source_entity_filter(
    layouts: Mapping[str, _Layout], queries: Mapping[str, DatasetQuery]
) -> dict[str, set[object]] | None:
    """Return a source-level union filter, or ``None`` when one view needs all entities."""
    filters: dict[str, set[object]] = {}
    for name, layout in layouts.items():
        entity_field = layout.entity_field
        if entity_field is None:
            # A date-/event-only logical view has no safe entity predicate. When it
            # shares a physical source with a target panel, that panel must filter only
            # after the complete source scan or rows required by this view would vanish.
            return None
        entity_ids = queries[name].entity_ids
        if entity_ids is None:
            return None
        filters.setdefault(entity_field.raw, set()).update(entity_ids)
    return filters or None


def _read_source(
    path: Path,
    layouts: Mapping[str, _Layout],
    queries: Mapping[str, DatasetQuery],
    *,
    chunk_size: int,
) -> pd.DataFrame:
    """Read one physical CSV once, with chunked predicate filtering when possible."""
    raw_columns = list(
        dict.fromkeys(field.raw for layout in layouts.values() for field in layout.fields)
    )
    dtype = cast(Any, _read_dtypes(layouts) or None)
    entity_filter = _source_entity_filter(layouts, queries)
    if entity_filter is None:
        return pd.read_csv(path, usecols=raw_columns, dtype=dtype)

    chunks: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        path,
        usecols=raw_columns,
        dtype=dtype,
        chunksize=chunk_size,
    ):
        keep = pd.Series(False, index=chunk.index)
        for raw_column, entity_ids in entity_filter.items():
            keep |= chunk[raw_column].isin(entity_ids)
        selected = chunk.loc[keep]
        if not selected.empty:
            chunks.append(selected)
    if not chunks:
        return pd.DataFrame(columns=raw_columns)
    return pd.concat(chunks, ignore_index=True)


def _materialize(raw: pd.DataFrame, layout: _Layout, query: DatasetQuery) -> pd.DataFrame:
    """Project, filter, type, rename, and index one logical view of a shared source."""
    frame = raw[[field.raw for field in layout.fields]].copy()
    entity_field = layout.entity_field
    if entity_field is not None and query.entity_ids is not None:
        frame = frame.loc[frame[entity_field.raw].isin(query.entity_ids)].copy()

    date_format = layout.spec.get("date_format")
    for field in layout.fields:
        if field.dtype == _DATE:
            frame[field.raw] = pd.to_datetime(
                frame[field.raw],
                format=str(date_format) if date_format is not None else None,
            )
        elif field.dtype == _NUMERIC:
            frame[field.raw] = pd.to_numeric(frame[field.raw], errors="coerce")

    renamed = frame.rename(columns={field.raw: field.canonical for field in layout.fields})
    index_names = [field.canonical for field in layout.index_fields]
    result = renamed.set_index(index_names).sort_index()
    result.attrs["llca.time_level"] = layout.index_fields[0].canonical
    result.attrs["llca.entity_level"] = entity_field.canonical if entity_field is not None else None
    return result


def load_dataset(spec: DictConfig, index: IndexSpec) -> pd.DataFrame:
    """Load one configured CSV into the canonical indexed panel representation.

    Only requested columns are read. Dates and explicitly typed index fields retain their
    configured types; value columns are coerced to numeric with invalid cells represented
    as missing. Raw names are mapped to canonical names before sorting by the global index
    contract.
    """
    return load_datasets(
        {"dataset": spec},
        index,
        {"dataset": DatasetQuery()},
    )["dataset"]


def load_datasets(
    specs: Mapping[str, DictConfig],
    index: IndexSpec,
    queries: Mapping[str, DatasetQuery],
    *,
    csv_chunk_size: int = 250_000,
) -> Panels:
    """Load arbitrary logical datasets while scanning each physical CSV at most once."""
    if set(specs) != set(queries):
        raise ValueError("dataset specs and queries must contain identical logical names")
    grouped: dict[Path, list[str]] = {}
    for name, spec in specs.items():
        grouped.setdefault((DATA_DIR / str(spec.path)).resolve(), []).append(name)

    panels: Panels = {}
    for path, names in grouped.items():
        header = pd.read_csv(path, nrows=0).columns
        layouts = {name: _layout(specs[name], index, header) for name in names}
        source_queries = {name: queries[name] for name in names}
        raw = _read_source(
            path,
            layouts,
            source_queries,
            chunk_size=csv_chunk_size,
        )
        panels.update(
            {
                name: _materialize(raw, layout, source_queries[name])
                for name, layout in layouts.items()
            }
        )
    return panels
