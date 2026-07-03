from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from omegaconf import DictConfig

from llca.data.modules.column_selection import is_all_columns


@dataclass(frozen=True, slots=True)
class ColumnRef:
    """Describe where a registered component references canonical dataset columns."""

    field: str
    kind: Literal["single", "list", "map", "map_keys"] = "single"
    required: bool = True


def referenced_columns(spec: DictConfig, refs: Sequence[ColumnRef]) -> list[str]:
    """Resolve declared single, list, mapping-key, or mapping-value column references."""
    columns: list[str] = []
    for ref in refs:
        value = spec.get(ref.field)
        if value is None or is_all_columns(value):
            continue
        if ref.kind == "single":
            if isinstance(value, str):
                columns.append(value)
        elif ref.kind == "list":
            columns += [str(column) for column in value]
        elif ref.kind == "map_keys":
            columns += [str(column) for column in value.keys()]
        else:
            columns += [str(column) for column in value.values() if column is not None]
    return columns
