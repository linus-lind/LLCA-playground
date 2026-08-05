from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from omegaconf import DictConfig, ListConfig

from llca.data.modules.column_selection import is_all_columns


@dataclass(frozen=True, slots=True)
class ColumnRef:
    """Describe where a registered component references canonical dataset columns."""

    field: str
    kind: Literal["single", "list", "map", "map_keys", "constraint_rules"] = "single"
    required: bool = True


def _constraint_columns(value: object) -> list[str]:
    """Resolve columns used by generic consistency expressions and invalidation groups."""
    if not isinstance(value, list | ListConfig):
        return []

    columns: list[str] = []
    for rule in value:
        if not isinstance(rule, DictConfig):
            continue
        expressions = rule.get("expressions")
        if isinstance(expressions, list | ListConfig):
            for expression in expressions:
                if not isinstance(expression, DictConfig):
                    continue
                left = expression.get("left")
                if isinstance(left, str):
                    columns.append(left)
                elif isinstance(left, list | ListConfig):
                    columns.extend(str(column) for column in left)
                right = expression.get("right")
                if isinstance(right, str):
                    columns.append(right)

        invalidate = rule.get("invalidate")
        if isinstance(invalidate, list | ListConfig):
            columns.extend(str(column) for column in invalidate)
    return columns


def referenced_columns(spec: DictConfig, refs: Sequence[ColumnRef]) -> list[str]:
    """Resolve every declaratively supported canonical dataset column reference."""
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
        elif ref.kind == "constraint_rules":
            columns += _constraint_columns(value)
        else:
            columns += [str(column) for column in value.values() if column is not None]
    return list(dict.fromkeys(columns))
