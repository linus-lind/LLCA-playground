"""Generic row-wise consistency constraints for numerical research datasets."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from typing import Literal

import numpy as np
import pandas as pd

type ComparisonOperator = Literal["gt", "ge", "lt", "le", "eq", "ne"]
type Invalidation = Literal["left", "operands", "raise"] | tuple[str, ...]
type RightOperand = str | int | float

COMPARISON_OPERATORS: tuple[ComparisonOperator, ...] = ("gt", "ge", "lt", "le", "eq", "ne")
INVALIDATION_ACTIONS = ("left", "operands", "raise")


@dataclass(frozen=True, slots=True)
class ConstraintExpression:
    """Compare one or more left columns against a column or numerical scalar."""

    left: tuple[str, ...]
    operator: ComparisonOperator
    right: RightOperand

    def __post_init__(self) -> None:
        if not self.left or any(not column for column in self.left):
            raise ValueError("constraint expression requires at least one left column")
        if len(self.left) != len(set(self.left)):
            raise ValueError("constraint expression left columns must be unique")
        if self.operator not in COMPARISON_OPERATORS:
            raise ValueError(f"unsupported comparison operator: {self.operator!r}")
        if isinstance(self.right, bool) or not isinstance(self.right, str | Real):
            raise TypeError("constraint right operand must be a column or numerical scalar")


@dataclass(frozen=True, slots=True)
class ConstraintRule:
    """Group expressions that share one auditable invalidation policy."""

    name: str
    expressions: tuple[ConstraintExpression, ...]
    invalidate: Invalidation = "left"

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("constraint rule name must not be empty")
        if not self.expressions:
            raise ValueError(f"constraint rule '{self.name}' requires expressions")
        if isinstance(self.invalidate, str):
            if self.invalidate not in INVALIDATION_ACTIONS:
                raise ValueError(
                    f"constraint rule '{self.name}' has unsupported invalidation "
                    f"action {self.invalidate!r}"
                )
        elif not self.invalidate or len(self.invalidate) != len(set(self.invalidate)):
            raise ValueError(
                f"constraint rule '{self.name}' invalidation columns must be non-empty and unique"
            )


def _predicate(
    left: pd.Series,
    operator: ComparisonOperator,
    right: pd.Series | int | float,
) -> pd.Series:
    if operator == "gt":
        return left > right
    if operator == "ge":
        return left >= right
    if operator == "lt":
        return left < right
    if operator == "le":
        return left <= right
    if operator == "eq":
        return left == right
    return left != right


def _expression_violations(
    panel: pd.DataFrame,
    expression: ConstraintExpression,
) -> dict[str, pd.Series]:
    """Return one violation mask per expanded left column, ignoring absent operands."""
    right = panel[expression.right] if isinstance(expression.right, str) else expression.right
    right_observed = right.notna() if isinstance(right, pd.Series) else True
    violations: dict[str, pd.Series] = {}
    for column in expression.left:
        left = panel[column]
        comparable = left.notna() & right_observed
        violations[column] = comparable & ~_predicate(left, expression.operator, right)
    return violations


def _rule_columns(rule: ConstraintRule) -> tuple[str, ...]:
    columns = [column for expression in rule.expressions for column in expression.left]
    columns.extend(
        expression.right for expression in rule.expressions if isinstance(expression.right, str)
    )
    return tuple(dict.fromkeys(columns))


def _require_columns(panel: pd.DataFrame, rules: tuple[ConstraintRule, ...]) -> None:
    referenced = {
        column
        for rule in rules
        for column in (
            *_rule_columns(rule),
            *(rule.invalidate if isinstance(rule.invalidate, tuple) else ()),
        )
    }
    missing = sorted(referenced - set(panel.columns))
    if missing:
        raise KeyError(f"consistency constraints reference missing columns: {missing}")


def consistency_check(
    panel: pd.DataFrame,
    rules: tuple[ConstraintRule, ...],
) -> pd.DataFrame:
    """Invalidate cells that violate configured scalar or column relationships.

    Missing operands do not constitute a relational violation. ``left`` invalidation acts
    cell-by-cell, ``operands`` invalidates all participating operands on violating rows,
    and an explicit column tuple invalidates an atomic configured group. A serializable
    report is appended to ``DataFrame.attrs`` without changing the panel fingerprint.
    """
    _require_columns(panel, rules)
    result = panel.copy()
    rule_reports: list[dict[str, object]] = []

    for rule in rules:
        masks: dict[str, pd.Series] = {}
        row_violation = pd.Series(False, index=result.index)
        for expression in rule.expressions:
            expression_masks = _expression_violations(result, expression)
            for column, mask in expression_masks.items():
                masks[column] = masks.get(column, pd.Series(False, index=result.index)) | mask
                row_violation |= mask

        violating_rows = int(row_violation.sum())
        invalidated_cells = 0
        if violating_rows and rule.invalidate == "raise":
            examples = [repr(index) for index in result.index[row_violation][:3]]
            raise ValueError(
                f"consistency rule '{rule.name}' failed for {violating_rows} row(s); "
                f"examples: {examples}"
            )
        if rule.invalidate == "left":
            for column, mask in masks.items():
                invalidated_cells += int(mask.sum())
                result.loc[mask, column] = np.nan
        elif rule.invalidate == "operands":
            columns = _rule_columns(rule)
            invalidated_cells = violating_rows * len(columns)
            result.loc[row_violation, list(columns)] = np.nan
        elif isinstance(rule.invalidate, tuple):
            invalidated_cells = violating_rows * len(rule.invalidate)
            result.loc[row_violation, list(rule.invalidate)] = np.nan

        rule_reports.append(
            {
                "name": rule.name,
                "violating_rows": violating_rows,
                "invalidated_cells": invalidated_cells,
            }
        )

    reports = list(result.attrs.get("consistency_reports", []))
    reports.append({"rules": rule_reports})
    result.attrs["consistency_reports"] = reports
    return result
