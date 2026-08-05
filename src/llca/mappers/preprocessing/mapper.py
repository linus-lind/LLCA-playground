from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import cast

import pandas as pd
from omegaconf import DictConfig, ListConfig

from llca.data.modules.column_selection import is_all_columns
from llca.data.modules.panels import Panels
from llca.mappers.modules.column_ref import ColumnRef, referenced_columns
from llca.mappers.modules.registry import Registry
from llca.preprocessing.consistency_check import (
    ComparisonOperator,
    ConstraintExpression,
    ConstraintRule,
    Invalidation,
    consistency_check,
)
from llca.preprocessing.corporate_adjustment import corporate_adjustment
from llca.preprocessing.deduplicate import deduplicate
from llca.preprocessing.impute import impute
from llca.preprocessing.missing_threshold_filter import missing_threshold_filter
from llca.preprocessing.non_negative import non_negative_check
from llca.preprocessing.trading_calendar_filter import trading_calendar_filter

Transform = Callable[[pd.DataFrame], pd.DataFrame]

preprocessing_registry: Registry[Transform] = Registry("preprocessing step")


@preprocessing_registry.register(
    "consistency_check",
    columns=[ColumnRef("constraints", "constraint_rules")],
)
def _consistency_check(spec: DictConfig) -> Transform:
    """Bind validated generic consistency rules into a reusable transform."""
    rules = tuple(
        ConstraintRule(
            name=str(rule.name),
            expressions=tuple(
                ConstraintExpression(
                    left=(str(expression.left),)
                    if isinstance(expression.left, str)
                    else tuple(str(column) for column in expression.left),
                    operator=cast(ComparisonOperator, str(expression.op)),
                    right=expression.right,
                )
                for expression in rule.expressions
            ),
            invalidate=cast(
                Invalidation,
                str(rule.get("invalidate", "left"))
                if isinstance(rule.get("invalidate", "left"), str)
                else tuple(str(column) for column in rule.invalidate),
            ),
        )
        for rule in spec.constraints
    )
    return partial(
        consistency_check,
        rules=rules,
    )


@preprocessing_registry.register(
    "corporate_adjustment",
    columns=[
        ColumnRef("price_columns", "list", required=False),
        ColumnRef("price_factor", required=False),
        ColumnRef("volume", required=False),
        ColumnRef("shares_outstanding", required=False),
        ColumnRef("share_factor", required=False),
    ],
)
def _corporate_adjustment(spec: DictConfig) -> Transform:
    """Bind corporate-action column roles into a reusable DataFrame transform."""
    price_columns = spec.get("price_columns")
    return partial(
        corporate_adjustment,
        price_columns=list(price_columns) if price_columns else None,
        price_factor=spec.get("price_factor"),
        volume=spec.get("volume"),
        shares_outstanding=spec.get("shares_outstanding"),
        share_factor=spec.get("share_factor"),
    )


@preprocessing_registry.register("deduplicate")
def _duplicate(*_: object) -> Transform:
    return deduplicate


@preprocessing_registry.register(
    "impute",
    columns=[
        ColumnRef("ffill", "list", required=False),
        ColumnRef("fill_zero", "list", required=False),
        ColumnRef("subgroup_keys", "list", required=False),
    ],
)
def _impute(spec: DictConfig) -> Transform:
    """Bind grouped imputation rules into a reusable DataFrame transform."""
    return partial(
        impute,
        ffill=list(spec.get("ffill", [])),
        fill_zero=list(spec.get("fill_zero", [])),
        subgroup_keys=list(spec.get("subgroup_keys", [])),
    )


@preprocessing_registry.register(
    "missing_threshold_filter",
    columns=[
        ColumnRef("columns", "list", required=False),
        ColumnRef("subgroup_keys", "list", required=False),
    ],
)
def _missing_threshold_filter(spec: DictConfig) -> Transform:
    """Bind global or subgroup sparsity settings into a reusable DataFrame transform."""
    columns = spec.get("columns")
    selected = list(columns) if columns and not is_all_columns(columns) else None
    return partial(
        missing_threshold_filter,
        threshold=spec.threshold,
        subgroup_keys=list(spec.get("subgroup_keys", [])),
        columns=selected,
    )


@preprocessing_registry.register("trading_calendar_filter")
def _trading_calendar_filter(spec: DictConfig) -> Transform:
    return partial(trading_calendar_filter, calendar=spec.calendar)


@preprocessing_registry.register(
    "non_negative_check", columns=[ColumnRef("columns", "list", required=False)]
)
def _non_negative_check(spec: DictConfig) -> Transform:
    return partial(non_negative_check, columns=list(spec.get("columns", [])))


def _dataset_steps(
    preprocessing: DictConfig | ListConfig | None, name: str, single: bool
) -> ListConfig | None:
    """Resolve flat single-dataset or dataset-keyed preprocessing configuration."""
    if preprocessing is None:
        return None
    if isinstance(preprocessing, ListConfig):
        return preprocessing if single else None
    steps = preprocessing.get(name)
    return steps if isinstance(steps, ListConfig) else None


def _require_columns(panel: pd.DataFrame, step: DictConfig) -> None:
    """Check registered column dependencies against the current transformed panel."""
    refs = preprocessing_registry.column_refs(step.name)
    missing = [column for column in referenced_columns(step, refs) if column not in panel.columns]
    if missing:
        raise KeyError(
            f"preprocessing step '{step.name}' references columns not in the dataset: {missing}"
        )


def _apply_steps(steps: ListConfig | None, panel: pd.DataFrame) -> pd.DataFrame:
    """Apply registered transforms in declared order with runtime column checks."""
    roles = dict(panel.attrs)
    for step in steps or []:
        _require_columns(panel, step)
        panel = preprocessing_registry.build(step.name, step)(panel)
        panel.attrs.update(roles)
    return panel


def build_preprocessing(
    preprocessing_cfg: DictConfig | ListConfig | None, datasets: Panels
) -> Panels:
    """Build and execute an independent preprocessing pipeline for every dataset."""
    single = len(datasets) == 1
    return {
        name: _apply_steps(_dataset_steps(preprocessing_cfg, name, single), panel)
        for name, panel in datasets.items()
    }
