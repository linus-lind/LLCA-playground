from collections.abc import Callable
from typing import cast

import numpy as np
import pandas as pd
from omegaconf import DictConfig, ListConfig

from llca.data.index_spec import entity_level
from llca.data.modules.panels import Panels
from llca.mappers.modules.column_ref import ColumnRef, referenced_columns
from llca.mappers.modules.registry import Registry
from llca.transforms.cross_sectional_median import cross_sectional_median
from llca.transforms.primitives import (
    log_change,
    log_difference,
    log_ratio,
    range_location,
    ratio,
    relative_spread,
    simple_change,
)

feature_registry: Registry[pd.Series] = Registry("feature")


def _per_series(
    panel: pd.DataFrame, series: pd.Series, fn: Callable[[np.ndarray], np.ndarray]
) -> pd.Series:
    """Apply a temporal transform independently per entity, or globally for date-only data."""
    entity = entity_level(panel)
    if entity is None:
        return pd.Series(fn(series.to_numpy(dtype=float)), index=panel.index)
    return series.groupby(level=entity).transform(lambda s: fn(s.to_numpy(dtype=float)))


@feature_registry.register(
    "log_change", columns=[ColumnRef("column"), ColumnRef("times", required=False)]
)
def _log_change(panel: pd.DataFrame, spec: DictConfig) -> pd.Series:
    """Compute an entity-local log change, optionally after scaling by another column."""
    horizon = spec.get("horizon", 1)
    base = panel[spec.column]
    if spec.get("times") is not None:
        base = base * panel[spec.times]
    values = _per_series(panel, base, lambda x: log_change(x, horizon=horizon))
    times = f"_{spec.times}" if spec.get("times") is not None else ""
    return values.rename(f"log_change_{spec.column}{times}_{horizon}")


@feature_registry.register(
    "simple_change", columns=[ColumnRef("column"), ColumnRef("times", required=False)]
)
def _simple_change(panel: pd.DataFrame, spec: DictConfig) -> pd.Series:
    """Compute an entity-local simple change, optionally on a column product."""
    horizon = spec.get("horizon", 1)
    base = panel[spec.column]
    if spec.get("times") is not None:
        base = base * panel[spec.times]
    values = _per_series(panel, base, lambda x: simple_change(x, horizon=horizon))
    times = f"_{spec.times}" if spec.get("times") is not None else ""
    return values.rename(f"simple_change_{spec.column}{times}_{horizon}")


@feature_registry.register("log_difference", columns=[ColumnRef("current"), ColumnRef("previous")])
def _log_difference(panel: pd.DataFrame, spec: DictConfig) -> pd.Series:
    """Compute ``log(current[t]) - log(previous[t-1])`` without crossing entities."""
    current, previous = spec.current, spec.previous
    entity = entity_level(panel)
    values = np.full(len(panel), np.nan, dtype=float)
    groups = (
        [np.arange(len(panel), dtype=int)]
        if entity is None
        else panel.groupby(level=entity, sort=False).indices.values()
    )
    for positions in groups:
        rows = np.asarray(positions, dtype=int)
        values[rows] = log_difference(
            panel.iloc[rows][current].to_numpy(dtype=float),
            panel.iloc[rows][previous].to_numpy(dtype=float),
        )
    return pd.Series(
        values,
        index=panel.index,
        name=f"log_difference_{current}_{previous}",
    )


@feature_registry.register("log_ratio", columns=[ColumnRef("numerator"), ColumnRef("denominator")])
def _log_ratio(panel: pd.DataFrame, spec: DictConfig) -> pd.Series:
    """Compute the element-wise logarithmic ratio of two configured columns."""
    values = log_ratio(
        panel[spec.numerator].to_numpy(dtype=float), panel[spec.denominator].to_numpy(dtype=float)
    )
    return pd.Series(
        values, index=panel.index, name=f"log_ratio_{spec.numerator}_{spec.denominator}"
    )


@feature_registry.register("ratio", columns=[ColumnRef("numerator"), ColumnRef("denominator")])
def _ratio(panel: pd.DataFrame, spec: DictConfig) -> pd.Series:
    values = ratio(
        panel[spec.numerator].to_numpy(dtype=float), panel[spec.denominator].to_numpy(dtype=float)
    )
    return pd.Series(values, index=panel.index, name=f"ratio_{spec.numerator}_{spec.denominator}")


@feature_registry.register(
    "range_location", columns=[ColumnRef("high"), ColumnRef("low"), ColumnRef("value")]
)
def _range_location(panel: pd.DataFrame, spec: DictConfig) -> pd.Series:
    """Map a configured value to its centered location between high and low columns."""
    values = range_location(
        panel[spec.high].to_numpy(dtype=float),
        panel[spec.low].to_numpy(dtype=float),
        panel[spec.value].to_numpy(dtype=float),
    )
    return pd.Series(values, index=panel.index, name=f"range_location_{spec.value}")


@feature_registry.register("relative_spread", columns=[ColumnRef("upper"), ColumnRef("lower")])
def _relative_spread(panel: pd.DataFrame, spec: DictConfig) -> pd.Series:
    values = relative_spread(
        panel[spec.upper].to_numpy(dtype=float), panel[spec.lower].to_numpy(dtype=float)
    )
    return pd.Series(values, index=panel.index, name=f"relative_spread_{spec.upper}_{spec.lower}")


@feature_registry.register("passthrough", columns=[ColumnRef("column")])
def _passthrough(panel: pd.DataFrame, spec: DictConfig) -> pd.Series:
    return cast(pd.Series, panel[spec.column]).rename(str(spec.column))


@feature_registry.register("cross_sectional_median", columns=[ColumnRef("column")])
def _cross_sectional_median(panel: pd.DataFrame, spec: DictConfig) -> pd.Series:
    return cross_sectional_median(panel, spec.get("horizon", 1), spec.column).rename(
        f"cross_sectional_median_{spec.column}"
    )


def _shift(panel: pd.DataFrame, series: pd.Series, periods: int) -> pd.Series:
    entity = entity_level(panel)
    if entity is None:
        return series.shift(periods)
    return series.groupby(level=entity).shift(periods)


def _require_feature_columns(panel: pd.DataFrame, spec: DictConfig) -> None:
    refs = feature_registry.column_refs(spec.name)
    missing = [column for column in referenced_columns(spec, refs) if column not in panel.columns]
    if missing:
        raise KeyError(f"feature '{spec.name}' references columns not in the dataset: {missing}")


def build_features(specs: ListConfig | None, panel: pd.DataFrame) -> pd.DataFrame:
    """Build configured feature columns on the input panel's unchanged row index.

    Registered transforms declare their source-column dependencies. Optional shifts are
    applied per entity to avoid cross-sectional leakage, and duplicate output names are
    rejected before the feature frame is returned.
    """
    columns: dict[str, pd.Series] = {}
    for spec in specs or []:
        _require_feature_columns(panel, spec)
        series = feature_registry.build(spec.name, panel, spec)
        shift = spec.get("shift")
        if shift is not None:
            series = _shift(panel, series, int(shift))
        name = str(spec.get("as") or series.name)
        if name in columns:
            raise ValueError(
                f"duplicate feature output name '{name}'; give each feature a unique 'as'"
            )
        columns[name] = series
    result = pd.DataFrame(columns, index=panel.index)
    result.attrs.update(panel.attrs)
    return result


def build_feature_panels(features_cfg: DictConfig | None, datasets: Panels) -> Panels:
    """Build features independently for every dataset, passing through unconfigured panels."""
    panels: Panels = {}
    for name, panel in datasets.items():
        specs = features_cfg.get(name) if features_cfg is not None else None
        panels[name] = build_features(specs, panel) if specs is not None else panel
    return panels
