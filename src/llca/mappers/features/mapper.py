from __future__ import annotations

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
    amihud_illiquidity,
    downside_deviation,
    high_proximity,
    log_change,
    log_difference,
    log_ratio,
    net_ratio,
    positive_indicator,
    range_location,
    ratio,
    relative_spread,
    rolling_skewness,
    rolling_volatility,
    simple_change,
    simple_difference,
    simple_ratio,
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


def _per_non_missing_series(
    panel: pd.DataFrame, series: pd.Series, fn: Callable[[np.ndarray], np.ndarray]
) -> pd.Series:
    """Apply an event-frequency transform without counting unrelated sparse rows.

    Point-in-time files may interleave quarterly and annual report rows.  A quarterly
    four-observation change must therefore count the previous four *quarterly values*,
    not the previous four physical rows.  Missing observations remain missing in the
    returned frame; this helper is opt-in because skipping a missing daily price would
    incorrectly bridge a trading gap.
    """
    values = np.full(len(panel), np.nan, dtype=float)
    entity = entity_level(panel)
    groups = (
        [np.arange(len(panel), dtype=int)]
        if entity is None
        else panel.groupby(level=entity, sort=False).indices.values()
    )
    source = series.to_numpy(dtype=float)
    for positions in groups:
        rows = np.asarray(positions, dtype=int)
        valid = np.isfinite(source[rows])
        if bool(valid.any()):
            values[rows[valid]] = fn(source[rows][valid])
    return pd.Series(values, index=panel.index)


def _per_entity_frame(
    panel: pd.DataFrame, name: str, fn: Callable[[pd.DataFrame], np.ndarray]
) -> pd.Series:
    """Apply a multi-column temporal transform per entity without crossing entity boundaries.

    Unlike :func:`_per_series`, the callee receives the entity's rows as a frame, so
    transforms depending on several columns (rolling maxima, price-impact ratios) observe an
    aligned, date-ordered slice of one entity at a time.
    """
    entity = entity_level(panel)
    values = np.full(len(panel), np.nan, dtype=float)
    groups = (
        [np.arange(len(panel), dtype=int)]
        if entity is None
        else panel.groupby(level=entity, sort=False).indices.values()
    )
    for positions in groups:
        rows = np.asarray(positions, dtype=int)
        values[rows] = fn(panel.iloc[rows])
    return pd.Series(values, index=panel.index, name=name)


def _window_params(spec: DictConfig) -> tuple[int, int | None]:
    """Read the trailing-window length and optional minimum finite-observation count."""
    window = int(spec.window)
    min_periods = spec.get("min_periods")
    return window, int(min_periods) if min_periods is not None else None


@feature_registry.register(
    "log_change", columns=[ColumnRef("column"), ColumnRef("times", required=False)]
)
def _log_change(panel: pd.DataFrame, spec: DictConfig) -> pd.Series:
    """Compute an entity-local log change, optionally after scaling by another column."""
    horizon = spec.get("horizon", 1)
    base = panel[spec.column]
    if spec.get("times") is not None:
        base = base * panel[spec.times]
    transform = _per_non_missing_series if bool(spec.get("skip_missing", False)) else _per_series
    values = transform(panel, base, lambda x: log_change(x, horizon=horizon))
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


@feature_registry.register("positive_indicator", columns=[ColumnRef("column")])
def _positive_indicator(panel: pd.DataFrame, spec: DictConfig) -> pd.Series:
    """Label each entity's horizon return direction as a classification target."""
    horizon = spec.get("horizon", 1)
    values = _per_series(
        panel, panel[spec.column], lambda x: positive_indicator(x, horizon=horizon)
    )
    return values.rename(f"positive_indicator_{spec.column}_{horizon}")


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


@feature_registry.register(
    "simple_difference", columns=[ColumnRef("current"), ColumnRef("previous")]
)
def _simple_difference(panel: pd.DataFrame, spec: DictConfig) -> pd.Series:
    """Compute ``current[t] / previous[t-1] - 1`` without crossing entities."""
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
        values[rows] = simple_difference(
            panel.iloc[rows][current].to_numpy(dtype=float),
            panel.iloc[rows][previous].to_numpy(dtype=float),
        )
    return pd.Series(
        values,
        index=panel.index,
        name=f"simple_difference_{current}_{previous}",
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


@feature_registry.register(
    "simple_ratio", columns=[ColumnRef("numerator"), ColumnRef("denominator")]
)
def _simple_ratio(panel: pd.DataFrame, spec: DictConfig) -> pd.Series:
    """Compute the element-wise simple return ``numerator / denominator - 1``."""
    values = simple_ratio(
        panel[spec.numerator].to_numpy(dtype=float), panel[spec.denominator].to_numpy(dtype=float)
    )
    return pd.Series(
        values, index=panel.index, name=f"simple_ratio_{spec.numerator}_{spec.denominator}"
    )


@feature_registry.register("ratio", columns=[ColumnRef("numerator"), ColumnRef("denominator")])
def _ratio(panel: pd.DataFrame, spec: DictConfig) -> pd.Series:
    values = ratio(
        panel[spec.numerator].to_numpy(dtype=float), panel[spec.denominator].to_numpy(dtype=float)
    )
    return pd.Series(values, index=panel.index, name=f"ratio_{spec.numerator}_{spec.denominator}")


@feature_registry.register(
    "net_ratio",
    columns=[
        ColumnRef("add", kind="list"),
        ColumnRef("subtract", kind="list", required=False),
        ColumnRef("denominator"),
    ],
)
def _net_ratio(panel: pd.DataFrame, spec: DictConfig) -> pd.Series:
    """Divide a signed sum of columns by a denominator column, element-wise."""
    add_columns = [str(column) for column in spec.add]
    subtract_columns = [str(column) for column in (spec.get("subtract") or [])]
    values = net_ratio(
        [panel[column].to_numpy(dtype=float) for column in add_columns],
        [panel[column].to_numpy(dtype=float) for column in subtract_columns],
        panel[spec.denominator].to_numpy(dtype=float),
    )
    numerator = "_".join(add_columns) + "".join(f"_minus_{column}" for column in subtract_columns)
    return pd.Series(
        values, index=panel.index, name=f"net_ratio_{numerator}_over_{spec.denominator}"
    )


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


@feature_registry.register("rolling_volatility", columns=[ColumnRef("column")])
def _rolling_volatility(panel: pd.DataFrame, spec: DictConfig) -> pd.Series:
    """Compute entity-local realized volatility of one-step log returns."""
    window, min_periods = _window_params(spec)
    values = _per_series(
        panel,
        panel[spec.column],
        lambda x: rolling_volatility(x, window=window, min_periods=min_periods),
    )
    return values.rename(f"rolling_volatility_{spec.column}_{window}")


@feature_registry.register("downside_deviation", columns=[ColumnRef("column")])
def _downside_deviation(panel: pd.DataFrame, spec: DictConfig) -> pd.Series:
    """Compute entity-local downside deviation of one-step log returns."""
    window, min_periods = _window_params(spec)
    values = _per_series(
        panel,
        panel[spec.column],
        lambda x: downside_deviation(x, window=window, min_periods=min_periods),
    )
    return values.rename(f"downside_deviation_{spec.column}_{window}")


@feature_registry.register("rolling_skewness", columns=[ColumnRef("column")])
def _rolling_skewness(panel: pd.DataFrame, spec: DictConfig) -> pd.Series:
    """Compute entity-local skewness of one-step log returns."""
    window, min_periods = _window_params(spec)
    values = _per_series(
        panel,
        panel[spec.column],
        lambda x: rolling_skewness(x, window=window, min_periods=min_periods),
    )
    return values.rename(f"rolling_skewness_{spec.column}_{window}")


@feature_registry.register("high_proximity", columns=[ColumnRef("value"), ColumnRef("high")])
def _high_proximity(panel: pd.DataFrame, spec: DictConfig) -> pd.Series:
    """Compute each entity's value relative to its trailing maximum high."""
    window, min_periods = _window_params(spec)
    return _per_entity_frame(
        panel,
        f"high_proximity_{spec.value}_{window}",
        lambda group: high_proximity(
            group[spec.value].to_numpy(dtype=float),
            group[spec.high].to_numpy(dtype=float),
            window=window,
            min_periods=min_periods,
        ),
    )


@feature_registry.register("amihud_illiquidity", columns=[ColumnRef("price"), ColumnRef("volume")])
def _amihud_illiquidity(panel: pd.DataFrame, spec: DictConfig) -> pd.Series:
    """Compute each entity's trailing Amihud price-impact illiquidity."""
    window, min_periods = _window_params(spec)
    log = bool(spec.get("log", False))
    suffix = "_log" if log else ""
    return _per_entity_frame(
        panel,
        f"amihud_illiquidity_{spec.price}_{spec.volume}_{window}{suffix}",
        lambda group: amihud_illiquidity(
            group[spec.price].to_numpy(dtype=float),
            group[spec.volume].to_numpy(dtype=float),
            window=window,
            min_periods=min_periods,
            log=log,
        ),
    )


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
