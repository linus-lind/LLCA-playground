from collections.abc import Sequence

import numpy as np
import pandas as pd

from llca.data.index_spec import entity_level, time_level
from llca.data.modules.masked_panel import MaskedPanel, MaskedPanels
from llca.data.modules.panels import Panels


def _segment_labels(
    grid: pd.Index, time: str, entity: str | None, spell: np.ndarray | None
) -> pd.Series:
    """Assign contiguous regime identifiers on an active time-entity grid.

    A new segment starts when the entity or any configured activity-spell boundary
    changes. Labels are restored to the original grid order after chronological grouping.
    """
    frame = pd.DataFrame({"__pos": np.arange(len(grid)), time: grid.get_level_values(time)})
    frame["__entity"] = grid.get_level_values(entity) if entity is not None else 0
    bound_columns: list[str] = []
    if spell is not None:
        bound_columns = [f"__bound{axis}" for axis in range(spell.shape[1])]
        for axis, column in enumerate(bound_columns):
            frame[column] = spell[:, axis]

    ordered = frame.sort_values(["__entity", time])
    keys = ordered[["__entity", *bound_columns]]
    changed = (keys != keys.shift()).any(axis=1)
    ordered["__seg"] = changed.cumsum().to_numpy() - 1

    labels = ordered.sort_values("__pos")["__seg"].to_numpy()
    return pd.Series(labels, index=grid, name="segment")


def _active_grid(
    primary_ds: pd.DataFrame, primary_feat: pd.DataFrame, subgroups: Sequence[tuple[str, str]]
) -> tuple[pd.Index, pd.Series]:
    """Restrict the primary feature grid to configured active date intervals.

    Each available start/end pair contributes an inclusive activity constraint. The
    result is the common row index used by all feature panels and a segment label per row.
    """
    index = primary_feat.index
    time = time_level(primary_feat)
    entity = entity_level(primary_feat)
    dates = pd.DatetimeIndex(index.get_level_values(time)).to_numpy()

    active = np.ones(len(index), dtype=bool)
    bounds: list[np.ndarray] = []
    for start_column, end_column in subgroups:
        if start_column not in primary_ds.columns or end_column not in primary_ds.columns:
            continue
        start = pd.to_datetime(primary_ds[start_column]).to_numpy()
        end = pd.to_datetime(primary_ds[end_column]).to_numpy()
        active &= ~pd.isna(start) & ~pd.isna(end) & (dates >= start) & (dates <= end)
        bounds.extend((start, end))

    spell = np.stack(bounds, axis=1) if bounds else None
    grid = index[active]
    segment = _segment_labels(grid, time, entity, spell[active] if spell is not None else None)
    return grid, segment


def _age(observed: pd.Series, segment: pd.Series) -> pd.Series:
    position = observed.groupby(segment).cumcount()
    last_observed = position.where(observed.to_numpy()).groupby(segment).ffill()
    return (position - last_observed).fillna(-1).astype(int)


def _first_effective(source: pd.Series, segment: pd.Series) -> pd.Series:
    previous = source.groupby(segment).shift()
    return source.notna() & (source != previous)


def _asof_align(
    series: pd.Series, grid: pd.Index, time: str, by: str | None
) -> tuple[pd.Series, pd.Series]:
    """Backward-align one series to the grid and retain each value's source date."""
    right = series.rename("__value").reset_index()
    right["__source"] = right[time]
    right = right.sort_values(time)
    left = grid.to_frame(index=False).sort_values(time)
    merged = pd.merge_asof(left, right, on=time, by=by, direction="backward")
    merged = merged.set_index(list(grid.names)).reindex(grid)
    return merged["__value"], merged["__source"]


def _mask_panel(
    panel: pd.DataFrame, grid: pd.Index, segment: pd.Series, time: str, entity: str | None
) -> MaskedPanel:
    """Align one panel to the common grid and derive availability metadata.

    Values are carried backward-as-of within an entity but never across a segment start.
    ``observed[R, F]`` marks the first effective appearance of a source value, while
    ``age[R, F]`` counts grid steps since that appearance and uses ``-1`` before any value
    has been observed in the segment.
    """
    by = entity if entity is not None and entity in panel.index.names else None
    grid_date = pd.Series(pd.DatetimeIndex(grid.get_level_values(time)), index=grid)
    segment_start = grid_date.groupby(segment).transform("min")
    values: dict[str, pd.Series] = {}
    observed: dict[str, pd.Series] = {}
    age: dict[str, pd.Series] = {}

    for column in panel.columns:
        value, source = _asof_align(panel[column].dropna(), grid, time, by)
        within = source >= segment_start
        value = value.where(within)
        source = source.where(within)
        obs = _first_effective(source, segment)
        values[str(column)] = value
        observed[str(column)] = obs
        age[str(column)] = _age(obs, segment)

    return MaskedPanel(
        values=pd.DataFrame(values, index=grid),
        observed=pd.DataFrame(observed, index=grid),
        age=pd.DataFrame(age, index=grid),
        segment=segment,
    )


def align_and_mask(
    datasets: Panels,
    feature_panels: Panels,
    primary: str,
    subgroups: Sequence[tuple[str, str]] = (),
) -> MaskedPanels:
    """Align all feature datasets to one leakage-safe primary grid.

    The primary dataset defines eligible ``R`` rows and segments. Every returned
    ``MaskedPanel`` shares that index and segment Series, enabling column-wise composition
    and synchronized slicing later in the pipeline.
    """
    grid, segment = _active_grid(datasets[primary], feature_panels[primary], subgroups)
    time = str(grid.names[0])
    entity = str(grid.names[1]) if grid.nlevels > 1 else None
    return {
        name: _mask_panel(panel, grid, segment, time, entity)
        for name, panel in feature_panels.items()
    }
