"""Leakage-safe inner walk-forward folds generated entirely within the outer training data.

This reuses the repository's chronological date-slicing primitive but deliberately does not go
through :class:`~llca.splitting.walk_forward.WalkForwardSplitter`, whose train/validation/test
geometry carries outer-evaluation semantics that do not belong inside model fitting. Each fold
advances a fixed train/validation block by ``step_size``; ``purge`` dates separate train from
validation so a forward-looking label cannot cross the boundary, and ``lookback`` warmup dates
are prepended to each slice for models that rebuild history-dependent inputs.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from llca.data.index_spec import time_level
from llca.data.modules.masked_panel import MaskedPanels
from llca.splitting.slice_by_date import slice_by_date
from llca.training.tuning.settings import InnerCvSettings


@dataclass(frozen=True, slots=True)
class InnerFold:
    """One inner fold: its train and validation panels plus the scored validation boundary.

    ``validation`` may include prepended ``lookback`` warmup rows; ``val_start`` and ``val_end``
    are the inclusive dates actually scored, excluding that warmup.
    """

    index: int
    train: MaskedPanels
    validation: MaskedPanels
    val_start: pd.Timestamp
    val_end: pd.Timestamp


def build_inner_folds(
    panels: MaskedPanels, primary: str, settings: InnerCvSettings
) -> list[InnerFold]:
    """Materialize every complete inner fold in chronological order over the outer train data.

    The inner geometry is end-anchored to mirror the outer split: the newest fold's validation
    ends on the last date legally available inside the outer training window, and earlier folds
    step backward by ``step_size``. Given ``M`` outer-train dates and ``block = train + purge +
    val`` there are ``F = (M - block) // step_size + 1`` folds; the fold at chronological index
    ``i`` (``0`` oldest) begins at ``M - block - (F - 1 - i) * step_size``. Any inner remainder
    is dropped from the oldest side, never the newest. Folds are sliced once and reused across
    all candidates, and no fold references a date beyond the outer training window. ``lookback``
    warmup rows are prepended to each slice without moving the scored validation boundary (zero
    for the precomputed-feature baselines that currently drive selection).
    """
    calendar = panels[primary].values
    dates = (
        pd.DatetimeIndex(calendar.index.get_level_values(time_level(calendar)))
        .unique()
        .sort_values()
    )
    total = len(dates)
    block = settings.train_size + settings.purge + settings.val_size
    if block > total:
        return []

    newest_train_start = total - block
    n_folds = newest_train_start // settings.step_size + 1
    folds: list[InnerFold] = []
    for index in range(n_folds):
        offset = (n_folds - 1 - index) * settings.step_size
        train_start_pos = newest_train_start - offset
        train_end_pos = train_start_pos + settings.train_size - 1
        val_start_pos = train_end_pos + settings.purge + 1
        val_end_pos = val_start_pos + settings.val_size - 1

        train = slice_by_date(
            panels, dates[max(0, train_start_pos - settings.lookback)], dates[train_end_pos]
        )
        validation = slice_by_date(
            panels, dates[max(0, val_start_pos - settings.lookback)], dates[val_end_pos]
        )
        folds.append(
            InnerFold(
                index=index,
                train=train,
                validation=validation,
                val_start=dates[val_start_pos],
                val_end=dates[val_end_pos],
            )
        )
    return folds
