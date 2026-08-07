from __future__ import annotations

from collections.abc import Iterator

import pandas as pd

from llca.data.index_spec import time_level
from llca.data.modules.masked_panel import MaskedPanels
from llca.splitting.fold import Fold
from llca.splitting.slice_by_date import slice_by_date
from llca.splitting.splitter import Splitter


class WalkForwardSplitter(Splitter[MaskedPanels]):
    """Generate fixed-width, end-anchored chronological folds over a shared calendar.

    Each fold contains train, purge, validation, purge, and test date ranges. The complete
    block ``block = train + 2*purge + val + test`` is anchored to the newest observation and
    earlier folds step **backward** by ``step_size``, so the final fold's last test date is
    always ``N - 1``. With ``F = (N - block) // step_size + 1`` feasible folds, the fold at
    chronological index ``i`` (``0`` oldest) begins at::

        train_start(i) = N - block - (F - 1 - i) * step_size

    and its remaining boundaries follow the fixed layout. Folds are yielded oldest-first; any
    excess observations fall before the oldest fold and are dropped from the beginning, never
    from the end. ``lookback`` rows are prepended to the train and validation input slices for
    sequence construction and excluded from the recorded ``Fold`` boundaries. No fold is
    yielded when the calendar is shorter than ``block``.
    """

    @property
    def name(self) -> str:
        return "walk_forward"

    def __init__(
        self,
        train_size: int,
        val_size: int,
        test_size: int,
        purge_size: int,
        step_size: int,
        *,
        lookback: int = 0,
    ) -> None:
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
        self.purge_size = purge_size
        self.step_size = step_size
        self.lookback = lookback

    @property
    def _block_size(self) -> int:
        return self.train_size + self.purge_size + self.val_size + self.purge_size + self.test_size

    def split(
        self, panels: MaskedPanels, primary: str
    ) -> Iterator[tuple[Fold, MaskedPanels, MaskedPanels]]:
        """Yield every complete rolling fold, oldest first, anchored to the newest date."""
        calendar = panels[primary].values
        dates = pd.DatetimeIndex(calendar.index.get_level_values(time_level(calendar)))
        sorted_dates = dates.unique().sort_values()
        n = len(sorted_dates)
        block_size = self._block_size

        if block_size > n:
            return

        newest_train_start = n - block_size
        n_folds = newest_train_start // self.step_size + 1

        for fold_index in range(n_folds):
            offset = (n_folds - 1 - fold_index) * self.step_size
            train_start_pos = newest_train_start - offset
            train_end_pos = train_start_pos + self.train_size - 1
            val_start_pos = train_end_pos + self.purge_size + 1
            val_end_pos = val_start_pos + self.val_size - 1
            test_start_pos = val_end_pos + self.purge_size + 1
            test_end_pos = test_start_pos + self.test_size - 1

            fold = Fold(
                index=fold_index,
                train_start=sorted_dates[train_start_pos],
                train_end=sorted_dates[train_end_pos],
                val_start=sorted_dates[val_start_pos],
                val_end=sorted_dates[val_end_pos],
                test_start=sorted_dates[test_start_pos],
                test_end=sorted_dates[test_end_pos],
            )

            train = slice_by_date(
                panels, sorted_dates[max(0, train_start_pos - self.lookback)], fold.train_end
            )
            val = slice_by_date(
                panels, sorted_dates[max(0, val_start_pos - self.lookback)], fold.val_end
            )

            yield fold, train, val
