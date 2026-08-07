from __future__ import annotations

from collections.abc import Iterator

import pandas as pd

from llca.data.index_spec import time_level
from llca.data.modules.masked_panel import MaskedPanels
from llca.splitting.fold import Fold
from llca.splitting.slice_by_date import slice_by_date
from llca.splitting.splitter import Splitter


class SingleSplitter(Splitter[MaskedPanels]):
    """Create at most one end-anchored chronological train/validation/test fold.

    The scored layout ``train | purge | validation | purge | test`` is laid out **backward
    from the newest observation**: the final scored test date is always the last date on the
    primary calendar. Given ``N`` ordered unique dates and ``block = train + 2*purge + val +
    test`` the inclusive scored positions are::

        test_end   = N - 1
        test_start = N - test
        val_end    = N - test - purge - 1
        val_start  = N - test - purge - val
        train_end  = N - test - purge - val - purge - 1
        train_start= N - block

    Any excess observations therefore fall *before* ``train_start`` and are dropped from the
    scored split; the last ``lookback`` of them are still attached to the train slice as input
    history. Because the scored positions are anchored to ``N - 1`` and offset only by the
    fixed sizes and purge, they are invariant to ``lookback`` -- two models differing only in
    history depth score exactly the same dates. ``lookback`` dates are prepended to the train
    and validation input slices for sequence construction and stay outside the ``Fold``
    evaluation windows; the test slice is materialized later by evaluation. No fold is yielded
    when the calendar is shorter than ``block``.
    """

    @property
    def name(self) -> str:
        return "single"

    def __init__(
        self,
        train_size: int,
        val_size: int,
        test_size: int,
        purge_size: int,
        *,
        lookback: int = 0,
    ) -> None:
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
        self.purge_size = purge_size
        self.lookback = lookback

    @property
    def _block_size(self) -> int:
        return self.train_size + self.purge_size + self.val_size + self.purge_size + self.test_size

    def split(
        self, panels: MaskedPanels, primary: str
    ) -> Iterator[tuple[Fold, MaskedPanels, MaskedPanels]]:
        """Yield the single end-anchored fold with synchronized panel slices."""
        calendar = panels[primary].values
        dates = pd.DatetimeIndex(calendar.index.get_level_values(time_level(calendar)))
        sorted_dates = dates.unique().sort_values()
        n = len(sorted_dates)

        if self._block_size > n:
            return

        test_end_pos = n - 1
        test_start_pos = test_end_pos - self.test_size + 1
        val_end_pos = test_start_pos - self.purge_size - 1
        val_start_pos = val_end_pos - self.val_size + 1
        train_end_pos = val_start_pos - self.purge_size - 1
        train_start_pos = train_end_pos - self.train_size + 1

        fold = Fold(
            index=0,
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
