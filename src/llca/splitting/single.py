from collections.abc import Iterator

import pandas as pd

from llca.data.index_spec import time_level
from llca.data.modules.masked_panel import MaskedPanels
from llca.splitting.fold import Fold
from llca.splitting.slice_by_date import slice_by_date
from llca.splitting.splitter import Splitter


class SingleSplitter(Splitter[MaskedPanels]):
    """Create at most one chronological train/validation/test fold.

    Sizes and purge gaps count unique dates from the primary panel. ``lookback`` dates are
    prepended to train and validation inputs for sequence construction but remain outside
    the evaluation windows recorded in ``Fold``. No fold is yielded when the calendar is
    too short for the complete layout.
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
        """Yield the single feasible fold with synchronized panel slices."""
        calendar = panels[primary].values
        dates = pd.DatetimeIndex(calendar.index.get_level_values(time_level(calendar)))
        sorted_dates = dates.unique().sort_values()
        n = len(sorted_dates)

        block_size = self._block_size
        start = self.lookback
        if start + block_size <= n:
            train_end_pos = start + self.train_size - 1
            val_start_pos = train_end_pos + self.purge_size + 1
            val_end_pos = val_start_pos + self.val_size - 1
            test_start_pos = val_end_pos + self.purge_size + 1
            test_end_pos = test_start_pos + self.test_size - 1

            fold = Fold(
                index=1,
                train_start=sorted_dates[start],
                train_end=sorted_dates[train_end_pos],
                val_start=sorted_dates[val_start_pos],
                val_end=sorted_dates[val_end_pos],
                test_start=sorted_dates[test_start_pos],
                test_end=sorted_dates[test_end_pos],
            )

            train = slice_by_date(panels, sorted_dates[start - self.lookback], fold.train_end)
            val = slice_by_date(panels, sorted_dates[val_start_pos - self.lookback], fold.val_end)

            yield fold, train, val
