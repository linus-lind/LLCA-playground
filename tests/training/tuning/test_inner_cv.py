"""Deterministic end-anchored inner-fold boundaries, purge, lookback, and history guards."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from llca.data.modules.masked_panel import MaskedPanel, MaskedPanels
from llca.training.tuning.inner_cv import build_inner_folds
from llca.training.tuning.settings import InnerCvSettings


def _calendar(periods: int) -> pd.DatetimeIndex:
    return pd.bdate_range("2021-01-01", periods=periods)


def _panels(dates: pd.DatetimeIndex) -> MaskedPanels:
    index = pd.MultiIndex.from_product([dates, [1]], names=["date", "entity"])
    values = pd.DataFrame({"f": np.arange(len(index), dtype=float)}, index=index)
    panel = MaskedPanel(
        values=values,
        observed=pd.DataFrame(True, index=index, columns=values.columns),
        age=pd.DataFrame(0.0, index=index, columns=values.columns),
        segment=pd.Series(index.get_level_values("entity"), index=index),
    )
    return {"daily_values": panel}


def _fold_dates(panels: MaskedPanels) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(panels["daily_values"].values.index.get_level_values("date")).unique()


class InnerFoldEndAnchorTest(unittest.TestCase):
    def test_exact_boundaries_are_end_anchored_with_purge_gap(self) -> None:
        dates = _calendar(20)
        folds = build_inner_folds(
            _panels(dates),
            "daily_values",
            InnerCvSettings(
                train_size=5, val_size=2, step_size=3, purge=1, lookback=0, min_folds=1
            ),
        )

        # block = 5+1+2 = 8; newest_train_start = 20-8 = 12; folds = 12//3 + 1 = 5.
        self.assertEqual(len(folds), 5)
        first = folds[0]
        self.assertEqual(list(_fold_dates(first.train)), list(dates[0:5]))
        self.assertEqual((first.val_start, first.val_end), (dates[6], dates[7]))
        self.assertEqual(list(_fold_dates(first.validation)), list(dates[6:8]))
        # Folds step backward from the end; chronological order, indices 0..4.
        self.assertEqual([fold.index for fold in folds], [0, 1, 2, 3, 4])
        self.assertEqual(list(_fold_dates(folds[1].train)), list(dates[3:8]))
        self.assertEqual((folds[1].val_start, folds[1].val_end), (dates[9], dates[10]))
        # The newest fold validates through the last available outer-train date.
        self.assertEqual(folds[-1].val_end, dates[19])

    def test_no_fold_references_a_date_beyond_the_training_window(self) -> None:
        dates = _calendar(20)
        folds = build_inner_folds(
            _panels(dates),
            "daily_values",
            InnerCvSettings(
                train_size=5, val_size=2, step_size=3, purge=1, lookback=0, min_folds=1
            ),
        )
        outer_max = dates.max()
        for fold in folds:
            self.assertLessEqual(_fold_dates(fold.train).max(), outer_max)
            self.assertLessEqual(_fold_dates(fold.validation).max(), outer_max)
            self.assertLessEqual(fold.val_end, outer_max)

    def test_lookback_prepends_warmup_without_moving_the_scored_window(self) -> None:
        dates = _calendar(20)
        folds = build_inner_folds(
            _panels(dates),
            "daily_values",
            InnerCvSettings(
                train_size=5, val_size=2, step_size=10, purge=1, lookback=3, min_folds=1
            ),
        )
        # newest_train_start = 12; step 10 -> 2 folds beginning at 2 and 12.
        self.assertEqual(len(folds), 2)
        first = folds[0]
        # Warmup prepends 3 rows (clamped at the calendar start) but the scored window is fixed.
        self.assertEqual(list(_fold_dates(first.train)), list(dates[0:7]))
        self.assertEqual((first.val_start, first.val_end), (dates[8], dates[9]))
        self.assertEqual(list(_fold_dates(first.validation)), list(dates[5:10]))

    def test_insufficient_history_yields_no_folds(self) -> None:
        dates = _calendar(20)
        folds = build_inner_folds(
            _panels(dates),
            "daily_values",
            InnerCvSettings(
                train_size=100, val_size=2, step_size=3, purge=1, lookback=0, min_folds=1
            ),
        )
        self.assertEqual(folds, [])


if __name__ == "__main__":
    unittest.main()
