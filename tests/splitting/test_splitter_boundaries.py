"""Deterministic boundary, end-anchoring, and leakage-invariant tests for the splitters.

Expected positions are derived by hand from a fixed toy calendar, independently of the splitter
implementation, so an anchoring or off-by-one regression fails loudly. Both splitters are
end-anchored: the newest observation is always the final scored test date, excess observations
are dropped from the beginning, and ``lookback`` prepends input history without moving any scored
boundary.
"""

from __future__ import annotations

import unittest
from typing import cast

import numpy as np
import pandas as pd

from llca.data.modules.masked_panel import MaskedPanel, MaskedPanels
from llca.models.utils.sequences import SequenceInput, build_sequences
from llca.splitting.single import SingleSplitter
from llca.splitting.walk_forward import WalkForwardSplitter

_CALENDAR = pd.bdate_range("2020-01-01", periods=80)


def _panel(n_dates: int, entities: tuple[int, ...] = (1,)) -> MaskedPanels:
    """One synthetic panel over ``n_dates`` business days and the given entities."""
    dates = _CALENDAR[:n_dates]
    index = pd.MultiIndex.from_product([dates, entities], names=["date", "entity"])
    values = pd.DataFrame(
        np.arange(len(index), dtype=float).reshape(-1, 1), index=index, columns=["f"]
    )
    return {
        "p": MaskedPanel(
            values=values,
            observed=pd.DataFrame(True, index=index, columns=["f"]),
            age=pd.DataFrame(0.0, index=index, columns=["f"]),
            segment=pd.Series(index.get_level_values("entity"), index=index),
        )
    }


def _pos(timestamp: pd.Timestamp) -> int:
    return int(cast(int, _CALENDAR.get_loc(timestamp)))


def _slice_positions(panels: MaskedPanels) -> tuple[int, int]:
    dates = pd.DatetimeIndex(panels["p"].values.index.get_level_values("date"))
    return _pos(dates.min()), _pos(dates.max())


def _windows(fold: object) -> tuple[int, ...]:
    return (
        _pos(fold.train_start),  # type: ignore[attr-defined]
        _pos(fold.train_end),  # type: ignore[attr-defined]
        _pos(fold.val_start),  # type: ignore[attr-defined]
        _pos(fold.val_end),  # type: ignore[attr-defined]
        _pos(fold.test_start),  # type: ignore[attr-defined]
        _pos(fold.test_end),  # type: ignore[attr-defined]
    )


class SingleSplitterEndAnchorTest(unittest.TestCase):
    def test_exact_windows_and_slices_are_end_anchored(self) -> None:
        splitter = SingleSplitter(train_size=10, val_size=4, test_size=4, purge_size=2, lookback=3)
        fold, train, val = next(iter(splitter.split(_panel(30), "p")))

        # block = 10+2+4+2+4 = 22; anchored to the last date (29), laid out backward.
        self.assertEqual(_windows(fold), (8, 17, 20, 23, 26, 29))
        self.assertEqual(_pos(fold.test_end), 29)  # newest observation
        self.assertEqual(_pos(fold.val_start) - _pos(fold.train_end) - 1, 2)  # purge gap
        self.assertEqual(_pos(fold.test_start) - _pos(fold.val_end) - 1, 2)
        # Train/val input slices prepend exactly ``lookback`` rows; test is not materialized.
        self.assertEqual(_slice_positions(train), (5, 17))
        self.assertEqual(_slice_positions(val), (17, 23))

    def test_lookback_does_not_move_scored_boundaries(self) -> None:
        scored = None
        for lookback in (0, 3, 9):
            splitter = SingleSplitter(10, 4, 4, purge_size=2, lookback=lookback)
            fold, train, _ = next(iter(splitter.split(_panel(30), "p")))
            if scored is None:
                scored = _windows(fold)
            self.assertEqual(_windows(fold), scored)
            # Only the prepended history depth changes.
            self.assertEqual(_slice_positions(train)[0], max(0, 8 - lookback))

    def test_purge_change_keeps_the_test_window_fixed(self) -> None:
        # The test window is the last ``test_size`` dates regardless of purge.
        for purge in (1, 2, 4):
            splitter = SingleSplitter(10, 4, 4, purge_size=purge, lookback=0)
            fold, _, _ = next(iter(splitter.split(_panel(40), "p")))
            self.assertEqual((_pos(fold.test_start), _pos(fold.test_end)), (36, 39))

    def test_excess_is_dropped_from_the_beginning(self) -> None:
        # block = 22; one excess observation sits before the scored train window.
        splitter = SingleSplitter(10, 4, 4, purge_size=2, lookback=0)
        fold, _, _ = next(iter(splitter.split(_panel(23), "p")))
        self.assertEqual(_pos(fold.train_start), 1)
        self.assertEqual(_pos(fold.test_end), 22)

    def test_exact_fit_starts_at_zero(self) -> None:
        splitter = SingleSplitter(10, 4, 4, purge_size=2, lookback=5)
        fold, train, _ = next(iter(splitter.split(_panel(22), "p")))
        self.assertEqual(_pos(fold.train_start), 0)
        self.assertEqual(_pos(fold.test_end), 21)
        # Lookback is clamped to the available history at the calendar start.
        self.assertEqual(_slice_positions(train)[0], 0)

    def test_no_fold_when_calendar_shorter_than_block(self) -> None:
        splitter = SingleSplitter(10, 4, 4, purge_size=2, lookback=0)
        self.assertEqual(list(splitter.split(_panel(21), "p")), [])


class WalkForwardEndAnchorTest(unittest.TestCase):
    def test_folds_step_backward_and_final_fold_ends_on_newest(self) -> None:
        splitter = WalkForwardSplitter(
            train_size=8, val_size=3, test_size=3, purge_size=1, step_size=5, lookback=2
        )
        folds = [fold for fold, _, _ in splitter.split(_panel(40), "p")]

        expected = [
            (4, 11, 13, 15, 17, 19),
            (9, 16, 18, 20, 22, 24),
            (14, 21, 23, 25, 27, 29),
            (19, 26, 28, 30, 32, 34),
            (24, 31, 33, 35, 37, 39),
        ]
        self.assertEqual([_windows(f) for f in folds], expected)
        self.assertEqual([f.index for f in folds], [0, 1, 2, 3, 4])
        self.assertEqual(_pos(folds[-1].test_end), 39)  # newest observation

        for f in folds:
            self.assertEqual(_pos(f.val_start) - _pos(f.train_end) - 1, 1)
            self.assertEqual(_pos(f.test_start) - _pos(f.val_end) - 1, 1)
        starts = [_pos(f.train_start) for f in folds]
        self.assertTrue(all(b - a == 5 for a, b in zip(starts, starts[1:], strict=False)))

    def test_lookback_does_not_move_scored_fold_dates(self) -> None:
        shallow = WalkForwardSplitter(8, 3, 3, purge_size=1, step_size=5, lookback=2)
        deep = WalkForwardSplitter(8, 3, 3, purge_size=1, step_size=5, lookback=20)
        shallow_folds = [_windows(f) for f, _, _ in shallow.split(_panel(40), "p")]
        deep_folds = [_windows(f) for f, _, _ in deep.split(_panel(40), "p")]
        self.assertEqual(shallow_folds, deep_folds)

    def test_no_fold_when_calendar_shorter_than_block(self) -> None:
        splitter = WalkForwardSplitter(8, 3, 3, purge_size=1, step_size=5, lookback=0)
        self.assertEqual(list(splitter.split(_panel(15), "p")), [])  # block = 16


class SplitterLeakageInvariantTest(unittest.TestCase):
    """Purge covers the label horizon, and warmup lands the first target on the scored start."""

    def test_one_step_label_cannot_realize_inside_validation_or_test(self) -> None:
        # With purge == label horizon (1), the last train label realizes in the purge band,
        # never on a scored validation date; likewise between validation and test.
        splitter = SingleSplitter(10, 4, 4, purge_size=1, lookback=0)
        fold, _, _ = next(iter(splitter.split(_panel(30), "p")))
        self.assertGreaterEqual(_pos(fold.val_start) - _pos(fold.train_end), 2)
        self.assertGreaterEqual(_pos(fold.test_start) - _pos(fold.val_end), 2)

    def test_lookback_of_window_minus_one_lands_first_target_on_val_start(self) -> None:
        # A model whose causal window is W needs W-1 history rows; the splitter's lookback is
        # exactly required_lookback == W-1, so the first constructible target is val_start with
        # no dropped and no purge-band predictions.
        for window in (3, 5, 8):
            splitter = SingleSplitter(15, 6, 6, purge_size=1, lookback=window - 1)
            fold, _, val = next(iter(splitter.split(_panel(60), "p")))
            _, index = build_sequences(
                val["p"],
                [SequenceInput("f", ["f"], windowed=True)],
                sequence_length=window,
                buffer_size=0,
            )
            first = _pos(pd.DatetimeIndex(index.get_level_values("date")).min())
            self.assertEqual(first, _pos(fold.val_start), f"window={window}")


if __name__ == "__main__":
    unittest.main()
