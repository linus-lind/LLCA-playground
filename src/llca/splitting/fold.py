from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True, slots=True)
class Fold:
    """Describe the inclusive train, validation, and test evaluation windows of one fold.

    These dates exclude any lookback rows prepended to model inputs. ``index`` is the
    stable fold identifier used by tracking and model registration.
    """

    index: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    val_start: pd.Timestamp
    val_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
