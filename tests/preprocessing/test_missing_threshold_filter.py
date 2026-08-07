"""Regression tests for the composite/per-entity sparsity filter."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from llca.preprocessing.missing_threshold_filter import missing_threshold_filter


def _entity_panel() -> pd.DataFrame:
    """Entity 11 fully missing in ``a``; entity 22 fully present."""
    rows = []
    for day in pd.date_range("2020-01-01", periods=4):
        rows.append((day, 11, np.nan))
        rows.append((day, 22, 1.0))
    index = pd.MultiIndex.from_tuples(
        [(day, ent) for day, ent, _ in rows], names=["date", "instrument_id"]
    )
    return pd.DataFrame({"a": [value for *_, value in rows]}, index=index)


class MissingThresholdFilterTest(unittest.TestCase):
    def test_per_entity_mode_drops_sparse_entity_without_crashing(self) -> None:
        # Single grouping key (entity level, no configured subgroup columns). Previously
        # raised TypeError because the row-group label was built as a 1-tuple MultiIndex.
        result = missing_threshold_filter(_entity_panel(), threshold=0.5, subgroup_keys=[])
        kept = sorted(result.index.get_level_values("instrument_id").unique().tolist())
        self.assertEqual(kept, [22])

    def test_date_only_single_string_key(self) -> None:
        dates = pd.date_range("2020-01-01", periods=6, name="date")
        panel = pd.DataFrame(
            {"a": [np.nan, np.nan, np.nan, 1.0, 2.0, 3.0], "grp": ["x", "x", "x", "y", "y", "y"]},
            index=dates,
        )
        result = missing_threshold_filter(panel, threshold=0.5, subgroup_keys=["grp"])
        self.assertEqual(sorted(result["grp"].unique().tolist()), ["y"])

    def test_composite_keys_drop_worst_group(self) -> None:
        panel = _entity_panel()
        panel = panel.assign(sector=["A"] * len(panel))
        result = missing_threshold_filter(panel, threshold=0.5, subgroup_keys=["sector"])
        kept = sorted(result.index.get_level_values("instrument_id").unique().tolist())
        self.assertEqual(kept, [22])

    def test_no_keys_drops_sparse_columns(self) -> None:
        dates = pd.date_range("2020-01-01", periods=4, name="date")
        panel = pd.DataFrame(
            {"a": [1.0, np.nan, np.nan, np.nan], "b": [1.0, 2.0, 3.0, 4.0]}, index=dates
        )
        result = missing_threshold_filter(panel, threshold=0.5, subgroup_keys=[])
        self.assertEqual(list(result.columns), ["b"])


if __name__ == "__main__":
    unittest.main()
