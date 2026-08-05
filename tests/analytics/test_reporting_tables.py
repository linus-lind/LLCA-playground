import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
from matplotlib import pyplot as plt

from llca.analytics.reporting.table_rendering import (
    _GROUP_HEADER_HORIZONTAL_PADDING_POINTS,
    _auto_size_columns,
    _export_table,
    _fit_group_header_widths,
    _formatted_frame,
    _group_label_width,
)
from llca.analytics.reporting.tables import MetricSpec, _metric_table


class ReportingTableTest(unittest.TestCase):
    def test_group_header_sets_a_minimum_without_shrinking_wider_subheaders(self) -> None:
        figure, axis = plt.subplots(figsize=(5.0, 2.0), dpi=100)
        self.addCleanup(plt.close, figure)
        axis.axis("off")
        artist = axis.table(
            cellText=[
                ["", "", "", ""],
                ["A", "B", "Very long model name alpha", "Very long model name beta"],
                ["1", "2", "3", "4"],
            ],
            cellLoc="center",
            loc="center",
        )
        artist.auto_set_font_size(False)
        artist.set_fontsize(8.5)
        cells = artist.get_celld()
        for (row, _column), cell in cells.items():
            if row < 2:
                cell.set_text_props(fontweight="bold")
        _auto_size_columns(figure, cells, range(4))
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()

        def span_width(start: int, end: int) -> float:
            boxes = [
                cells[(0, column)].get_window_extent(renderer) for column in range(start, end + 1)
            ]
            return float(max(box.x1 for box in boxes) - min(box.x0 for box in boxes))

        long_label = "Annualized long return contribution"
        spans = [(0, 1, long_label), (2, 3, "Fit")]
        narrow_before = span_width(0, 1)
        wide_before = span_width(2, 3)
        required = _group_label_width(figure, renderer, long_label) + (
            2.0 * _GROUP_HEADER_HORIZONTAL_PADDING_POINTS * float(figure.dpi) / 72.0
        )
        self.assertLess(narrow_before, required)

        _fit_group_header_widths(figure, axis, cells, spans)
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()

        self.assertGreaterEqual(span_width(0, 1), required - 0.5)
        self.assertGreater(span_width(0, 1), narrow_before)
        self.assertAlmostEqual(span_width(2, 3), wide_before, places=6)

    def test_metric_pvalue_becomes_inline_stars_without_extra_row(self) -> None:
        source = pd.DataFrame({"estimate": [0.25, 0.10]}, index=["A", "B"])
        significance = pd.DataFrame({"estimate_p": [0.009, 0.20]}, index=["A", "B"])
        table = _metric_table(
            source,
            (MetricSpec("estimate", "Estimate"),),
            name="estimate",
            title="Estimate",
            caption="Estimate with inline significance.",
            significance=significance,
            pvalue_for=(("estimate", "estimate_p"),),
        )
        assert table is not None
        self.assertEqual(list(table.frame.index), ["Estimate"])
        self.assertFalse(any("p-value" in str(label) for label in table.frame.index))
        display = _formatted_frame(table)
        self.assertIn("(***)", str(display.loc["Estimate", "A"]))
        self.assertNotIn("(", str(display.loc["Estimate", "B"]))

    def test_csv_export_preserves_inline_significance_markers(self) -> None:
        source = pd.DataFrame({"estimate": [0.25]}, index=["A"])
        significance = pd.DataFrame({"estimate_p": [0.009]}, index=["A"])
        table = _metric_table(
            source,
            (MetricSpec("estimate", "Estimate"),),
            name="estimate",
            title="Estimate",
            caption="Estimate with inline significance.",
            significance=significance,
            pvalue_for=(("estimate", "estimate_p"),),
        )
        assert table is not None

        with TemporaryDirectory() as directory:
            (path,) = _export_table(table, Path(directory), ("csv",), dpi=72)
            exported = pd.read_csv(path, index_col=0, dtype=str)

        self.assertIn("(***)", exported.loc["Estimate", "A"])


if __name__ == "__main__":
    unittest.main()
