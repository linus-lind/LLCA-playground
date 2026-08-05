from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from matplotlib.figure import Figure
from matplotlib.table import Cell

from llca.analytics.comparison import ComparisonEvaluation
from llca.analytics.comparison.plots import build_comparison_figures
from llca.analytics.modules.analytics_config import PlotFormat


def style_report_table(
    cells: dict[tuple[int, int], Cell],
    *,
    header_rows: int = 1,
    blank_corner: bool = False,
) -> None:
    """Apply the report's shared table styling to a Matplotlib table's cells.

    Draws thin grey gridlines throughout, fills the top ``header_rows`` rows as a dark header
    band with white bold centred text, and shades alternating body rows a light zebra tint. With
    ``blank_corner`` set, the header cells above the row-label column (negative column index) are
    painted out so the band does not extend over the row labels.
    """
    for (row, column), cell in cells.items():
        cell.set_edgecolor("#D6DCE4")
        cell.set_linewidth(0.5)
        if row < header_rows and column >= 0:
            cell.set_facecolor("#1F3A5F")
            cell.set_text_props(color="white", fontweight="bold", ha="center", va="center")
        elif row < header_rows:
            if blank_corner:
                cell.set_facecolor("white")
                cell.set_edgecolor("white")
        elif (row - header_rows) % 2 == 1:
            cell.set_facecolor("#F3F6F9")


def build_report_figures(comparison: ComparisonEvaluation) -> list[tuple[str, Figure]]:
    """Build the portfolio and signal overlay figures for a comparison, single model or many.

    A one-model comparison simply produces overlays with a single series; the genuinely
    cross-model statistics figures are built elsewhere. Figures are returned open for the caller
    to save and then close.
    """
    return build_comparison_figures(comparison)


def save_figures(
    figures: Sequence[tuple[str, Figure]],
    directory: Path,
    formats: tuple[PlotFormat, ...],
    dpi: int,
) -> dict[str, tuple[Path, ...]]:
    """Write each figure into ``directory`` once per requested format.

    Returns a map from figure name to the tuple of files written for it.
    """
    artifacts: dict[str, tuple[Path, ...]] = {}
    for name, figure in figures:
        paths: list[Path] = []
        for output_format in formats:
            path = directory / f"{name}.{output_format}"
            figure.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
            paths.append(path)
        artifacts[name] = tuple(paths)
    return artifacts
