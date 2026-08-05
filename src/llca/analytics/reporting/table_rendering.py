"""Render publication tables to styled figures and export the full paper-ready report.

The table *content* is built in :mod:`llca.analytics.reporting.tables`; this module owns the
Matplotlib layout (grouped headers, auto-sized columns, alternating rows), the CSV/TeX/figure
exporters, and :func:`export_publication_report`, which assembles every table and figure
artifact for one comparison into an immutable output directory.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path
from textwrap import fill
from uuid import uuid4

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.backend_bases import RendererBase
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.table import Cell
from matplotlib.transforms import Bbox, TransformedBbox

from llca.analytics.comparison import ComparisonEvaluation, ComparisonInference
from llca.analytics.modules.analytics_config import ModelEvaluationConfig, TableFormat
from llca.analytics.reporting.factor_tables import (
    FactorAnalysis,
    build_additional_statistics_table,
    build_alpha_difference_matrix,
    build_factor_alpha_tables,
    build_factor_figures,
)
from llca.analytics.reporting.figures import save_figures, style_report_table
from llca.analytics.reporting.statistics_figures import build_statistics_comparison_figure
from llca.analytics.reporting.statistics_tables import build_statistical_tables
from llca.analytics.reporting.table_types import PublicationReport, PublicationTable
from llca.analytics.reporting.tables import build_publication_tables
from llca.analytics.stats.statistics import significance_marker, significance_marker_bold

_GROUP_HEADER_FONT_SIZE = 8.5
_GROUP_HEADER_HORIZONTAL_PADDING_POINTS = 6.0


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-") or "report"


def _formatted_frame(table: PublicationTable, *, bold_stars: bool = False) -> pd.DataFrame:
    """Convert a table's numeric frame into its formatted display strings.

    Each cell is formatted per its column or row number format (percent, integer, p-value, or
    decimal), with missing values shown as ``--``. Where the table carries paired p-values, the
    corresponding significance stars are appended to the estimate and the p-value itself stays
    hidden. ``bold_stars`` selects the mathtext-bold markers for figure rendering; CSV and TeX
    keep the plain-text form.
    """
    marker = significance_marker_bold if bold_stars else significance_marker
    frame = table.frame.copy().astype(object)
    for position in range(len(frame)):
        for column_position, _column in enumerate(frame.columns):
            number_format = (
                table.column_formats[column_position]
                if table.column_formats
                else table.row_formats[position]
            )
            value = frame.iat[position, column_position]
            estimate_marker = ""
            if table.cell_p_values is not None and number_format != "pvalue":
                p_value = table.cell_p_values.reindex(
                    index=table.frame.index, columns=table.frame.columns
                ).iat[position, column_position]
                if isinstance(p_value, int | float | np.integer | np.floating):
                    estimate_marker = marker(float(p_value))
            if pd.isna(value):
                rendered = "--"
            elif not isinstance(value, int | float | np.integer | np.floating):
                rendered = str(value)
            elif number_format == "percent":
                rendered = f"{float(value):.5%}{estimate_marker}"
            elif number_format == "integer":
                rendered = f"{int(value):,}{estimate_marker}"
            elif number_format == "pvalue":
                rendered = f"{float(value):.4f}{marker(float(value))}"
            else:
                rendered = f"{float(value):.5f}{estimate_marker}"
            frame.iat[position, column_position] = rendered
    return frame


def _group_spans(labels: list[str]) -> list[tuple[int, int, str]]:
    """Collapse a header level into its runs of identical labels.

    Returns one ``(start, end_inclusive, label)`` tuple per maximal run of equal adjacent
    labels, used to span a grouped column header across its sub-columns.
    """
    spans: list[tuple[int, int, str]] = []
    index = 0
    while index < len(labels):
        end = index
        while end + 1 < len(labels) and labels[end + 1] == labels[index]:
            end += 1
        spans.append((index, end, labels[index]))
        index = end + 1
    return spans


def _render_figure(
    table: PublicationTable,
    path: Path,
    *,
    dpi: int,
) -> None:
    if table.panels:
        _render_panel_figure(table, path, dpi=dpi)
        return
    display = _formatted_frame(table, bold_stars=True)
    grouped = isinstance(display.columns, pd.MultiIndex)
    ncols = len(display.columns)
    sub_labels = [str(column[-1]) if grouped else str(column) for column in display.columns]

    header_rows: list[list[str]] = []
    spans: list[tuple[int, int, str]] = []
    if grouped:
        spans = _group_spans([str(column[0]) for column in display.columns])
        # The group labels are drawn as centered overlays so long labels are not clipped
        # to a single cell; the band itself stays blank.
        header_rows.append([""] * ncols)
    header_rows.append(sub_labels)
    n_header = len(header_rows)

    body = [[str(value) for value in row] for row in display.to_numpy().tolist()]
    cell_text = header_rows + body
    row_labels = [""] * n_header + [str(value) for value in display.index]

    width = max(7.2, 1.45 * (ncols + 1))
    height = max(2.0, 0.36 * (len(body) + 2 + n_header))
    # Build at the final dpi so the cell geometry measured for the group-band overlay in
    # _draw_group_labels matches what savefig renders (auto column widths are dpi-sensitive,
    # and a mismatch shifts the over-paint off its divider the further it is from centre).
    figure, axis = plt.subplots(figsize=(width, height), dpi=dpi)
    axis.axis("off")
    axis.set_title(table.title, fontsize=12, fontweight="bold", pad=16)
    artist = axis.table(
        cellText=cell_text,
        rowLabels=row_labels,
        cellLoc="right",
        rowLoc="left",
        loc="center",
    )
    artist.auto_set_font_size(False)
    artist.set_fontsize(8.5)
    artist.scale(1.0, 1.4)

    cells = artist.get_celld()
    style_report_table(cells, header_rows=n_header, blank_corner=True)
    _auto_size_columns(figure, cells, range(ncols))
    if grouped:
        _fit_group_header_widths(figure, axis, cells, spans)
        _draw_group_labels(figure, cells, spans)
    figure.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _render_panel_figure(table: PublicationTable, path: Path, *, dpi: int) -> None:
    """Render a multi-panel table as a grid of individually styled sub-tables in one figure.

    Lays the table's panels out in ``layout_columns`` columns, draws each as a styled table with
    a shared header/zebra treatment, hides unused cells, adds the wrapped caption, and saves the
    figure to ``path``.
    """
    columns = max(1, table.layout_columns)
    rows = int(np.ceil(len(table.panels) / columns))
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(8.5 * columns, 7.0 * rows),
        dpi=dpi,
        squeeze=False,
    )
    figure.suptitle(table.title, fontsize=14, fontweight="bold")
    flat_axes = list(axes.flat)
    for axis, panel in zip(flat_axes, table.panels, strict=False):
        axis.axis("off")
        axis.set_title(panel.title, fontsize=11, fontweight="bold", pad=10)
        display = _formatted_frame(panel, bold_stars=True)
        artist = axis.table(
            cellText=[[str(value) for value in row] for row in display.to_numpy().tolist()],
            rowLabels=[str(value) for value in display.index],
            colLabels=[str(value) for value in display.columns],
            cellLoc="right",
            rowLoc="left",
            colLoc="center",
            loc="upper center",
        )
        artist.auto_set_font_size(False)
        artist.set_fontsize(7.8)
        artist.auto_set_column_width(range(len(display.columns)))
        artist.scale(1.0, 1.28)
        style_report_table(artist.get_celld())
    for axis in flat_axes[len(table.panels) :]:
        axis.set_visible(False)
    figure.text(
        0.5,
        0.015,
        fill(table.caption, width=180),
        ha="center",
        va="bottom",
        fontsize=7.5,
        color="#4A4A4A",
    )
    figure.subplots_adjust(
        left=0.16,
        right=0.98,
        top=0.93,
        bottom=0.07,
        wspace=0.48,
        hspace=0.25,
    )
    figure.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _draw_group_labels(
    figure: Figure,
    cells: dict[tuple[int, int], Cell],
    spans: list[tuple[int, int, str]],
) -> None:
    """Draw the spanning group labels over the top header band and hide interior dividers.

    Each group's label is added as a figure-level text centred over its member cells, so a long
    label neither widens a sub-column nor gets clipped. The sub-column dividers that fall inside
    a group are over-painted in the band's own fill colour, clipped to the band, leaving only the
    true group-boundary lines visible.
    """
    if not spans:
        return
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()  # type: ignore[attr-defined]
    inverse = figure.transFigure.inverted()
    for start, end, label in spans:
        boxes = [cells[(0, column)].get_window_extent(renderer) for column in range(start, end + 1)]
        left = min(box.x0 for box in boxes)
        right = max(box.x1 for box in boxes)
        top = max(box.y1 for box in boxes)
        bottom = min(box.y0 for box in boxes)
        x_left = float(inverse.transform((left, 0.0))[0])
        x_right = float(inverse.transform((right, 0.0))[0])
        y_low = float(inverse.transform((0.0, bottom))[1])
        y_high = float(inverse.transform((0.0, top))[1])
        # Over-paint the internal dividers (right edge of every cell but the group's last),
        # clipped to the band so a thick over-paint line cannot poke past its edges. The clip
        # box is expressed in figure fractions (via transFigure) so it stays correct when
        # savefig re-renders at a different dpi than the draw that measured the cells.
        band_box = TransformedBbox(
            Bbox.from_extents(x_left, y_low, x_right, y_high), figure.transFigure
        )
        for box in boxes[:-1]:
            x_fig = float(inverse.transform((box.x1, 0.0))[0])
            line = Line2D(
                [x_fig, x_fig],
                [y_low, y_high],
                color="#1F3A5F",
                linewidth=2.5,
                solid_capstyle="butt",
                transform=figure.transFigure,
                zorder=4,
            )
            line.set_clip_box(band_box)
            line.set_clip_on(True)
            figure.add_artist(line)
        center = inverse.transform(((left + right) / 2, (top + bottom) / 2))
        figure.text(
            float(center[0]),
            float(center[1]),
            label,
            ha="center",
            va="center",
            color="white",
            fontweight="bold",
            fontsize=_GROUP_HEADER_FONT_SIZE,
            zorder=5,
        )


def _group_label_width(
    figure: Figure,
    renderer: RendererBase,
    label: str,
) -> float:
    """Return the pixel width of ``label`` rendered in the bold group-header font.

    Adds a temporary text artist purely to measure it and removes it before returning, so the
    figure is left unchanged.
    """
    artist = figure.text(
        0.0,
        0.0,
        label,
        fontweight="bold",
        fontsize=_GROUP_HEADER_FONT_SIZE,
    )
    try:
        return float(artist.get_window_extent(renderer).width)
    finally:
        artist.remove()


def _auto_size_columns(
    figure: Figure,
    cells: dict[tuple[int, int], Cell],
    columns: Sequence[int],
) -> None:
    """Set each of ``columns`` to the width its widest cell content requires.

    Draws once to measure required widths, then applies the per-column maximum so a subsequent
    minimum-width pass has a stable baseline to build on.
    """
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()  # type: ignore[attr-defined]
    for column in columns:
        column_cells = [cell for (_row, key), cell in cells.items() if key == column]
        width = max((cell.get_required_width(renderer) for cell in column_cells), default=0.0)
        for cell in column_cells:
            cell.set_width(width)
    figure.canvas.draw()


def _fit_group_header_widths(
    figure: Figure,
    axis: Axes,
    cells: dict[tuple[int, int], Cell],
    spans: list[tuple[int, int, str]],
) -> None:
    """Widen column groups just enough that each spanning group label fits.

    Column widths come from the sub-header and body cells, which ignore the overlaid group
    labels. For any group whose label plus padding exceeds its members' combined width, the
    shortfall is spread evenly across those columns; groups already wide enough are left alone.
    """
    if not spans:
        return
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()  # type: ignore[attr-defined]
    axis_width = float(axis.get_window_extent(renderer).width)
    if axis_width <= 0.0:
        return
    horizontal_padding = 2.0 * _GROUP_HEADER_HORIZONTAL_PADDING_POINTS * float(figure.dpi) / 72.0
    increases: dict[int, float] = {}
    for start, end, label in spans:
        boxes = [cells[(0, column)].get_window_extent(renderer) for column in range(start, end + 1)]
        available = max(box.x1 for box in boxes) - min(box.x0 for box in boxes)
        required = _group_label_width(figure, renderer, label) + horizontal_padding
        deficit = required - available
        if deficit <= 0.0:
            continue
        per_column = deficit / axis_width / len(boxes)
        for column in range(start, end + 1):
            increases[column] = increases.get(column, 0.0) + per_column
    for column, increase in increases.items():
        width = cells[(0, column)].get_width() + increase
        for (_row, cell_column), cell in cells.items():
            if cell_column == column:
                cell.set_width(width)
    if increases:
        figure.canvas.draw()


def _export_table(
    table: PublicationTable,
    directory: Path,
    formats: tuple[TableFormat, ...],
    dpi: int,
) -> tuple[Path, ...]:
    paths: list[Path] = []
    formatted = _formatted_frame(table)
    for output_format in formats:
        path = directory / f"{table.name}.{output_format}"
        if output_format == "csv":
            # CSV is a publication artifact, not a raw-data dump: preserve the same inline
            # significance markers that readers see in TeX and rendered table figures.
            formatted.to_csv(path)
        elif output_format == "tex":
            path.write_text(
                formatted.to_latex(
                    caption=table.caption,
                    label=f"tab:{_slug(table.name)}",
                    escape=True,
                    na_rep="--",
                    column_format="l" + "r" * len(formatted.columns),
                ),
                encoding="utf-8",
            )
        else:
            _render_figure(table, path, dpi=dpi)
        paths.append(path)
    return tuple(paths)


def export_publication_report(
    comparison: ComparisonEvaluation,
    config: ModelEvaluationConfig,
    inference: ComparisonInference,
    figures: Sequence[tuple[str, Figure]] = (),
    factor_analysis: FactorAnalysis | None = None,
) -> PublicationReport:
    """Write the complete publication report — tables and figures — to a fresh directory.

    Builds and exports the metric, significance, and (when ``factor_analysis`` is given) factor
    tables in every configured format, then the caller's ``figures`` together with the factor and
    cross-model comparison figures. With a factor analysis, its additional statistics join the
    significance table and its alpha-difference matrix joins the comparison grid. Each run owns a
    uniquely named directory so it never overwrites another run's artifacts. Locally created
    figures are closed before returning; caller-supplied ones are left open.
    """
    labels = "-vs-".join(_slug(result.label) for result in comparison.results)
    # Every invocation owns an immutable directory. Reusing the deterministic comparison
    # name could upload stale artifacts from an earlier run with different output formats.
    directory = config.output_dir / (
        f"{comparison.start:%Y%m%d}-{comparison.end:%Y%m%d}_{labels}_{uuid4().hex[:12]}"
    )
    directory.mkdir(parents=True, exist_ok=True)
    factor_tables = (
        build_factor_alpha_tables(factor_analysis) if factor_analysis is not None else ()
    )
    additional_statistics = (
        build_additional_statistics_table(factor_analysis) if factor_analysis is not None else None
    )
    tables = (
        *build_publication_tables(comparison, config, inference.model_significance),
        *build_statistical_tables(inference.model_significance, additional_statistics),
        *factor_tables,
    )
    artifacts = {
        table.name: _export_table(
            table,
            directory,
            config.table_formats,
            config.table_dpi,
        )
        for table in tables
    }
    extra_matrices = []
    factor_figures: list[tuple[str, Figure]] = []
    if factor_analysis is not None:
        matrix = build_alpha_difference_matrix(factor_analysis)
        extra_matrices = [matrix] if matrix is not None else []
        factor_figures = build_factor_figures(factor_analysis)
    comparison_figures = build_statistics_comparison_figure(
        inference.comparison_matrices, inference.model_confidence, extra_matrices
    )
    generated_figures = (*factor_figures, *comparison_figures)
    try:
        artifacts |= save_figures(
            (*figures, *generated_figures),
            directory,
            config.plot_formats,
            config.plot_dpi,
        )
    finally:
        # Caller-owned dashboard figures may still be displayed after export. Factor and
        # statistical figures are created locally and must be closed here on success or error.
        for _, figure in generated_figures:
            plt.close(figure)
    return PublicationReport(directory=directory, artifacts=artifacts)
