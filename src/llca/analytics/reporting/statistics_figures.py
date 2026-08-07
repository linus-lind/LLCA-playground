"""Render the cross-model comparison statistics as report figures.

Two figures are produced. The pairwise statistics — Diebold-Mariano and Sharpe-difference
p-values plus net-return, signal, and position similarity — are laid out as a grid of
symmetric heatmaps in the ``model_comparison`` figure. The model confidence set, being a
per-model table rather than a matrix, is drawn on its own as the ``model_confidence_set``
figure so it is not squeezed into the heatmap grid. Each heatmap shows only its strict lower
triangle, shades cells over the statistic's theoretical range (0..1 for p-values, -1..1 for
correlations), and annotates p-value panels with bold significance stars.
"""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import Normalize
from matplotlib.figure import Figure

from llca.analytics.comparison import ComparisonMatrix, ModelConfidenceSummary
from llca.analytics.evaluation.plots import DIVERGING_CMAP, text_color
from llca.analytics.reporting.figures import style_report_table
from llca.analytics.stats.statistics import significance_marker_bold


def _cell_text(value: float, is_pvalue: bool) -> str:
    """Format a single heatmap cell.

    Non-p-value statistics are shown to two decimals. P-value cells are shown to three decimals
    with their bold significance marker; the marker is dropped onto a second line so the value
    and its ``(***)``/``(ns)`` suffix never overflow a narrow cell.
    """
    if not is_pvalue:
        return f"{value:.2f}"
    marker = significance_marker_bold(value)
    return f"{value:.3f}\n{marker.strip()}" if marker else f"{value:.3f}"


def _draw_matrix_panel(axis: Axes, matrix: ComparisonMatrix) -> None:
    """Render one comparison matrix as an annotated lower-triangular heatmap on ``axis``.

    Only the strict lower triangle is coloured and labelled; the diagonal and upper triangle
    are left blank because the matrix is symmetric. Cells are normalised over the matrix's
    declared value range, each annotated with its formatted value in a contrasting colour.
    """
    labels = [str(label) for label in matrix.frame.index]
    data = matrix.frame.to_numpy(dtype=float)
    n = len(labels)
    keep = np.tril(np.ones((n, n), dtype=bool), k=-1)
    norm = Normalize(vmin=matrix.value_range[0], vmax=matrix.value_range[1])
    shown = np.where(keep & np.isfinite(data), data, np.nan)
    masked = np.ma.masked_invalid(shown)
    axis.imshow(masked, cmap=DIVERGING_CMAP, norm=norm, aspect="equal")
    for i in range(n):
        for j in range(n):
            if not keep[i, j] or not np.isfinite(data[i, j]):
                continue
            rgba = DIVERGING_CMAP(norm(data[i, j]))
            axis.text(
                j,
                i,
                _cell_text(float(data[i, j]), matrix.is_pvalue),
                ha="center",
                va="center",
                fontsize=7,
                color=text_color(rgba),
            )
    axis.set_xticks(range(n))
    axis.set_yticks(range(n))
    axis.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    axis.set_yticklabels(labels, fontsize=7)
    axis.set_xticks(np.arange(-0.5, n, 1), minor=True)
    axis.set_yticks(np.arange(-0.5, n, 1), minor=True)
    axis.grid(which="minor", color="white", linewidth=1.0)
    axis.tick_params(which="minor", length=0)
    for spine in axis.spines.values():
        spine.set_visible(False)
    axis.set_title(matrix.title, fontsize=9, fontweight="bold")


def _draw_mcs_panel(axis: Axes, summary: ModelConfidenceSummary) -> None:
    """Render the model confidence set as a one-row-per-model table on ``axis``.

    Each row reports the model's MCS p-value (with bold significance stars, or ``--`` when
    undefined), whether it belongs to the confidence set, and its mean loss. The table uses
    the report's shared styling: a dark header band, alternating light body rows, and thin
    grey gridlines.
    """
    axis.axis("off")
    axis.set_title(summary.title, fontsize=9, fontweight="bold")
    frame = summary.frame
    models = [str(index) for index in frame.index]
    p_values = frame["mcs_p_value"].to_numpy(dtype=float)
    in_confidence = frame["in_confidence_set"].to_numpy()
    mean_loss = frame["mean_loss"].to_numpy(dtype=float)

    cell_text = []
    for index in range(len(models)):
        p = float(p_values[index])
        p_text = "--" if not np.isfinite(p) else f"{p:.3f}{significance_marker_bold(p)}"
        cell_text.append(
            [
                p_text,
                "yes" if bool(in_confidence[index]) else "no",
                f"{float(mean_loss[index]):.4f}",
            ]
        )
    table = axis.table(
        cellText=cell_text,
        rowLabels=models,
        colLabels=["MCS p-value", "In set", "Mean loss"],
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1.0, 1.4)
    style_report_table(table.get_celld(), blank_corner=True)


def _matrix_grid_figure(matrices: Sequence[ComparisonMatrix]) -> Figure:
    """Lay the comparison matrices out as a two-column grid of heatmap panels.

    An odd matrix count leaves the trailing half-row cell blank so the grid keeps clean
    whitespace rather than stretching the final panel.
    """
    ncols = 2
    nrows = -(-len(matrices) // ncols)
    figure = plt.figure(figsize=(4.8 * ncols, 4.4 * nrows))
    figure.suptitle("Cross-Model Statistical Comparison", fontsize=13, fontweight="bold")
    grid = figure.add_gridspec(nrows, ncols)
    for index, matrix in enumerate(matrices):
        _draw_matrix_panel(figure.add_subplot(grid[index // ncols, index % ncols]), matrix)
    if len(matrices) % ncols:
        figure.add_subplot(grid[nrows - 1, ncols - 1]).axis("off")
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    return figure


def _model_confidence_figure(summary: ModelConfidenceSummary) -> Figure:
    """Place the model-confidence-set table alone in a compact, single-panel figure."""
    figure = plt.figure(figsize=(6.0, 1.2 + 0.4 * len(summary.frame.index)))
    _draw_mcs_panel(figure.add_subplot(1, 1, 1), summary)
    figure.tight_layout()
    return figure


def build_statistics_comparison_figure(
    comparison_matrices: Sequence[ComparisonMatrix],
    model_confidence: ModelConfidenceSummary | None,
    extra_matrices: Sequence[ComparisonMatrix] = (),
) -> list[tuple[str, Figure]]:
    """Render the cross-model comparison figures from precomputed inference.

    Returns the ``model_comparison`` heatmap grid and, when a model confidence set is
    supplied, a separate ``model_confidence_set`` figure, each paired with its artifact name.
    ``extra_matrices`` (such as the factor alpha-difference matrix) join the comparison
    matrices in the grid. The list is empty when there is nothing to draw.
    """
    matrices = [*comparison_matrices, *extra_matrices]
    figures: list[tuple[str, Figure]] = []
    if matrices:
        figures.append(("model_comparison", _matrix_grid_figure(matrices)))
    if model_confidence is not None:
        figures.append(("model_confidence_set", _model_confidence_figure(model_confidence)))
    return figures
