"""Single-model figure primitives reused by the comparison figure layer."""

from __future__ import annotations

import pandas as pd
from matplotlib.axes import Axes


def draw_confusion(
    axis: Axes,
    confusion: pd.DataFrame,
    *,
    title: str,
    compact: bool = False,
) -> None:
    """Render a directional confusion matrix as an annotated heatmap on ``axis``.

    Cells show integer counts, coloured for contrast, with predicted classes on the x-axis and
    actual classes on the y-axis. ``compact`` shrinks the title, ticks, and annotations for a
    packed multi-panel layout; left unset, Matplotlib's defaults apply.
    """
    matrix = confusion.to_numpy(dtype=float)
    axis.imshow(matrix, cmap="Blues")
    predicted = [str(column).replace("predicted_", "") for column in confusion.columns]
    actual = [str(index).replace("actual_", "") for index in confusion.index]
    axis.set_xticks(range(matrix.shape[1]))
    axis.set_yticks(range(matrix.shape[0]))
    if compact:
        axis.set_title(title, fontsize=10)
        axis.set_xticklabels(predicted, fontsize=8)
        axis.set_yticklabels(actual, fontsize=8)
    else:
        axis.set_title(title)
        axis.set_xticklabels(predicted)
        axis.set_yticklabels(actual)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("Actual")
    threshold = matrix.max() / 2.0 if matrix.size else 0.0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            color = "white" if matrix[row, column] > threshold else "black"
            value = f"{int(matrix[row, column]):,}"
            if compact:
                axis.text(column, row, value, ha="center", va="center", color=color, fontsize=9)
            else:
                axis.text(column, row, value, ha="center", va="center", color=color)
