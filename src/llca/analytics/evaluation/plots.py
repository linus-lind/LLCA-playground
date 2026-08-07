"""Single-model figure primitives reused by the comparison figure layer."""

from __future__ import annotations

import pandas as pd
from matplotlib.axes import Axes
from matplotlib.colors import Colormap, LinearSegmentedColormap, Normalize, to_rgba

# Shared diverging orange->grey->blue palette (a muted, colour-blind-friendly scientific scheme)
# used by every report heatmap -- the confusion matrices here and the cross-model comparison
# panels -- so they read as one figure family: low values blue, high values orange, neutral grey
# midpoint.
DIVERGING_CMAP: Colormap = LinearSegmentedColormap.from_list(
    "orange_grey_blue", ["#2166AC", "#CCCCCC", "#E08214"]
).with_extremes(bad="white")


def text_color(color: object) -> str:
    """Return ``"black"`` or ``"white"``, whichever stays legible on ``color`` as a fill.

    The choice follows the perceived luminance of the background: dark fills take white text,
    light fills take black.
    """
    red, green, blue, _ = to_rgba(color)  # type: ignore[arg-type]
    luminance = 0.299 * red + 0.587 * green + 0.114 * blue
    return "black" if luminance > 0.55 else "white"


def draw_confusion(
    axis: Axes,
    confusion: pd.DataFrame,
    *,
    title: str,
    compact: bool = False,
) -> None:
    """Render a directional confusion matrix as an annotated heatmap on ``axis``.

    Cells show integer counts on the shared diverging palette (low counts blue, high counts
    orange), with predicted classes on the x-axis and actual classes on the y-axis. ``compact``
    shrinks the title, ticks, and annotations for a packed multi-panel layout; left unset,
    Matplotlib's defaults apply.
    """
    matrix = confusion.to_numpy(dtype=float)
    vmin = float(matrix.min()) if matrix.size else 0.0
    vmax = float(matrix.max()) if matrix.size else 1.0
    norm = Normalize(vmin=vmin, vmax=vmax if vmax > vmin else vmin + 1.0)
    axis.imshow(matrix, cmap=DIVERGING_CMAP, norm=norm)
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
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            color = text_color(DIVERGING_CMAP(norm(matrix[row, column])))
            value = f"{int(matrix[row, column]):,}"
            if compact:
                axis.text(column, row, value, ha="center", va="center", color=color, fontsize=9)
            else:
                axis.text(column, row, value, ha="center", va="center", color=color)
