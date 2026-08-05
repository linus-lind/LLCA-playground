"""Render the per-model significance table from precomputed inference statistics.

The single-model significance table answers "is this model's signal real?" (directional
content, information coefficient, risk-adjusted return) and is the only table this module
renders. The cross-model question — "are the models actually different?" (Diebold-Mariano
accuracy, the model confidence set, Sharpe-ratio differences, and return/signal/position
similarity) — is computed in :mod:`llca.analytics.comparison.inference` and drawn as one
combined figure by :mod:`llca.analytics.reporting.statistics_figures`; it is not a table.

Every statistic consumed here is computed up front by
:func:`llca.analytics.comparison.inference.evaluate_comparison_inference`; this module only
assembles the standalone rows into a :class:`PublicationTable`. The significance table keeps
only content that is *not* already reported elsewhere: whenever a test has a displayed
estimate or statistic, its p-value is represented by significance stars in that value's cell,
and a p-value row is retained only when no corresponding value exists.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from llca.analytics.reporting.table_types import NumberFormat, PublicationTable

# Canonical row order, numeric format, and paired p-value of the significance table. Point
# estimates already shown on the signal/portfolio metric tables are intentionally absent.
# Rows that no configured model can populate are dropped before rendering. A paired p-value
# becomes cell-level stars; a standalone p-value remains a normally formatted p-value row.
#
# The rank-IC, Sharpe, and directional p-values are *not* here: they are attached to their
# point estimates in the signal/portfolio metric tables. Only tests whose
# estimate lives nowhere else remain — directional content (Pesaran-Timmermann, excess
# profitability).
_SIGNIFICANCE_ROWS: tuple[tuple[str, str, NumberFormat, str | None], ...] = (
    ("pt_statistic", "Pesaran-Timmermann statistic", "decimal", "pt_p_value"),
    (
        "excess_profitability",
        "Excess profitability (HAC)",
        "decimal",
        "excess_profitability_p_value",
    ),
)


def _significance_table(model_significance: pd.DataFrame) -> PublicationTable | None:
    """Build the standalone significance table from the per-model statistics frame.

    Emits only the rows whose test has no estimate shown elsewhere: where a model has a point
    estimate the row displays it and encodes its p-value as inline stars, and where only a
    p-value exists the row shows that instead. Returns ``None`` when no row survives.
    """
    columns = {
        str(label): {str(key): float(value) for key, value in row.items()}
        for label, row in model_significance.iterrows()
    }
    rows: list[tuple[str, str, NumberFormat, str | None]] = []
    for key, label, number_format, p_value_key in _SIGNIFICANCE_ROWS:
        value_available = any(np.isfinite(values.get(key, np.nan)) for values in columns.values())
        p_value_available = p_value_key is not None and any(
            np.isfinite(values.get(p_value_key, np.nan)) for values in columns.values()
        )
        if value_available:
            rows.append((key, label, number_format, p_value_key))
        elif p_value_available and p_value_key is not None:
            # Retain a p-value row only when the test exposes no corresponding value.
            rows.append((p_value_key, f"{label} p-value", "pvalue", None))
    if not rows:
        return None
    frame = pd.DataFrame(
        {
            model_label: [values.get(key, np.nan) for key, _, _, _ in rows]
            for model_label, values in columns.items()
        },
        index=pd.Index([label for _, label, _, _ in rows], name="Statistic"),
    )
    p_values = pd.DataFrame(
        {
            model_label: [
                values.get(p_value_key, np.nan) if p_value_key is not None else np.nan
                for _, _, _, p_value_key in rows
            ]
            for model_label, values in columns.items()
        },
        index=frame.index,
    )
    frame.columns.name = "Model"
    p_values.columns.name = frame.columns.name
    return PublicationTable(
        name="statistical_significance",
        title="Statistical Significance of Predictive Content",
        caption=(
            "Time-series directional (Pesaran-Timmermann) and excess-profitability tests "
            "where applicable; genuine panels use date-level directional HAC inference. "
            "HAC-corrected rows are labelled and one-sided where a direction is implied. "
            "Information-coefficient, directional-accuracy, and Sharpe significance appears "
            "inline on their metric tables. Paired p-values are represented inline: "
            "*** p<0.01, ** p<0.05, * p<0.10."
        ),
        frame=frame,
        row_formats=tuple(number_format for _, _, number_format, _ in rows),
        cell_p_values=p_values,
    )


def _merge_additional_statistics(
    significance: PublicationTable | None,
    additional: PublicationTable,
) -> PublicationTable:
    """Stack the factor-model ``additional`` statistics beneath the significance table.

    Unions the two tables' model columns, concatenates their rows and per-cell p-values, and
    merges their captions and row formats. When the significance table is absent the additional
    statistics stand alone.
    """
    if significance is None:
        columns = list(additional.frame.columns)
        frame = additional.frame.copy()
        p_values = (
            additional.cell_p_values.copy()
            if additional.cell_p_values is not None
            else pd.DataFrame(np.nan, index=frame.index, columns=columns)
        )
        row_formats = additional.row_formats
        caption = additional.caption
    else:
        columns = list(dict.fromkeys([*significance.frame.columns, *additional.frame.columns]))
        frame = pd.concat(
            [
                significance.frame.reindex(columns=columns),
                additional.frame.reindex(columns=columns),
            ]
        )

        def p_values_for(table: PublicationTable) -> pd.DataFrame:
            if table.cell_p_values is None:
                return pd.DataFrame(np.nan, index=table.frame.index, columns=columns)
            return table.cell_p_values.reindex(index=table.frame.index, columns=columns)

        p_values = pd.concat([p_values_for(significance), p_values_for(additional)])
        row_formats = (*significance.row_formats, *additional.row_formats)
        caption = f"{significance.caption} {additional.caption}"
    frame = frame.reindex(columns=columns)
    p_values = p_values.reindex(index=frame.index, columns=columns)
    frame.columns.name = "Model"
    p_values.columns.name = frame.columns.name
    return PublicationTable(
        name="statistical_significance",
        title="Statistical Significance and Additional Statistics",
        caption=caption,
        frame=frame,
        row_formats=row_formats,
        cell_p_values=p_values,
    )


def build_statistical_tables(
    model_significance: pd.DataFrame,
    additional_statistics: PublicationTable | None = None,
) -> tuple[PublicationTable, ...]:
    """Produce the significance table (with any factor-model statistics) as a one-tuple or empty.

    Assembles the standalone significance table from ``model_significance`` and, when
    ``additional_statistics`` is given, appends them. Returns an empty tuple if there is nothing
    to show. Cross-model comparisons are rendered as a figure elsewhere, not here.
    """
    significance = _significance_table(model_significance)
    if additional_statistics is not None:
        significance = _merge_additional_statistics(significance, additional_statistics)
    return (significance,) if significance is not None else ()
