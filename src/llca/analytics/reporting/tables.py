"""Build the paper-facing publication tables (content only) from one comparison.

Every function here produces :class:`PublicationTable` data — humanized labels, metric-per-row
frames, and item-aligned detail tables. Turning those tables into styled figures and exporting
the report lives in :mod:`llca.analytics.reporting.table_rendering`.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd

from llca.analytics.comparison import ComparisonEvaluation
from llca.analytics.modules.analytics_config import ModelEvaluationConfig
from llca.analytics.modules.test_evaluation import TestEvaluation
from llca.analytics.reporting.table_types import NumberFormat, PublicationTable

# Tokens that must keep a fixed capitalization when snake_case labels are humanized.
_LABEL_ACRONYMS = {
    "hhi": "HHI",
    "ic": "IC",
    "icir": "ICIR",
    "ir": "IR",
    "var": "VaR",
    "es": "ES",
    "roc": "ROC",
    "auc": "AUC",
    "l1": "L1",
    "cagr": "CAGR",
    "pnl": "PnL",
}


def _humanize_text(text: str) -> str:
    """Turn a ``snake_case`` token into readable words, respecting known acronyms.

    Splits on underscores/hyphens, capitalizes the first word, and substitutes fixed casings for
    recognised acronyms. Text that already reads as prose — containing spaces or interior
    capitals — is returned untouched so curated labels are preserved.
    """
    stripped = text.strip()
    if not stripped or " " in stripped or stripped != stripped.lower():
        return stripped
    tokens = [token for token in re.split(r"[_\-]+", stripped) if token]
    if not tokens:
        return stripped
    rendered = [
        _LABEL_ACRONYMS.get(token, token.capitalize() if index == 0 else token)
        for index, token in enumerate(tokens)
    ]
    return " ".join(rendered)


def _humanize_index_value(value: object) -> object:
    """Render a row-index value for display, formatting dates and numbers cleanly.

    A year-end timestamp becomes its year and other timestamps their ISO date; integer-valued
    numbers lose their decimals; strings are humanized; anything else is returned as-is.
    """
    if isinstance(value, pd.Timestamp | datetime):
        if value.month == 12 and value.day == 31:
            return str(value.year)
        return value.date().isoformat()
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int | np.integer):
        return str(int(value))
    if isinstance(value, float) and float(value).is_integer():
        return str(int(value))
    if isinstance(value, str):
        return _humanize_text(value)
    return value


def _prettify_detail_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Humanize a detail frame's statistic headers and row labels, leaving model labels intact.

    On a two-level column index only the ``Statistic`` level is prettified; the ``Model`` level
    holds user-facing identifiers and is preserved. Single-level column headers and the row index
    are humanized as well.
    """
    frame = frame.copy()
    if isinstance(frame.columns, pd.MultiIndex):
        names = list(frame.columns.names)
        frame.columns = pd.MultiIndex.from_tuples(
            [
                tuple(
                    _humanize_text(str(value)) if name == "Statistic" else value
                    for name, value in zip(names, column, strict=True)
                )
                for column in frame.columns
            ],
            names=names,
        )
    else:
        frame.columns = pd.Index(
            [_humanize_text(str(column)) for column in frame.columns],
            name=frame.columns.name,
        )
    index_name = _humanize_text(str(frame.index.name)) if frame.index.name else None
    frame.index = pd.Index(
        [_humanize_index_value(value) for value in frame.index],
        name=index_name,
    )
    return frame


def _statistic_label(column: object, names: Sequence[object]) -> str:
    """Pull the statistic name out of a column key, grouped or flat.

    For a tuple key it returns the entry under the ``Statistic`` level (falling back to the last
    element); a plain key is returned as a string.
    """
    if isinstance(column, tuple):
        for name, value in zip(names, column, strict=True):
            if name == "Statistic":
                return str(value)
        return str(column[-1])
    return str(column)


def _stat_major(frame: pd.DataFrame, comparison: ComparisonEvaluation) -> pd.DataFrame:
    """Pivot a ``(Model, Statistic)`` column index to statistic-major ``(Statistic, Model)``.

    Statistics become the spanning top-level header and models the sub-columns, keeping the
    original statistic order and the configured model order. When only one model was evaluated,
    the redundant model level is dropped, leaving just the statistic header.
    """
    swapped = frame.swaplevel(0, 1, axis=1)
    stat_order = list(dict.fromkeys(frame.columns.get_level_values("Statistic")))
    present = set(swapped.columns.get_level_values("Model"))
    model_order = [result.label for result in comparison.results if result.label in present]
    columns = pd.MultiIndex.from_product([stat_order, model_order], names=["Statistic", "Model"])
    ordered = swapped.reindex(columns=columns)
    if len(comparison.results) == 1:
        return ordered.droplevel("Model", axis=1)
    return ordered


@dataclass(frozen=True, slots=True)
class MetricSpec:
    """Define the publication label and numerical format of one scalar metric."""

    key: str
    label: str
    number_format: NumberFormat = "decimal"


_OBJECTIVE = (
    MetricSpec("loss", "Objective loss"),
    MetricSpec("mean_return", "Mean return", "percent"),
    MetricSpec("variance", "Return variance"),
    MetricSpec("turnover", "L1 turnover", "percent"),
    MetricSpec("cost", "Mean cost", "percent"),
    MetricSpec("gross_exposure", "Gross exposure", "percent"),
    MetricSpec("net_exposure", "Net exposure", "percent"),
    MetricSpec("long_exposure", "Long exposure", "percent"),
    MetricSpec("short_exposure", "Short exposure", "percent"),
    MetricSpec("concentration", "Concentration (HHI)"),
)

_SIGNAL = (
    MetricSpec("pearson_correlation", "Pearson correlation"),
    MetricSpec("spearman_correlation", "Spearman correlation"),
    MetricSpec("mean_daily_pearson_ic", "Mean Pearson IC (recorded basis)"),
    MetricSpec("mean_daily_rank_ic", "Mean rank IC (recorded basis)"),
    MetricSpec("rank_ic_ir", "Rank ICIR"),
    MetricSpec("annualized_rank_ic_ir", "Annualized rank ICIR (cross-sectional only)"),
    MetricSpec("directional_accuracy", "Directional accuracy", "percent"),
    MetricSpec("magnitude_weighted_directional_accuracy", "Magnitude-weighted accuracy", "percent"),
    MetricSpec("roc_auc", "ROC AUC"),
    MetricSpec("top_minus_bottom_outcome", "Top-minus-bottom outcome", "percent"),
    MetricSpec("bucket_monotonicity", "Bucket monotonicity"),
)

_PORTFOLIO_PERFORMANCE = (
    MetricSpec("net_total_return", "Net total return", "percent"),
    MetricSpec("net_cagr", "Net CAGR", "percent"),
    MetricSpec("net_annualized_arithmetic_return", "Net annualized return", "percent"),
    MetricSpec("net_annualized_volatility", "Net annualized volatility", "percent"),
    MetricSpec("net_sharpe_ratio", "Net Sharpe ratio"),
    MetricSpec("net_sortino_ratio", "Net Sortino ratio"),
    MetricSpec("net_calmar_ratio", "Net Calmar ratio"),
    MetricSpec("net_maximum_drawdown", "Net maximum drawdown", "percent"),
    MetricSpec("net_expected_shortfall_95", "Net expected shortfall 95%", "percent"),
    MetricSpec("net_expected_shortfall_99", "Net expected shortfall 99%", "percent"),
    MetricSpec("net_skewness", "Net return skewness"),
    MetricSpec("net_excess_kurtosis", "Net excess kurtosis"),
    MetricSpec("net_profit_factor", "Net profit factor"),
)

_PORTFOLIO_CONSTRUCTION = (
    MetricSpec("annualized_cost_drag", "Annualized cost drag", "percent"),
    MetricSpec("mean_daily_one_way_turnover", "Mean one-way turnover", "percent"),
    MetricSpec("annualized_l1_turnover", "Annualized L1 turnover"),
    MetricSpec("mean_gross_exposure", "Mean gross exposure", "percent"),
    MetricSpec("mean_net_exposure", "Mean net exposure", "percent"),
    MetricSpec("mean_long_exposure", "Mean long exposure", "percent"),
    MetricSpec("mean_short_exposure", "Mean short exposure", "percent"),
    MetricSpec("mean_effective_positions", "Mean effective positions"),
    MetricSpec("maximum_absolute_weight", "Maximum absolute weight", "percent"),
    MetricSpec("average_position_holding_period", "Average holding period", "decimal"),
    MetricSpec("annualized_long_return_contribution", "Annualized long contribution", "percent"),
    MetricSpec("annualized_short_return_contribution", "Annualized short contribution", "percent"),
)


def _metric_table(
    source: pd.DataFrame,
    specs: tuple[MetricSpec, ...],
    *,
    name: str,
    title: str,
    caption: str,
    significance: pd.DataFrame | None = None,
    pvalue_for: tuple[tuple[str, str], ...] = (),
) -> PublicationTable | None:
    """Build a table of the given metrics, one row per metric and one column per model.

    Only metrics present with at least one non-missing value are kept. ``pvalue_for`` pairs a
    displayed metric with a hidden p-value from ``significance``, whose stars are attached to the
    metric's own cell. Returns ``None`` when no metric qualifies.
    """
    available = [spec for spec in specs if spec.key in source and not source[spec.key].isna().all()]
    if not available:
        return None
    p_value_keys = dict(pvalue_for)
    models = list(source.index)
    labels: list[str] = []
    formats: list[NumberFormat] = []
    data: dict[object, list[float]] = {model: [] for model in models}
    p_values: dict[object, list[float]] = {model: [] for model in models}

    def emit(label: str, series: pd.Series, number_format: NumberFormat) -> None:
        labels.append(label)
        formats.append(number_format)
        for model in models:
            data[model].append(float(series.get(model, np.nan)))

    for spec in available:
        emit(spec.label, source[spec.key], spec.number_format)
        p_value_key = p_value_keys.get(spec.key)
        for model in models:
            value = (
                significance[p_value_key].get(model, np.nan)
                if significance is not None
                and p_value_key is not None
                and p_value_key in significance.columns
                else np.nan
            )
            p_values[model].append(float(value))

    frame = pd.DataFrame(data, index=pd.Index(labels, name="Metric"))[models]
    significance_frame = pd.DataFrame(p_values, index=frame.index)[models]
    return PublicationTable(
        name=name,
        title=title,
        caption=caption,
        frame=frame,
        row_formats=tuple(formats),
        cell_p_values=significance_frame,
    )


def _overview(comparison: ComparisonEvaluation) -> PublicationTable:
    rows = []
    for result in comparison.results:
        rows.append(
            {
                "Model": result.label,
                "Registry version": result.metadata.config.version,
                "Output": result.evaluation.predictions.kind,
                "Observations": result.evaluation.valid_observations,
                "Dates": result.evaluation.dates,
                "Test start": comparison.start.date().isoformat(),
                "Test end": comparison.end.date().isoformat(),
            }
        )
    frame = pd.DataFrame(rows).set_index("Model").transpose()
    return PublicationTable(
        name="model_overview",
        title="Model Evaluation Sample",
        caption="Registry versions and the common held-out evaluation universe.",
        frame=frame,
        row_formats=("integer", "decimal", "integer", "integer", "decimal", "decimal"),
    )


def _combined_detail(
    comparison: ComparisonEvaluation,
    selector: Callable[[TestEvaluation], pd.DataFrame | None],
) -> pd.DataFrame | None:
    """Stack one detail table per model into grouped ``(Model, Statistic)`` columns.

    Applies ``selector`` to each model's evaluation, skips models that return nothing or an empty
    frame, and concatenates the survivors side by side. Returns ``None`` if no model contributes.
    """
    frames: dict[str, pd.DataFrame] = {}
    for result in comparison.results:
        frame = selector(result.evaluation)
        if frame is None or frame.empty:
            continue
        frames[result.label] = frame
    if not frames:
        return None
    return pd.concat(frames, axis=1, names=["Model", "Statistic"])


def _signal_bucket_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Replace a bucket table's low/high score bounds with a single formatted ``range`` column."""
    frame = frame.copy()
    frame.insert(
        0,
        "range",
        [
            f"[{low:.5f} - {high:.5f}]"
            for low, high in zip(frame["score_low"], frame["score_high"], strict=True)
        ],
    )
    return frame.drop(columns=["score_low", "score_high"])


def _detailed_tables(comparison: ComparisonEvaluation) -> list[PublicationTable]:
    """Build the per-item detail tables: yearly returns, side attribution, and signal buckets.

    Each is assembled across models, pivoted to statistic-major grouped columns, humanized, and
    given inferred column number formats. Tables with no data are omitted.
    """
    candidates = (
        (
            "yearly_returns",
            "Calendar-Year Portfolio Returns",
            _combined_detail(
                comparison,
                lambda result: result.portfolio.yearly_returns,
            ),
        ),
        (
            "side_attribution",
            "Long, Short, Cash, and Cost Attribution",
            _combined_detail(
                comparison,
                lambda result: result.portfolio.side_attribution,
            ),
        ),
        (
            "signal_bucket_analysis",
            "Signal Bucket Analysis",
            _combined_detail(
                comparison,
                lambda result: _signal_bucket_frame(result.portfolio.signal_attribution),
            ),
        ),
    )
    tables: list[PublicationTable] = []
    for name, title, frame in candidates:
        if frame is None or frame.empty:
            continue
        stat = _stat_major(frame, comparison)
        column_formats = tuple(
            _detailed_column_format(_statistic_label(column, stat.columns.names))
            for column in stat.columns
        )
        tables.append(
            PublicationTable(
                name=name,
                title=title,
                caption=f"{title} on the common held-out sample.",
                frame=_prettify_detail_frame(stat),
                column_formats=column_formats,
            )
        )
    return tables


def _detailed_column_format(column: str) -> NumberFormat:
    """Guess a number format for a detail column from its name.

    Count-like names format as integers, return/weight/rate-like names as percentages, and
    everything else as decimals.
    """
    normalized = column.lower()
    if any(token in normalized for token in ("observation", "count", "periods")):
        return "integer"
    if any(
        token in normalized
        for token in (
            "return",
            "contribution",
            "weight",
            "rate",
            "accuracy",
            "cost",
        )
    ):
        return "percent"
    return "decimal"


def build_publication_tables(
    comparison: ComparisonEvaluation,
    config: ModelEvaluationConfig,
    model_significance: pd.DataFrame,
) -> tuple[PublicationTable, ...]:
    """Build the report's headline tables for one comparison.

    Produces the model overview, the objective/signal/portfolio metric tables, and the per-item
    detail tables. Rank-IC, directional-accuracy, and Sharpe p-values from ``model_significance``
    are rendered as stars on their estimates, leaving only estimate-less tests for the separate
    significance table. Tables with no data are dropped.
    """
    significance = model_significance
    optional = (
        _metric_table(
            comparison.loss_metrics,
            _OBJECTIVE,
            name="objective_metrics",
            title="Training Objective on the Test Set",
            caption="Objective components recomputed on the common held-out sample.",
        ),
        _metric_table(
            comparison.signal_metrics,
            _SIGNAL,
            name="signal_metrics",
            title="Signal Performance",
            caption="Signal quality metrics computed on identical prediction items.",
            significance=significance,
            pvalue_for=(
                ("mean_daily_rank_ic", "ic_p_value"),
                ("directional_accuracy", "hit_rate_p_value"),
            ),
        ),
        _metric_table(
            comparison.portfolio_metrics,
            _PORTFOLIO_PERFORMANCE,
            name="portfolio_performance",
            title="Portfolio Performance and Risk",
            caption="Net realized performance after configured transaction and borrow costs.",
            significance=significance,
            pvalue_for=(("net_sharpe_ratio", "sharpe_p_value"),),
        ),
        _metric_table(
            comparison.portfolio_metrics,
            _PORTFOLIO_CONSTRUCTION,
            name="portfolio_construction",
            title="Portfolio Construction and Trading",
            caption="Exposure, concentration, turnover, and cost characteristics.",
        ),
    )
    return tuple(
        [
            _overview(comparison),
            *(table for table in optional if table is not None),
            *_detailed_tables(comparison),
        ]
    )
