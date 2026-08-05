"""Alpha and factor-model analysis of the realized net portfolio returns.

This assembles the report's factor section from three tradable/estimated factor models — the
Fama-French 6, IPCA, and a conditional timing model — plus the joint asset-pricing tests. Each
model's excess net return (already stored as ``excess_net_return`` on its portfolio) is the
left-hand side, so single-asset and many-asset strategies are handled identically.

Rendered outputs:

* separate **factor-model table figures** for Fama-French + Momentum, IPCA, and Conditional
  Timing. Estimated coefficients carry HAC significance stars directly in their cells;
* an **additional-statistics table** that is merged into the statistical-significance output;
* a **cross-model alpha-difference matrix** (HAC test that one model's FF6 alpha beats another's,
  multiple-testing corrected) drawn into the combined comparison figure;
* one two-column **rolling factor-beta figure** (one panel per portfolio model);
* one shared **cumulative-alpha figure** overlaying every model's running FF6 abnormal return.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from llca.analytics.comparison import (
    ComparisonEvaluation,
    ComparisonMatrix,
    ModelEvaluationResult,
)
from llca.analytics.comparison.inference import correction_label, pairwise_matrix
from llca.analytics.factors import factor_models
from llca.analytics.factors.factor_models import FactorAlpha, TimingModel
from llca.analytics.modules.analytics_config import ModelEvaluationConfig
from llca.analytics.modules.factor_settings import FactorSources
from llca.analytics.reporting.table_types import NumberFormat, PublicationTable
from llca.analytics.stats.inference import adjust_pairwise


@dataclass(frozen=True, slots=True)
class _ModelFactors:
    """Per-model factor-analysis results keyed by the model's comparison label."""

    label: str
    excess: pd.Series
    ff6: FactorAlpha | None
    ipca: FactorAlpha | None
    timing: TimingModel | None
    spanning_statistic: float
    spanning_p_value: float
    rolling_betas: pd.DataFrame
    cumulative_alpha: pd.Series


@dataclass(frozen=True, slots=True)
class FactorAnalysis:
    """Complete factor section, including cross-model and joint zero-alpha tests."""

    models: tuple[_ModelFactors, ...]
    ff6_columns: tuple[str, ...]
    ipca_columns: tuple[str, ...]
    market_column: str
    rolling_beta_window: int
    alpha_difference: pd.DataFrame
    joint_alpha_statistic: float
    joint_alpha_p_value: float
    correction_label: str


def _excess(result: ModelEvaluationResult) -> pd.Series:
    """Take a model's daily excess net-return series, labelled by the model."""
    return result.evaluation.portfolio.daily["excess_net_return"].rename(result.label)


def build_factor_analysis(
    comparison: ComparisonEvaluation,
    config: ModelEvaluationConfig,
    sources: FactorSources,
    ipca_factors: pd.DataFrame | None,
) -> FactorAnalysis | None:
    """Run the complete factor section for the comparison, or ``None`` if no models qualify.

    For each portfolio model it fits the FF6 and (when available) IPCA alphas, the conditional
    timing model, the spanning test, rolling betas, and cumulative alpha off the model's excess
    net returns. Across models it builds the multiple-testing-corrected alpha-difference matrix
    and, with two or more models, the joint zero-alpha J-test, packaging everything into a
    ``FactorAnalysis``.
    """
    annualization = config.annualization_periods
    lag = config.hac_lag
    entries: list[_ModelFactors] = []
    excess_by_label: dict[str, pd.Series] = {}
    for result in comparison.results:
        excess = _excess(result)
        excess_by_label[result.label] = excess
        ff6 = factor_models.factor_alpha(
            excess, sources.ff6, annualization_periods=annualization, lag=lag
        )
        ipca = (
            factor_models.factor_alpha(
                excess, ipca_factors, annualization_periods=annualization, lag=lag
            )
            if ipca_factors is not None
            else None
        )
        timing = factor_models.timing_model(
            excess,
            sources.ff6,
            sources.market_column,
            sources.timing_instruments,
            annualization_periods=annualization,
            instrument_lag=sources.timing_instrument_lag,
            market_squared=sources.market_squared,
            conditional_alpha=sources.conditional_alpha,
            lag=lag,
        )
        spanning = factor_models.spanning_test(excess, sources.spanning_benchmark, lag=lag)
        rolling = factor_models.rolling_betas(
            excess, sources.ff6, window=sources.rolling_beta_window
        )
        cumulative = factor_models.cumulative_alpha(excess, sources.ff6)
        entries.append(
            _ModelFactors(
                label=result.label,
                excess=excess,
                ff6=ff6,
                ipca=ipca,
                timing=timing,
                spanning_statistic=spanning["spanning_statistic"],
                spanning_p_value=spanning["spanning_p_value"],
                rolling_betas=rolling,
                cumulative_alpha=cumulative,
            )
        )
    if not entries:
        return None

    labels = [entry.label for entry in entries]
    difference = _alpha_difference_matrix(labels, excess_by_label, sources.ff6, lag)
    difference = adjust_pairwise(difference, config.multiple_testing_correction)
    if len(labels) >= 2:
        portfolios = pd.concat({label: excess_by_label[label] for label in labels}, axis=1)
        joint_alpha = factor_models.joint_alpha_test(portfolios, sources.ff6, lag=lag)
    else:
        joint_alpha = {
            "joint_alpha_statistic": float("nan"),
            "joint_alpha_p_value": float("nan"),
        }
    correction = correction_label(config)
    return FactorAnalysis(
        models=tuple(entries),
        ff6_columns=tuple(str(column) for column in sources.ff6.columns),
        ipca_columns=(
            tuple(str(column) for column in ipca_factors.columns)
            if ipca_factors is not None
            else ()
        ),
        market_column=sources.market_column,
        rolling_beta_window=sources.rolling_beta_window,
        alpha_difference=difference,
        joint_alpha_statistic=joint_alpha["joint_alpha_statistic"],
        joint_alpha_p_value=joint_alpha["joint_alpha_p_value"],
        correction_label=correction,
    )


def _alpha_difference_matrix(
    labels: list[str],
    excess: dict[str, pd.Series],
    factors: pd.DataFrame,
    lag: int | None,
) -> pd.DataFrame:
    def difference_p_value(left: str, right: str) -> float:
        outcome = factor_models.alpha_difference(excess[left], excess[right], factors, lag=lag)
        return outcome["alpha_difference_p_value"]

    return pairwise_matrix(labels, difference_p_value, diagonal=float("nan"))


# --------------------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------------------
# Human-readable names for the Fama-French / Carhart factor columns, used both in the alpha
# table's loading rows and the rolling-beta plot legends.
FACTOR_LABELS = {
    "mktrf": "Market",
    "smb": "Size",
    "hml": "Value",
    "rmw": "Profitability",
    "cma": "Investment",
    "umd": "Momentum",
}

INSTRUMENT_LABELS = {
    "baa10y_spread": "BAA-10Y spread",
    "tips_10y_yield": "10Y TIPS yield",
    "nfci": "NFCI",
    "vix": "VIX",
}


def factor_label(column: str) -> str:
    return FACTOR_LABELS.get(str(column), str(column))


def _coefficient_formats(count: int) -> tuple[NumberFormat, ...]:
    percent: NumberFormat = "percent"
    decimal: NumberFormat = "decimal"
    return (percent,) + (decimal,) * count + (decimal,)


_FACTOR_ALPHA_CAPTION = (
    "Annualized alpha, factor loadings, and regression fit. Coefficient stars use "
    "two-sided HAC p-values: *** p<0.01, ** p<0.05, * p<0.10."
)


def _factor_alpha_panel(
    analysis: FactorAnalysis,
    *,
    name: str,
    title: str,
    caption: str,
    columns: tuple[str, ...],
    result_for: Callable[[_ModelFactors], FactorAlpha | TimingModel | None],
    label_for: Callable[[str], str],
    loadings_of: Callable[[Any], Mapping[str, float]] = lambda result: result.betas,
    loading_p_values_of: Callable[[Any], Mapping[str, float]] = lambda result: result.beta_p_values,
) -> PublicationTable:
    """Build a factor-model table of annualized alpha, factor loadings, and R-squared.

    One column per model and one row per statistic, with each coefficient's HAC p-value encoded
    as inline stars. ``columns`` names the factors, ``result_for`` selects a model's fitted
    result, ``label_for`` renders each factor's row label, and ``loadings_of`` /
    ``loading_p_values_of`` read the coefficient and p-value maps off that result. Models without
    a fitted result show blanks.
    """
    row_labels = [
        "Annualized alpha",
        *(label_for(column) for column in columns),
        "R-squared",
    ]
    values: dict[str, list[float]] = {}
    p_values: dict[str, list[float]] = {}
    for entry in analysis.models:
        result = result_for(entry)
        values[entry.label] = (
            [result.annualized_alpha]
            + [loadings_of(result).get(column, np.nan) for column in columns]
            + [result.r_squared]
            if result is not None
            else [np.nan] * len(row_labels)
        )
        p_values[entry.label] = (
            [result.alpha_p_value]
            + [loading_p_values_of(result).get(column, np.nan) for column in columns]
            + [np.nan]
            if result is not None
            else [np.nan] * len(row_labels)
        )
    index = pd.Index(row_labels, name="Statistic")
    frame = pd.DataFrame(values, index=index)
    frame.columns.name = "Portfolio model"
    significance = pd.DataFrame(p_values, index=index)
    significance.columns.name = frame.columns.name
    return PublicationTable(
        name=name,
        title=title,
        caption=caption,
        frame=frame,
        row_formats=_coefficient_formats(len(columns)),
        cell_p_values=significance,
    )


def _instrument_label(column: str) -> str:
    return INSTRUMENT_LABELS.get(column, column.replace("_", " ").title())


def _timing_coefficient_label(column: str, market_column: str) -> str:
    if column == market_column:
        return factor_label(column)
    if column == f"{market_column}_squared":
        return "Treynor-Mazuy convexity (Market squared)"
    interaction_prefix = f"{market_column}_x_"
    if column.startswith(interaction_prefix):
        instrument = column.removeprefix(interaction_prefix)
        return f"Conditional market beta: {_instrument_label(instrument)}"
    if column.startswith("alpha_"):
        return f"Conditional alpha: {_instrument_label(column.removeprefix('alpha_'))}"
    return factor_label(column)


def _timing_panel(analysis: FactorAnalysis) -> PublicationTable:
    coefficient_columns = tuple(
        dict.fromkeys(
            coefficient
            for entry in analysis.models
            if entry.timing is not None
            for coefficient in entry.timing.coefficients
        )
    )
    return _factor_alpha_panel(
        analysis,
        name="factor_alpha_timing",
        title="Conditional Timing",
        caption=(
            "Conditional alpha, factor exposures, timing terms, and regression fit. "
            "Coefficient stars use two-sided HAC p-values: *** p<0.01, ** p<0.05, * p<0.10."
        ),
        columns=coefficient_columns,
        result_for=lambda entry: entry.timing,
        label_for=lambda column: _timing_coefficient_label(column, analysis.market_column),
        loadings_of=lambda result: result.coefficients,
        loading_p_values_of=lambda result: result.coefficient_p_values,
    )


def build_additional_statistics_table(analysis: FactorAnalysis) -> PublicationTable:
    """Build the additional-statistics table of spanning and joint zero-alpha tests.

    Gives each model a mean-variance spanning statistic row and, when two or more models exist,
    adds a joint zero-alpha J-statistic in a dedicated joint column. p-values are carried as
    inline stars. This table is later merged into the significance table.
    """
    model_labels = [entry.label for entry in analysis.models]
    include_joint = len(analysis.models) >= 2
    rows = ["Mean-variance spanning statistic (HAC)"]
    values = {entry.label: [entry.spanning_statistic] for entry in analysis.models}
    significance = {entry.label: [entry.spanning_p_value] for entry in analysis.models}
    if include_joint:
        joint_label = "Joint model set"
        rows.append("Joint zero-alpha J-statistic (HAC)")
        for entry in analysis.models:
            values[entry.label].append(np.nan)
            significance[entry.label].append(np.nan)
        values[joint_label] = [np.nan, analysis.joint_alpha_statistic]
        significance[joint_label] = [np.nan, analysis.joint_alpha_p_value]
    index = pd.Index(rows, name="Statistic")
    frame = pd.DataFrame(values, index=index)
    p_values = pd.DataFrame(significance, index=index)
    frame = frame[model_labels + (["Joint model set"] if include_joint else [])]
    p_values = p_values[frame.columns]
    frame.columns.name = "Model"
    p_values.columns.name = frame.columns.name
    decimal: NumberFormat = "decimal"
    return PublicationTable(
        name="additional_statistics",
        title="Additional Statistics",
        caption=(
            "Huberman-Kandel mean-variance spanning tests against the 2x3 size-sorted "
            "benchmark portfolios"
            + (" and a kernel-HAC J-test of jointly zero portfolio alphas" if include_joint else "")
            + ". Stars use their HAC p-values."
        ),
        frame=frame,
        row_formats=(decimal,) * len(rows),
        cell_p_values=p_values,
    )


def build_factor_alpha_tables(analysis: FactorAnalysis) -> tuple[PublicationTable, ...]:
    """Build a separate table per factor-model specification that any model fitted.

    Emits the Fama-French + Momentum, IPCA, and Conditional Timing panels, each included only
    when at least one model produced that fit, so an unavailable specification is silently
    skipped.
    """
    if not analysis.models:
        return ()
    tables: list[PublicationTable] = []
    if any(entry.ff6 is not None for entry in analysis.models):
        tables.append(
            _factor_alpha_panel(
                analysis,
                name="factor_alpha_ff6",
                title="Fama-French + Momentum",
                caption=_FACTOR_ALPHA_CAPTION,
                columns=analysis.ff6_columns,
                result_for=lambda entry: entry.ff6,
                label_for=factor_label,
            )
        )
    if analysis.ipca_columns and any(entry.ipca is not None for entry in analysis.models):
        tables.append(
            _factor_alpha_panel(
                analysis,
                name="factor_alpha_ipca",
                title="IPCA",
                caption=_FACTOR_ALPHA_CAPTION,
                columns=analysis.ipca_columns,
                result_for=lambda entry: entry.ipca,
                label_for=lambda column: column.upper().replace("_", " "),
            )
        )
    if any(entry.timing is not None for entry in analysis.models):
        tables.append(_timing_panel(analysis))
    return tuple(tables)


def build_alpha_difference_matrix(analysis: FactorAnalysis) -> ComparisonMatrix | None:
    """Package the FF6 alpha-difference p-values as a comparison-figure matrix, or ``None``.

    Returns ``None`` for fewer than two models, since a pairwise matrix needs at least two.
    """
    if len(analysis.models) < 2:
        return None
    return ComparisonMatrix(
        name="alpha_difference_pvalues",
        title="FF6 Alpha-Difference p-values",
        frame=analysis.alpha_difference,
        is_pvalue=True,
        value_range=(0.0, 1.0),
        caption=f"Equal FF6 alpha (HAC test on the return difference); {analysis.correction_label}.",
    )


# --------------------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------------------
def build_factor_figures(analysis: FactorAnalysis) -> list[tuple[str, Figure]]:
    """Build the factor-section figures: rolling betas per model and a cumulative-alpha overlay.

    Each figure is included only when it has data to show.
    """
    figures: list[tuple[str, Figure]] = []
    rolling = _rolling_betas_figure(analysis)
    if rolling is not None:
        figures.append(rolling)
    cumulative = _cumulative_alpha_figure(analysis)
    if cumulative is not None:
        figures.append(cumulative)
    return figures


def _rolling_betas_figure(analysis: FactorAnalysis) -> tuple[str, Figure] | None:
    """Draw one panel per model tracing its rolling FF6 factor betas over time.

    Panels are laid out in two columns and share an x-axis. Returns ``None`` when no model has
    rolling betas to plot; otherwise it is tagged ``rolling_factor_betas``.
    """
    entries = [entry for entry in analysis.models if not entry.rolling_betas.empty]
    if not entries:
        return None
    palette = plt.get_cmap("tab10")
    columns = 2
    rows = int(np.ceil(len(entries) / columns))
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(14, 4.0 * rows),
        squeeze=False,
        sharex=True,
    )
    figure.suptitle(
        f"Rolling Fama-French 6 Factor Betas ({analysis.rolling_beta_window}-Observation Window)",
        fontsize=13,
        fontweight="bold",
    )
    flat_axes = list(axes.flat)
    for axis, entry in zip(flat_axes, entries, strict=False):
        for index, column in enumerate(entry.rolling_betas.columns):
            axis.plot(
                entry.rolling_betas.index,
                entry.rolling_betas[column],
                label=factor_label(str(column)),
                color=palette(index % 10),
            )
        axis.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
        axis.set_title(entry.label, fontsize=10)
        axis.set_ylabel("Beta")
        axis.grid(True, alpha=0.2)
        axis.legend(fontsize="small", ncol=3)
    for axis in flat_axes[len(entries) :]:
        axis.set_visible(False)
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    return "rolling_factor_betas", figure


def _cumulative_alpha_figure(analysis: FactorAnalysis) -> tuple[str, Figure] | None:
    palette = plt.get_cmap("tab10")
    figure, axis = plt.subplots(figsize=(11, 5))
    plotted = False
    for index, entry in enumerate(analysis.models):
        series = entry.cumulative_alpha
        if series.empty:
            continue
        axis.plot(series.index, series, label=entry.label, color=palette(index % 10))
        plotted = True
    if not plotted:
        plt.close(figure)
        return None
    axis.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    axis.set_title("Cumulative FF6 Alpha (Abnormal Return)")
    axis.set_ylabel("Cumulative abnormal return")
    axis.grid(True, alpha=0.2)
    axis.legend(fontsize="small")
    figure.tight_layout()
    return "cumulative_factor_alpha", figure
