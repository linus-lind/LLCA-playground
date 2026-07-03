from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from llca.analytics.comparison import ComparisonEvaluation
from llca.analytics.modules.test_evaluation import TestEvaluation
from llca.analytics.utils.config import ModelEvaluationConfig, TableFormat

type NumberFormat = Literal["decimal", "percent", "integer"]


@dataclass(frozen=True, slots=True)
class MetricSpec:
    """Define the publication label and numerical format of one scalar metric."""

    key: str
    label: str
    number_format: NumberFormat = "decimal"


@dataclass(frozen=True, slots=True)
class PublicationTable:
    """Hold one numeric or textual table together with paper-facing metadata."""

    name: str
    title: str
    caption: str
    frame: pd.DataFrame
    row_formats: tuple[NumberFormat, ...] = ()
    column_formats: tuple[NumberFormat, ...] = ()


@dataclass(frozen=True, slots=True)
class PublicationReport:
    """Describe the report directory and every generated table artifact."""

    directory: Path
    artifacts: dict[str, tuple[Path, ...]]


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
    MetricSpec("prediction_coverage", "Prediction coverage", "percent"),
    MetricSpec("pearson_correlation", "Pearson correlation"),
    MetricSpec("spearman_correlation", "Spearman correlation"),
    MetricSpec("mean_daily_pearson_ic", "Mean daily Pearson IC"),
    MetricSpec("mean_daily_rank_ic", "Mean daily rank IC"),
    MetricSpec("rank_ic_ir", "Rank ICIR"),
    MetricSpec("annualized_rank_ic_ir", "Annualized rank ICIR"),
    MetricSpec("directional_accuracy", "Directional accuracy", "percent"),
    MetricSpec("magnitude_weighted_directional_accuracy", "Magnitude-weighted accuracy", "percent"),
    MetricSpec("accuracy", "Accuracy", "percent"),
    MetricSpec("balanced_accuracy", "Balanced accuracy", "percent"),
    MetricSpec("roc_auc", "ROC AUC"),
    MetricSpec("average_precision", "Average precision"),
    MetricSpec("brier_score", "Brier score"),
    MetricSpec("matthews_correlation", "Matthews correlation"),
    MetricSpec("mae", "Mean absolute error"),
    MetricSpec("rmse", "Root mean squared error"),
    MetricSpec("r_squared", "R-squared"),
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


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-") or "report"


def _metric_table(
    source: pd.DataFrame,
    specs: tuple[MetricSpec, ...],
    *,
    name: str,
    title: str,
    caption: str,
) -> PublicationTable | None:
    available = [spec for spec in specs if spec.key in source and not source[spec.key].isna().all()]
    if not available:
        return None
    frame = source[[spec.key for spec in available]].transpose()
    frame.index = pd.Index([spec.label for spec in available], name="Metric")
    return PublicationTable(
        name=name,
        title=title,
        caption=caption,
        frame=frame,
        row_formats=tuple(spec.number_format for spec in available),
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
    *,
    rows: int | None = None,
) -> pd.DataFrame | None:
    """Align a task-specific table by item and place every model in grouped columns."""
    frames: dict[str, pd.DataFrame] = {}
    for result in comparison.results:
        frame = selector(result.evaluation)
        if frame is None or frame.empty:
            continue
        frames[result.label] = frame.head(rows) if rows is not None else frame
    if not frames:
        return None
    return pd.concat(frames, axis=1, names=["Model", "Statistic"])


def _detailed_tables(comparison: ComparisonEvaluation) -> list[PublicationTable]:
    """Build item-aligned detail tables with grouped columns for all models."""
    candidates = (
        (
            "signal_buckets",
            "Signal Bucket Analysis",
            _combined_detail(comparison, lambda result: result.signal.buckets),
            False,
        ),
        (
            "confusion_matrix",
            "Classification Confusion Matrix",
            _combined_detail(comparison, lambda result: result.signal.confusion),
            True,
        ),
        (
            "signal_calibration",
            "Signal Calibration",
            _combined_detail(comparison, lambda result: result.signal.calibration),
            False,
        ),
        (
            "signal_decay",
            "Signal Decay",
            _combined_detail(comparison, lambda result: result.signal.decay),
            False,
        ),
        (
            "yearly_returns",
            "Calendar-Year Portfolio Returns",
            _combined_detail(
                comparison,
                lambda result: (
                    result.portfolio.yearly_returns if result.portfolio is not None else None
                ),
            ),
            False,
        ),
        (
            "side_attribution",
            "Long, Short, and Cost Attribution",
            _combined_detail(
                comparison,
                lambda result: (
                    result.portfolio.side_attribution if result.portfolio is not None else None
                ),
            ),
            False,
        ),
        (
            "signal_attribution",
            "Portfolio Contribution by Signal Bucket",
            _combined_detail(
                comparison,
                lambda result: (
                    result.portfolio.signal_attribution if result.portfolio is not None else None
                ),
            ),
            False,
        ),
        (
            "asset_attribution",
            "Largest Asset Return Contributions",
            _combined_detail(
                comparison,
                lambda result: (
                    result.portfolio.asset_attribution if result.portfolio is not None else None
                ),
                rows=10,
            ),
            False,
        ),
        (
            "maximum_drawdown_attribution",
            "Largest Maximum-Drawdown Contributions",
            _combined_detail(
                comparison,
                lambda result: (
                    result.portfolio.maximum_drawdown_attribution
                    if result.portfolio is not None
                    else None
                ),
                rows=10,
            ),
            False,
        ),
    )
    tables: list[PublicationTable] = []
    for name, title, frame, integer_columns in candidates:
        if frame is None or frame.empty:
            continue
        column_formats: tuple[NumberFormat, ...]
        if integer_columns:
            column_formats = tuple("integer" for _ in frame.columns)
        else:
            column_formats = tuple(
                _detailed_column_format(str(column[-1] if isinstance(column, tuple) else column))
                for column in frame.columns
            )
        tables.append(
            PublicationTable(
                name=name,
                title=title,
                caption=f"{title} on the common held-out sample.",
                frame=frame.copy(),
                column_formats=column_formats,
            )
        )
    return tables


def _detailed_column_format(column: str) -> NumberFormat:
    """Infer readable formats for task-specific table columns."""
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
            "coverage",
            "cost",
        )
    ):
        return "percent"
    return "decimal"


def build_publication_tables(comparison: ComparisonEvaluation) -> tuple[PublicationTable, ...]:
    """Build compact comparison and task-specific tables from one evaluation object."""
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
        ),
        _metric_table(
            comparison.portfolio_metrics,
            _PORTFOLIO_PERFORMANCE,
            name="portfolio_performance",
            title="Portfolio Performance and Risk",
            caption="Net realized performance after configured transaction and borrow costs.",
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


def _formatted_frame(table: PublicationTable) -> pd.DataFrame:
    frame = table.frame.copy().astype(object)
    for position in range(len(frame)):
        for column_position, _column in enumerate(frame.columns):
            number_format = (
                table.column_formats[column_position]
                if table.column_formats
                else table.row_formats[position]
            )
            value = frame.iat[position, column_position]
            if pd.isna(value):
                rendered = "--"
            elif not isinstance(value, int | float | np.integer | np.floating):
                rendered = str(value)
            elif number_format == "percent":
                rendered = f"{float(value):.2%}"
            elif number_format == "integer":
                rendered = f"{int(value):,}"
            else:
                rendered = f"{float(value):.3f}"
            frame.iat[position, column_position] = rendered
    return frame


def _render_figure(
    table: PublicationTable,
    path: Path,
    *,
    dpi: int,
) -> None:
    display = _formatted_frame(table)
    width = max(7.2, 1.45 * (len(display.columns) + 1))
    height = max(2.0, 0.36 * (len(display) + 3))
    figure, axis = plt.subplots(figsize=(width, height))
    axis.axis("off")
    axis.set_title(table.title, fontsize=12, fontweight="bold", pad=14)
    cell_text = [[str(value) for value in row] for row in display.to_numpy().tolist()]
    artist = axis.table(
        cellText=cell_text,
        rowLabels=[str(value) for value in display.index],
        colLabels=[str(value) for value in display.columns],
        cellLoc="right",
        rowLoc="left",
        colLoc="center",
        loc="center",
    )
    artist.auto_set_font_size(False)
    artist.set_fontsize(8.5)
    artist.scale(1.0, 1.35)
    for (row, _column), cell in artist.get_celld().items():
        cell.set_edgecolor("#D6DCE4")
        cell.set_linewidth(0.5)
        if row == 0:
            cell.set_facecolor("#1F3A5F")
            cell.set_text_props(color="white", fontweight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#F3F6F9")
    figure.text(0.5, 0.02, table.caption, ha="center", fontsize=8, color="#4A4A4A")
    figure.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(figure)


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
            table.frame.to_csv(path)
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
) -> PublicationReport:
    """Export paper-ready tables without writing tabular output to the console."""
    labels = "-vs-".join(_slug(result.label) for result in comparison.results)
    directory = config.output_dir / (f"{comparison.start:%Y%m%d}-{comparison.end:%Y%m%d}_{labels}")
    directory.mkdir(parents=True, exist_ok=True)
    artifacts = {
        table.name: _export_table(
            table,
            directory,
            config.table_formats,
            config.table_dpi,
        )
        for table in build_publication_tables(comparison)
    }
    return PublicationReport(directory=directory, artifacts=artifacts)
