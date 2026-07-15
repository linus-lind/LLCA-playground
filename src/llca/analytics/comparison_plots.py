from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

from llca.analytics.comparison import ComparisonEvaluation


def _colors(comparison: ComparisonEvaluation) -> dict[str, object]:
    palette = plt.get_cmap("tab10")
    return {result.label: palette(index % 10) for index, result in enumerate(comparison.results)}


def _plot_portfolio_comparison(comparison: ComparisonEvaluation) -> None:
    """Overlay comparable return, risk, tail-loss, and turnover paths for portfolio models."""
    colors = _colors(comparison)
    figure, axes = plt.subplots(3, 2, figsize=(16, 13))
    figure.suptitle("Cross-Model Portfolio Comparison")
    plotted = False
    for result in comparison.results:
        portfolio = result.evaluation.portfolio
        if portfolio is None:
            continue
        plotted = True
        label = result.label
        color = colors[label]
        cumulative = (1.0 + portfolio.daily["net_return"]).cumprod() - 1.0
        axes[0, 0].plot(cumulative.index, cumulative, label=label, color=color)
        axes[0, 1].plot(
            portfolio.drawdowns.index,
            portfolio.drawdowns["drawdown"],
            label=label,
            color=color,
        )
        axes[1, 0].plot(
            portfolio.rolling.index,
            portfolio.rolling["sharpe_ratio"],
            label=label,
            color=color,
        )
        axes[1, 1].plot(
            portfolio.rolling.index,
            portfolio.rolling["annualized_volatility"],
            label=label,
            color=color,
        )
        if not portfolio.tail_risk.empty:
            var_columns = [
                str(column) for column in portfolio.tail_risk if str(column).startswith("var_")
            ]
            if var_columns:
                var_column = var_columns[0]
                es_column = var_column.replace("var_", "expected_shortfall_")
                axes[2, 0].plot(
                    portfolio.tail_risk.index,
                    portfolio.tail_risk[var_column],
                    label=f"{label} VaR",
                    color=color,
                    linestyle="--",
                )
                axes[2, 0].plot(
                    portfolio.tail_risk.index,
                    portfolio.tail_risk[es_column],
                    label=f"{label} ES",
                    color=color,
                )
        axes[2, 1].plot(
            portfolio.rolling.index,
            portfolio.rolling["one_way_turnover"],
            label=label,
            color=color,
        )
    if not plotted:
        plt.close(figure)
        return
    titles = (
        "Net cumulative return",
        "Net drawdown",
        "Rolling Sharpe ratio",
        "Rolling annualized volatility",
        "Rolling historical VaR and ES",
        "Rolling one-way turnover",
    )
    for axis, title in zip(axes.flat, titles, strict=True):
        axis.set_title(title)
        axis.grid(True, alpha=0.2)
        axis.legend(fontsize="small")
    for axis in (axes[0, 0], axes[0, 1], axes[1, 1], axes[2, 0]):
        axis.yaxis.set_major_formatter(PercentFormatter(1.0))
    figure.tight_layout()


def _plot_signal_comparison(comparison: ComparisonEvaluation) -> None:
    """Overlay rolling signal quality and outcome-lead decay on shared axes."""
    colors = _colors(comparison)
    figure, axes = plt.subplots(2, 2, figsize=(15, 10))
    figure.suptitle("Cross-Model Signal Comparison")
    plotted = False
    for result in comparison.results:
        signal = result.evaluation.signal
        label = result.label
        color = colors[label]
        if "mean_rank_ic" in signal.rolling and signal.rolling["mean_rank_ic"].notna().any():
            axes[0, 0].plot(
                signal.rolling.index,
                signal.rolling["mean_rank_ic"],
                label=label,
                color=color,
            )
            axes[0, 1].plot(
                signal.rolling.index,
                signal.rolling["rank_ic_ir"],
                label=label,
                color=color,
            )
            plotted = True
        if "mean_hit_rate" in signal.rolling and signal.rolling["mean_hit_rate"].notna().any():
            axes[1, 0].plot(
                signal.rolling.index,
                signal.rolling["mean_hit_rate"],
                label=label,
                color=color,
            )
            plotted = True
        elif "mean_accuracy" in signal.rolling and signal.rolling["mean_accuracy"].notna().any():
            axes[1, 0].plot(
                signal.rolling.index,
                signal.rolling["mean_accuracy"],
                label=label,
                color=color,
            )
            plotted = True
        decay_column = next(
            (
                column
                for column in (
                    "mean_daily_rank_ic",
                    "mean_daily_pearson_ic",
                    "pooled_rank",
                    "pooled_pearson",
                )
                if column in signal.decay and signal.decay[column].notna().any()
            ),
            None,
        )
        if decay_column is not None:
            axes[1, 1].plot(
                signal.decay.index,
                signal.decay[decay_column],
                marker="o",
                label=label,
                color=color,
            )
            plotted = True
    if not plotted:
        plt.close(figure)
        return
    titles = (
        "Rolling mean Rank IC",
        "Rolling Rank ICIR",
        "Rolling hit rate / accuracy",
        "Signal decay by outcome lead",
    )
    for axis, title in zip(axes.flat, titles, strict=True):
        axis.set_title(title)
        axis.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
        axis.grid(True, alpha=0.2)
        axis.legend(fontsize="small")
    axes[1, 1].set_xlabel("Lead periods")
    figure.tight_layout()


def _plot_classification_comparison(comparison: ComparisonEvaluation) -> None:
    """Overlay available discrimination and probability-calibration curves for classifiers."""
    classifiers = [
        result
        for result in comparison.results
        if result.evaluation.signal.kind in ("binary", "multiclass")
    ]
    if not classifiers:
        return
    colors = _colors(comparison)
    figure, axes = plt.subplots(1, 3, figsize=(17, 5))
    figure.suptitle("Cross-Model Classification Comparison")
    for result in classifiers:
        signal = result.evaluation.signal
        color = colors[result.label]
        if signal.roc is not None:
            axes[0].plot(
                signal.roc["false_positive_rate"],
                signal.roc["true_positive_rate"],
                label=result.label,
                color=color,
            )
        if signal.precision_recall is not None:
            axes[1].plot(
                signal.precision_recall["recall"],
                signal.precision_recall["precision"],
                label=result.label,
                color=color,
            )
        if signal.calibration is not None and "mean_predicted_probability" in signal.calibration:
            axes[2].plot(
                signal.calibration["mean_predicted_probability"],
                signal.calibration["observed_positive_rate"],
                marker="o",
                label=result.label,
                color=color,
            )
    axes[0].plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=0.8)
    axes[2].plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=0.8)
    for axis, title in zip(
        axes,
        ("ROC curves", "Precision-recall curves", "Probability calibration"),
        strict=True,
    ):
        axis.set_title(title)
        axis.grid(True, alpha=0.2)
        axis.legend(fontsize="small")
    figure.tight_layout()


def plot_comparison(comparison: ComparisonEvaluation) -> None:
    """Overlay every model on shared items, dates, scales and metric definitions."""
    _plot_portfolio_comparison(comparison)
    _plot_signal_comparison(comparison)
    _plot_classification_comparison(comparison)
    plt.show()
