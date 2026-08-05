from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.ticker import PercentFormatter

from llca.analytics.comparison import ComparisonEvaluation
from llca.analytics.evaluation.plots import draw_confusion


def _colors(comparison: ComparisonEvaluation) -> dict[str, object]:
    palette = plt.get_cmap("tab10")
    return {result.label: palette(index % 10) for index, result in enumerate(comparison.results)}


def _plot_portfolio_comparison(comparison: ComparisonEvaluation) -> list[tuple[str, Figure]]:
    """Overlay every model's time-series portfolio diagnostics on one 4x2 figure.

    The panels trace cumulative net return, drawdown, rolling Sharpe and volatility, rolling
    historical VaR and expected shortfall, and rolling one-way turnover and short exposure,
    with a consistent per-model colour and percentage axes where appropriate. Returned tagged
    ``portfolio_comparison``.
    """
    colors = _colors(comparison)
    figure, axes = plt.subplots(4, 2, figsize=(16, 17))
    figure.suptitle("Portfolio Performance and Risk")
    for result in comparison.results:
        portfolio = result.evaluation.portfolio
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
                    label=label,
                    color=color,
                )
                axes[2, 1].plot(
                    portfolio.tail_risk.index,
                    portfolio.tail_risk[es_column],
                    label=label,
                    color=color,
                )
        axes[3, 0].plot(
            portfolio.rolling.index,
            portfolio.rolling["one_way_turnover"],
            label=label,
            color=color,
        )
        axes[3, 1].plot(
            portfolio.rolling.index,
            portfolio.rolling["short_exposure"],
            label=label,
            color=color,
        )
    titled_axes = (
        (axes[0, 0], "Net cumulative return"),
        (axes[0, 1], "Net drawdown"),
        (axes[1, 0], "Rolling Sharpe ratio"),
        (axes[1, 1], "Rolling annualized volatility"),
        (axes[2, 0], "Rolling historical VaR"),
        (axes[2, 1], "Rolling historical expected shortfall"),
        (axes[3, 0], "Rolling one-way turnover"),
        (axes[3, 1], "Rolling short exposure"),
    )
    for axis, title in titled_axes:
        axis.set_title(title)
        axis.grid(True, alpha=0.2)
        axis.legend(fontsize="small")
    for axis in (axes[0, 0], axes[0, 1], axes[1, 1], axes[2, 0], axes[2, 1], axes[3, 1]):
        axis.yaxis.set_major_formatter(PercentFormatter(1.0))
    figure.tight_layout()
    return [("portfolio_comparison", figure)]


def _plot_signal_comparison(comparison: ComparisonEvaluation) -> list[tuple[str, Figure]]:
    """Overlay every model's rolling signal quality and decay on one 2x2 figure.

    Panels show rolling mean rank IC, rolling rank ICIR, rolling hit rate, and information
    decay against the outcome lead; each model is labelled by its IC basis. Series are only
    drawn where the underlying columns hold data, and the figure is discarded (returning an
    empty list) when no model contributes anything. Otherwise returned tagged
    ``signal_comparison``.
    """
    colors = _colors(comparison)
    figure, axes = plt.subplots(2, 2, figsize=(15, 10))
    figure.suptitle("Signal Performance Through Time")
    plotted = False
    for result in comparison.results:
        signal = result.evaluation.signal
        label = result.label
        signal_label = (
            f"{label} (cross-sectional)"
            if signal.ic_basis == "cross_sectional"
            else f"{label} (rolling time-series)"
        )
        color = colors[label]
        if "mean_rank_ic" in signal.rolling and signal.rolling["mean_rank_ic"].notna().any():
            axes[0, 0].plot(
                signal.rolling.index,
                signal.rolling["mean_rank_ic"],
                label=signal_label,
                color=color,
            )
            axes[0, 1].plot(
                signal.rolling.index,
                signal.rolling["rank_ic_ir"],
                label=signal_label,
                color=color,
            )
            plotted = True
        if "mean_hit_rate" in signal.rolling and signal.rolling["mean_hit_rate"].notna().any():
            axes[1, 0].plot(
                signal.rolling.index,
                signal.rolling["mean_hit_rate"],
                label=signal_label,
                color=color,
            )
            plotted = True
        decay_column = next(
            (
                column
                for column in (
                    "basis_rank_ic",
                    "basis_pearson_ic",
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
                label=signal_label,
                color=color,
            )
            plotted = True
    if not plotted:
        plt.close(figure)
        return []
    titles = (
        "Rolling mean Rank IC",
        "Rolling Rank ICIR",
        "Rolling hit rate",
        "Signal decay by outcome lead",
    )
    for axis, title in zip(axes.flat, titles, strict=True):
        axis.set_title(title)
        axis.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
        axis.grid(True, alpha=0.2)
        axis.legend(fontsize="small")
    axes[1, 1].set_xlabel("Lead periods")
    figure.tight_layout()
    return [("signal_comparison", figure)]


def _plot_confusion_and_roc(comparison: ComparisonEvaluation) -> list[tuple[str, Figure]]:
    """Draw one directional confusion matrix per model plus a shared ROC panel.

    Each model gets an annotated confusion heatmap; the final panel overlays every model's ROC
    curve, labelled with its AUC, against the chance diagonal. Returned tagged ``confusion_roc``.
    """
    results = list(comparison.results)
    colors = _colors(comparison)
    panels = len(results) + 1
    figure, axes = plt.subplots(1, panels, figsize=(4.2 * panels, 4.4), squeeze=False)
    row = axes[0]
    figure.suptitle("Directional Confusion Matrices and ROC")
    for axis, result in zip(row[:-1], results, strict=True):
        draw_confusion(axis, result.evaluation.signal.confusion, title=result.label, compact=True)
    roc_axis = row[-1]
    for result in results:
        signal = result.evaluation.signal
        if signal.roc is None:
            continue
        auc = float(signal.metrics.get("roc_auc", float("nan")))
        roc_axis.plot(
            signal.roc["false_positive_rate"],
            signal.roc["true_positive_rate"],
            color=colors[result.label],
            label=f"{result.label} (AUC {auc:.3f})",
        )
    roc_axis.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=0.8)
    roc_axis.set_title("ROC curve")
    roc_axis.set_xlabel("False positive rate")
    roc_axis.set_ylabel("True positive rate")
    roc_axis.grid(True, alpha=0.2)
    roc_axis.legend(fontsize="small")
    figure.tight_layout()
    return [("confusion_roc", figure)]


def build_comparison_figures(comparison: ComparisonEvaluation) -> list[tuple[str, Figure]]:
    """Build every cross-model comparison figure and return them without showing any.

    Concatenates the portfolio, signal, and confusion/ROC figures, omitting any that had no
    data to plot.
    """
    return [
        *_plot_portfolio_comparison(comparison),
        *_plot_signal_comparison(comparison),
        *_plot_confusion_and_roc(comparison),
    ]
