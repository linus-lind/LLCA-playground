from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import PercentFormatter

from llca.analytics.modules.test_evaluation import TestEvaluation


def plot_cumulative_returns(portfolio_returns: pd.Series) -> None:
    """Display one compounded return series for backward-compatible callers."""
    if (portfolio_returns <= -1.0).any():
        raise ValueError("cannot compound portfolio returns containing values <= -100%")
    cumulative = (1.0 + portfolio_returns).cumprod() - 1.0
    figure, axis = plt.subplots(figsize=(12, 6))
    axis.plot(cumulative.index, cumulative.to_numpy(), color="#0B4F8A", linewidth=1.8)
    axis.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    axis.set_title("Test Set Cumulative Return")
    axis.set_xlabel("Date")
    axis.set_ylabel("Cumulative return")
    axis.yaxis.set_major_formatter(PercentFormatter(1.0))
    axis.grid(True, alpha=0.25)
    figure.tight_layout()
    plt.show()


def _plot_signal(evaluation: TestEvaluation) -> None:
    """Plot task-aware signal distributions, buckets, date dispersion, and calibration."""
    signal = evaluation.signal
    figure, axes = plt.subplots(2, 2, figsize=(14, 9))
    figure.suptitle(f"{signal.kind.title()} Signal Diagnostics")

    values = evaluation.predictions.values
    if isinstance(values, pd.Series):
        axes[0, 0].hist(values.to_numpy(dtype=float), bins=60, color="#0B4F8A", alpha=0.85)
        axes[0, 0].set_title("Signal distribution")
        axes[0, 0].set_xlabel("Signal")
    else:
        for column in values:
            axes[0, 0].hist(values[column], bins=40, alpha=0.45, label=str(column))
        axes[0, 0].legend()
        axes[0, 0].set_title("Class-score distributions")

    if not signal.buckets.empty:
        outcome_column = next(
            (
                name
                for name in ("mean_outcome", "accuracy", "observed_positive_rate")
                if name in signal.buckets
            ),
            None,
        )
        if outcome_column is not None:
            axes[0, 1].bar(
                signal.buckets.index.astype(str),
                signal.buckets[outcome_column],
                color="#2E8B57",
            )
            axes[0, 1].set_title(f"{outcome_column.replace('_', ' ').title()} by signal bucket")
            axes[0, 1].set_xlabel("Bucket")
    elif signal.confusion is not None:
        image = axes[0, 1].imshow(signal.confusion.to_numpy(), cmap="Blues")
        figure.colorbar(image, ax=axes[0, 1])
        axes[0, 1].set_title("Confusion matrix")

    diagnostic_column = next(
        (column for column in ("rank_ic", "pearson_ic", "accuracy") if column in signal.per_date),
        None,
    )
    if diagnostic_column is not None:
        values_per_date = signal.per_date[diagnostic_column].dropna()
        axes[1, 0].hist(values_per_date, bins=50, color="#7A5195", alpha=0.85)
        axes[1, 0].axvline(values_per_date.mean(), color="black", linestyle="--", linewidth=1)
        axes[1, 0].set_title(f"Daily {diagnostic_column.replace('_', ' ')} distribution")

    if signal.roc is not None:
        axes[1, 1].plot(
            signal.roc["false_positive_rate"],
            signal.roc["true_positive_rate"],
            color="#D95F02",
            label="ROC",
        )
        axes[1, 1].plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=0.8)
        axes[1, 1].set_xlabel("False positive rate")
        axes[1, 1].set_ylabel("True positive rate")
        axes[1, 1].set_title("ROC curve")
    elif signal.calibration is not None:
        if "mean_predicted_probability" in signal.calibration:
            axes[1, 1].plot(
                signal.calibration["mean_predicted_probability"],
                signal.calibration["observed_positive_rate"],
                marker="o",
            )
            axes[1, 1].set_title("Probability calibration")
        else:
            axes[1, 1].plot(
                signal.calibration.index,
                signal.calibration["empirical_coverage"],
                marker="o",
            )
            axes[1, 1].set_title("Quantile calibration")
        axes[1, 1].plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=0.8)
    else:
        axes[1, 1].axis("off")

    for axis in axes.flat:
        axis.grid(True, alpha=0.2)
    figure.tight_layout()

    if signal.kind == "classification":
        classification_figure, classification_axes = plt.subplots(2, 2, figsize=(12, 9))
        classification_figure.suptitle("Classification Discrimination and Calibration")
        if signal.confusion is not None:
            image = classification_axes[0, 0].imshow(signal.confusion.to_numpy(), cmap="Blues")
            classification_figure.colorbar(image, ax=classification_axes[0, 0])
            classification_axes[0, 0].set_title("Confusion matrix")
        if signal.roc is not None:
            classification_axes[0, 1].plot(
                signal.roc["false_positive_rate"], signal.roc["true_positive_rate"]
            )
            classification_axes[0, 1].plot(
                [0, 1], [0, 1], color="black", linestyle="--", linewidth=0.8
            )
            classification_axes[0, 1].set_title("ROC curve")
        if signal.precision_recall is not None:
            classification_axes[1, 0].plot(
                signal.precision_recall["recall"], signal.precision_recall["precision"]
            )
            classification_axes[1, 0].set_title("Precision-recall curve")
        if signal.calibration is not None:
            classification_axes[1, 1].plot(
                signal.calibration["mean_predicted_probability"],
                signal.calibration["observed_positive_rate"],
                marker="o",
            )
            classification_axes[1, 1].plot(
                [0, 1], [0, 1], color="black", linestyle="--", linewidth=0.8
            )
            classification_axes[1, 1].set_title("Probability calibration")
        for axis in classification_axes.flat:
            axis.grid(True, alpha=0.2)
        classification_figure.tight_layout()


def _plot_signal_time(evaluation: TestEvaluation) -> None:
    """Display signal strength, stability and decay without mixing in portfolio returns."""
    signal = evaluation.signal
    if signal.rolling.empty and signal.decay.empty:
        return

    figure, axes = plt.subplots(2, 2, figsize=(15, 10))
    figure.suptitle(f"{signal.kind.title()} Signal Through Time")

    daily_column = next(
        (
            column
            for column in ("rank_ic", "pearson_ic", "accuracy", "hit_rate")
            if column in signal.per_date and signal.per_date[column].notna().any()
        ),
        None,
    )
    if daily_column is not None:
        axes[0, 0].plot(
            signal.per_date.index,
            signal.per_date[daily_column],
            color="#7A5195",
            linewidth=0.7,
            alpha=0.45,
            label=f"Daily {daily_column.replace('_', ' ')}",
        )
        mean_column = f"mean_{daily_column}"
        if mean_column in signal.rolling:
            axes[0, 0].plot(
                signal.rolling.index,
                signal.rolling[mean_column],
                color="#7A5195",
                linewidth=2.0,
                label="Rolling mean",
            )
        axes[0, 0].legend()
    else:
        axes[0, 0].axis("off")
    axes[0, 0].set_title("Daily signal quality and rolling mean")

    ic_column = next(
        (
            column
            for column in ("rank_ic_ir", "pearson_ic_ir")
            if column in signal.rolling and signal.rolling[column].notna().any()
        ),
        None,
    )
    if ic_column is not None:
        axes[0, 1].plot(
            signal.rolling.index,
            signal.rolling[ic_column],
            color="#D95F02",
            linewidth=1.6,
        )
    else:
        axes[0, 1].axis("off")
    axes[0, 1].set_title("Rolling ICIR")

    secondary_column = next(
        (
            column
            for column in ("mean_hit_rate", "mean_accuracy", "mean_balanced_accuracy")
            if column in signal.rolling and signal.rolling[column].notna().any()
        ),
        None,
    )
    if secondary_column is not None:
        axes[1, 0].plot(
            signal.rolling.index,
            signal.rolling[secondary_column],
            color="#2E8B57",
            linewidth=1.6,
        )
        if secondary_column in ("mean_hit_rate", "mean_accuracy", "mean_balanced_accuracy"):
            axes[1, 0].axhline(0.5, color="black", linestyle="--", linewidth=0.8)
    else:
        axes[1, 0].axis("off")
    axes[1, 0].set_title("Rolling directional quality")

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
            color="#0B4F8A",
            marker="o",
            linewidth=1.6,
        )
        axes[1, 1].set_xlabel("Outcome lead periods")
    else:
        axes[1, 1].axis("off")
    axes[1, 1].set_title("Signal decay")

    for axis in axes.flat:
        if axis.axison:
            axis.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
            axis.grid(True, alpha=0.2)
    figure.tight_layout()


def _plot_portfolio(evaluation: TestEvaluation) -> None:
    """Plot the reconciled portfolio return, risk, exposure, and trading paths."""
    portfolio = evaluation.portfolio
    if portfolio is None:
        return
    figure, axes = plt.subplots(4, 2, figsize=(15, 15))
    figure.suptitle("Portfolio Performance and Construction")

    cumulative = (1.0 + portfolio.daily[["gross_return", "net_return"]]).cumprod() - 1.0
    axes[0, 0].plot(cumulative.index, cumulative["gross_return"], label="Gross", linewidth=1.6)
    axes[0, 0].plot(cumulative.index, cumulative["net_return"], label="Net", linewidth=1.6)
    axes[0, 0].set_title("Cumulative returns")
    axes[0, 0].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[0, 0].legend()

    axes[0, 1].fill_between(
        portfolio.drawdowns.index,
        portfolio.drawdowns["drawdown"],
        0.0,
        color="#C44E52",
        alpha=0.65,
    )
    axes[0, 1].set_title("Net drawdown")
    axes[0, 1].yaxis.set_major_formatter(PercentFormatter(1.0))

    axes[1, 0].plot(
        portfolio.rolling.index,
        portfolio.rolling["sharpe_ratio"],
        color="#55A868",
    )
    axes[1, 0].axhline(0.0, color="black", linewidth=0.8)
    axes[1, 0].set_title("Rolling Sharpe ratio")

    axes[1, 1].plot(
        portfolio.rolling.index,
        portfolio.rolling["annualized_volatility"],
        color="#4C72B0",
    )
    axes[1, 1].set_title("Rolling annualized volatility")
    axes[1, 1].yaxis.set_major_formatter(PercentFormatter(1.0))

    var_columns = [str(column) for column in portfolio.tail_risk if str(column).startswith("var_")]
    if var_columns:
        var_column = var_columns[0]
        es_column = var_column.replace("var_", "expected_shortfall_")
        axes[2, 0].plot(
            portfolio.tail_risk.index,
            portfolio.tail_risk[var_column],
            label="VaR",
            color="#C44E52",
            linestyle="--",
        )
        axes[2, 0].plot(
            portfolio.tail_risk.index,
            portfolio.tail_risk[es_column],
            label="Expected shortfall",
            color="#C44E52",
        )
        axes[2, 0].legend()
    axes[2, 0].set_title("Rolling historical tail loss")
    axes[2, 0].yaxis.set_major_formatter(PercentFormatter(1.0))

    axes[2, 1].hist(
        portfolio.daily["net_return"].to_numpy(),
        bins=60,
        color="#4C72B0",
        alpha=0.85,
    )
    axes[2, 1].axvline(0.0, color="black", linewidth=0.8)
    axes[2, 1].set_title("Net return distribution")
    axes[2, 1].xaxis.set_major_formatter(PercentFormatter(1.0))

    axes[3, 0].plot(portfolio.exposures.index, portfolio.exposures["long"], label="Long")
    axes[3, 0].plot(portfolio.exposures.index, -portfolio.exposures["short"], label="Short")
    axes[3, 0].plot(
        portfolio.exposures.index,
        portfolio.exposures["net"],
        label="Net",
        alpha=0.8,
    )
    axes[3, 0].set_title("Portfolio exposure")
    axes[3, 0].legend()

    axes[3, 1].plot(
        portfolio.turnover.index,
        portfolio.turnover["one_way_turnover"],
        color="#DD8452",
    )
    axes[3, 1].set_title("One-way turnover")

    for axis in axes.flat:
        axis.grid(True, alpha=0.2)
    figure.tight_layout()


def plot_evaluation(evaluation: TestEvaluation) -> None:
    """Display task-aware signal and portfolio dashboards at the end of evaluation."""
    _plot_signal(evaluation)
    _plot_signal_time(evaluation)
    _plot_portfolio(evaluation)
    plt.show()
