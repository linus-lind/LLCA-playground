from __future__ import annotations

from typing import Any, Literal, cast

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew
from sklearn.metrics import (  # type: ignore[import-untyped]
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from llca.analytics.modules.signal_evaluation import SignalEvaluation
from llca.data.index_spec import entity_level, time_level
from llca.models.estimators.prediction import PredictionOutput

_EPS = 1e-12


def _finite_correlation(
    left: pd.Series,
    right: pd.Series,
    method: Literal["pearson", "spearman"],
) -> float:
    """Return a correlation only when both aligned series contain usable variation."""
    if len(left) < 2 or left.nunique() < 2 or right.nunique() < 2:
        return float("nan")
    return float(left.corr(right, method=method))


def _safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if abs(denominator) > _EPS else float("nan")


def _shape_statistics(values: pd.Series) -> tuple[float, float]:
    """Return bias-corrected skewness and excess kurtosis when statistically defined."""
    if len(values) < 4 or float(values.std(ddof=1)) <= _EPS:
        return float("nan"), float("nan")
    array = values.to_numpy(dtype=float)
    return (
        float(skew(array, bias=False)),
        float(kurtosis(array, fisher=True, bias=False)),
    )


def _binary_directions(
    scores: pd.Series, target: pd.Series, target_threshold: float
) -> tuple[pd.Series, pd.Series]:
    return scores > 0.0, target > target_threshold


def _panel_diagnostics(
    scores: pd.Series,
    target: pd.Series,
    target_threshold: float,
    annualization_periods: int,
) -> tuple[dict[str, float], pd.DataFrame]:
    """Measure cross-sectional signal quality independently on each date.

    For each date, Pearson IC, rank IC, and directional hit rate are computed over that
    date's entities. Aggregate metrics describe their distribution through time, including
    ICIR and coverage; no date receives extra weight because it contains more entities.
    """
    frame = pd.DataFrame({"score": scores, "target": target})
    time = str(time_level(frame))
    rows: list[dict[str, float | pd.Timestamp]] = []
    for date, group in frame.groupby(level=time, sort=True):
        predicted_positive, actual_positive = _binary_directions(
            group["score"], group["target"], target_threshold
        )
        rows.append(
            {
                "date": pd.Timestamp(cast(Any, date)),
                "observations": float(len(group)),
                "pearson_ic": _finite_correlation(group["score"], group["target"], "pearson"),
                "rank_ic": _finite_correlation(group["score"], group["target"], "spearman"),
                "hit_rate": float((predicted_positive == actual_positive).mean()),
            }
        )
    per_date = pd.DataFrame(rows).set_index("date") if rows else pd.DataFrame()

    metrics: dict[str, float] = {}
    for column, prefix in (("pearson_ic", "pearson_ic"), ("rank_ic", "rank_ic")):
        values = per_date[column].dropna()
        mean = float(values.mean()) if not values.empty else float("nan")
        std = float(values.std(ddof=1)) if len(values) > 1 else float("nan")
        metrics |= {
            f"mean_daily_{prefix}": mean,
            f"median_daily_{prefix}": (
                float(values.median()) if not values.empty else float("nan")
            ),
            f"std_daily_{prefix}": std,
            f"positive_daily_{prefix}_rate": (
                float((values > 0.0).mean()) if not values.empty else float("nan")
            ),
            f"{prefix}_ir": _safe_ratio(mean, std),
            f"annualized_{prefix}_ir": _safe_ratio(mean, std) * np.sqrt(annualization_periods),
        }
    hit_rates = per_date["hit_rate"].dropna()
    metrics |= {
        "mean_daily_hit_rate": float(hit_rates.mean()),
        "median_daily_hit_rate": float(hit_rates.median()),
        "std_daily_hit_rate": float(hit_rates.std(ddof=1)),
        "minimum_daily_coverage": float(per_date["observations"].min()),
        "median_daily_coverage": float(per_date["observations"].median()),
        "maximum_daily_coverage": float(per_date["observations"].max()),
    }
    return metrics, per_date


def _rolling_diagnostics(
    per_date: pd.DataFrame,
    rolling_window: int,
    annualization_periods: int,
) -> pd.DataFrame:
    """Compute full-window rolling means, dispersion, and annualized IC information ratios."""
    rolling = pd.DataFrame(index=per_date.index)
    for column in ("pearson_ic", "rank_ic", "hit_rate", "accuracy", "balanced_accuracy"):
        if column not in per_date:
            continue
        mean = per_date[column].rolling(rolling_window, min_periods=rolling_window).mean()
        std = per_date[column].rolling(rolling_window, min_periods=rolling_window).std(ddof=1)
        rolling[f"mean_{column}"] = mean
        rolling[f"std_{column}"] = std
        if column in ("pearson_ic", "rank_ic"):
            rolling[f"{column}_ir"] = mean / std.replace(0.0, np.nan)
            rolling[f"annualized_{column}_ir"] = rolling[f"{column}_ir"] * np.sqrt(
                annualization_periods
            )
    return rolling


def _shift_target(target: pd.Series, periods: int) -> pd.Series:
    """Align future targets to current rows without shifting values across entities."""
    if periods == 0:
        return target
    entity = entity_level(target)
    return (
        target.shift(-periods) if entity is None else target.groupby(level=entity).shift(-periods)
    )


def _signal_decay(
    scores: pd.Series,
    target: pd.Series,
    periods: tuple[int, ...],
) -> pd.DataFrame:
    """Relate today's score to outcomes at each configured future lead.

    A lead ``L`` pairs a score at row ``t`` with the same entity's target at ``t + L``.
    Both pooled and equal-date-weighted correlations are returned, so cross-sectional
    coverage changes do not silently determine the reported decay profile.
    """
    rows = []
    time = time_level(scores)
    for lead in periods:
        shifted = _shift_target(target, lead)
        valid = shifted.notna()
        lead_scores = scores[valid]
        lead_target = shifted[valid]
        daily_pearson: list[float] = []
        daily_rank: list[float] = []
        frame = pd.DataFrame({"score": lead_scores, "target": lead_target})
        for _, group in frame.groupby(level=time, sort=True):
            daily_pearson.append(_finite_correlation(group["score"], group["target"], "pearson"))
            daily_rank.append(_finite_correlation(group["score"], group["target"], "spearman"))
        pearson_series = pd.Series(daily_pearson, dtype=float).dropna()
        rank_series = pd.Series(daily_rank, dtype=float).dropna()
        pearson_mean = float(pearson_series.mean())
        rank_mean = float(rank_series.mean())
        pearson_std = float(pearson_series.std(ddof=1))
        rank_std = float(rank_series.std(ddof=1))
        rows.append(
            {
                "lead": lead,
                "observations": float(valid.sum()),
                "dates": float(frame.index.get_level_values(time).nunique()),
                "pooled_pearson": _finite_correlation(lead_scores, lead_target, "pearson"),
                "pooled_rank": _finite_correlation(lead_scores, lead_target, "spearman"),
                "mean_daily_pearson_ic": pearson_mean,
                "mean_daily_rank_ic": rank_mean,
                "pearson_ic_ir": _safe_ratio(pearson_mean, pearson_std),
                "rank_ic_ir": _safe_ratio(rank_mean, rank_std),
                "positive_daily_rank_ic_rate": float((rank_series > 0.0).mean()),
            }
        )
    return pd.DataFrame(rows).set_index("lead")


def _bucket_numbers(scores: pd.Series, bucket_count: int) -> pd.Series:
    """Assign equal-count rank buckets within each date, or globally for date-only data."""
    if scores.index.nlevels > 1:
        time = time_level(scores)
        percentiles = scores.groupby(level=time).rank(method="first", pct=True)
    else:
        percentiles = scores.rank(method="first", pct=True)
    numbers = np.ceil(percentiles * bucket_count).clip(1, bucket_count).astype(int)
    return pd.Series(numbers, index=scores.index, name="bucket")


def _signal_buckets(
    scores: pd.Series,
    target: pd.Series,
    bucket_count: int,
    target_threshold: float,
) -> pd.DataFrame:
    """Summarize outcome level and directional accuracy across ordered signal buckets."""
    frame = pd.DataFrame(
        {
            "score": scores,
            "target": target,
            "bucket": _bucket_numbers(scores, bucket_count),
        }
    )
    frame["correct_direction"] = (frame["score"] > 0.0) == (frame["target"] > target_threshold)
    return frame.groupby("bucket", observed=True).agg(
        observations=("target", "size"),
        mean_score=("score", "mean"),
        mean_outcome=("target", "mean"),
        median_outcome=("target", "median"),
        hit_rate=("correct_direction", "mean"),
    )


def _continuous_metrics(
    scores: pd.Series,
    target: pd.Series,
    target_threshold: float,
    annualization_periods: int,
    include_forecast_errors: bool,
) -> tuple[dict[str, float], pd.DataFrame]:
    """Evaluate scalar ranking or regression outputs at pooled and per-date levels.

    Both tasks receive correlation, direction, score-shape, IC, ICIR, and coverage metrics.
    Absolute forecast errors and linear calibration are added only for regression because
    ranking-score magnitudes need not share the target's units.
    """
    predicted_positive, actual_positive = _binary_directions(scores, target, target_threshold)
    correct = predicted_positive == actual_positive
    target_scale = float(target.abs().sum())
    score_skewness, score_kurtosis = _shape_statistics(scores)
    metrics = {
        "observations": float(len(target)),
        "score_mean": float(scores.mean()),
        "score_std": float(scores.std(ddof=1)),
        "score_skewness": score_skewness,
        "score_excess_kurtosis": score_kurtosis,
        "outcome_mean": float(target.mean()),
        "pearson_correlation": _finite_correlation(scores, target, "pearson"),
        "spearman_correlation": _finite_correlation(scores, target, "spearman"),
        "directional_accuracy": float(correct.mean()),
        "magnitude_weighted_directional_accuracy": (
            float((target.abs() * correct.astype(float)).sum()) / target_scale
            if target_scale > _EPS
            else float("nan")
        ),
    }
    if include_forecast_errors:
        errors = scores - target
        design = np.column_stack([np.ones(len(scores)), scores.to_numpy(dtype=float)])
        intercept, slope = np.linalg.lstsq(design, target.to_numpy(dtype=float), rcond=None)[0]
        metrics |= {
            "mean_error": float(errors.mean()),
            "mae": float(mean_absolute_error(target, scores)),
            "rmse": float(np.sqrt(mean_squared_error(target, scores))),
            "r_squared": (
                float(r2_score(target, scores)) if target.nunique() > 1 else float("nan")
            ),
            "calibration_intercept": float(intercept),
            "calibration_slope": float(slope),
        }
    panel_metrics, per_date = _panel_diagnostics(
        scores,
        target,
        target_threshold,
        annualization_periods,
    )
    return metrics | panel_metrics, per_date


def _quantile_metrics(
    quantiles: pd.DataFrame,
    target: pd.Series,
) -> tuple[dict[str, float], pd.DataFrame]:
    """Evaluate quantile forecasts without assuming a parametric predictive distribution."""
    levels = np.asarray(quantiles.columns, dtype=float)
    forecasts = quantiles.to_numpy(dtype=float)
    outcomes = target.to_numpy(dtype=float)[:, np.newaxis]
    errors = outcomes - forecasts
    losses = np.maximum(levels * errors, (levels - 1.0) * errors)
    mean_losses = losses.mean(axis=0)
    empirical_coverage = (outcomes <= forecasts).mean(axis=0)
    calibration = pd.DataFrame(
        {
            "observations": len(target),
            "mean_pinball_loss": mean_losses,
            "empirical_coverage": empirical_coverage,
            "calibration_error": empirical_coverage - levels,
        },
        index=pd.Index(levels, name="quantile"),
    )
    lower = forecasts[:, 0]
    upper = forecasts[:, -1]
    interval_coverage = ((target.to_numpy() >= lower) & (target.to_numpy() <= upper)).mean()
    metrics = {
        "quantile_crps_approximation": float(np.mean(2.0 * np.trapezoid(losses, x=levels, axis=1))),
        "quantile_calibration_mae": float(np.mean(np.abs(empirical_coverage - levels))),
        "central_interval_nominal_coverage": float(levels[-1] - levels[0]),
        "central_interval_empirical_coverage": float(interval_coverage),
        "central_interval_average_width": float(np.mean(upper - lower)),
    }
    for level, loss in zip(levels, mean_losses, strict=True):
        metrics[f"pinball_loss_q{level:g}"] = float(loss)
    return metrics, calibration


def _evaluate_continuous(
    predictions: PredictionOutput,
    target: pd.Series,
    *,
    bucket_count: int,
    target_threshold: float,
    annualization_periods: int,
    rolling_window: int,
    signal_decay_periods: tuple[int, ...],
) -> SignalEvaluation:
    """Assemble scalar signal metrics, buckets, rolling stability, decay, and quantiles."""
    if not isinstance(predictions.values, pd.Series):
        raise TypeError(f"{predictions.kind} evaluation requires one scalar prediction per row")
    scores = predictions.values.astype(float)
    metrics, per_date = _continuous_metrics(
        scores,
        target,
        target_threshold=target_threshold,
        annualization_periods=annualization_periods,
        include_forecast_errors=predictions.kind == "regression",
    )
    buckets = _signal_buckets(scores, target, bucket_count, target_threshold)
    if len(buckets) > 1:
        metrics["top_minus_bottom_outcome"] = float(
            buckets["mean_outcome"].iloc[-1] - buckets["mean_outcome"].iloc[0]
        )
        metrics["bucket_monotonicity"] = _finite_correlation(
            pd.Series(buckets.index, index=buckets.index, dtype=float),
            buckets["mean_outcome"],
            "spearman",
        )
    calibration = None
    if predictions.kind == "regression" and predictions.quantiles is not None:
        quantile_metrics, calibration = _quantile_metrics(predictions.quantiles, target)
        metrics |= quantile_metrics
    return SignalEvaluation(
        kind=predictions.kind,
        metrics=metrics,
        per_date=per_date,
        rolling=_rolling_diagnostics(per_date, rolling_window, annualization_periods),
        decay=_signal_decay(scores, target, signal_decay_periods),
        buckets=buckets,
        calibration=calibration,
    )


def _classification_per_date(actual: pd.Series, predicted: pd.Series) -> pd.DataFrame:
    """Measure classification accuracy per date without weighting by cross-section size."""
    frame = pd.DataFrame({"actual": actual, "predicted": predicted})
    time = time_level(frame)
    rows = []
    for date, group in frame.groupby(level=time, sort=True):
        rows.append(
            {
                "date": pd.Timestamp(cast(Any, date)),
                "observations": float(len(group)),
                "accuracy": float(accuracy_score(group["actual"], group["predicted"])),
                "balanced_accuracy": (
                    float(balanced_accuracy_score(group["actual"], group["predicted"]))
                    if group["actual"].nunique() > 1
                    else float("nan")
                ),
            }
        )
    return pd.DataFrame(rows).set_index("date")


def _binary_classification(
    predictions: PredictionOutput,
    target: pd.Series,
    *,
    classification_threshold: float,
    probability_bins: int,
    bucket_count: int,
    rolling_window: int,
    annualization_periods: int,
) -> SignalEvaluation:
    """Evaluate binary discrimination, threshold decisions, calibration, and stability.

    The sorted target labels define negative and positive classes. Probabilities, when
    supplied, drive thresholding and proper scoring rules; otherwise native decision scores
    are used. ROC and precision-recall metrics remain threshold-independent, while the
    confusion matrix and accuracy metrics use ``classification_threshold``.
    """
    if not isinstance(predictions.values, pd.Series):
        raise TypeError("binary classification requires Series decision scores")
    classes = list(pd.Index(target.unique()).sort_values())
    if len(classes) != 2:
        raise ValueError(
            f"binary classification requires exactly two target classes, got {classes}"
        )
    negative_label, positive_label = classes
    probability = (
        predictions.probabilities if isinstance(predictions.probabilities, pd.Series) else None
    )
    decision = probability if probability is not None else predictions.values.astype(float)
    predicted = pd.Series(
        np.where(decision >= classification_threshold, positive_label, negative_label),
        index=target.index,
        name="predicted_class",
    )
    actual_positive = target == positive_label
    predicted_positive = predicted == positive_label
    tn, fp, fn, tp = confusion_matrix(
        actual_positive, predicted_positive, labels=[False, True]
    ).ravel()
    metrics = {
        "observations": float(len(target)),
        "positive_prevalence": float(actual_positive.mean()),
        "accuracy": float(accuracy_score(actual_positive, predicted_positive)),
        "balanced_accuracy": float(balanced_accuracy_score(actual_positive, predicted_positive)),
        "precision": float(precision_score(actual_positive, predicted_positive, zero_division=0)),
        "recall_sensitivity": float(
            recall_score(actual_positive, predicted_positive, zero_division=0)
        ),
        "specificity": _safe_ratio(float(tn), float(tn + fp)),
        "negative_predictive_value": _safe_ratio(float(tn), float(tn + fn)),
        "f1": float(f1_score(actual_positive, predicted_positive, zero_division=0)),
        "matthews_correlation": float(matthews_corrcoef(actual_positive, predicted_positive)),
        "roc_auc": float(roc_auc_score(actual_positive, decision)),
        "average_precision": float(average_precision_score(actual_positive, decision)),
    }
    calibration = None
    if probability is not None:
        metrics["brier_score"] = float(
            mean_squared_error(actual_positive.astype(float), probability)
        )
        metrics["log_loss"] = float(log_loss(actual_positive, probability, labels=[False, True]))
        quantile_bins = pd.qcut(
            probability.rank(method="first"),
            q=min(probability_bins, len(probability)),
            labels=False,
            duplicates="drop",
        )
        calibration_frame = pd.DataFrame(
            {
                "probability": probability,
                "actual_positive": actual_positive.astype(float),
                "bin": quantile_bins,
            }
        )
        calibration = calibration_frame.groupby("bin", observed=True).agg(
            observations=("actual_positive", "size"),
            mean_predicted_probability=("probability", "mean"),
            observed_positive_rate=("actual_positive", "mean"),
        )
        metrics["expected_calibration_error"] = float(
            np.average(
                np.abs(
                    calibration["observed_positive_rate"]
                    - calibration["mean_predicted_probability"]
                ),
                weights=calibration["observations"],
            )
        )

    fpr, tpr, roc_thresholds = roc_curve(actual_positive, decision)
    precision, recall, pr_thresholds = precision_recall_curve(actual_positive, decision)
    roc_data = pd.DataFrame(
        {"false_positive_rate": fpr, "true_positive_rate": tpr, "threshold": roc_thresholds}
    )
    pr_data = pd.DataFrame(
        {
            "precision": precision,
            "recall": recall,
            "threshold": np.append(pr_thresholds, np.nan),
        }
    )
    confusion = pd.DataFrame(
        [[tn, fp], [fn, tp]],
        index=pd.Index([f"actual_{negative_label}", f"actual_{positive_label}"], name="actual"),
        columns=pd.Index(
            [f"predicted_{negative_label}", f"predicted_{positive_label}"],
            name="predicted",
        ),
    )
    per_date = _classification_per_date(target, predicted)
    metrics |= {
        "mean_daily_accuracy": float(per_date["accuracy"].mean()),
        "std_daily_accuracy": float(per_date["accuracy"].std(ddof=1)),
    }

    confidence = (decision - classification_threshold).abs()
    bucket = _bucket_numbers(confidence, bucket_count)
    bucket_frame = pd.DataFrame(
        {
            "bucket": bucket,
            "confidence": confidence,
            "correct": (predicted == target).astype(float),
            "positive_rate": actual_positive.astype(float),
        }
    )
    buckets = bucket_frame.groupby("bucket", observed=True).agg(
        observations=("correct", "size"),
        mean_confidence=("confidence", "mean"),
        accuracy=("correct", "mean"),
        observed_positive_rate=("positive_rate", "mean"),
    )
    return SignalEvaluation(
        kind="classification",
        metrics=metrics,
        per_date=per_date,
        rolling=_rolling_diagnostics(per_date, rolling_window, annualization_periods),
        decay=pd.DataFrame(),
        buckets=buckets,
        confusion=confusion,
        calibration=calibration,
        roc=roc_data,
        precision_recall=pr_data,
    )


def _multiclass_classification(
    predictions: PredictionOutput,
    target: pd.Series,
    *,
    rolling_window: int,
    annualization_periods: int,
) -> SignalEvaluation:
    """Evaluate one-score-column-per-class outputs with macro and probabilistic metrics."""
    if not isinstance(predictions.values, pd.DataFrame):
        raise TypeError("multiclass classification requires one score column per class")
    scores = predictions.values
    predicted = scores.idxmax(axis=1)
    classes = list(scores.columns)
    metrics = {
        "observations": float(len(target)),
        "accuracy": float(accuracy_score(target, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(target, predicted)),
        "macro_precision": float(
            precision_score(target, predicted, average="macro", zero_division=0)
        ),
        "macro_recall": float(recall_score(target, predicted, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(target, predicted, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(target, predicted, average="weighted", zero_division=0)),
        "matthews_correlation": float(matthews_corrcoef(target, predicted)),
    }
    probabilities = (
        predictions.probabilities if isinstance(predictions.probabilities, pd.DataFrame) else None
    )
    if probabilities is not None:
        encoded = pd.get_dummies(target).reindex(columns=classes, fill_value=False).astype(float)
        metrics |= {
            "log_loss": float(log_loss(target, probabilities, labels=classes)),
            "multiclass_brier_score": float(
                np.square(probabilities.to_numpy() - encoded.to_numpy()).sum(axis=1).mean()
            ),
            "macro_ovr_roc_auc": float(
                roc_auc_score(target, probabilities, labels=classes, multi_class="ovr")
            ),
        }
    matrix = confusion_matrix(target, predicted, labels=classes)
    confusion = pd.DataFrame(
        matrix,
        index=pd.Index([f"actual_{label}" for label in classes], name="actual"),
        columns=pd.Index([f"predicted_{label}" for label in classes], name="predicted"),
    )
    per_date = _classification_per_date(target, predicted)
    buckets = pd.DataFrame()
    return SignalEvaluation(
        kind="classification",
        metrics=metrics,
        per_date=per_date,
        rolling=_rolling_diagnostics(per_date, rolling_window, annualization_periods),
        decay=pd.DataFrame(),
        buckets=buckets,
        confusion=confusion,
    )


def evaluate_signal(
    predictions: PredictionOutput,
    target: pd.Series,
    *,
    bucket_count: int,
    probability_bins: int,
    classification_threshold: float,
    target_threshold: float,
    annualization_periods: int,
    rolling_window: int,
    signal_decay_periods: tuple[int, ...],
) -> SignalEvaluation:
    """Dispatch an aligned prediction contract to ranking, regression, or classification analytics.

    Ranking and regression require one scalar per row. Multiclass classification is
    identified by a score DataFrame; a Series denotes binary classification. Every path
    returns the same ``SignalEvaluation`` container with task-inapplicable tables omitted.
    """
    if not predictions.values.index.equals(target.index):
        raise ValueError("signal predictions and target must have identical aligned indices")
    if predictions.kind in ("ranking", "regression", "allocation"):
        return _evaluate_continuous(
            predictions,
            target.astype(float),
            bucket_count=bucket_count,
            target_threshold=target_threshold,
            annualization_periods=annualization_periods,
            rolling_window=rolling_window,
            signal_decay_periods=signal_decay_periods,
        )
    if isinstance(predictions.values, pd.DataFrame):
        return _multiclass_classification(
            predictions,
            target,
            rolling_window=rolling_window,
            annualization_periods=annualization_periods,
        )
    return _binary_classification(
        predictions,
        target,
        classification_threshold=classification_threshold,
        probability_bins=probability_bins,
        bucket_count=bucket_count,
        rolling_window=rolling_window,
        annualization_periods=annualization_periods,
    )
