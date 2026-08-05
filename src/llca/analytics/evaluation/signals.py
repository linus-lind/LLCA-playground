"""Portfolio-score diagnostics on a common held-out return sample."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import (  # type: ignore[import-untyped]
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

from llca.analytics.modules.signal_evaluation import IcBasis, SignalEvaluation
from llca.analytics.stats.statistics import EPS, rank_buckets, shape_statistics
from llca.data.index_spec import entity_level, time_level
from llca.models.estimators.prediction import PredictionOutput


def _unsupported_kind(kind: str) -> NotImplementedError:
    return NotImplementedError(
        f"analytics for prediction kind '{kind}' is not implemented; "
        "register a dedicated evaluator before enabling this prediction contract"
    )


def _finite_correlation(
    left: pd.Series,
    right: pd.Series,
    method: Literal["pearson", "spearman"],
) -> float:
    """Correlate two series on their shared index, or return ``nan`` if it is ill-defined.

    Requires at least two overlapping observations and genuine variation in both series;
    otherwise the correlation is undefined and ``nan`` is returned. Callers in the per-date and
    rolling hot loops pass index-aligned series, so the shared-index case works on numpy arrays
    directly; a differing index still falls back to an inner join. For finite inputs this equals
    ``concat(join="inner").dropna()`` followed by a guarded pandas ``corr`` but is several times
    cheaper; non-finite values (``NaN`` or ``inf``) are excluded, and it never evaluates a
    degenerate correlation (so it emits no warnings).
    """
    if left.index.equals(right.index):
        left_values = left.to_numpy(dtype=float)
        right_values = right.to_numpy(dtype=float)
    else:
        joint = pd.concat([left, right], axis=1, join="inner")
        left_values = joint.iloc[:, 0].to_numpy(dtype=float)
        right_values = joint.iloc[:, 1].to_numpy(dtype=float)
    mask = np.isfinite(left_values) & np.isfinite(right_values)
    if int(mask.sum()) < 2:
        return float("nan")
    first = left_values[mask]
    second = right_values[mask]
    if bool((first == first[0]).all()) or bool((second == second[0]).all()):
        return float("nan")
    if method == "spearman":
        first = rankdata(first)
        second = rankdata(second)
    return float(np.corrcoef(first, second)[0, 1])


def _safe_ratio(numerator: float, denominator: float) -> float:
    return (
        numerator / denominator
        if np.isfinite(denominator) and abs(denominator) > EPS
        else float("nan")
    )


def _rolling_pair_correlation(
    scores: pd.Series,
    target: pd.Series,
    window: int,
    method: Literal["pearson", "spearman"],
) -> pd.Series:
    """Compute the trailing ``window``-period correlation of scores and target at each date.

    Each point holds the correlation over the preceding ``window`` observations; positions with
    fewer than a full window, or a window without usable variation, are left ``nan``.
    """
    result = pd.Series(np.nan, index=scores.index, dtype=float)
    if window < 2 or len(scores) < window:
        return result
    for end in range(window, len(scores) + 1):
        block = slice(end - window, end)
        result.iloc[end - 1] = _finite_correlation(
            scores.iloc[block],
            target.iloc[block],
            method,
        )
    return result


def _date_diagnostics(
    scores: pd.Series,
    decisions: pd.Series,
    target: pd.Series,
    target_threshold: float,
    active_weight_threshold: float,
    rolling_window: int,
) -> tuple[IcBasis, pd.DataFrame]:
    """Produce the canonical per-date coverage, hit-rate, and IC frame plus its IC basis.

    Direction is scored only where the portfolio takes an active position. Information
    coefficients are computed cross-sectionally per date when a usable cross-section exists,
    signalled by the ``"cross_sectional"`` basis; otherwise the basis is
    ``"rolling_time_series"`` and the ICs are trailing ``rolling_window`` correlations of the
    date-collapsed series. The returned frame is the single source feeding headline metrics,
    rolling diagnostics, and the significance tests.
    """
    frame = pd.DataFrame({"score": scores, "target": target})
    time = time_level(frame)
    entity = entity_level(frame)
    active = decisions.abs() > active_weight_threshold
    correct = ((decisions > 0.0) == (frame["target"] > target_threshold)).astype(float)
    correct = correct.where(active)
    magnitude = frame["target"].abs()

    records: dict[object, dict[str, float]] = {}
    pearson: dict[object, float] = {}
    rank: dict[object, float] = {}
    usable_cross_section = False
    for date, group in frame.groupby(level=time, sort=True):
        positions = group.index
        group_correct = correct.reindex(positions)
        group_magnitude = magnitude.reindex(positions).where(group_correct.notna())
        magnitude_total = float(group_magnitude.sum(min_count=1))
        records[date] = {
            "observations": float(len(group)),
            "directional_observations": float(group_correct.notna().sum()),
            "hit_rate": float(group_correct.mean()),
            "magnitude_weighted_hit_rate": (
                float((group_magnitude * group_correct).sum(min_count=1)) / magnitude_total
                if np.isfinite(magnitude_total) and magnitude_total > EPS
                else float("nan")
            ),
        }
        # The per-date cross-sectional ICs need the same grouping, so compute them in this pass
        # instead of regrouping the panel a second time.
        if entity is not None:
            pearson_value = _finite_correlation(group["score"], group["target"], "pearson")
            rank_value = _finite_correlation(group["score"], group["target"], "spearman")
            pearson[date] = pearson_value
            rank[date] = rank_value
            usable_cross_section |= bool(np.isfinite(pearson_value) and np.isfinite(rank_value))
    per_date = pd.DataFrame.from_dict(records, orient="index").sort_index()
    per_date.index.name = time

    if usable_cross_section:
        basis: IcBasis = "cross_sectional"
        per_date["pearson_ic"] = pd.Series(pearson, dtype=float)
        per_date["rank_ic"] = pd.Series(rank, dtype=float)
        return basis, per_date

    basis = "rolling_time_series"
    collapsed = frame.groupby(level=time, sort=True).mean(numeric_only=True)
    per_date["pearson_ic"] = _rolling_pair_correlation(
        collapsed["score"], collapsed["target"], rolling_window, "pearson"
    ).reindex(per_date.index)
    per_date["rank_ic"] = _rolling_pair_correlation(
        collapsed["score"], collapsed["target"], rolling_window, "spearman"
    ).reindex(per_date.index)
    return basis, per_date


def _ic_metrics(
    per_date: pd.DataFrame,
    annualization_periods: int,
    basis: IcBasis,
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for column in ("pearson_ic", "rank_ic"):
        values = per_date[column].dropna()
        mean = float(values.mean()) if not values.empty else float("nan")
        std = float(values.std(ddof=1)) if len(values) > 1 else float("nan")
        information_ratio = _safe_ratio(mean, std)
        metrics |= {
            f"mean_daily_{column}": mean,
            f"median_daily_{column}": (
                float(values.median()) if not values.empty else float("nan")
            ),
            f"std_daily_{column}": std,
            f"positive_daily_{column}_rate": (
                float((values > 0.0).mean()) if not values.empty else float("nan")
            ),
            f"{column}_ir": information_ratio,
            f"annualized_{column}_ir": (
                information_ratio * np.sqrt(annualization_periods)
                if basis == "cross_sectional"
                else float("nan")
            ),
        }
    return metrics


def _rolling_diagnostics(
    per_date: pd.DataFrame,
    rolling_window: int,
    annualization_periods: int,
    basis: IcBasis,
) -> pd.DataFrame:
    """Roll the per-date IC and hit-rate columns into trailing means and dispersions.

    For each metric it emits a rolling mean and standard deviation over ``rolling_window``; for
    the IC columns it also emits the rolling information ratio and, on a cross-sectional basis,
    its annualization. Windows shorter than ``rolling_window`` stay ``nan``.
    """
    rolling = pd.DataFrame(index=per_date.index)
    for column in ("pearson_ic", "rank_ic", "hit_rate", "magnitude_weighted_hit_rate"):
        values = per_date[column]
        mean = values.rolling(rolling_window, min_periods=rolling_window).mean()
        std = values.rolling(rolling_window, min_periods=rolling_window).std(ddof=1)
        rolling[f"mean_{column}"] = mean
        rolling[f"std_{column}"] = std
        if column in ("pearson_ic", "rank_ic"):
            rolling[f"{column}_ir"] = mean / std.replace(0.0, np.nan)
            rolling[f"annualized_{column}_ir"] = (
                rolling[f"{column}_ir"] * np.sqrt(annualization_periods)
                if basis == "cross_sectional"
                else np.nan
            )
    return rolling


def _shift_target(target: pd.Series, periods: int) -> pd.Series:
    """Bring each row's outcome ``periods`` steps into the future onto the current row.

    For a panel the shift is applied per entity, in date order, so an outcome never leaks from
    one instrument into another; a date-only series is shifted directly. A zero lead returns the
    target unchanged.
    """
    if periods == 0:
        return target
    entity = entity_level(target)
    time = time_level(target)
    if entity is None:
        ordered = target.sort_index()
        return ordered.shift(-periods).reindex(target.index)
    ordered = target.sort_index(level=[entity, time])
    return ordered.groupby(level=entity, sort=False).shift(-periods).reindex(target.index)


def _basis_ic(
    scores: pd.Series,
    target: pd.Series,
    basis: IcBasis,
    rolling_window: int,
) -> tuple[float, float, float, float]:
    """Return mean Pearson and rank IC with their series standard deviations for one basis.

    Uses per-date cross-sectional correlations on a ``"cross_sectional"`` basis and trailing
    ``rolling_window`` correlations of the date-collapsed series otherwise, so the estimates
    match the report's IC convention. Returns ``(mean_pearson, mean_rank, std_pearson,
    std_rank)``, with ``nan`` where a quantity is undefined.
    """
    if basis == "cross_sectional":
        time = time_level(scores)
        pearson: list[float] = []
        rank: list[float] = []
        frame = pd.DataFrame({"score": scores, "target": target})
        for _, group in frame.groupby(level=time, sort=True):
            pearson.append(_finite_correlation(group["score"], group["target"], "pearson"))
            rank.append(_finite_correlation(group["score"], group["target"], "spearman"))
        pearson_values = pd.Series(pearson, dtype=float).dropna()
        rank_values = pd.Series(rank, dtype=float).dropna()
    else:
        time = time_level(scores)
        frame = (
            pd.DataFrame({"score": scores, "target": target})
            .groupby(level=time, sort=True)
            .mean(numeric_only=True)
        )
        pearson_values = _rolling_pair_correlation(
            frame["score"], frame["target"], rolling_window, "pearson"
        ).dropna()
        rank_values = _rolling_pair_correlation(
            frame["score"], frame["target"], rolling_window, "spearman"
        ).dropna()
    return (
        float(pearson_values.mean()) if not pearson_values.empty else float("nan"),
        float(rank_values.mean()) if not rank_values.empty else float("nan"),
        float(pearson_values.std(ddof=1)) if len(pearson_values) > 1 else float("nan"),
        float(rank_values.std(ddof=1)) if len(rank_values) > 1 else float("nan"),
    )


def _signal_decay(
    scores: pd.Series,
    target: pd.Series,
    periods: tuple[int, ...],
    basis: IcBasis,
    rolling_window: int,
) -> pd.DataFrame:
    """Measure how the signal's information coefficient decays against outcomes further ahead.

    For each lead in ``periods`` the target is shifted forward that many steps, and the surviving
    pairs are summarised by basis-consistent and pooled Pearson/rank ICs, their information
    ratios, and coverage counts. Returns one row per lead, indexed by the lead length.
    """
    rows: list[dict[str, float]] = []
    time = time_level(scores)
    for lead in periods:
        shifted = _shift_target(target, lead)
        valid = shifted.notna()
        lead_scores = scores[valid]
        lead_target = shifted[valid]
        pearson, rank, pearson_std, rank_std = _basis_ic(
            lead_scores,
            lead_target,
            basis,
            rolling_window,
        )
        rows.append(
            {
                "lead": float(lead),
                "observations": float(valid.sum()),
                "dates": float(lead_scores.index.get_level_values(time).nunique()),
                "basis_pearson_ic": pearson,
                "basis_rank_ic": rank,
                "pooled_pearson": _finite_correlation(lead_scores, lead_target, "pearson"),
                "pooled_rank": _finite_correlation(lead_scores, lead_target, "spearman"),
                "pearson_ic_ir": _safe_ratio(pearson, pearson_std),
                "rank_ic_ir": _safe_ratio(rank, rank_std),
            }
        )
    result = pd.DataFrame(rows).set_index("lead")
    result.index = result.index.astype(int)
    return result


def _signal_buckets(
    scores: pd.Series,
    decisions: pd.Series,
    target: pd.Series,
    bucket_count: int,
    target_threshold: float,
    active_weight_threshold: float,
    basis: IcBasis,
) -> pd.DataFrame:
    """Group items into score-rank buckets and summarise each bucket's outcomes.

    Scores are ranked into ``bucket_count`` groups — pooled across all items on a time-series
    basis, within each date otherwise — and each bucket reports its coverage, mean score, mean
    and median outcome, and active-position hit rate. This exposes whether higher-scored items
    realize better outcomes.
    """
    frame = pd.DataFrame(
        {
            "score": scores,
            "target": target,
            "bucket": rank_buckets(
                scores,
                bucket_count,
                pooled=basis == "rolling_time_series",
            ),
        }
    )
    active = decisions.abs() > active_weight_threshold
    frame["correct_direction"] = (
        ((decisions > 0.0) == (frame["target"] > target_threshold)).astype(float).where(active)
    )
    return frame.groupby("bucket", observed=True).agg(
        observations=("target", "size"),
        directional_observations=("correct_direction", "count"),
        mean_score=("score", "mean"),
        mean_outcome=("target", "mean"),
        median_outcome=("target", "median"),
        hit_rate=("correct_direction", "mean"),
    )


def _directional_discrimination(
    decisions: pd.Series,
    target: pd.Series,
    target_threshold: float,
    active_weight_threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame | None, float]:
    """Score the sign of active positions against realized outcome direction.

    Restricting to dates where the portfolio is active, it classifies each call as up or down
    versus the realized outcome and returns the confusion matrix, the ROC curve, and the ROC
    AUC. The curve and AUC are omitted (``None``/``nan``) when the active outcomes lack both
    directions.
    """
    active = decisions.abs() > active_weight_threshold
    active_decisions = decisions[active]
    active_target = target[active]
    predicted_positive = active_decisions > 0.0
    actual_positive = active_target > target_threshold
    if actual_positive.empty:
        tn = fp = fn = tp = 0
    else:
        tn, fp, fn, tp = confusion_matrix(
            actual_positive,
            predicted_positive,
            labels=[False, True],
        ).ravel()
    confusion = pd.DataFrame(
        [[int(tn), int(fp)], [int(fn), int(tp)]],
        index=pd.Index(["actual_down", "actual_up"], name="actual"),
        columns=pd.Index(["predicted_down", "predicted_up"], name="predicted"),
    )
    if actual_positive.empty or actual_positive.nunique() < 2:
        return confusion, None, float("nan")
    false_positive, true_positive, thresholds = roc_curve(actual_positive, active_decisions)
    roc_data = pd.DataFrame(
        {
            "false_positive_rate": false_positive,
            "true_positive_rate": true_positive,
            "threshold": thresholds,
        }
    )
    return confusion, roc_data, float(roc_auc_score(actual_positive, active_decisions))


def _evaluate_portfolio_signal(
    predictions: PredictionOutput,
    target: pd.Series,
    decisions: pd.Series,
    *,
    bucket_count: int,
    target_threshold: float,
    active_weight_threshold: float,
    annualization_periods: int,
    rolling_window: int,
    signal_decay_periods: tuple[int, ...],
) -> SignalEvaluation:
    if not isinstance(predictions.values, pd.Series):
        raise TypeError("portfolio evaluation requires one scalar score per row")
    scores = predictions.values.astype(float)
    basis, per_date = _date_diagnostics(
        scores,
        decisions,
        target,
        target_threshold,
        active_weight_threshold,
        rolling_window,
    )
    active = decisions.abs() > active_weight_threshold
    correct = ((decisions > 0.0) == (target > target_threshold)).astype(float).where(active)
    score_skewness, score_kurtosis = shape_statistics(scores)
    hit_rates = per_date["hit_rate"].dropna()
    magnitude_hit_rates = per_date["magnitude_weighted_hit_rate"].dropna()
    metrics = {
        "observations": float(len(target)),
        "score_mean": float(scores.mean()),
        "score_std": float(scores.std(ddof=1)),
        "score_skewness": score_skewness,
        "score_excess_kurtosis": score_kurtosis,
        "outcome_mean": float(target.mean()),
        "pearson_correlation": _finite_correlation(scores, target, "pearson"),
        "spearman_correlation": _finite_correlation(scores, target, "spearman"),
        "directional_accuracy": (float(hit_rates.mean()) if not hit_rates.empty else float("nan")),
        "active_directional_observations": float(active.sum()),
        "active_directional_rate": float(active.mean()),
        "pooled_directional_accuracy": float(correct.mean()),
        "magnitude_weighted_directional_accuracy": (
            float(magnitude_hit_rates.mean()) if not magnitude_hit_rates.empty else float("nan")
        ),
        "mean_daily_hit_rate": (float(hit_rates.mean()) if not hit_rates.empty else float("nan")),
        "median_daily_hit_rate": (
            float(hit_rates.median()) if not hit_rates.empty else float("nan")
        ),
        "std_daily_hit_rate": (
            float(hit_rates.std(ddof=1)) if len(hit_rates) > 1 else float("nan")
        ),
        "minimum_daily_coverage": float(per_date["observations"].min()),
        "median_daily_coverage": float(per_date["observations"].median()),
        "maximum_daily_coverage": float(per_date["observations"].max()),
    }
    metrics |= _ic_metrics(per_date, annualization_periods, basis)

    buckets = _signal_buckets(
        scores,
        decisions,
        target,
        bucket_count,
        target_threshold,
        active_weight_threshold,
        basis,
    )
    if len(buckets) > 1:
        metrics["top_minus_bottom_outcome"] = float(
            buckets["mean_outcome"].iloc[-1] - buckets["mean_outcome"].iloc[0]
        )
        metrics["bucket_monotonicity"] = _finite_correlation(
            pd.Series(buckets.index, index=buckets.index, dtype=float),
            buckets["mean_outcome"],
            "spearman",
        )
    confusion, roc_data, roc_auc = _directional_discrimination(
        decisions,
        target,
        target_threshold,
        active_weight_threshold,
    )
    metrics["roc_auc"] = roc_auc
    rolling = _rolling_diagnostics(per_date, rolling_window, annualization_periods, basis)
    decay = _signal_decay(
        scores,
        target,
        signal_decay_periods,
        basis,
        rolling_window,
    )
    return SignalEvaluation(
        kind="portfolio",
        ic_basis=basis,
        metrics=metrics,
        per_date=per_date,
        rolling=rolling,
        decay=decay,
        buckets=buckets,
        confusion=confusion,
        roc=roc_data,
    )


def evaluate_signal(
    predictions: PredictionOutput,
    target: pd.Series,
    *,
    decisions: pd.Series | None = None,
    bucket_count: int,
    target_threshold: float,
    active_weight_threshold: float,
    annualization_periods: int,
    rolling_window: int,
    signal_decay_periods: tuple[int, ...],
) -> SignalEvaluation:
    """Evaluate a portfolio model's scores into a full signal-quality report.

    Requires the predictions, target, and optional ``decisions`` (defaulting to the scores) to
    share one aligned index, and rejects any non-portfolio prediction kind. Delegates the actual
    diagnostics — ICs, hit rates, buckets, decay, and directional discrimination — to the
    portfolio signal evaluator.
    """
    if not predictions.index.equals(target.index):
        raise ValueError("signal predictions and target must have identical aligned indices")
    if predictions.kind != "portfolio":
        raise _unsupported_kind(predictions.kind)
    direction_values = predictions.values if decisions is None else decisions
    if not isinstance(direction_values, pd.Series):
        raise TypeError("portfolio directional diagnostics require scalar allocation values")
    if not direction_values.index.equals(target.index):
        raise ValueError("portfolio decisions and target must have identical aligned indices")
    return _evaluate_portfolio_signal(
        predictions,
        target,
        direction_values.astype(float),
        bucket_count=bucket_count,
        target_threshold=target_threshold,
        active_weight_threshold=active_weight_threshold,
        annualization_periods=annualization_periods,
        rolling_window=rolling_window,
        signal_decay_periods=signal_decay_periods,
    )
