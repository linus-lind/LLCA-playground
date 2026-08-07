"""The conservative baseline-versus-candidate adoption rule.

Because the baseline and every candidate are scored on the *same* inner folds, the comparison
is paired. Let ``L_base,i`` and ``L_cand,i`` be their fold losses (lower is better) and define
the per-fold improvement ``d_i = L_base,i - L_cand,i`` (positive when the candidate improves on
the baseline). With ``K`` folds the rule adopts the candidate only when

    mean(d) > margin * standard_error(d),   standard_error(d) = std(d, ddof=1) / sqrt(K)

so a margin of ``1.0`` is a one-standard-error rule and ``0.0`` adopts on any mean improvement.
When every fold agrees exactly (zero standard error) the candidate is adopted iff it strictly
improves the mean; otherwise the baseline is retained. This deliberately prefers the baseline
whenever the measured improvement is statistically indistinguishable from noise.
"""

from __future__ import annotations

import math
from collections.abc import Sequence


def paired_improvement(
    baseline_losses: Sequence[float], candidate_losses: Sequence[float]
) -> tuple[float, float]:
    """Return the mean and standard error of the per-fold improvement ``baseline - candidate``.

    The standard error uses the sample standard deviation (``ddof=1``) over the fold-level
    differences, so at least two folds are required for a defined value.
    """
    if len(baseline_losses) != len(candidate_losses):
        raise ValueError("baseline and candidate must be scored on the same folds")
    folds = len(baseline_losses)
    if folds < 2:
        raise ValueError("the paired standard-error rule requires at least two folds")
    differences = [
        baseline - candidate
        for baseline, candidate in zip(baseline_losses, candidate_losses, strict=True)
    ]
    mean = math.fsum(differences) / folds
    variance = math.fsum((value - mean) ** 2 for value in differences) / (folds - 1)
    standard_error = math.sqrt(variance) / math.sqrt(folds)
    return mean, standard_error


def adopt_candidate(
    baseline_losses: Sequence[float],
    candidate_losses: Sequence[float],
    *,
    standard_error_margin: float,
) -> bool:
    """Return whether the candidate beats the baseline by the configured margin of evidence."""
    mean, standard_error = paired_improvement(baseline_losses, candidate_losses)
    if standard_error == 0.0:
        return mean > 0.0
    return mean > standard_error_margin * standard_error
