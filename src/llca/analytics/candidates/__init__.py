"""Turn configured registry models into a validated, aligned set of prediction candidates.

The stage runs in three steps: :mod:`.model_set` guards the comparability of the configured
registry models and derives their shared window, :mod:`.prediction` scores each model once into
an :class:`EvaluationCandidate`, and :mod:`.alignment` derives the common item universe and
rejects non-comparable supervision targets.
"""

from llca.analytics.candidates.alignment import assert_common_targets, common_target_index
from llca.analytics.candidates.model_set import (
    assert_portfolio_accounting_contract,
    assert_realization_lag_contract,
    comparison_window,
)
from llca.analytics.candidates.prediction import EvaluationCandidate, build_evaluation_candidates

__all__ = [
    "EvaluationCandidate",
    "assert_common_targets",
    "assert_portfolio_accounting_contract",
    "assert_realization_lag_contract",
    "build_evaluation_candidates",
    "common_target_index",
    "comparison_window",
]
