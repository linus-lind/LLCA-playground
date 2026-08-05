"""Cross-model target alignment guards over predicted candidates.

After every model has produced its native prediction, these guards derive the shared item
overlap and reject a comparison whose aligned supervision outcomes are not actually the same.
"""

from __future__ import annotations

from functools import reduce

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from llca.analytics.candidates.prediction import EvaluationCandidate
from llca.analytics.evaluation.predictions import valid_supervision


def common_target_index(
    candidates: list[EvaluationCandidate],
) -> pd.Index:
    """Return the items every model predicts and holds valid supervision for.

    Intersects the candidates' prediction indices, then keeps only items whose supervision is
    observed and finite for all of them. Models are still each scored on their own native
    universe; this shared set exists solely for the item-by-item cross-model checks (target
    equality and signal correlation). Raises if the prediction overlap or the valid overlap is
    empty.
    """
    common = reduce(
        lambda left, right: left.intersection(right),
        (candidate.predictions.index for candidate in candidates),
    )
    common = common.sort_values()
    if common.empty:
        raise ValueError("configured models have no common prediction items")

    valid = pd.Series(True, index=common, dtype=bool)
    for candidate in candidates:
        valid &= valid_supervision(
            candidate.supervision,
            candidate.evaluation_spec.supervision_column,
            common,
        )
    evaluation_index = common[valid.to_numpy(dtype=bool)]
    if evaluation_index.empty:
        raise ValueError("configured models have no common items with valid supervision")
    return evaluation_index


def assert_common_targets(candidates: list[EvaluationCandidate], common_index: pd.Index) -> None:
    """Verify every model sees the same supervision outcome on the shared item set.

    Compares each model's aligned targets on ``common_index`` against the first model's, using a
    numerical tolerance for numeric labels and exact equality otherwise. Raises ``ValueError``
    naming the offending pair if any two disagree, since a comparison across different realized
    outcomes would be meaningless.
    """
    reference_candidate = candidates[0]
    reference_column = reference_candidate.evaluation_spec.supervision_column
    reference = reference_candidate.supervision.values[reference_column].reindex(common_index)
    for candidate in candidates[1:]:
        target = candidate.supervision.values[candidate.evaluation_spec.supervision_column].reindex(
            common_index
        )
        if is_numeric_dtype(reference.dtype) and is_numeric_dtype(target.dtype):
            equal = np.allclose(
                reference.to_numpy(dtype=float),
                target.to_numpy(dtype=float),
                rtol=1e-10,
                atol=1e-12,
            )
        else:
            equal = reference.equals(target)
        if not equal:
            raise ValueError(
                "configured models use different supervision values on the common "
                f"universe ({reference_candidate.metadata.config.label} versus "
                f"{candidate.metadata.config.label})"
            )
