"""Pack aligned scores and realized returns into the dense tensors an objective consumes.

Portfolio-style objectives operate on ``[dates, entities]`` matrices with a companion validity
mask and require every retained date to carry at least one valid entity. This helper pivots an
aligned ``(score, return)`` pair to that layout and drops any date with no jointly observed
entity, so a single-asset fold whose only entity is missing on some date does not violate that
invariant. It is intentionally distinct from the analytics common-sample packer, which relies
on a pre-intersected universe and therefore never needs to drop dates.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from torch import Tensor, from_numpy

from llca.data.index_spec import entity_level


def pack_objective_panel(
    scores: pd.Series, returns: pd.Series
) -> tuple[Tensor, Tensor, Tensor, pd.Index]:
    """Return dense ``(scores, returns, valid_mask, dates)`` over jointly observed dates.

    ``returns`` is aligned to the score index. A cell is valid where both the score and the
    return are present; dates with no valid entity are removed entirely. Invalid cells are
    zero-filled and marked ``False`` in the mask. ``dates`` is the retained row axis, used to
    align a per-date risk-free rate to the packed tensors. Raises when no date survives.
    """
    aligned_returns = returns.reindex(scores.index)
    entity = entity_level(scores)
    if entity is None:
        score_frame = scores.to_frame("value").sort_index()
        return_frame = aligned_returns.to_frame("value").reindex_like(score_frame)
    else:
        score_frame = scores.unstack(level=entity).sort_index()
        return_frame = aligned_returns.unstack(level=entity).reindex_like(score_frame)

    valid = score_frame.notna() & return_frame.notna()
    retained = np.flatnonzero(valid.to_numpy(dtype=bool).any(axis=1))
    if retained.size == 0:
        raise ValueError("objective evaluation has no jointly observed score and return")
    valid = valid.iloc[retained]
    score_frame = score_frame.iloc[retained]
    return_frame = return_frame.iloc[retained]
    mask = valid.to_numpy(dtype=bool)
    return (
        from_numpy(score_frame.to_numpy(dtype=np.float32)),
        from_numpy(np.where(mask, return_frame.to_numpy(dtype=np.float32), np.float32(0.0))),
        from_numpy(mask),
        score_frame.index,
    )
