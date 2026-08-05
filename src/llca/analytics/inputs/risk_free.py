"""Shared causal alignment for daily risk-free return inputs."""

from __future__ import annotations

import numpy as np
import pandas as pd


def align_risk_free(risk_free: pd.Series, dates: pd.Index) -> pd.Series:
    """Reindex the risk-free series onto ``dates`` using only current or past observations.

    Each requested date takes the most recent risk-free value at or before it (forward fill), so
    no future information leaks into excess-return accounting. Raises ``ValueError`` if the source
    is empty, has duplicate dates, holds a non-finite value, or cannot cover a requested date from
    the past.
    """
    if risk_free.empty:
        raise ValueError("risk-free series must not be empty")
    if risk_free.index.has_duplicates:
        raise ValueError("risk-free series must have a unique date index")
    ordered = risk_free.sort_index().astype(float)
    if not np.isfinite(ordered.to_numpy(dtype=float)).all():
        raise ValueError("risk-free series contains non-finite values")
    aligned = ordered.reindex(dates, method="ffill")
    if aligned.isna().any():
        first = aligned.index[aligned.isna()][0]
        raise ValueError(
            f"risk-free series has no current or prior observation for evaluation date {first!s}"
        )
    return aligned
