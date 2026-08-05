from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

type IcBasis = Literal["cross_sectional", "rolling_time_series"]


@dataclass(frozen=True, slots=True)
class SignalEvaluation:
    """Collect portfolio-score quality summaries on the common evaluation sample.

    ``metrics`` contains scalar whole-sample statistics. ``per_date`` and ``rolling`` use
    a date index; ``decay`` is indexed by outcome lead; and ``buckets`` is indexed by
    signal bucket. ``ic_basis`` makes the IC convention explicit: a genuine panel uses
    same-date cross-sectional correlations, while a single-asset output uses trailing
    time-series correlations. Association metrics retain native scores; directional metrics
    use the objective-normalized allocation sign.
    """

    kind: Literal["portfolio"]
    ic_basis: IcBasis
    metrics: dict[str, float]
    per_date: pd.DataFrame
    rolling: pd.DataFrame
    decay: pd.DataFrame
    buckets: pd.DataFrame
    confusion: pd.DataFrame
    roc: pd.DataFrame | None = None
