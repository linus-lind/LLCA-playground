from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True, slots=True)
class SignalEvaluation:
    """Collect task-aware signal quality summaries without portfolio assumptions.

    ``metrics`` contains scalar whole-sample statistics. ``per_date`` and ``rolling`` use
    a date index; ``decay`` is indexed by outcome lead; ``buckets`` is indexed by signal
    or confidence bucket. Classification and probabilistic tables are optional because
    their contracts do not apply to portfolio or regression outputs.
    """

    kind: str
    metrics: dict[str, float]
    per_date: pd.DataFrame
    rolling: pd.DataFrame
    decay: pd.DataFrame
    buckets: pd.DataFrame
    confusion: pd.DataFrame | None = None
    calibration: pd.DataFrame | None = None
    roc: pd.DataFrame | None = None
    precision_recall: pd.DataFrame | None = None
