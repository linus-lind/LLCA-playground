from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from llca.analytics.modules.portfolio_evaluation import PortfolioEvaluation
from llca.analytics.modules.signal_evaluation import SignalEvaluation
from llca.models.estimators.prediction import PredictionOutput


@dataclass(frozen=True, slots=True)
class TestEvaluation:
    """Combine native portfolio predictions, diagnostics, and reconciled accounting.

    All sections share the same aligned test observations. ``target`` retains the realized
    return aligned to ``predictions`` so portfolio inference can be recomputed downstream
    without reloading panels. Other prediction kinds attach through the analytics evaluator
    registry once their own typed result contract exists; they never produce a partial
    portfolio result here.
    """

    predictions: PredictionOutput
    target: pd.Series
    signal: SignalEvaluation
    objective_metrics: dict[str, float]
    portfolio: PortfolioEvaluation
    valid_observations: int
    dates: int
