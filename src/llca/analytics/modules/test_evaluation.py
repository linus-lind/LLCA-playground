from dataclasses import dataclass

import pandas as pd

from llca.analytics.modules.portfolio_evaluation import PortfolioEvaluation
from llca.analytics.modules.signal_evaluation import SignalEvaluation
from llca.models.estimators.prediction import PredictionOutput


@dataclass(frozen=True, slots=True)
class TestEvaluation:
    """Combine native predictions, signal quality, objective diagnostics, and portfolio results.

    Objective and portfolio sections are optional: some estimators expose outputs for
    which the configured objective cannot be reconstructed or which have no allocation
    normalization contract. All present sections share the same aligned test observations.
    """

    predictions: PredictionOutput
    signal: SignalEvaluation
    objective_metrics: dict[str, float]
    portfolio: PortfolioEvaluation | None
    valid_observations: int
    dates: int

    @property
    def portfolio_returns(self) -> pd.Series:
        """Backward-compatible access to gross returns for simple plotting clients."""
        if self.portfolio is None:
            raise ValueError("this model output has no portfolio construction")
        return self.portfolio.daily["gross_return"]
