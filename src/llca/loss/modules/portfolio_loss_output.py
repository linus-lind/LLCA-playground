from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor

from llca.loss.modules.loss_output import LossOutput


@dataclass(frozen=True, slots=True)
class PortfolioLossOutput(LossOutput):
    """Expose detached-ready scalar components of the portfolio training objective.

    Every field is a scalar tensor averaged over the date block. ``loss`` is the negative
    utility used for backpropagation; remaining fields explain its return, risk,
    concentration, turnover, financing, and execution contributions. ``mean_return`` is the
    cash-inclusive gross portfolio return and ``cash_return`` isolates the residual-cash
    contribution earned at the risk-free rate under the ``residual_cash_at_risk_free``
    convention.
    """

    mean_return: Tensor
    cash_return: Tensor
    variance: Tensor
    turnover: Tensor
    cost: Tensor
    gross_exposure: Tensor
    net_exposure: Tensor
    long_exposure: Tensor
    short_exposure: Tensor
    concentration: Tensor
    common_score_penalty: Tensor
    net_exposure_penalty: Tensor
    market_penalty: Tensor
