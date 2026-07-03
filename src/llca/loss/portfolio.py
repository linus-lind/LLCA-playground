from typing import Literal, cast

import torch
from torch import Tensor, nn

from llca.core.returns import ReturnType
from llca.loss.modules.portfolio_loss_output import PortfolioLossOutput

_EPS = 1e-12
type PortfolioNormalization = Literal["bounded", "gross", "market_neutral"]
PORTFOLIO_NORMALIZATIONS: tuple[PortfolioNormalization, ...] = (
    "bounded",
    "gross",
    "market_neutral",
)


def _scores_to_weights(
    scores: Tensor,
    leverage: float,
    valid_mask: Tensor,
    normalization: PortfolioNormalization,
) -> Tensor:
    """Convert raw scores to masked allocations under the configured exposure rule.

    ``bounded`` keeps every valid score unchanged while its cross-sectional L1 norm is at
    most ``leverage`` and rescales only rows exceeding that maximum. It therefore lets the
    model choose cash usage, net exposure, side balance, and relative asset weights.
    ``market_neutral`` subtracts each date's valid cross-sectional mean before fixed-L1
    scaling, producing zero net exposure whenever at least two scores differ. Constant
    scores and single-entity dates map to zero instead of creating arbitrary exposure.
    ``gross`` applies fixed-L1 scaling without demeaning and supports directional or
    single-asset models.
    """
    scores = torch.where(valid_mask, scores, scores.new_zeros(()))
    gross = scores.abs().sum(dim=-1, keepdim=True)
    if normalization == "bounded":
        scale = torch.minimum(
            torch.ones_like(gross),
            leverage / gross.clamp_min(_EPS),
        )
        return scores * scale

    if normalization == "market_neutral":
        counts = valid_mask.sum(dim=-1, keepdim=True).clamp_min(1)
        means = scores.sum(dim=-1, keepdim=True) / counts
        scores = torch.where(valid_mask, scores - means, scores.new_zeros(()))
        gross = scores.abs().sum(dim=-1, keepdim=True)
    scale = torch.where(gross > _EPS, leverage / gross.clamp_min(_EPS), 0.0)
    return scores * scale


def _to_simple_returns(returns: Tensor, return_type: ReturnType) -> Tensor:
    """Convert configured periodic returns to the simple convention used in accounting."""
    simple = torch.expm1(returns) if return_type == "log" else returns
    invalid = ~torch.isfinite(simple) | (simple <= -1.0)
    if bool(invalid.any().item()):
        raise ValueError("portfolio returns must be finite and greater than -100%")
    return simple


def _portfolio_returns(weights: Tensor, returns: Tensor) -> Tensor:
    return (weights * returns).sum(dim=-1)


def _drifted_turnover(weights: Tensor, returns: Tensor) -> Tensor:
    """Measure trades relative to positions drifted by the previous simple return.

    ``weights`` and ``returns`` are ``[D, N]``. For dates after the first, previous
    holdings are advanced through asset and portfolio NAV growth before comparison with
    current target weights. Because no pre-sample portfolio is known, the first entry is
    filled with the within-block mean turnover and does not introduce an artificial trade.
    """
    if weights.shape[0] < 2:
        return weights.new_zeros(weights.shape[0])

    prev_weights = weights[:-1]
    prev_returns = returns[:-1]
    nav_growth = (1.0 + (prev_weights * prev_returns).sum(dim=-1, keepdim=True)).clamp_min(_EPS)
    drifted = prev_weights * (1.0 + prev_returns) / nav_growth
    turnover = (weights[1:] - drifted).abs().sum(dim=-1)

    first = turnover.mean().reshape(1)
    return torch.cat([first, turnover])


def _short_exposure(weights: Tensor) -> Tensor:
    return weights.clamp_max(0.0).abs().sum(dim=-1)


def _concentration(weights: Tensor) -> Tensor:
    """Per-date Herfindahl index (sum of squared weights); higher = more concentrated."""
    return weights.pow(2).sum(dim=-1)


def _common_score_penalty(scores: Tensor, valid_mask: Tensor) -> Tensor:
    """Penalize a shared directional offset without removing relative asset signals."""
    masked_scores = torch.where(valid_mask, scores, scores.new_zeros(()))
    counts = valid_mask.sum(dim=-1).clamp_min(1)
    common_score = masked_scores.sum(dim=-1) / counts
    return common_score.square().mean()


def _net_exposure_penalty(net_exposure: Tensor, tolerance: float) -> Tensor:
    """Penalize only net exposure outside the symmetric tolerance band."""
    violation = (net_exposure.abs() - tolerance).clamp_min(0.0)
    return violation.square().mean()


class PortfolioLoss(nn.Module):
    """Optimize mean portfolio utility from unnormalized cross-sectional scores.

    Inputs use dense ``[D, N]`` date-by-entity matrices with a same-shaped validity mask.
    Each date's valid scores are converted through a configured bounded, directional, or
    market-neutral allocation rule. Depending on that rule, ``leverage`` is either a
    maximum gross-exposure cap or a fixed gross-exposure target. Raw log or simple
    outcomes are converted to simple returns before the objective rewards their mean and
    penalizes variance, Herfindahl concentration, drift-adjusted turnover, and short
    borrow costs.
    """

    def __init__(
        self,
        leverage: float,
        risk_aversion: float,
        concentration_aversion: float,
        execution_fee: float,
        bid_ask_spread: float,
        slippage: float,
        borrow_cost: float,
        normalization: PortfolioNormalization = "market_neutral",
        return_type: ReturnType = "simple",
        common_score_aversion: float = 0.0,
        net_exposure_aversion: float = 0.0,
        net_exposure_tolerance: float = 0.0,
    ) -> None:
        super().__init__()
        if normalization not in PORTFOLIO_NORMALIZATIONS:
            raise ValueError(
                f"unknown portfolio normalization '{normalization}', "
                f"available: {list(PORTFOLIO_NORMALIZATIONS)}"
            )
        if return_type not in ("simple", "log"):
            raise ValueError("portfolio return_type must be 'simple' or 'log'")
        if common_score_aversion < 0.0:
            raise ValueError("common_score_aversion must be non-negative")
        if net_exposure_aversion < 0.0:
            raise ValueError("net_exposure_aversion must be non-negative")
        if not 0.0 <= net_exposure_tolerance <= leverage:
            raise ValueError("net_exposure_tolerance must be between zero and leverage")
        self.leverage = leverage
        self.risk_aversion = risk_aversion
        self.concentration_aversion = concentration_aversion
        self.execution_fee = execution_fee
        self.bid_ask_spread = bid_ask_spread
        self.slippage = slippage
        self.borrow_cost = borrow_cost
        self.common_score_aversion = common_score_aversion
        self.net_exposure_aversion = net_exposure_aversion
        self.net_exposure_tolerance = net_exposure_tolerance
        self.normalization = normalization
        self.return_type = return_type

    def normalize_weights(self, scores: Tensor, valid_mask: Tensor) -> Tensor:
        """Apply the configured cross-sectional allocation rule to raw model scores."""
        normalization = cast(
            PortfolioNormalization,
            getattr(self, "normalization", "gross"),
        )
        return _scores_to_weights(scores, self.leverage, valid_mask, normalization)

    def forward(
        self, weights: Tensor, returns: Tensor, valid_mask: Tensor | None = None
    ) -> PortfolioLossOutput:
        """Evaluate scalar utility and its components over one block of dates.

        Despite the historical ``weights`` argument name, the first tensor contains raw
        signed scores and is normalized internally. Raw outcomes follow ``return_type``
        and are converted to simple returns before position drift and accounting.
        """
        if weights.dim() != 2 or returns.shape != weights.shape:
            raise ValueError(
                f"expected scores and returns of shape [B, N], got {tuple(weights.shape)} and {tuple(returns.shape)}"
            )
        if valid_mask is None:
            valid_mask = torch.ones_like(weights, dtype=torch.bool)

        common_score_penalty = _common_score_penalty(weights, valid_mask)
        returns = torch.where(valid_mask, returns, returns.new_zeros(()))
        return_type = cast(ReturnType, getattr(self, "return_type", "simple"))
        returns = _to_simple_returns(returns, return_type)
        weights = self.normalize_weights(weights, valid_mask)

        realised = _portfolio_returns(weights, returns)
        turnover = _drifted_turnover(weights, returns)
        gross = weights.abs().sum(dim=-1)
        net = weights.sum(dim=-1)
        long = weights.clamp_min(0.0).sum(dim=-1)
        short = _short_exposure(weights)
        concentration = _concentration(weights)

        turnover_rate = self.execution_fee + self.bid_ask_spread + self.slippage
        cost = turnover_rate * turnover + self.borrow_cost * short

        mean_return = realised.mean()
        variance = realised.var(unbiased=False)
        mean_cost = cost.mean()
        mean_concentration = concentration.mean()
        net_exposure_penalty = _net_exposure_penalty(
            net,
            getattr(self, "net_exposure_tolerance", 0.0),
        )
        market_penalty = (
            getattr(self, "common_score_aversion", 0.0) * common_score_penalty
            + getattr(self, "net_exposure_aversion", 0.0) * net_exposure_penalty
        )
        utility = (
            mean_return
            - self.risk_aversion * variance
            - self.concentration_aversion * mean_concentration
            - mean_cost
            - market_penalty
        )

        return PortfolioLossOutput(
            loss=-utility,
            mean_return=mean_return,
            variance=variance,
            turnover=turnover.mean(),
            cost=mean_cost,
            gross_exposure=gross.mean(),
            net_exposure=net.mean(),
            long_exposure=long.mean(),
            short_exposure=short.mean(),
            concentration=mean_concentration,
            common_score_penalty=common_score_penalty,
            net_exposure_penalty=net_exposure_penalty,
            market_penalty=market_penalty,
        )
