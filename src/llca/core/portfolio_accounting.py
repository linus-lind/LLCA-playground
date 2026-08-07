"""Canonical residual-cash-at-risk-free portfolio return accounting.

Single source of truth for how risky asset weights, realized simple returns, and a per-date
risk-free rate combine into portfolio returns and one-period drift. Both the training
objective (:mod:`llca.loss.portfolio`) and analytics evaluation
(:mod:`llca.analytics.evaluation.portfolio`) consume these primitives so the two cannot
diverge on the funding convention.

All functions operate on dense ``[D, N]`` date-by-entity weight and simple-return tensors
with a per-date ``[D]`` risk-free vector. Risky net exposure is ``n_t = Σ_i w_{i,t}`` and
residual cash weight is ``c_t = 1 - n_t``; that residual (positive when underinvested,
negative under net leverage) earns the risk-free rate. Gross portfolio return before trading
and financing costs is therefore

    r^gross_{t} = Σ_i w_{i,t} r_{i,t} + (1 - Σ_i w_{i,t}) r^f_{t}
                = r^f_{t} + Σ_i w_{i,t} (r_{i,t} - r^f_{t}),

the ``residual_cash_at_risk_free`` funding convention. Residual cash is never clamped to
``[0, 1]``: an explicit borrowing spread would be a separate funding-cost extension.

Returns must already be simple (callers convert from log). ``risk_free`` is optional and
interpreted as zero when absent, which keeps low-level accounting usable in isolated tests;
production training and analytics always pass an explicit rate.
"""

from __future__ import annotations

from torch import Tensor

_EPS = 1e-12

FUNDING_CONVENTION = "residual_cash_at_risk_free"


def _resolve_risk_free(risk_free: Tensor | None, weights: Tensor) -> Tensor:
    """Return a per-date risk-free vector matching ``weights``' date axis, defaulting to zero.

    ``weights`` is ``[D, N]`` so the date axis is ``weights.shape[:-1]``. A supplied
    ``risk_free`` must have exactly that shape — one rate per date, never one per asset — so a
    caller cannot accidentally broadcast a per-entity series into cash accounting.
    """
    dates_shape = weights.shape[:-1]
    if risk_free is None:
        return weights.new_zeros(dates_shape)
    if risk_free.shape != dates_shape:
        raise ValueError(
            f"risk_free must have one rate per date with shape {tuple(dates_shape)}, "
            f"got {tuple(risk_free.shape)}"
        )
    return risk_free


def net_exposure(weights: Tensor) -> Tensor:
    """Return risky net exposure ``n_t = Σ_i w_{i,t}`` per date."""
    return weights.sum(dim=-1)


def residual_cash_weight(weights: Tensor) -> Tensor:
    """Return residual cash weight ``c_t = 1 - Σ_i w_{i,t}``; may be negative under leverage."""
    return 1.0 - net_exposure(weights)


def risky_return(weights: Tensor, returns: Tensor) -> Tensor:
    """Return the risky contribution ``Σ_i w_{i,t} r_{i,t}`` per date."""
    return (weights * returns).sum(dim=-1)


def cash_return_contribution(weights: Tensor, risk_free: Tensor | None) -> Tensor:
    """Return the residual-cash contribution ``(1 - Σ_i w_{i,t}) r^f_{t}`` per date."""
    return residual_cash_weight(weights) * _resolve_risk_free(risk_free, weights)


def gross_return(weights: Tensor, returns: Tensor, risk_free: Tensor | None = None) -> Tensor:
    """Return cash-inclusive gross portfolio return: risky contribution plus residual cash at rf."""
    return risky_return(weights, returns) + cash_return_contribution(weights, risk_free)


def portfolio_nav_growth(
    weights: Tensor, returns: Tensor, risk_free: Tensor | None = None
) -> Tensor:
    """Return one-period wealth growth ``1 + r^gross`` under the residual-cash funding convention.

    This is the total-portfolio denominator that renormalizes drifted weights, so risky
    holdings and residual cash are advanced under one consistent wealth path.
    """
    return 1.0 + gross_return(weights, returns, risk_free)


def drifted_weights(weights: Tensor, returns: Tensor, risk_free: Tensor | None = None) -> Tensor:
    """Return weights after one holding period's passive drift, before any rebalancing.

    Each risky holding grows by its own return while the whole book — including residual cash
    growing at the risk-free rate — renormalizes total wealth:

        w^drift_{i} = w_{i} (1 + r_{i}) / (1 + r^gross).

    ``weights`` and ``returns`` are ``[D, N]`` and ``risk_free`` is ``[D]``; the result is
    ``[D, N]``. The denominator is floored at a tiny epsilon so a fully wiped-out book yields
    finite (near-zero) drifted weights rather than a division by zero; callers that treat NAV
    exhaustion as an error should check :func:`portfolio_nav_growth` before drifting.
    """
    growth = portfolio_nav_growth(weights, returns, risk_free).clamp_min(_EPS)
    return weights * (1.0 + returns) / growth.unsqueeze(-1)
