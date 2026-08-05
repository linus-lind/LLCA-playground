from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True, slots=True)
class PortfolioEvaluation:
    """Collect one reconciled realized portfolio path and its derived reports.

    Date-by-entity tables include ``weights``, ``asset_returns``, and additive
    ``asset_contributions``. Date-indexed tables describe returns, exposures, turnover,
    costs, composition, drawdowns, rolling metrics, and tail risk. Period and attribution
    tables are derived from the same path so headline, side, asset, and drawdown totals can
    be checked against each other.
    """

    metrics: dict[str, float]
    daily: pd.DataFrame
    weights: pd.DataFrame
    asset_returns: pd.DataFrame
    asset_contributions: pd.DataFrame
    exposures: pd.DataFrame
    turnover: pd.DataFrame
    costs: pd.DataFrame
    composition: pd.DataFrame
    drawdowns: pd.DataFrame
    rolling: pd.DataFrame
    tail_risk: pd.DataFrame
    monthly_returns: pd.DataFrame
    yearly_returns: pd.DataFrame
    asset_attribution: pd.DataFrame
    side_attribution: pd.DataFrame
    signal_attribution: pd.DataFrame
    maximum_drawdown_attribution: pd.DataFrame
