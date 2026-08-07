"""The single-asset tabular CV objective funds residual cash at the risk-free rate."""

import unittest

import pandas as pd
from omegaconf import OmegaConf

from llca.loss.portfolio import PortfolioLoss
from llca.models.estimators.logistic_net import LogisticNetEstimator

_TARGET = 53613


def _gross_loss() -> PortfolioLoss:
    return PortfolioLoss(
        leverage=1.0,
        risk_aversion=0.0,
        concentration_aversion=0.0,
        execution_fee=0.0,
        bid_ask_spread=0.0,
        slippage=0.0,
        borrow_cost=0.0,
        normalization="gross",
        return_type="simple",
        common_score_aversion=0.0,
        net_exposure_aversion=0.0,
        net_exposure_tolerance=1.0,
    )


def _estimator() -> LogisticNetEstimator:
    config = OmegaConf.create(
        {
            "name": "elastic-net",
            "target": {"entity_id": _TARGET},
            "inputs": {"features": "daily_values", "context": ["macro"]},
            "supervision": {"dataset": "loss", "column": "fwd_return"},
            "classification": {"dataset": "loss", "column": "fwd_direction"},
            "risk_free": {"dataset": "risk_free", "column": "rf"},
            "l1_ratio": 0.5,
            "C": 1.0,
        }
    )
    return LogisticNetEstimator(config, "portfolio", cv_objective=_gross_loss())


def _single_asset(values: list[float], name: str) -> pd.Series:
    dates = pd.bdate_range("2021-01-01", periods=len(values))
    index = pd.MultiIndex.from_product([dates, [_TARGET]], names=["date", "instrument_id"])
    return pd.Series(values, index=index, name=name)


class SingleAssetTabularRiskFreeTest(unittest.TestCase):
    def test_short_position_earns_two_units_of_cash_at_the_risk_free_rate(self) -> None:
        # Gross normalization maps a negative score to w = -1, leaving residual cash 1-(-1)=2.
        scores = _single_asset([-1.0, -1.0, -1.0], "score")
        returns = _single_asset([0.01, 0.01, 0.01], "fwd_return")
        dates = scores.index.get_level_values("date").unique()
        estimator = _estimator()

        estimator._risk_free_by_date = pd.Series(0.0, index=dates)
        loss_without_rf = estimator._fold_objective(scores, returns)
        estimator._risk_free_by_date = pd.Series(0.002, index=dates)
        loss_with_rf = estimator._fold_objective(scores, returns)

        # Return per date is -1*0.01 (short) + 2*0.002 (cash) = -0.006 vs -0.01; higher utility,
        # hence a lower (better) portfolio loss once residual cash earns the risk-free rate.
        self.assertAlmostEqual(loss_without_rf, 0.01, places=6)
        self.assertAlmostEqual(loss_with_rf, 0.006, places=6)
        self.assertLess(loss_with_rf, loss_without_rf)

    def test_missing_cv_risk_free_rate_raises(self) -> None:
        scores = _single_asset([-1.0, -1.0, -1.0], "score")
        returns = _single_asset([0.01, 0.01, 0.01], "fwd_return")
        estimator = _estimator()
        # A rate series that omits the scored dates must fail loudly, not silently drop cash.
        estimator._risk_free_by_date = pd.Series(
            [0.001], index=pd.bdate_range("2019-01-01", periods=1)
        )
        with self.assertRaisesRegex(ValueError, "risk-free rate is missing"):
            estimator._fold_objective(scores, returns)


if __name__ == "__main__":
    unittest.main()
