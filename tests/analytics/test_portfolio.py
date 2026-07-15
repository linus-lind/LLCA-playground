import unittest

import numpy as np
import pandas as pd

from llca.analytics.portfolio import build_portfolio_evaluation
from llca.loss.portfolio import PortfolioLoss


class PortfolioEvaluationTest(unittest.TestCase):
    def setUp(self) -> None:
        dates = pd.date_range("2024-01-01", periods=6)
        self.index = pd.MultiIndex.from_product([dates, ["A", "B"]], names=["date", "instrument"])
        self.loss = PortfolioLoss(
            leverage=1.0,
            risk_aversion=1.0,
            concentration_aversion=0.0,
            execution_fee=0.001,
            bid_ask_spread=0.0,
            slippage=0.0,
            borrow_cost=0.0001,
        )

    def test_portfolio_path_reconciles_returns_costs_and_attribution(self) -> None:
        scores = pd.Series(np.tile([1.0, -1.0], 6), index=self.index)
        returns = pd.Series(np.tile([0.02, -0.01], 6), index=self.index)

        result = build_portfolio_evaluation(
            scores,
            returns,
            normalize=self.loss.normalize_weights,
            return_type="simple",
            annualization_periods=252,
            risk_free_rate=0.0,
            minimum_acceptable_return=0.0,
            var_levels=(0.95, 0.99),
            rolling_window=3,
            signal_buckets=2,
            active_weight_threshold=0.0001,
            include_initial_trade=True,
            execution_fee=self.loss.execution_fee,
            bid_ask_spread=self.loss.bid_ask_spread,
            slippage=self.loss.slippage,
            borrow_cost=self.loss.borrow_cost,
        )

        np.testing.assert_allclose(result.exposures["gross"], 1.0)
        np.testing.assert_allclose(
            result.daily["gross_return"],
            result.daily["long_return_contribution"] + result.daily["short_return_contribution"],
        )
        np.testing.assert_allclose(
            result.daily["net_return"],
            result.daily["gross_return"] - result.costs["total"],
        )
        self.assertGreater(result.metrics["gross_total_return"], 0.0)
        self.assertLess(result.metrics["net_total_return"], result.metrics["gross_total_return"])
        self.assertEqual(len(result.asset_attribution), 2)
        self.assertEqual(len(result.signal_attribution), 2)
        self.assertIn("var_95", result.tail_risk)
        self.assertTrue(np.isfinite(result.tail_risk["var_95"].iloc[-1]))

    def test_log_returns_are_converted_before_portfolio_accounting(self) -> None:
        scores = pd.Series(np.tile([1.0, -1.0], 6), index=self.index)
        log_returns = pd.Series(np.tile([np.log1p(0.1), np.log1p(-0.1)], 6), index=self.index)

        result = build_portfolio_evaluation(
            scores,
            log_returns,
            normalize=self.loss.normalize_weights,
            return_type="log",
            annualization_periods=252,
            risk_free_rate=0.0,
            minimum_acceptable_return=0.0,
            var_levels=(0.95,),
            rolling_window=3,
            signal_buckets=2,
            active_weight_threshold=0.0001,
            include_initial_trade=False,
            execution_fee=0.0,
            bid_ask_spread=0.0,
            slippage=0.0,
            borrow_cost=0.0,
        )

        np.testing.assert_allclose(result.daily["gross_return"], 0.1)

    def test_bounded_scores_retain_dynamic_exposure_in_analytics(self) -> None:
        objective = PortfolioLoss(
            leverage=1.0,
            risk_aversion=0.0,
            concentration_aversion=0.0,
            execution_fee=0.0,
            bid_ask_spread=0.0,
            slippage=0.0,
            borrow_cost=0.0,
            normalization="bounded",
            return_type="simple",
        )
        scores = pd.Series(np.tile([0.2, -0.1], 6), index=self.index)
        returns = pd.Series(np.tile([0.02, -0.01], 6), index=self.index)

        result = build_portfolio_evaluation(
            scores,
            returns,
            normalize=objective.normalize_weights,
            return_type="simple",
            annualization_periods=252,
            risk_free_rate=0.0,
            minimum_acceptable_return=0.0,
            var_levels=(0.95,),
            rolling_window=3,
            signal_buckets=2,
            active_weight_threshold=0.0001,
            include_initial_trade=False,
            execution_fee=0.0,
            bid_ask_spread=0.0,
            slippage=0.0,
            borrow_cost=0.0,
        )

        np.testing.assert_allclose(result.exposures["gross"], 0.3)
        np.testing.assert_allclose(result.exposures["net"], 0.1)
        np.testing.assert_allclose(result.exposures["long"], 0.2)
        np.testing.assert_allclose(result.exposures["short"], 0.1)


if __name__ == "__main__":
    unittest.main()
