import unittest

import numpy as np
import pandas as pd

from llca.analytics.evaluation.portfolio import build_portfolio_evaluation
from llca.analytics.inputs.risk_free import align_risk_free
from llca.analytics.modules.portfolio_evaluation import PortfolioEvaluation
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
            common_score_aversion=0.0,
            net_exposure_aversion=0.0,
            net_exposure_tolerance=0.0,
            normalization="market_neutral",
            return_type="simple",
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
            risk_free=pd.Series(0.0, index=self.index.get_level_values("date").unique()),
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
            result.daily["long_return_contribution"]
            + result.daily["short_return_contribution"]
            + result.daily["cash_return_contribution"],
        )
        np.testing.assert_allclose(
            result.daily["net_return"],
            result.daily["gross_return"] - result.costs["total"],
        )
        self.assertGreater(result.metrics["gross_total_return"], 0.0)
        self.assertLess(result.metrics["net_total_return"], result.metrics["gross_total_return"])
        self.assertEqual(len(result.asset_attribution), 3)
        self.assertIn("Cash", result.asset_attribution.index)
        self.assertEqual(len(result.signal_attribution), 2)
        expected_daily = result.signal_attribution["total_return_contribution"] / len(result.daily)
        np.testing.assert_allclose(
            result.signal_attribution["mean_daily_contribution"], expected_daily
        )
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
            risk_free=pd.Series(0.0, index=self.index.get_level_values("date").unique()),
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

    def test_cash_only_weights_are_abstentions_in_signal_attribution(self) -> None:
        scores = pd.Series(0.0, index=self.index)
        returns = pd.Series(np.tile([0.02, -0.01], 6), index=self.index)

        result = build_portfolio_evaluation(
            scores,
            returns,
            normalize=self.loss.normalize_weights,
            return_type="simple",
            annualization_periods=252,
            risk_free=pd.Series(0.0, index=self.index.get_level_values("date").unique()),
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

        self.assertEqual(int(result.signal_attribution["directional_observations"].sum()), 0)
        self.assertTrue(result.signal_attribution["hit_rate"].isna().all())

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
            common_score_aversion=0.0,
            net_exposure_aversion=0.0,
            net_exposure_tolerance=0.0,
        )
        scores = pd.Series(np.tile([0.2, -0.1], 6), index=self.index)
        returns = pd.Series(np.tile([0.02, -0.01], 6), index=self.index)

        result = build_portfolio_evaluation(
            scores,
            returns,
            normalize=objective.normalize_weights,
            return_type="simple",
            annualization_periods=252,
            risk_free=pd.Series(0.0, index=self.index.get_level_values("date").unique()),
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
        np.testing.assert_allclose(result.composition["concentration_hhi"], 5.0 / 9.0, rtol=1e-6)
        np.testing.assert_allclose(result.composition["effective_positions"], 1.8, rtol=1e-6)
        np.testing.assert_allclose(result.composition["top_5_weight_share"], 1.0)

    def test_risk_free_alignment_never_backfills_from_the_future(self) -> None:
        scores = pd.Series(np.tile([1.0, -1.0], 6), index=self.index)
        returns = pd.Series(np.tile([0.02, -0.01], 6), index=self.index)
        dates = self.index.get_level_values("date").unique()

        with self.assertRaisesRegex(ValueError, "no current or prior observation"):
            build_portfolio_evaluation(
                scores,
                returns,
                normalize=self.loss.normalize_weights,
                return_type="simple",
                annualization_periods=252,
                risk_free=pd.Series(0.001, index=dates[1:]),
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

        with self.assertRaisesRegex(ValueError, "non-finite"):
            align_risk_free(
                pd.Series([0.001, np.nan], index=dates[:2]),
                dates[:1],
            )

    def test_rolling_sharpe_uses_the_aligned_daily_excess_return(self) -> None:
        scores = pd.Series(np.tile([1.0, -1.0], 6), index=self.index)
        returns = pd.Series(np.repeat([0.01, 0.03, -0.01, 0.02, 0.04, 0.00], 2), index=self.index)
        dates = self.index.get_level_values("date").unique()
        risk_free = pd.Series([0.001, 0.002, 0.001, 0.003, 0.002, 0.001], index=dates)

        result = build_portfolio_evaluation(
            scores,
            returns,
            normalize=self.loss.normalize_weights,
            return_type="simple",
            annualization_periods=252,
            risk_free=risk_free,
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

        excess = result.daily["net_return"] - risk_free
        expected = excess.rolling(3).mean() / excess.rolling(3).std(ddof=1) * np.sqrt(252)
        np.testing.assert_allclose(
            result.rolling["sharpe_ratio"].dropna(), expected.dropna(), rtol=1e-12
        )

    def test_residual_cash_earns_risk_free_and_zero_weights_have_zero_excess(self) -> None:
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
            common_score_aversion=0.0,
            net_exposure_aversion=0.0,
            net_exposure_tolerance=0.0,
        )
        scores = pd.Series(0.0, index=self.index)
        returns = pd.Series(np.tile([0.10, -0.10], 6), index=self.index)
        dates = self.index.get_level_values("date").unique()
        risk_free = pd.Series(np.linspace(0.0001, 0.0006, len(dates)), index=dates)

        result = build_portfolio_evaluation(
            scores,
            returns,
            normalize=objective.normalize_weights,
            return_type="simple",
            annualization_periods=252,
            risk_free=risk_free,
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

        np.testing.assert_allclose(result.daily["gross_return"], risk_free)
        np.testing.assert_allclose(result.daily["excess_net_return"], 0.0, atol=1e-15)
        np.testing.assert_allclose(result.daily["cash_return_contribution"], risk_free)
        np.testing.assert_allclose(result.turnover["l1_turnover"], 0.0)

    def test_maximum_drawdown_attribution_excludes_the_peak_date(self) -> None:
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
            common_score_aversion=0.0,
            net_exposure_aversion=0.0,
            net_exposure_tolerance=0.0,
        )
        scores = pd.Series(np.tile([1.0, 0.0], 6), index=self.index)
        first_asset_returns = np.array([0.10, -0.20, 0.05, 0.01, 0.01, 0.01])
        returns = pd.Series(
            np.column_stack([first_asset_returns, np.zeros(6)]).ravel(), index=self.index
        )
        dates = self.index.get_level_values("date").unique()

        result = build_portfolio_evaluation(
            scores,
            returns,
            normalize=objective.normalize_weights,
            return_type="simple",
            annualization_periods=252,
            risk_free=pd.Series(0.0, index=dates),
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

        attribution = result.maximum_drawdown_attribution
        self.assertEqual(attribution.loc["A", "peak"], dates[0])
        self.assertEqual(attribution.loc["A", "trough"], dates[1])
        self.assertAlmostEqual(attribution.loc["A", "return_contribution"], -0.20)

    def test_initial_loss_drawdown_starts_from_presample_nav(self) -> None:
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
            common_score_aversion=0.0,
            net_exposure_aversion=0.0,
            net_exposure_tolerance=0.0,
        )
        scores = pd.Series(np.tile([1.0, 0.0], 6), index=self.index)
        first_asset_returns = np.array([-0.10, -0.10, 0.0, 0.0, 0.0, 0.0])
        returns = pd.Series(
            np.column_stack([first_asset_returns, np.zeros(6)]).ravel(), index=self.index
        )
        dates = self.index.get_level_values("date").unique()

        result = build_portfolio_evaluation(
            scores,
            returns,
            normalize=objective.normalize_weights,
            return_type="simple",
            annualization_periods=252,
            risk_free=pd.Series(0.0, index=dates),
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

        self.assertAlmostEqual(result.drawdowns["drawdown"].iloc[0], -0.10)
        self.assertAlmostEqual(result.metrics["net_maximum_drawdown"], -0.19)
        attribution = result.maximum_drawdown_attribution
        self.assertTrue(pd.isna(attribution.loc["A", "peak"]))
        self.assertEqual(attribution.loc["A", "trough"], dates[1])
        self.assertAlmostEqual(attribution.loc["A", "return_contribution"], -0.20)

    def test_signal_attribution_pools_when_no_date_has_a_cross_section(self) -> None:
        dates = pd.bdate_range("2024-02-01", periods=6, name="date")
        index = pd.MultiIndex.from_arrays(
            [dates, ["A", "B", "A", "B", "A", "B"]],
            names=["date", "instrument"],
        )
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
            common_score_aversion=0.0,
            net_exposure_aversion=0.0,
            net_exposure_tolerance=0.0,
        )

        result = build_portfolio_evaluation(
            pd.Series(np.arange(1.0, 7.0), index=index),
            pd.Series(np.arange(0.01, 0.07, 0.01), index=index),
            normalize=objective.normalize_weights,
            return_type="simple",
            annualization_periods=252,
            risk_free=pd.Series(0.0, index=dates),
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

        self.assertEqual(list(result.signal_attribution.index), [1, 2])


class TrainingAnalyticsParityTest(unittest.TestCase):
    """The training objective and analytics evaluation must agree on the shared accounting."""

    def setUp(self) -> None:
        self.dates = pd.date_range("2024-01-01", periods=4)
        self.index = pd.MultiIndex.from_product(
            [self.dates, ["A", "B"]], names=["date", "instrument"]
        )
        # Bounded weights whose per-date gross <= 1, so normalize is the identity and the
        # scores are exactly the weights fed to both the loss and analytics.
        self.weights = np.array([[0.3, -0.2], [0.1, 0.4], [-0.5, 0.2], [0.25, 0.25]])
        self.asset_returns = np.array([[0.02, -0.03], [0.01, 0.04], [-0.02, 0.03], [0.05, -0.01]])
        self.rf = np.array([0.001, 0.002, 0.003, 0.004])
        self.scores = pd.Series(self.weights.ravel(), index=self.index)
        self.returns = pd.Series(self.asset_returns.ravel(), index=self.index)
        self.risk_free = pd.Series(self.rf, index=self.dates)
        self.objective = PortfolioLoss(
            leverage=1.0,
            risk_aversion=0.0,
            concentration_aversion=0.0,
            execution_fee=0.0,
            bid_ask_spread=0.0,
            slippage=0.0,
            borrow_cost=0.0,
            normalization="bounded",
            return_type="simple",
            common_score_aversion=0.0,
            net_exposure_aversion=0.0,
            net_exposure_tolerance=0.0,
        )

    def _analytics(self) -> PortfolioEvaluation:
        return build_portfolio_evaluation(
            self.scores,
            self.returns,
            normalize=self.objective.normalize_weights,
            return_type="simple",
            annualization_periods=252,
            risk_free=self.risk_free,
            minimum_acceptable_return=0.0,
            var_levels=(0.95,),
            rolling_window=2,
            signal_buckets=2,
            active_weight_threshold=0.0001,
            include_initial_trade=False,
            execution_fee=0.0,
            bid_ask_spread=0.0,
            slippage=0.0,
            borrow_cost=0.0,
        )

    def test_gross_return_matches_the_shared_primitive_used_by_training(self) -> None:
        import torch

        from llca.core.portfolio_accounting import (
            cash_return_contribution,
            drifted_weights,
            gross_return,
            residual_cash_weight,
        )

        result = self._analytics()
        # Feed analytics' own (float32-normalized) weights into the shared primitive so the
        # comparison isolates the accounting identity, not weight-normalization precision.
        weights = torch.from_numpy(result.weights.to_numpy(dtype=float))
        returns = torch.from_numpy(self.asset_returns)
        rf = torch.from_numpy(self.rf)

        # Gross return agrees on every date (no boundary ambiguity), applying rf exactly once.
        np.testing.assert_allclose(
            result.daily["gross_return"].to_numpy(),
            gross_return(weights, returns, rf).numpy(),
            rtol=0,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            result.daily["cash_return_contribution"].to_numpy(),
            cash_return_contribution(weights, rf).numpy(),
            rtol=0,
            atol=1e-12,
        )
        # Residual cash earned exactly once: cash contribution equals cash weight times rf.
        np.testing.assert_allclose(
            result.daily["cash_return_contribution"].to_numpy(),
            (residual_cash_weight(weights) * rf).numpy(),
            rtol=0,
            atol=1e-12,
        )

        # Interior drifted turnover (t >= 1) agrees; t = 0 is a per-caller boundary by design.
        drifted = drifted_weights(weights[:-1], returns[:-1], rf[:-1])
        expected_interior = (weights[1:] - drifted).abs().sum(dim=-1).numpy()
        np.testing.assert_allclose(
            result.turnover["l1_turnover"].to_numpy()[1:], expected_interior, rtol=0, atol=1e-12
        )

    def test_objective_mean_return_equals_analytics_gross_return(self) -> None:
        import torch

        result = self._analytics()
        scores = torch.from_numpy(self.weights).float()
        returns = torch.from_numpy(self.asset_returns).float()
        rf = torch.from_numpy(self.rf).float()
        output = self.objective(
            scores, returns, torch.ones_like(scores, dtype=torch.bool), risk_free=rf
        )
        self.assertAlmostEqual(
            float(output.mean_return), float(result.daily["gross_return"].mean()), places=6
        )
        self.assertAlmostEqual(
            float(output.cash_return),
            float(result.daily["cash_return_contribution"].mean()),
            places=6,
        )


if __name__ == "__main__":
    unittest.main()
