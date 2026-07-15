import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from torch import Tensor, nn

from llca.analytics.evaluation import evaluate_predictions
from llca.analytics.plots import plot_evaluation
from llca.analytics.utils.config import ModelEvaluationConfig, RegisteredModelConfig
from llca.data.modules.masked_panel import MaskedPanel
from llca.loss.portfolio import PortfolioLoss
from llca.models.estimators.prediction import PredictionOutput


class _TensorObjective(nn.Module):
    def forward(self, scores: Tensor, target: Tensor, mask: Tensor) -> Tensor:
        return (scores[mask] - target[mask]).square().mean()


def _config() -> ModelEvaluationConfig:
    return ModelEvaluationConfig(
        models=(RegisteredModelConfig(name="test", version=1, label="test-v1"),),
        device="cpu",
        annualization_periods=252,
        return_type="simple",
        signal_buckets=2,
        probability_bins=2,
        classification_threshold=0.5,
        target_threshold=0.0,
        risk_free_rate=0.0,
        minimum_acceptable_return=0.0,
        var_levels=(0.95, 0.99),
        rolling_window=2,
        signal_decay_periods=(0, 1),
        active_weight_threshold=0.0001,
        include_initial_trade=True,
        show_plots=False,
        evaluation_end=None,
    )


def _supervision(index: pd.MultiIndex) -> MaskedPanel:
    returns = pd.DataFrame(
        {"fwd_return": [0.01, -0.01, 0.02, 0.00, -0.01, 0.01]},
        index=index,
    )
    return MaskedPanel(
        values=returns,
        observed=pd.DataFrame(True, index=index, columns=["fwd_return"]),
        age=pd.DataFrame(0, index=index, columns=["fwd_return"]),
        segment=pd.Series(np.tile([0, 1], 3), index=index),
    )


class PredictionEvaluationTest(unittest.TestCase):
    def setUp(self) -> None:
        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        self.index = pd.MultiIndex.from_product([dates, ["A", "B"]], names=["date", "instrument"])
        self.objective = PortfolioLoss(
            leverage=1.0,
            risk_aversion=1.0,
            concentration_aversion=0.002,
            execution_fee=0.0001,
            bid_ask_spread=0.0003,
            slippage=0.0002,
            borrow_cost=0.00002,
        )

    def test_returns_complete_portfolio_loss_output(self) -> None:
        predictions = PredictionOutput(
            kind="portfolio",
            values=pd.Series(
                [0.6, -0.4, 0.5, -0.5, 0.4, -0.6],
                index=self.index,
                name="score",
            ),
        )

        evaluation = evaluate_predictions(
            predictions,
            _supervision(self.index),
            "fwd_return",
            self.objective,
            _config(),
        )

        self.assertEqual(evaluation.valid_observations, 6)
        self.assertEqual(evaluation.dates, 3)
        self.assertIsNotNone(evaluation.portfolio)
        assert evaluation.portfolio is not None
        self.assertEqual(len(evaluation.portfolio.daily["gross_return"]), 3)
        self.assertTrue(np.isfinite(evaluation.portfolio.daily["gross_return"]).all())
        self.assertEqual(
            set(evaluation.objective_metrics),
            {
                "loss",
                "mean_return",
                "variance",
                "turnover",
                "cost",
                "gross_exposure",
                "net_exposure",
                "long_exposure",
                "short_exposure",
                "concentration",
                "common_score_penalty",
                "net_exposure_penalty",
                "market_penalty",
            },
        )

    def test_bare_tensor_objective_uses_row_layout(self) -> None:
        predictions = PredictionOutput(
            kind="regression",
            values=pd.Series(
                [0.01, -0.01, 0.02, 0.00, -0.01, 0.01],
                index=self.index,
                name="forecast",
            ),
        )

        evaluation = evaluate_predictions(
            predictions,
            _supervision(self.index),
            "fwd_return",
            _TensorObjective(),
            _config(),
            objective_layout="rows",
        )

        self.assertEqual(evaluation.objective_metrics, {"loss": 0.0})
        self.assertIsNone(evaluation.portfolio)

    def test_direct_allocations_bypass_objective_normalisation(self) -> None:
        allocations = pd.Series(
            [0.6, -0.4, 0.5, -0.5, 0.4, -0.6],
            index=self.index,
            name="weight",
        )
        evaluation = evaluate_predictions(
            PredictionOutput(kind="portfolio", values=allocations),
            _supervision(self.index),
            "fwd_return",
            None,
            _config(),
        )

        assert evaluation.portfolio is not None
        expected = allocations.unstack("instrument")
        pd.testing.assert_frame_equal(
            evaluation.portfolio.weights,
            expected,
            check_dtype=False,
            rtol=1e-6,
            atol=1e-7,
        )
        self.assertEqual(evaluation.objective_metrics, {})

    def test_objective_return_type_drives_test_portfolio_accounting(self) -> None:
        values = pd.DataFrame(
            {"fwd_return": np.tile([np.log1p(0.1), np.log1p(-0.1)], 3)},
            index=self.index,
        )
        supervision = MaskedPanel(
            values=values,
            observed=pd.DataFrame(True, index=self.index, columns=values.columns),
            age=pd.DataFrame(0, index=self.index, columns=values.columns),
            segment=pd.Series(np.tile([0, 1], 3), index=self.index),
        )
        objective = PortfolioLoss(
            leverage=1.0,
            risk_aversion=0.0,
            concentration_aversion=0.0,
            execution_fee=0.0,
            bid_ask_spread=0.0,
            slippage=0.0,
            borrow_cost=0.0,
            normalization="market_neutral",
            return_type="log",
        )
        predictions = PredictionOutput(
            kind="portfolio",
            values=pd.Series(np.tile([1.0, -1.0], 3), index=self.index),
        )

        evaluation = evaluate_predictions(
            predictions,
            supervision,
            "fwd_return",
            objective,
            _config(),  # Deliberately says simple; the model objective is authoritative.
        )

        assert evaluation.portfolio is not None
        np.testing.assert_allclose(evaluation.portfolio.daily["gross_return"], 0.1)
        self.assertAlmostEqual(evaluation.objective_metrics["mean_return"], 0.1, places=6)

    def test_rejects_non_finite_predictions(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-finite values"):
            PredictionOutput(
                kind="portfolio",
                values=pd.Series(
                    [np.nan, -0.4, 0.5, -0.5, 0.4, -0.6],
                    index=self.index,
                    name="score",
                ),
            )

    def test_raw_scores_are_normalised_once_for_portfolio(self) -> None:
        predictions = PredictionOutput(
            kind="portfolio",
            values=pd.Series(
                [6.0, -4.0, 5.0, -5.0, 4.0, -6.0],
                index=self.index,
                name="score",
            ),
        )

        evaluation = evaluate_predictions(
            predictions,
            _supervision(self.index),
            "fwd_return",
            self.objective,
            _config(),
        )

        assert evaluation.portfolio is not None
        gross = evaluation.portfolio.weights.abs().sum(axis=1)
        net = evaluation.portfolio.weights.sum(axis=1)
        np.testing.assert_allclose(gross.to_numpy(), 1.0)
        np.testing.assert_allclose(net.to_numpy(), 0.0, atol=1e-7)

    def test_complete_evaluation_dashboard_can_be_rendered(self) -> None:
        predictions = PredictionOutput(
            kind="portfolio",
            values=pd.Series(
                [0.6, -0.4, 0.5, -0.5, 0.4, -0.6],
                index=self.index,
                name="score",
            ),
        )
        evaluation = evaluate_predictions(
            predictions,
            _supervision(self.index),
            "fwd_return",
            self.objective,
            _config(),
        )

        with patch("matplotlib.pyplot.show") as show:
            plot_evaluation(evaluation)

        show.assert_called_once()


if __name__ == "__main__":
    unittest.main()
