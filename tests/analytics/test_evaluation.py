import unittest

import numpy as np
import pandas as pd
import torch

from llca.analytics.evaluation import evaluate_predictions
from llca.analytics.modules.analytics_config import ModelEvaluationConfig, RegisteredModelConfig
from llca.data.modules.masked_panel import MaskedPanel
from llca.loss.portfolio import PortfolioLoss
from llca.models.estimators.prediction import PredictionOutput


def _config() -> ModelEvaluationConfig:
    return ModelEvaluationConfig(
        models=(RegisteredModelConfig(name="test", version=1, label="test-v1"),),
        device="cpu",
        annualization_periods=252,
        return_type="simple",
        return_realization_lag=2,
        signal_buckets=2,
        target_threshold=0.0,
        minimum_acceptable_return=0.0,
        var_levels=(0.95, 0.99),
        autocorrelation_lags=(1,),
        worst_rolling_windows=(2,),
        rolling_window=2,
        signal_decay_periods=(0, 1),
        active_weight_threshold=0.0001,
        include_initial_trade=True,
        show_plots=False,
        evaluation_end=None,
    )


def _rf(index: pd.MultiIndex) -> pd.Series:
    return pd.Series(0.0, index=index.get_level_values("date").unique())


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
            common_score_aversion=0.0,
            net_exposure_aversion=0.0,
            net_exposure_tolerance=0.0,
            normalization="market_neutral",
            return_type="simple",
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
            _rf(self.index),
        )

        self.assertEqual(evaluation.valid_observations, 6)
        self.assertEqual(evaluation.dates, 3)
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

    def test_row_layout_requires_an_explicit_objective_adapter(self) -> None:
        predictions = PredictionOutput(
            kind="portfolio",
            values=pd.Series(
                [0.01, -0.01, 0.02, 0.00, -0.01, 0.01],
                index=self.index,
                name="forecast",
            ),
        )

        with self.assertRaisesRegex(NotImplementedError, "objective_adapter"):
            evaluate_predictions(
                predictions,
                _supervision(self.index),
                "fwd_return",
                self.objective,
                _config(),
                _rf(self.index),
                objective_layout="rows",
            )

    def test_explicit_objective_adapter_remains_an_extension_contract(self) -> None:
        predictions = PredictionOutput(
            kind="portfolio",
            values=pd.Series(
                [0.6, -0.4, 0.5, -0.5, 0.4, -0.6],
                index=self.index,
                name="score",
            ),
        )
        calls = 0

        def adapter(
            output: PredictionOutput, target: pd.Series
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            nonlocal calls
            calls += 1
            assert isinstance(output.values, pd.Series)
            scores = output.values.unstack("instrument")
            outcomes = target.unstack("instrument").reindex_like(scores)
            valid = scores.notna() & outcomes.notna()
            return (
                torch.from_numpy(scores.fillna(0.0).to_numpy(dtype=np.float32)),
                torch.from_numpy(outcomes.fillna(0.0).to_numpy(dtype=np.float32)),
                torch.from_numpy(valid.to_numpy(dtype=bool)),
            )

        evaluation = evaluate_predictions(
            predictions,
            _supervision(self.index),
            "fwd_return",
            self.objective,
            _config(),
            _rf(self.index),
            objective_layout="rows",
            objective_adapter=adapter,
        )

        self.assertEqual(calls, 1)
        self.assertIn("loss", evaluation.objective_metrics)
        self.assertEqual(len(evaluation.portfolio.daily), 3)

    def test_predictions_without_allocation_contract_are_rejected(self) -> None:
        allocations = pd.Series(
            [0.6, -0.4, 0.5, -0.5, 0.4, -0.6],
            index=self.index,
            name="weight",
        )
        with self.assertRaisesRegex(TypeError, "no implicit weight semantics"):
            evaluate_predictions(
                PredictionOutput(kind="portfolio", values=allocations),
                _supervision(self.index),
                "fwd_return",
                None,
                _config(),
                _rf(self.index),
            )

    def test_unimplemented_prediction_kinds_fail_before_evaluation(self) -> None:
        values = pd.Series(np.arange(len(self.index), dtype=float), index=self.index)
        cases = (
            PredictionOutput(kind="regression", values=values),
            PredictionOutput(kind="binary", values=values),
            PredictionOutput(
                kind="multiclass",
                values=pd.DataFrame(
                    np.tile([1.0, 0.0, 0.0], (len(self.index), 1)),
                    index=self.index,
                    columns=["a", "b", "c"],
                ),
            ),
        )
        for predictions in cases:
            with self.subTest(kind=predictions.kind):
                with self.assertRaisesRegex(NotImplementedError, predictions.kind):
                    evaluate_predictions(
                        predictions,
                        _supervision(self.index),
                        "fwd_return",
                        None,
                        _config(),
                        _rf(self.index),
                    )

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
            common_score_aversion=0.0,
            net_exposure_aversion=0.0,
            net_exposure_tolerance=0.0,
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
            _rf(self.index),
        )

        np.testing.assert_allclose(evaluation.portfolio.daily["gross_return"], 0.1)
        np.testing.assert_allclose(evaluation.target, np.tile([0.1, -0.1], 3))
        self.assertAlmostEqual(evaluation.objective_metrics["mean_return"], 0.1, places=6)

    def test_directional_diagnostics_follow_normalized_allocations(self) -> None:
        values = pd.DataFrame(
            {"fwd_return": np.tile([0.01, -0.01], 3)},
            index=self.index,
        )
        supervision = MaskedPanel(
            values=values,
            observed=pd.DataFrame(True, index=self.index, columns=values.columns),
            age=pd.DataFrame(0, index=self.index, columns=values.columns),
            segment=pd.Series(np.tile([0, 1], 3), index=self.index),
        )
        predictions = PredictionOutput(
            kind="portfolio",
            # Both raw scores are positive, but market-neutral normalization makes the
            # lower-ranked asset a short allocation.
            values=pd.Series(np.tile([2.0, 1.0], 3), index=self.index),
        )

        evaluation = evaluate_predictions(
            predictions,
            supervision,
            "fwd_return",
            self.objective,
            _config(),
            _rf(self.index),
        )

        self.assertEqual(evaluation.signal.metrics["directional_accuracy"], 1.0)
        np.testing.assert_allclose(evaluation.portfolio.signal_attribution["hit_rate"], 1.0)
        self.assertTrue((evaluation.portfolio.weights["A"] > 0.0).all())
        self.assertTrue((evaluation.portfolio.weights["B"] < 0.0).all())

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
            _rf(self.index),
        )

        gross = evaluation.portfolio.weights.abs().sum(axis=1)
        net = evaluation.portfolio.weights.sum(axis=1)
        np.testing.assert_allclose(gross.to_numpy(), 1.0)
        np.testing.assert_allclose(net.to_numpy(), 0.0, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
