import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from llca.analytics.modules.test_evaluation import TestEvaluation
from llca.analytics.plots import plot_evaluation
from llca.analytics.signals import evaluate_signal
from llca.models.estimators.prediction import PredictionOutput


def _panel_index() -> pd.MultiIndex:
    return pd.MultiIndex.from_product(
        [pd.date_range("2024-01-01", periods=4), ["A", "B", "C", "D"]],
        names=["date", "instrument"],
    )


class SignalEvaluationTest(unittest.TestCase):
    def test_portfolio_scores_report_cross_sectional_ic_and_monotone_buckets(self) -> None:
        index = _panel_index()
        target = pd.Series(np.tile([-0.02, -0.01, 0.01, 0.02], 4), index=index)
        predictions = PredictionOutput(kind="portfolio", values=target.rename("score"))

        result = evaluate_signal(
            predictions,
            target,
            bucket_count=4,
            probability_bins=4,
            classification_threshold=0.5,
            target_threshold=0.0,
            annualization_periods=252,
            rolling_window=2,
            signal_decay_periods=(0, 1),
        )

        self.assertAlmostEqual(result.metrics["mean_daily_rank_ic"], 1.0)
        self.assertAlmostEqual(result.metrics["directional_accuracy"], 1.0)
        self.assertAlmostEqual(result.metrics["bucket_monotonicity"], 1.0)
        self.assertGreater(result.metrics["top_minus_bottom_outcome"], 0.0)
        self.assertIn("rank_ic_ir", result.rolling)
        self.assertEqual(list(result.decay.index), [0, 1])

    def test_regression_reports_error_and_calibration_metrics(self) -> None:
        index = _panel_index()
        target = pd.Series(np.linspace(-0.03, 0.03, len(index)), index=index)
        predictions = PredictionOutput(kind="regression", values=target.rename("forecast"))

        result = evaluate_signal(
            predictions,
            target,
            bucket_count=4,
            probability_bins=4,
            classification_threshold=0.5,
            target_threshold=0.0,
            annualization_periods=252,
            rolling_window=2,
            signal_decay_periods=(0, 1),
        )

        self.assertAlmostEqual(result.metrics["mae"], 0.0)
        self.assertAlmostEqual(result.metrics["rmse"], 0.0)
        self.assertAlmostEqual(result.metrics["r_squared"], 1.0)
        self.assertAlmostEqual(result.metrics["calibration_slope"], 1.0)

    def test_probabilistic_regression_reports_quantile_quality_and_coverage(self) -> None:
        index = _panel_index()
        target = pd.Series(np.linspace(-0.03, 0.03, len(index)), index=index)
        quantiles = pd.DataFrame(
            {
                0.1: target - 0.01,
                0.5: target,
                0.9: target + 0.01,
            },
            index=index,
        )
        predictions = PredictionOutput(
            kind="regression",
            values=target.rename("forecast"),
            quantiles=quantiles,
        )

        result = evaluate_signal(
            predictions,
            target,
            bucket_count=4,
            probability_bins=4,
            classification_threshold=0.5,
            target_threshold=0.0,
            annualization_periods=252,
            rolling_window=2,
            signal_decay_periods=(0, 1),
        )

        self.assertAlmostEqual(result.metrics["pinball_loss_q0.5"], 0.0)
        self.assertAlmostEqual(result.metrics["central_interval_empirical_coverage"], 1.0)
        self.assertAlmostEqual(result.metrics["central_interval_average_width"], 0.02)
        self.assertIsNotNone(result.calibration)

        evaluation = TestEvaluation(
            predictions=predictions,
            signal=result,
            objective_metrics={},
            portfolio=None,
            valid_observations=len(target),
            dates=4,
        )
        with patch("matplotlib.pyplot.show") as show:
            plot_evaluation(evaluation)
        show.assert_called_once()

    def test_binary_classification_reports_discrimination_and_calibration(self) -> None:
        index = _panel_index()
        actual = pd.Series(np.tile([0, 0, 1, 1], 4), index=index)
        probability = pd.Series(np.tile([0.05, 0.2, 0.8, 0.95], 4), index=index)
        predictions = PredictionOutput(
            kind="binary",
            values=probability.rename("decision_score"),
            probabilities=probability.rename("probability"),
        )

        result = evaluate_signal(
            predictions,
            actual,
            bucket_count=4,
            probability_bins=4,
            classification_threshold=0.5,
            target_threshold=0.0,
            annualization_periods=252,
            rolling_window=2,
            signal_decay_periods=(0, 1),
        )

        self.assertAlmostEqual(result.metrics["accuracy"], 1.0)
        self.assertAlmostEqual(result.metrics["roc_auc"], 1.0)
        self.assertLess(result.metrics["brier_score"], 0.05)
        self.assertIsNotNone(result.confusion)
        self.assertIsNotNone(result.roc)
        self.assertIsNotNone(result.precision_recall)

    def test_multiclass_classification_reports_macro_metrics(self) -> None:
        dates = pd.date_range("2024-01-01", periods=2)
        index = pd.MultiIndex.from_product([dates, ["A", "B", "C"]], names=["date", "instrument"])
        target = pd.Series(["down", "flat", "up"] * 2, index=index)
        probabilities = pd.DataFrame(
            np.tile(np.eye(3), (2, 1)),
            index=index,
            columns=["down", "flat", "up"],
        )
        predictions = PredictionOutput(
            kind="multiclass",
            values=probabilities,
            probabilities=probabilities,
        )

        result = evaluate_signal(
            predictions,
            target,
            bucket_count=3,
            probability_bins=3,
            classification_threshold=0.5,
            target_threshold=0.0,
            annualization_periods=252,
            rolling_window=2,
            signal_decay_periods=(0, 1),
        )

        self.assertAlmostEqual(result.metrics["accuracy"], 1.0)
        self.assertAlmostEqual(result.metrics["macro_f1"], 1.0)
        self.assertAlmostEqual(result.metrics["multiclass_brier_score"], 0.0)


if __name__ == "__main__":
    unittest.main()
