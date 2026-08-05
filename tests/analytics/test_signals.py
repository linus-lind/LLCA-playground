import unittest

import numpy as np
import pandas as pd

from llca.analytics.evaluation.signals import evaluate_signal
from llca.analytics.modules.signal_evaluation import SignalEvaluation
from llca.models.estimators.prediction import PredictionOutput


def _evaluate(
    predictions: PredictionOutput,
    target: pd.Series,
    *,
    decisions: pd.Series | None = None,
    rolling_window: int = 2,
    bucket_count: int = 2,
    active_weight_threshold: float = 0.0001,
) -> SignalEvaluation:
    return evaluate_signal(
        predictions,
        target,
        decisions=decisions,
        bucket_count=bucket_count,
        target_threshold=0.0,
        active_weight_threshold=active_weight_threshold,
        annualization_periods=252,
        rolling_window=rolling_window,
        signal_decay_periods=(0, 1),
    )


class SignalEvaluationTest(unittest.TestCase):
    def test_cross_section_uses_same_date_ic_and_keeps_directional_roc(self) -> None:
        dates = pd.date_range("2024-01-01", periods=4, name="date")
        index = pd.MultiIndex.from_product(
            [dates, ["A", "B", "C", "D"]],
            names=["date", "instrument"],
        )
        target = pd.Series(np.tile([-0.02, -0.01, 0.01, 0.02], 4), index=index)
        predictions = PredictionOutput(kind="portfolio", values=target.rename("score"))

        result = _evaluate(predictions, target)

        self.assertEqual(result.ic_basis, "cross_sectional")
        np.testing.assert_allclose(result.per_date["pearson_ic"], 1.0)
        np.testing.assert_allclose(result.per_date["rank_ic"], 1.0)
        self.assertAlmostEqual(result.metrics["mean_daily_rank_ic"], 1.0)
        self.assertAlmostEqual(result.metrics["directional_accuracy"], 1.0)
        self.assertAlmostEqual(result.metrics["roc_auc"], 1.0)
        self.assertEqual(result.confusion.to_numpy().trace(), len(target))
        self.assertIsNotNone(result.roc)
        self.assertIn("rank_ic_ir", result.rolling)
        self.assertEqual(list(result.decay.index), [0, 1])
        self.assertIn("basis_rank_ic", result.decay)

    def test_single_asset_uses_trailing_time_series_ic(self) -> None:
        dates = pd.date_range("2024-01-01", periods=6, name="date")
        index = pd.MultiIndex.from_product(
            [dates, ["A"]],
            names=["date", "instrument"],
        )
        target = pd.Series(np.arange(1.0, 7.0), index=index)
        predictions = PredictionOutput(kind="portfolio", values=target.rename("score"))

        result = _evaluate(predictions, target, rolling_window=3)

        self.assertEqual(result.ic_basis, "rolling_time_series")
        self.assertTrue(result.per_date["rank_ic"].iloc[:2].isna().all())
        np.testing.assert_allclose(result.per_date["rank_ic"].iloc[2:], 1.0)
        np.testing.assert_allclose(result.per_date["pearson_ic"].iloc[2:], 1.0)
        self.assertAlmostEqual(result.metrics["mean_daily_rank_ic"], 1.0)

    def test_alternating_entities_without_a_cross_section_use_time_series_ic(self) -> None:
        dates = pd.date_range("2024-01-01", periods=6, name="date")
        index = pd.MultiIndex.from_arrays(
            [dates, ["A", "B", "A", "B", "A", "B"]],
            names=["date", "instrument"],
        )
        target = pd.Series(np.arange(1.0, 7.0), index=index)

        result = _evaluate(
            PredictionOutput(kind="portfolio", values=target.rename("score")),
            target,
            rolling_window=3,
        )

        self.assertEqual(result.ic_basis, "rolling_time_series")
        np.testing.assert_allclose(result.per_date["rank_ic"].iloc[2:], 1.0)

    def test_cross_sectional_buckets_rank_within_each_date(self) -> None:
        dates = pd.date_range("2024-01-01", periods=2, name="date")
        index = pd.MultiIndex.from_product(
            [dates, ["A", "B"]],
            names=["date", "instrument"],
        )
        scores = pd.Series([1.0, 2.0, 100.0, 200.0], index=index)
        target = pd.Series([0.0, 10.0, 100.0, 110.0], index=index)

        result = _evaluate(PredictionOutput(kind="portfolio", values=scores), target)

        self.assertEqual(result.ic_basis, "cross_sectional")
        np.testing.assert_allclose(result.buckets["mean_outcome"], [50.0, 60.0])

    def test_cross_sectional_buckets_drop_singletons_and_span_requested_range(self) -> None:
        dates = pd.date_range("2024-01-01", periods=2, name="date")
        index = pd.MultiIndex.from_tuples(
            [(dates[0], "A"), (dates[0], "B"), (dates[1], "A")],
            names=["date", "instrument"],
        )
        scores = pd.Series([1.0, 2.0, 100.0], index=index)
        target = pd.Series([0.0, 1.0, 100.0], index=index)

        result = _evaluate(
            PredictionOutput(kind="portfolio", values=scores),
            target,
            bucket_count=10,
        )

        self.assertEqual(list(result.buckets.index), [1, 10])
        self.assertEqual(int(result.buckets["observations"].sum()), 2)

    def test_directional_accuracy_equal_weights_dates_not_observations(self) -> None:
        date_a, date_b = pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02")
        index = pd.MultiIndex.from_tuples(
            [
                (date_a, "A"),
                (date_a, "B"),
                (date_a, "C"),
                (date_a, "D"),
                (date_b, "A"),
                (date_b, "B"),
            ],
            names=["date", "instrument"],
        )
        target = pd.Series([1.0, 2.0, 3.0, 4.0, 1.0, 2.0], index=index)
        scores = pd.Series([1.0, 2.0, 3.0, 4.0, -1.0, -2.0], index=index)

        result = _evaluate(PredictionOutput(kind="portfolio", values=scores), target)

        self.assertAlmostEqual(result.metrics["directional_accuracy"], 0.5)
        self.assertAlmostEqual(result.metrics["pooled_directional_accuracy"], 4.0 / 6.0)
        np.testing.assert_allclose(result.per_date["hit_rate"], [1.0, 0.0])

    def test_neutral_allocations_are_directional_abstentions(self) -> None:
        date = pd.Timestamp("2024-01-01")
        index = pd.MultiIndex.from_product(
            [[date], ["A", "B", "C", "D"]],
            names=["date", "instrument"],
        )
        target = pd.Series([1.0, -1.0, -1.0, 1.0], index=index)
        scores = pd.Series([-4.0, -3.0, 3.0, 4.0], index=index)
        decisions = pd.Series([0.0, 0.00001, -0.5, 0.5], index=index)

        result = _evaluate(
            PredictionOutput(kind="portfolio", values=scores),
            target,
            decisions=decisions,
        )

        self.assertEqual(result.metrics["active_directional_observations"], 2.0)
        self.assertAlmostEqual(result.metrics["directional_accuracy"], 1.0)
        self.assertAlmostEqual(result.metrics["pooled_directional_accuracy"], 1.0)
        self.assertEqual(int(result.confusion.to_numpy().sum()), 2)
        self.assertAlmostEqual(result.metrics["roc_auc"], 1.0)
        self.assertEqual(int(result.buckets["directional_observations"].sum()), 2)

    def test_cash_only_allocation_has_no_directional_skill(self) -> None:
        dates = pd.date_range("2024-01-01", periods=4, name="date")
        target = pd.Series([-1.0, 1.0, -1.0, 1.0], index=dates)
        scores = target.rename("score")
        decisions = pd.Series(0.0, index=dates)

        result = _evaluate(
            PredictionOutput(kind="portfolio", values=scores),
            target,
            decisions=decisions,
        )

        self.assertTrue(np.isnan(result.metrics["directional_accuracy"]))
        self.assertTrue(np.isnan(result.metrics["pooled_directional_accuracy"]))
        self.assertTrue(np.isnan(result.metrics["roc_auc"]))
        self.assertEqual(int(result.confusion.to_numpy().sum()), 0)
        self.assertTrue(result.per_date["hit_rate"].isna().all())
        self.assertTrue(result.buckets["hit_rate"].isna().all())

    def test_unimplemented_prediction_kinds_fail_explicitly(self) -> None:
        index = pd.date_range("2024-01-01", periods=3, name="date")
        target = pd.Series([0.0, 1.0, 2.0], index=index)
        cases = (
            PredictionOutput(kind="regression", values=target.rename("forecast")),
            PredictionOutput(kind="binary", values=target.rename("score")),
            PredictionOutput(
                kind="multiclass",
                values=pd.DataFrame(
                    {
                        "a": [1.0, 0.0, 0.0],
                        "b": [0.0, 1.0, 0.0],
                        "c": [0.0, 0.0, 1.0],
                    },
                    index=index,
                ),
            ),
        )
        for predictions in cases:
            with self.subTest(kind=predictions.kind):
                with self.assertRaisesRegex(NotImplementedError, predictions.kind):
                    _evaluate(predictions, target)


if __name__ == "__main__":
    unittest.main()
