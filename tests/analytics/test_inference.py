import unittest
from typing import cast

import numpy as np
import pandas as pd

from llca.analytics.stats import inference


class InferenceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.rng = np.random.default_rng(20240208)

    def test_newey_west_bandwidth_grows_with_sample(self) -> None:
        self.assertEqual(inference.newey_west_bandwidth(1), 0)
        self.assertLessEqual(
            inference.newey_west_bandwidth(100), inference.newey_west_bandwidth(10000)
        )

    def test_stationary_bootstrap_indices_are_valid_positions(self) -> None:
        rng = np.random.default_rng(0)
        indices = inference.stationary_bootstrap_indices(50, 8.0, 25, rng)
        self.assertEqual(indices.shape, (25, 50))
        self.assertTrue((indices >= 0).all() and (indices < 50).all())

    def test_pesaran_timmermann_detects_and_rejects_directional_content(self) -> None:
        actual = self.rng.normal(size=600)
        informative = actual + self.rng.normal(scale=0.5, size=600)
        good = inference.pesaran_timmermann(informative > 0, actual > 0)
        random = inference.pesaran_timmermann(self.rng.normal(size=600) > 0, actual > 0)
        self.assertGreater(good["pt_statistic"], 3.0)
        self.assertLess(good["pt_p_value"], 0.01)
        self.assertGreater(random["pt_p_value"], 0.05)

    def test_directional_and_excess_profitability_are_significant_when_present(self) -> None:
        daily_hits = 0.56 + 0.02 * self.rng.normal(size=250)
        directional = inference.directional_accuracy_test(daily_hits, baseline=0.5)
        self.assertGreater(directional["mean_hit_rate"], 0.5)
        self.assertLess(directional["hit_rate_p_value"], 0.01)

        ep = inference.excess_profitability_test(0.001 + 0.002 * self.rng.normal(size=250))
        self.assertLess(ep["excess_profitability_p_value"], 0.05)

    def test_information_coefficient_reports_ratio_and_interval(self) -> None:
        daily_ic = 0.03 + 0.05 * self.rng.normal(size=252)
        result = inference.information_coefficient_test(daily_ic, annualization_periods=252)
        self.assertLess(result["ic_p_value"], 0.01)
        self.assertTrue(result["ic_ci_low"] < result["mean_ic"] < result["ic_ci_high"])
        self.assertGreater(result["annualized_information_ratio"], 0.0)

    def test_sharpe_significance_brackets_positive_ratio(self) -> None:
        returns = 0.0008 + 0.01 * self.rng.normal(size=1000)
        result = inference.sharpe_significance(
            returns, annualization_periods=252, n_boot=400, seed=1
        )
        self.assertGreater(result["annualized_sharpe"], 0.0)
        self.assertLess(result["sharpe_p_value"], 0.05)
        self.assertTrue(
            result["sharpe_ci_low"] < result["annualized_sharpe"] < result["sharpe_ci_high"]
        )

    def test_diebold_mariano_flags_worse_model_and_direction(self) -> None:
        loss_a = np.abs(self.rng.normal(size=500)) + 0.2
        loss_b = np.abs(self.rng.normal(size=500))
        outcome = inference.diebold_mariano(loss_a, loss_b)
        self.assertLess(outcome["dm_p_value"], 0.01)
        self.assertGreater(outcome["mean_difference"], 0.0)

    def test_sharpe_difference_is_symmetric_in_pvalue(self) -> None:
        a = 0.001 + 0.01 * self.rng.normal(size=800)
        b = 0.0002 + 0.01 * self.rng.normal(size=800)
        forward = inference.sharpe_difference(a, b, annualization_periods=252, n_boot=200, seed=3)
        backward = inference.sharpe_difference(b, a, annualization_periods=252, n_boot=200, seed=3)
        self.assertAlmostEqual(forward["memmel_p_value"], backward["memmel_p_value"], places=10)
        self.assertAlmostEqual(
            forward["bootstrap_p_value"], backward["bootstrap_p_value"], places=10
        )

        identical = inference.sharpe_difference(
            a,
            a,
            annualization_periods=252,
            n_boot=200,
            block_length=5.0,
            seed=7,
        )
        self.assertEqual(identical["bootstrap_p_value"], 1.0)
        self.assertAlmostEqual(forward["delta_sharpe"], -backward["delta_sharpe"], places=10)

    def test_model_confidence_set_excludes_worst_and_keeps_best(self) -> None:
        losses = pd.DataFrame(
            {
                "A": np.abs(self.rng.normal(size=400)),
                "B": np.abs(self.rng.normal(size=400)) + 0.02,
                "C": np.abs(self.rng.normal(size=400)) + 0.5,
            }
        )
        mcs = inference.model_confidence_set(
            losses, alpha=0.1, n_boot=400, block_length=5.0, seed=5
        )
        self.assertTrue(mcs.loc["A", "in_confidence_set"])
        self.assertFalse(mcs.loc["C", "in_confidence_set"])
        self.assertLess(cast(float, mcs.loc["C", "mcs_p_value"]), 0.1)
        self.assertGreater(cast(float, mcs.loc["A", "mean_loss"]), 0.0)

    def test_model_confidence_set_keeps_equivalent_duplicate_models(self) -> None:
        losses = pd.DataFrame(
            {
                "A": np.linspace(-0.01, 0.01, 40),
                "A duplicate": np.linspace(-0.01, 0.01, 40),
                "worse": np.linspace(-0.01, 0.01, 40) + 0.1,
            }
        )

        result = inference.model_confidence_set(
            losses,
            alpha=0.05,
            n_boot=200,
            block_length=5.0,
            seed=11,
        )

        self.assertEqual(result.loc["A", "mcs_p_value"], result.loc["A duplicate", "mcs_p_value"])
        self.assertEqual(
            result.loc["A", "in_confidence_set"],
            result.loc["A duplicate", "in_confidence_set"],
        )

    def test_model_confidence_set_retains_all_models_when_too_few_dates(self) -> None:
        # Only one row survives dropna, so the bootstrap cannot run. No model can be eliminated:
        # the MCS stays non-empty (every model retained) with undefined p-values.
        losses = pd.DataFrame({"A": [0.1, np.nan], "B": [0.2, 0.3], "C": [0.05, np.nan]})
        result = inference.model_confidence_set(
            losses, alpha=0.1, n_boot=100, block_length=5.0, seed=0
        )
        self.assertTrue(bool(result["in_confidence_set"].all()))
        self.assertTrue(bool(result["mcs_p_value"].isna().all()))
        self.assertAlmostEqual(float(result.loc["C", "mean_loss"]), 0.05)

    def test_holm_and_bh_are_monotone_and_bounded(self) -> None:
        raw = np.array([0.001, 0.02, 0.03, 0.5])
        holm = inference.holm_adjust(raw)
        bh = inference.benjamini_hochberg(raw)
        self.assertTrue(np.all(holm >= raw - 1e-12))
        self.assertTrue(np.all(bh >= raw - 1e-12))
        self.assertTrue(np.all(holm <= 1.0) and np.all(bh <= 1.0))

    def test_adjust_pairwise_preserves_diagonal_and_symmetry(self) -> None:
        matrix = pd.DataFrame(
            [[np.nan, 0.01, 0.04], [0.01, np.nan, 0.20], [0.04, 0.20, np.nan]],
            index=["A", "B", "C"],
            columns=["A", "B", "C"],
        )
        adjusted = inference.adjust_pairwise(matrix, "holm")
        self.assertTrue(np.isnan(np.diag(adjusted.to_numpy())).all())
        self.assertAlmostEqual(
            cast(float, adjusted.loc["A", "B"]), cast(float, adjusted.loc["B", "A"])
        )
        self.assertGreaterEqual(
            cast(float, adjusted.loc["A", "B"]), cast(float, matrix.loc["A", "B"])
        )
        pd.testing.assert_frame_equal(inference.adjust_pairwise(matrix, "none"), matrix)


if __name__ == "__main__":
    unittest.main()
