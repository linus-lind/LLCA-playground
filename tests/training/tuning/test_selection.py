"""Exact paired standard-error mathematics for baseline-versus-candidate adoption."""

from __future__ import annotations

import math
import unittest

from llca.training.tuning.selection import adopt_candidate, paired_improvement


class PairedImprovementTest(unittest.TestCase):
    def test_mean_and_standard_error_of_fold_differences(self) -> None:
        baseline = [1.0, 2.0, 3.0]
        candidate = [0.9, 1.5, 2.9]
        mean, standard_error = paired_improvement(baseline, candidate)
        # d = [0.1, 0.5, 0.1]; mean = 0.7/3; sd(ddof=1) = sqrt(0.106667/2); se = sd/sqrt(3)
        self.assertAlmostEqual(mean, 0.7 / 3.0)
        expected_se = math.sqrt(0.10666666666666667 / 2.0) / math.sqrt(3.0)
        self.assertAlmostEqual(standard_error, expected_se)

    def test_fewer_than_two_folds_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least two folds"):
            paired_improvement([1.0], [0.5])

    def test_mismatched_lengths_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "same folds"):
            paired_improvement([1.0, 2.0], [0.5])


class AdoptCandidateTest(unittest.TestCase):
    def test_uniform_improvement_is_adopted(self) -> None:
        # Zero-variance strictly positive improvement is adopted at any margin.
        self.assertTrue(
            adopt_candidate([1.0, 1.0, 1.0], [0.5, 0.5, 0.5], standard_error_margin=1.0)
        )

    def test_one_standard_error_threshold(self) -> None:
        baseline = [1.0, 2.0, 3.0]
        candidate = [0.9, 1.5, 2.9]
        # mean ~ 0.233, se ~ 0.133: beats a 1-SE margin but not a 2-SE margin.
        self.assertTrue(adopt_candidate(baseline, candidate, standard_error_margin=1.0))
        self.assertFalse(adopt_candidate(baseline, candidate, standard_error_margin=2.0))

    def test_zero_margin_adopts_on_any_mean_improvement(self) -> None:
        baseline = [1.0, 2.0, 3.0]
        candidate = [0.9, 1.5, 2.9]
        self.assertTrue(adopt_candidate(baseline, candidate, standard_error_margin=0.0))

    def test_worse_candidate_is_rejected(self) -> None:
        self.assertFalse(adopt_candidate([1.0, 1.0], [2.0, 2.0], standard_error_margin=1.0))

    def test_no_improvement_is_rejected(self) -> None:
        self.assertFalse(adopt_candidate([1.0, 1.0], [1.0, 1.0], standard_error_margin=0.0))


if __name__ == "__main__":
    unittest.main()
