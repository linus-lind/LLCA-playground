"""Deterministic grid enumeration and reproducible, de-duplicated random search."""

from __future__ import annotations

import unittest

from llca.training.tuning.search import generate_candidates
from llca.training.tuning.search_space import ChoiceDimension, LogRangeDimension, SearchSpace
from llca.training.tuning.settings import SearchSettings

_BASELINE = {"l1_ratio": 0.5, "C": 1.0}


class GridSearchTest(unittest.TestCase):
    def test_enumerates_cartesian_product_in_stable_order(self) -> None:
        space = SearchSpace(
            (
                ChoiceDimension("l1_ratio", (0.2, 0.5)),
                ChoiceDimension("C", (0.1, 1.0, 10.0)),
            )
        )
        candidates = generate_candidates(SearchSettings("grid", 0, 0), space, _BASELINE)

        self.assertEqual(
            candidates,
            [
                {"l1_ratio": 0.2, "C": 0.1},
                {"l1_ratio": 0.2, "C": 1.0},
                {"l1_ratio": 0.2, "C": 10.0},
                {"l1_ratio": 0.5, "C": 0.1},
                {"l1_ratio": 0.5, "C": 1.0},
                {"l1_ratio": 0.5, "C": 10.0},
            ],
        )

    def test_partial_space_completes_against_baseline(self) -> None:
        space = SearchSpace((ChoiceDimension("C", (0.1, 10.0)),))
        candidates = generate_candidates(SearchSettings("grid", 0, 0), space, _BASELINE)

        # The un-searched l1_ratio is taken from the baseline in every candidate.
        self.assertEqual(candidates, [{"l1_ratio": 0.5, "C": 0.1}, {"l1_ratio": 0.5, "C": 10.0}])

    def test_empty_space_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "empty search space"):
            generate_candidates(SearchSettings("grid", 0, 0), SearchSpace(()), _BASELINE)


class RandomSearchTest(unittest.TestCase):
    def _space(self) -> SearchSpace:
        return SearchSpace(
            (
                LogRangeDimension("C", 1e-3, 1e2, 12),
                ChoiceDimension("l1_ratio", (0.2, 0.5, 0.8)),
            )
        )

    def test_is_reproducible_from_seed(self) -> None:
        settings = SearchSettings("random", 6, 7)
        first = generate_candidates(settings, self._space(), _BASELINE)
        second = generate_candidates(settings, self._space(), _BASELINE)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 6)

    def test_produces_no_duplicate_candidates(self) -> None:
        settings = SearchSettings("random", 25, 1)
        candidates = generate_candidates(settings, self._space(), _BASELINE)
        keys = [tuple(sorted(candidate.items())) for candidate in candidates]
        self.assertEqual(len(keys), len(set(keys)))

    def test_exhausted_discrete_space_returns_only_unique(self) -> None:
        # Three discrete choices cannot yield more than three unique candidates.
        space = SearchSpace((ChoiceDimension("l1_ratio", (0.2, 0.5, 0.8)),))
        candidates = generate_candidates(SearchSettings("random", 10, 3), space, _BASELINE)
        self.assertLessEqual(len(candidates), 3)
        self.assertEqual(len({tuple(sorted(c.items())) for c in candidates}), len(candidates))

    def test_zero_trials_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "n_trials"):
            generate_candidates(SearchSettings("random", 0, 0), self._space(), _BASELINE)


class UnknownMethodTest(unittest.TestCase):
    def test_unknown_method_is_rejected(self) -> None:
        space = SearchSpace((ChoiceDimension("C", (1.0,)),))
        with self.assertRaisesRegex(ValueError, "unknown search method"):
            generate_candidates(SearchSettings("bayesian", 1, 0), space, _BASELINE)


if __name__ == "__main__":
    unittest.main()
