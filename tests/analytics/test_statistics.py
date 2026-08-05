"""Unit tests for :mod:`llca.analytics.stats.statistics`.

These validate the ``rank_buckets`` equal-count (``N >= K``) and spanning (``N < K``)
regimes with analytically understandable expectations rather than by reproducing the
implementation formula.
"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from llca.analytics.stats.statistics import rank_buckets


def _sizes(labels: pd.Series, bucket_count: int) -> list[int]:
    counts = labels.dropna().astype(int).value_counts()
    return [int(counts.get(bucket, 0)) for bucket in range(1, bucket_count + 1)]


class RankBucketsEqualCountTest(unittest.TestCase):
    def test_n500_k10_gives_fifty_per_bucket(self) -> None:
        scores = pd.Series(np.random.RandomState(0).permutation(500).astype(float))
        self.assertEqual(_sizes(rank_buckets(scores, 10), 10), [50] * 10)

    def test_n100_k5_gives_twenty_per_bucket(self) -> None:
        scores = pd.Series(np.random.RandomState(1).permutation(100).astype(float))
        self.assertEqual(_sizes(rank_buckets(scores, 5), 5), [20] * 5)

    def test_n11_k10_sizes_differ_by_at_most_one(self) -> None:
        scores = pd.Series(np.random.RandomState(2).permutation(11).astype(float))
        sizes = _sizes(rank_buckets(scores, 10), 10)
        self.assertEqual(sum(sizes), 11)
        self.assertTrue(all(size >= 1 for size in sizes))  # every bucket populated
        self.assertLessEqual(max(sizes) - min(sizes), 1)

    def test_n10_k10_one_per_bucket(self) -> None:
        scores = pd.Series(np.random.RandomState(3).permutation(10).astype(float))
        self.assertEqual(_sizes(rank_buckets(scores, 10), 10), [1] * 10)

    def test_equal_count_invariant_holds_across_random_panels(self) -> None:
        rng = np.random.RandomState(4)
        for n, k in [(23, 5), (37, 4), (200, 7), (64, 8), (99, 10)]:
            scores = pd.Series(rng.normal(size=n))
            sizes = _sizes(rank_buckets(scores, k), k)
            self.assertEqual(sum(sizes), n, msg=f"N={n} K={k}")
            self.assertTrue(all(size >= 1 for size in sizes), msg=f"N={n} K={k}")
            self.assertLessEqual(max(sizes) - min(sizes), 1, msg=f"N={n} K={k}")


class RankBucketsSpanningTest(unittest.TestCase):
    def test_n2_k10_labels_the_extremes(self) -> None:
        scores = pd.Series([10.0, 20.0])
        self.assertEqual(rank_buckets(scores, 10).tolist(), [1, 10])

    def test_n9_k10_spreads_across_the_full_range(self) -> None:
        # Ascending scores so ordinal rank r == position + 1; spanning maps r in 1..9 across
        # the [1, 10] label range, hitting both endpoints and skipping exactly one interior label.
        scores = pd.Series(np.arange(9.0))
        result = rank_buckets(scores, 10).tolist()
        self.assertEqual(result, [1, 2, 3, 4, 5, 6, 7, 8, 10])
        self.assertEqual(min(result), 1)
        self.assertEqual(max(result), 10)

    def test_single_observation_is_unlabelled(self) -> None:
        scores = pd.Series([42.0])
        self.assertTrue(rank_buckets(scores, 5).isna().all())


class RankBucketsTiesAndMissingTest(unittest.TestCase):
    def test_all_equal_scores_are_split_into_equal_count_buckets(self) -> None:
        scores = pd.Series([5.0] * 10, index=[f"e{i}" for i in range(10)])
        self.assertEqual(_sizes(rank_buckets(scores, 5), 5), [2, 2, 2, 2, 2])

    def test_ties_are_broken_deterministically_by_index(self) -> None:
        scores = pd.Series([1.0, 1.0, 1.0, 1.0], index=["d", "a", "c", "b"])
        first = rank_buckets(scores, 2)
        second = rank_buckets(scores.sample(frac=1.0, random_state=7), 2)
        # Same label for the same index regardless of incoming row order.
        self.assertEqual(first.to_dict(), second.to_dict())
        # Lower entity labels fall in the lower bucket (tie broken by index).
        self.assertEqual(first["a"], 1)
        self.assertEqual(first["b"], 1)
        self.assertEqual(first["c"], 2)
        self.assertEqual(first["d"], 2)

    def test_missing_scores_are_neither_ranked_nor_counted(self) -> None:
        scores = pd.Series([1.0, 2.0, np.nan, 4.0, 5.0], index=list("abcde"))
        result = rank_buckets(scores, 2)
        self.assertTrue(pd.isna(result["c"]))
        # Four finite scores over two buckets -> two per bucket.
        self.assertEqual(_sizes(result, 2), [2, 2])

    def test_permutation_of_distinct_scores_does_not_change_labels(self) -> None:
        scores = pd.Series([3.0, 1.0, 2.0, 5.0, 4.0], index=list("abcde"))
        base = rank_buckets(scores, 5)
        shuffled = rank_buckets(scores.sample(frac=1.0, random_state=11), 5)
        self.assertEqual(base.to_dict(), shuffled.reindex(base.index).to_dict())

    def test_result_index_matches_input_index_order(self) -> None:
        scores = pd.Series([2.0, 0.0, 1.0], index=["z", "x", "y"])
        result = rank_buckets(scores, 3)
        self.assertEqual(list(result.index), ["z", "x", "y"])

    def test_duplicate_index_labels_are_handled_positionally(self) -> None:
        # A non-unique, unsorted index must neither raise nor misplace labels.
        scores = pd.Series([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], index=pd.Index([3, 1, 1, 3, 2, 4]))
        result = rank_buckets(scores, 2)
        self.assertEqual(result.tolist(), [1, 1, 1, 2, 2, 2])
        self.assertEqual(list(result.index), [3, 1, 1, 3, 2, 4])


class RankBucketsPanelTest(unittest.TestCase):
    def _panel(self, per_date: list[list[float]]) -> pd.Series:
        dates = pd.date_range("2024-01-01", periods=len(per_date), name="date")
        frames = []
        for date, row in zip(dates, per_date, strict=True):
            index = pd.MultiIndex.from_product(
                [[date], [f"e{i}" for i in range(len(row))]], names=["date", "entity"]
            )
            frames.append(pd.Series(row, index=index))
        return pd.concat(frames)

    def test_buckets_are_assigned_within_each_date(self) -> None:
        panel = self._panel([[1.0, 2.0, 3.0, 4.0], [40.0, 30.0, 20.0, 10.0]])
        result = rank_buckets(panel, 2)
        # Date 0 ascending -> [1,1,2,2]; date 1 descending -> [2,2,1,1].
        self.assertEqual(result.tolist(), [1, 1, 2, 2, 2, 2, 1, 1])

    def test_pooled_ranks_all_observations_together(self) -> None:
        panel = self._panel([[1.0, 2.0], [3.0, 4.0]])
        pooled = rank_buckets(panel, 2, pooled=True)
        # Pooled ordering 1<2<3<4 -> lower two in bucket 1, upper two in bucket 2.
        self.assertEqual(pooled.tolist(), [1, 1, 2, 2])

    def test_singleton_date_is_unlabelled(self) -> None:
        dates = pd.date_range("2024-01-01", periods=2, name="date")
        index = pd.MultiIndex.from_tuples(
            [(dates[0], "a"), (dates[0], "b"), (dates[1], "a")], names=["date", "entity"]
        )
        result = rank_buckets(pd.Series([1.0, 2.0, 9.0], index=index), 4)
        self.assertEqual(result.iloc[0], 1)
        self.assertEqual(result.iloc[1], 4)
        self.assertTrue(pd.isna(result.iloc[2]))


class FiniteCorrelationTest(unittest.TestCase):
    """Guards the optimized ``_finite_correlation`` (must equal a guarded pandas ``corr``)."""

    def test_matches_pandas_corr_across_random_inputs(self) -> None:
        from llca.analytics.evaluation.signals import _finite_correlation

        rng = np.random.RandomState(0)
        for method in ("pearson", "spearman"):
            for _ in range(80):
                n = int(rng.randint(2, 40))
                left = pd.Series(rng.normal(size=n))
                right = pd.Series(rng.normal(size=n))
                left[rng.rand(n) < 0.2] = np.nan
                right[rng.rand(n) < 0.2] = np.nan
                expected = left.corr(right, method=method)
                got = _finite_correlation(left, right, method)  # type: ignore[arg-type]
                if np.isnan(expected):
                    self.assertTrue(np.isnan(got))
                else:
                    self.assertAlmostEqual(got, float(expected), places=10)

    def test_degenerate_inputs_return_nan(self) -> None:
        from llca.analytics.evaluation.signals import _finite_correlation

        constant = _finite_correlation(
            pd.Series([1.0, 1.0, 1.0]), pd.Series([2.0, 3.0, 4.0]), "pearson"
        )
        too_short = _finite_correlation(
            pd.Series([1.0], index=[0]), pd.Series([2.0], index=[0]), "pearson"
        )
        self.assertTrue(np.isnan(constant))
        self.assertTrue(np.isnan(too_short))

    def test_is_invariant_to_row_order(self) -> None:
        from llca.analytics.evaluation.signals import _finite_correlation

        left = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=list("abcde"))
        right = pd.Series([2.0, 1.0, 4.0, 3.0, 5.0], index=list("abcde"))
        base = _finite_correlation(left, right, "spearman")
        shuffled = _finite_correlation(left.sample(frac=1.0, random_state=3), right, "spearman")
        self.assertAlmostEqual(base, shuffled, places=12)


if __name__ == "__main__":
    unittest.main()
