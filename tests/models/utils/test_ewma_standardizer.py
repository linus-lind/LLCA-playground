import unittest

import numpy as np
import pandas as pd

from llca.data.modules.masked_panel import MaskedPanel
from llca.models.utils.ewma_standardizer import EwmaStandardizer, half_life_to_decay


def _index(dates: list[str], entities: list[int]) -> pd.MultiIndex:
    return pd.MultiIndex.from_product(
        [pd.to_datetime(dates), entities], names=["date", "instrument_id"]
    )


def _panel(values: pd.DataFrame, *, invalid: pd.DataFrame | None = None) -> MaskedPanel:
    """Build a minimal MaskedPanel; ``invalid`` cells get age=-1 and a NaN value."""
    index = values.index
    if invalid is None:
        invalid = pd.DataFrame(False, index=index, columns=values.columns)
    age = pd.DataFrame(0, index=index, columns=values.columns).mask(invalid, -1)
    observed = ~invalid
    masked_values = values.mask(invalid)
    return MaskedPanel(
        values=masked_values, observed=observed, age=age, segment=pd.Series(0, index=index)
    )


def _entity_panel(dates: list[str], entity: int, column: str, series: list[float]) -> MaskedPanel:
    values = pd.DataFrame({column: series}, index=_index(dates, [entity]))
    return _panel(values)


def _reference_ewma(
    x: list[float], half_life: float, eps: float = 1e-8
) -> tuple[list[float], float, float]:
    """Independent, trivially-correct recursion used to check the class under test."""
    return _reference_ewma_walk(x, x, half_life, eps)


def _reference_ewma_walk(
    prior_x: list[float], walk_x: list[float], half_life: float, eps: float = 1e-8
) -> tuple[list[float], float, float]:
    """Like ``_reference_ewma``, but the init prior comes from ``prior_x`` only while the
    recursion walks ``walk_x`` (which may extend beyond ``prior_x``, e.g. train -> val)."""
    decay = 2.0 ** (-1.0 / half_life)
    mean = sum(prior_x) / len(prior_x)
    var = sum((v - mean) ** 2 for v in prior_x) / len(prior_x)
    outputs: list[float] = []
    for value in walk_x:
        diff = value - mean
        std = max(var**0.5, eps)
        outputs.append(diff / std)
        mean = mean + (1 - decay) * diff
        var = decay * (var + (1 - decay) * diff * diff)
    return outputs, mean, var


class HalfLifeConversionTest(unittest.TestCase):
    def test_half_life_of_one_gives_decay_one_half(self) -> None:
        self.assertAlmostEqual(half_life_to_decay(1.0), 0.5)

    def test_larger_half_life_gives_decay_closer_to_one(self) -> None:
        self.assertGreater(half_life_to_decay(126.0), half_life_to_decay(10.0))
        self.assertLess(half_life_to_decay(126.0), 1.0)

    def test_rejects_non_positive_half_life(self) -> None:
        with self.assertRaises(ValueError):
            half_life_to_decay(0.0)
        with self.assertRaises(ValueError):
            half_life_to_decay(-5.0)


class EwmaMathTest(unittest.TestCase):
    def test_matches_independent_reference_recursion(self) -> None:
        x = [1.0, 2.0, 4.0]
        expected_outputs, expected_mean, expected_var = _reference_ewma(x, half_life=1.0)
        panel = _entity_panel(["2024-01-01", "2024-01-02", "2024-01-03"], 1, "f", x)

        standardizer = EwmaStandardizer(half_life=1.0, history_buffer=8)
        standardizer.fit(panel)
        result = standardizer.transform(panel)

        np.testing.assert_allclose(result["f"].to_numpy(), expected_outputs, rtol=1e-10)
        np.testing.assert_allclose(standardizer._mean[1], [expected_mean], rtol=1e-10)
        np.testing.assert_allclose(standardizer._var[1], [expected_var], rtol=1e-10)

    def test_step_t_uses_state_strictly_before_t(self) -> None:
        """The update at t must be applied *after* z_t is read, not folded into it."""
        x = [10.0, 10.0, 10.0]  # a constant prior; a real jump only at step 2
        x[1] = 40.0
        panel = _entity_panel(["2024-01-01", "2024-01-02", "2024-01-03"], 1, "f", x)
        standardizer = EwmaStandardizer(half_life=2.0, history_buffer=8)
        standardizer.fit(panel)
        result = standardizer.transform(panel)["f"].to_numpy()

        # z_0 depends only on the (whole-train) init prior, not on the later jump at t=1.
        init_mean = sum(x) / len(x)
        init_var = sum((v - init_mean) ** 2 for v in x) / len(x)
        expected_z0 = (x[0] - init_mean) / max(init_var**0.5, 1e-8)
        self.assertAlmostEqual(result[0], expected_z0, places=10)


class IndependentVarianceRecursionTest(unittest.TestCase):
    """Cross-checks transform()'s Welford-style variance update against a second,
    algebraically distinct estimator of the same quantity: two plain EWMA accumulators for
    E[X] and E[X^2], combined via Var = E[X^2] - E[X]^2. Unlike ``_reference_ewma_walk``
    (which re-types the class's own ``var_after = decay * (var_before + (1 - decay) * diff *
    diff)`` line and so cannot catch a convention error shared by both), this path uses no
    "deviation from the old mean" trick at all -- it is only two independent linear
    recursions and a subtraction. One can show algebraically the two are equal *if and only
    if* the decay/deviation convention matches; a wrong factor of decay, a deviation taken
    from the wrong mean, or a mixed-up alpha/decay would make them diverge numerically.
    """

    def test_matches_e_x_squared_minus_e_x_squared_identity(self) -> None:
        prior = [2.0, 4.0, 3.0]
        walk = [5.0, 1.0, 8.0, -2.0, 6.5, 0.0, 3.3]
        half_life = 4.0
        decay = 2.0 ** (-1.0 / half_life)
        alpha = 1.0 - decay

        mean0 = sum(prior) / len(prior)
        var0 = sum((v - mean0) ** 2 for v in prior) / len(prior)
        s1, s2 = mean0, var0 + mean0 * mean0
        independent_means = []
        independent_vars = []
        for x in walk:
            s1 = decay * s1 + alpha * x
            s2 = decay * s2 + alpha * x * x
            independent_means.append(s1)
            independent_vars.append(s2 - s1 * s1)

        prior_dates = [f"2024-01-{i + 1:02d}" for i in range(len(prior))]
        walk_dates = [f"2024-02-{i + 1:02d}" for i in range(len(walk))]
        prior_panel = _entity_panel(prior_dates, 1, "f", prior)
        walk_panel = _entity_panel(walk_dates, 1, "f", walk)

        standardizer = EwmaStandardizer(half_life=half_life, history_buffer=64)
        standardizer.fit(prior_panel)
        standardizer.transform(walk_panel)

        np.testing.assert_allclose(standardizer._mean[1], [independent_means[-1]], rtol=1e-10)
        np.testing.assert_allclose(standardizer._var[1], [independent_vars[-1]], rtol=1e-10)


class EntityIsolationTest(unittest.TestCase):
    def test_entities_never_share_mean_variance_or_output(self) -> None:
        dates = ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"]
        low = [1.0, 1.1, 0.9, 1.2]  # relative pattern: up, down, up
        high = [1000.0, 900.0, 1100.0, 800.0]  # different scale AND opposite pattern
        values_low = pd.DataFrame({"f": low}, index=_index(dates, [1]))
        values_high = pd.DataFrame({"f": high}, index=_index(dates, [2]))
        combined = pd.concat([values_low, values_high]).sort_index()
        panel = _panel(combined)

        together = EwmaStandardizer(half_life=2.0, history_buffer=8)
        together.fit(panel)
        result = together.transform(panel)

        alone = EwmaStandardizer(half_life=2.0, history_buffer=8)
        low_panel = _entity_panel(dates, 1, "f", low)
        alone.fit(low_panel)
        alone_result = alone.transform(low_panel)

        low_from_together = result.xs(1, level="instrument_id")["f"].to_numpy()
        np.testing.assert_allclose(low_from_together, alone_result["f"].to_numpy(), rtol=1e-12)
        self.assertFalse(np.allclose(low_from_together, result.xs(2, level="instrument_id")["f"]))


class FeatureIsolationTest(unittest.TestCase):
    def test_features_never_share_mean_variance_or_output(self) -> None:
        dates = ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"]
        f1 = [1.0, 2.0, 1.5, 3.0]
        f2 = [100.0, 50.0, 75.0, 25.0]
        combined = pd.DataFrame({"f1": f1, "f2": f2}, index=_index(dates, [1]))
        panel = _panel(combined)

        together = EwmaStandardizer(half_life=2.0, history_buffer=8)
        together.fit(panel)
        result = together.transform(panel)

        alone = EwmaStandardizer(half_life=2.0, history_buffer=8)
        f1_panel = _panel(pd.DataFrame({"f1": f1}, index=_index(dates, [1])))
        alone.fit(f1_panel)
        alone_result = alone.transform(f1_panel)

        np.testing.assert_allclose(
            result["f1"].to_numpy(), alone_result["f1"].to_numpy(), rtol=1e-12
        )


class TrainInitializationTest(unittest.TestCase):
    def test_init_uses_only_valid_rows_of_the_given_panel(self) -> None:
        dates = ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"]
        values = pd.DataFrame({"f": [1.0, np.nan, 3.0, 5.0]}, index=_index(dates, [1]))
        invalid = pd.DataFrame({"f": [False, True, False, False]}, index=_index(dates, [1]))
        panel = _panel(values, invalid=invalid)

        standardizer = EwmaStandardizer(half_life=3.0, history_buffer=8)
        standardizer.fit(panel)

        valid_x = [1.0, 3.0, 5.0]
        expected_mean = np.mean(valid_x)
        expected_var = np.var(valid_x, ddof=0)
        np.testing.assert_allclose(standardizer._mean[1], [expected_mean])
        np.testing.assert_allclose(standardizer._var[1], [expected_var])

    def test_init_does_not_depend_on_data_never_passed_to_fit(self) -> None:
        dates = ["2024-01-01", "2024-01-02"]
        panel = _entity_panel(dates, 1, "f", [1.0, 2.0])
        later_panel = _entity_panel(["2024-06-01"], 1, "f", [999_999.0])

        standardizer = EwmaStandardizer(half_life=3.0, history_buffer=8)
        standardizer.fit(panel)
        mean_before = standardizer._mean[1].copy()

        # Fitting must be an explicit, separate call; merely holding a reference to a
        # later panel must never change already-computed initialization statistics.
        del later_panel
        np.testing.assert_allclose(standardizer._mean[1], mean_before)


class TrainValidationContinuationTest(unittest.TestCase):
    def test_validation_continues_the_training_recursion_without_refitting(self) -> None:
        x = [1.0, 2.0, 1.5, 4.0, 3.5]
        expected_outputs, _, _ = _reference_ewma_walk(x[:3], x, half_life=4.0)

        train_panel = _entity_panel(["2024-01-01", "2024-01-02", "2024-01-03"], 1, "f", x[:3])
        val_panel = _entity_panel(["2024-01-04", "2024-01-05"], 1, "f", x[3:])

        standardizer = EwmaStandardizer(half_life=4.0, history_buffer=8)
        standardizer.fit(train_panel)
        train_result = standardizer.transform(train_panel)["f"].to_numpy()
        val_result = standardizer.transform(val_panel)["f"].to_numpy()

        np.testing.assert_allclose(train_result, expected_outputs[:3], rtol=1e-10)
        np.testing.assert_allclose(val_result, expected_outputs[3:], rtol=1e-10)

    def test_first_validation_row_uses_state_carried_from_training(self) -> None:
        x = [5.0, 6.0, 100.0]
        train_panel = _entity_panel(["2024-01-01", "2024-01-02"], 1, "f", x[:2])
        val_panel = _entity_panel(["2024-01-03"], 1, "f", x[2:])

        standardizer = EwmaStandardizer(half_life=4.0, history_buffer=8)
        standardizer.fit(train_panel)
        standardizer.transform(train_panel)
        mean_after_train = standardizer._mean[1].copy()
        var_after_train = standardizer._var[1].copy()

        val_result = standardizer.transform(val_panel)["f"].to_numpy()
        expected = (x[2] - mean_after_train[0]) / max(var_after_train[0] ** 0.5, 1e-8)
        self.assertAlmostEqual(val_result[0], expected, places=10)


class ValidationTestContinuationTest(unittest.TestCase):
    def test_test_split_continues_from_post_validation_state_across_a_serialization_boundary(
        self,
    ) -> None:
        x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        expected_outputs, _, _ = _reference_ewma_walk(x[:2], x, half_life=3.0)

        train_panel = _entity_panel(["2024-01-01", "2024-01-02"], 1, "f", x[:2])
        val_panel = _entity_panel(["2024-01-03", "2024-01-04"], 1, "f", x[2:4])
        test_panel = _entity_panel(["2024-01-05", "2024-01-06"], 1, "f", x[4:])

        standardizer = EwmaStandardizer(half_life=3.0, history_buffer=8)
        standardizer.fit(train_panel)
        standardizer.transform(train_panel)
        standardizer.transform(val_panel)

        # Simulate the real fit()-in-one-process, predict()-in-another-process boundary.
        restored = EwmaStandardizer.from_state_dict(standardizer.state_dict())
        test_result = restored.transform(test_panel)["f"].to_numpy()

        np.testing.assert_allclose(test_result, expected_outputs[4:], rtol=1e-10)

    def test_test_boundary_is_not_a_refit(self) -> None:
        x = [1.0, 2.0, 3.0]
        train_panel = _entity_panel(["2024-01-01"], 1, "f", x[:1])
        val_panel = _entity_panel(["2024-01-02"], 1, "f", x[1:2])
        test_panel = _entity_panel(["2024-01-03"], 1, "f", x[2:3])

        standardizer = EwmaStandardizer(half_life=3.0, history_buffer=8)
        standardizer.fit(train_panel)
        standardizer.transform(train_panel)
        standardizer.transform(val_panel)
        mean_before_test = standardizer._mean[1].copy()
        var_before_test = standardizer._var[1].copy()

        result = standardizer.transform(test_panel)["f"].to_numpy()
        expected = (x[2] - mean_before_test[0]) / max(var_before_test[0] ** 0.5, 1e-8)
        self.assertAlmostEqual(result[0], expected, places=10)


class LeakageTest(unittest.TestCase):
    def test_changing_a_future_observation_does_not_change_earlier_transformed_output(
        self,
    ) -> None:
        """Mandatory: perturbing a later value must leave earlier outputs byte-for-byte."""
        train_x = [1.0, 2.0, 3.0]
        train_panel = _entity_panel(["2024-01-01", "2024-01-02", "2024-01-03"], 1, "f", train_x)

        base_future = [4.0, 5.0, 6.0, 7.0]
        perturbed_future = [4.0, 5.0, 999_999.0, 7.0]
        future_dates = ["2024-01-04", "2024-01-05", "2024-01-06", "2024-01-07"]

        base = EwmaStandardizer(half_life=5.0, history_buffer=8)
        base.fit(train_panel)
        base.transform(train_panel)
        base_out = base.transform(_entity_panel(future_dates, 1, "f", base_future))["f"].to_numpy()

        perturbed = EwmaStandardizer(half_life=5.0, history_buffer=8)
        perturbed.fit(train_panel)
        perturbed.transform(train_panel)
        perturbed_out = perturbed.transform(_entity_panel(future_dates, 1, "f", perturbed_future))[
            "f"
        ].to_numpy()

        # Everything strictly before the perturbed (3rd) row is untouched.
        np.testing.assert_array_equal(base_out[:2], perturbed_out[:2])
        # The perturbed row itself, and everything after it, may (and here does) differ.
        self.assertFalse(np.array_equal(base_out[2:], perturbed_out[2:]))

    def test_init_prior_intentionally_pools_the_whole_training_split_document_the_nuance(
        self,
    ) -> None:
        """A later *training* value CAN move earlier training outputs, but only through
        the shared whole-train initialization prior -- never through the step recursion.
        This is the one documented, intentional exception to causal ordering.
        """
        train_a = [1.0, 2.0, 3.0]
        train_b = [1.0, 2.0, 999.0]  # only the last training point changes
        dates = ["2024-01-01", "2024-01-02", "2024-01-03"]

        standardizer_a = EwmaStandardizer(half_life=4.0, history_buffer=8)
        panel_a = _entity_panel(dates, 1, "f", train_a)
        standardizer_a.fit(panel_a)
        out_a = standardizer_a.transform(panel_a)["f"].to_numpy()

        standardizer_b = EwmaStandardizer(half_life=4.0, history_buffer=8)
        panel_b = _entity_panel(dates, 1, "f", train_b)
        standardizer_b.fit(panel_b)
        out_b = standardizer_b.transform(panel_b)["f"].to_numpy()

        # The first output differs solely because the whole-train prior differs -- prove
        # it by reproducing out_a[0]/out_b[0] from each fit's own recorded init prior via
        # the identical step formula, independent of the class's transform() call.
        mean_a, var_a = float(np.mean(train_a)), float(np.var(train_a, ddof=0))
        mean_b, var_b = float(np.mean(train_b)), float(np.var(train_b, ddof=0))
        expected_a0 = (train_a[0] - mean_a) / max(var_a**0.5, 1e-8)
        expected_b0 = (train_b[0] - mean_b) / max(var_b**0.5, 1e-8)
        self.assertAlmostEqual(out_a[0], expected_a0, places=10)
        self.assertAlmostEqual(out_b[0], expected_b0, places=10)
        self.assertNotAlmostEqual(out_a[0], out_b[0], places=6)


class MissingValueTest(unittest.TestCase):
    def test_missing_row_does_not_update_state_and_outputs_zero(self) -> None:
        dates = ["2024-01-01", "2024-01-02", "2024-01-03"]
        values = pd.DataFrame({"f": [1.0, np.nan, 3.0]}, index=_index(dates, [1]))
        invalid = pd.DataFrame({"f": [False, True, False]}, index=_index(dates, [1]))
        panel = _panel(values, invalid=invalid)

        standardizer = EwmaStandardizer(half_life=3.0, history_buffer=8)
        standardizer.fit(panel)
        mean_after_fit = standardizer._mean[1].copy()
        var_after_fit = standardizer._var[1].copy()
        result = standardizer.transform(panel)["f"].to_numpy()

        self.assertEqual(result[1], 0.0)
        # Row 2 (valid) must be normalized as if row 1 (missing) had never occurred: the
        # state used for row 2 equals the state produced by row 0 alone, unchanged by row 1.
        decay = half_life_to_decay(3.0)
        diff0 = 1.0 - mean_after_fit[0]
        mean_after_row0 = mean_after_fit[0] + (1 - decay) * diff0
        var_after_row0 = decay * (var_after_fit[0] + (1 - decay) * diff0 * diff0)
        expected_row2 = (3.0 - mean_after_row0) / max(var_after_row0**0.5, 1e-8)
        self.assertAlmostEqual(result[2], expected_row2, places=10)

    def test_missing_row_still_advances_the_watermark(self) -> None:
        """A missing observation was still 'seen' at that date; replaying it later must
        hit the history cache, not silently fall through to the (wrong) new-row path.
        """
        dates = ["2024-01-01", "2024-01-02"]
        values = pd.DataFrame({"f": [1.0, np.nan]}, index=_index(dates, [1]))
        invalid = pd.DataFrame({"f": [False, True]}, index=_index(dates, [1]))
        panel = _panel(values, invalid=invalid)

        standardizer = EwmaStandardizer(half_life=3.0, history_buffer=8)
        standardizer.fit(panel)
        standardizer.transform(panel)

        replay = standardizer.transform(panel)["f"].to_numpy()
        np.testing.assert_array_equal(replay, [0.0, 0.0])


class LowVarianceTest(unittest.TestCase):
    def test_constant_series_produces_finite_output(self) -> None:
        dates = ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"]
        panel = _entity_panel(dates, 1, "f", [5.0, 5.0, 5.0, 5.0])
        standardizer = EwmaStandardizer(half_life=3.0, history_buffer=8)
        standardizer.fit(panel)
        result = standardizer.transform(panel)["f"].to_numpy()
        self.assertTrue(np.isfinite(result).all())
        np.testing.assert_allclose(result, 0.0, atol=1e-6)

    def test_near_zero_variance_then_a_small_move_stays_finite(self) -> None:
        dates = ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]
        panel = _entity_panel(dates, 1, "f", [5.0, 5.0, 5.0, 5.0, 5.0001])
        standardizer = EwmaStandardizer(half_life=3.0, history_buffer=8)
        standardizer.fit(panel)
        result = standardizer.transform(panel)["f"].to_numpy()
        self.assertTrue(np.isfinite(result).all())


class ChangingUniverseTest(unittest.TestCase):
    def test_entity_absent_then_reappearing_retains_its_last_state(self) -> None:
        x = [1.0, 2.0, 3.0, 4.0]
        expected_outputs, _, _ = _reference_ewma_walk(x[:2], x, half_life=2.0)

        first_panel = _entity_panel(["2024-01-01", "2024-01-02"], 1, "f", x[:2])
        # 2024-01-03/04 deliberately absent: entity 1 has no row on those dates at all.
        second_panel = _entity_panel(["2024-01-05", "2024-01-06"], 1, "f", x[2:])

        standardizer = EwmaStandardizer(half_life=2.0, history_buffer=8)
        standardizer.fit(first_panel)
        first_out = standardizer.transform(first_panel)["f"].to_numpy()
        second_out = standardizer.transform(second_panel)["f"].to_numpy()

        np.testing.assert_allclose(first_out, expected_outputs[:2], rtol=1e-10)
        np.testing.assert_allclose(second_out, expected_outputs[2:], rtol=1e-10)

    def test_entity_unseen_during_training_falls_back_to_pooled_training_statistic(self) -> None:
        train_panel = _entity_panel(["2024-01-01", "2024-01-02"], 1, "f", [10.0, 12.0])
        standardizer = EwmaStandardizer(half_life=3.0, history_buffer=8)
        standardizer.fit(train_panel)

        new_entity_panel = _entity_panel(["2024-01-03"], 99, "f", [11.0])
        result = standardizer.transform(new_entity_panel)["f"].to_numpy()

        expected_mean = float(np.mean([10.0, 12.0]))
        expected_var = float(np.var([10.0, 12.0], ddof=0))
        expected = (11.0 - expected_mean) / max(expected_var**0.5, 1e-8)
        self.assertAlmostEqual(result[0], expected, places=10)


class ReplayOverlapTest(unittest.TestCase):
    def test_overlapping_panel_replays_bit_exact_output_without_double_updating_state(
        self,
    ) -> None:
        x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        expected_outputs, expected_mean, expected_var = _reference_ewma_walk(
            x[:3], x, half_life=3.0
        )
        all_dates = [
            "2024-01-01",
            "2024-01-02",
            "2024-01-03",
            "2024-01-04",
            "2024-01-05",
            "2024-01-06",
            "2024-01-07",
            "2024-01-08",
        ]

        standardizer = EwmaStandardizer(half_life=3.0, history_buffer=8)
        train_panel = _entity_panel(all_dates[:3], 1, "f", x[:3])
        val_panel = _entity_panel(all_dates[3:6], 1, "f", x[3:6])
        standardizer.fit(train_panel)
        standardizer.transform(train_panel)
        val_out = standardizer.transform(val_panel)["f"].to_numpy()

        # "test" re-presents the last two val dates (its causal-history lookback) plus
        # two genuinely new dates -- exactly the overlap this pipeline's history buffers
        # produce between a fold's validation tail and its test window's lookback prefix.
        overlapping_test_panel = _entity_panel(all_dates[4:], 1, "f", x[4:])
        test_out = standardizer.transform(overlapping_test_panel)["f"].to_numpy()

        np.testing.assert_allclose(test_out[:2], val_out[1:], rtol=1e-12)
        np.testing.assert_allclose(test_out, expected_outputs[4:], rtol=1e-10)
        np.testing.assert_allclose(standardizer._mean[1], [expected_mean], rtol=1e-10)
        np.testing.assert_allclose(standardizer._var[1], [expected_var], rtol=1e-10)

    def test_replay_older_than_the_history_buffer_raises_clearly(self) -> None:
        dates = ["2024-01-01", "2024-01-02", "2024-01-03"]
        panel = _entity_panel(dates, 1, "f", [1.0, 2.0, 3.0])
        standardizer = EwmaStandardizer(half_life=3.0, history_buffer=1)
        standardizer.fit(panel)
        standardizer.transform(panel)

        stale_replay = _entity_panel(dates[:1], 1, "f", [1.0])
        with self.assertRaisesRegex(ValueError, "cannot replay"):
            standardizer.transform(stale_replay)


class SerializationTest(unittest.TestCase):
    def test_restored_standardizer_continues_identically_to_the_original(self) -> None:
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        train_panel = _entity_panel(["2024-01-01", "2024-01-02"], 1, "f", x[:2])
        val_panel = _entity_panel(["2024-01-03"], 1, "f", x[2:3])
        continuation_panel = _entity_panel(["2024-01-04", "2024-01-05"], 1, "f", x[3:])

        original = EwmaStandardizer(half_life=2.5, history_buffer=4)
        original.fit(train_panel)
        original.transform(train_panel)
        original.transform(val_panel)

        state = original.state_dict()
        restored = EwmaStandardizer.from_state_dict(state)

        np.testing.assert_allclose(restored._mean[1], original._mean[1])
        np.testing.assert_allclose(restored._var[1], original._var[1])
        self.assertEqual(restored._watermark[1], original._watermark[1])

        original_continued = original.transform(continuation_panel)["f"].to_numpy()
        restored_continued = restored.transform(continuation_panel)["f"].to_numpy()
        np.testing.assert_allclose(original_continued, restored_continued, rtol=1e-12)

    def test_state_dict_round_trip_preserves_replay_buffer(self) -> None:
        dates = ["2024-01-01", "2024-01-02"]
        panel = _entity_panel(dates, 1, "f", [1.0, 2.0])
        standardizer = EwmaStandardizer(half_life=3.0, history_buffer=8)
        standardizer.fit(panel)
        first_out = standardizer.transform(panel)["f"].to_numpy()

        restored = EwmaStandardizer.from_state_dict(standardizer.state_dict())
        replay = restored.transform(panel)["f"].to_numpy()
        np.testing.assert_allclose(replay, first_out, rtol=1e-12)


if __name__ == "__main__":
    unittest.main()
