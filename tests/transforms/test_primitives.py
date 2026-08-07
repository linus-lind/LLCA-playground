import unittest

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from llca.mappers.features.mapper import build_features
from llca.transforms.primitives import (
    amihud_illiquidity,
    downside_deviation,
    high_proximity,
    net_ratio,
    positive_indicator,
    rolling_skewness,
    rolling_volatility,
    simple_change,
    simple_difference,
    simple_ratio,
)


def _log_returns(prices: np.ndarray) -> np.ndarray:
    return np.concatenate([[np.nan], np.diff(np.log(prices))])


class SimpleChangeTest(unittest.TestCase):
    def test_simple_change_returns_fractional_period_returns(self) -> None:
        result = simple_change(np.array([100.0, 110.0, 99.0]))
        np.testing.assert_allclose(result, [np.nan, 0.1, -0.1], equal_nan=True)

    def test_registered_feature_is_computed_independently_per_entity(self) -> None:
        index = pd.MultiIndex.from_product(
            [pd.date_range("2024-01-01", periods=3), ["A", "B"]],
            names=["date", "instrument"],
        )
        panel = pd.DataFrame(
            {"close": [100.0, 200.0, 110.0, 180.0, 99.0, 198.0]},
            index=index,
        )
        specs = OmegaConf.create(
            [{"name": "simple_change", "column": "close", "horizon": 1, "as": "return"}]
        )

        result = build_features(specs, panel)

        np.testing.assert_allclose(
            result["return"].to_numpy(),
            [np.nan, np.nan, 0.1, -0.1, -0.1, 0.1],
            equal_nan=True,
        )

    def test_log_change_can_count_non_missing_report_events(self) -> None:
        index = pd.MultiIndex.from_product(
            [pd.date_range("2024-01-01", periods=6), ["A"]],
            names=["date", "instrument"],
        )
        panel = pd.DataFrame({"quarterly": [1.0, np.nan, 2.0, np.nan, 4.0, 8.0]}, index=index)
        specs = OmegaConf.create(
            [
                {
                    "name": "log_change",
                    "column": "quarterly",
                    "horizon": 2,
                    "skip_missing": True,
                    "as": "growth",
                }
            ]
        )

        result = build_features(specs, panel)

        np.testing.assert_allclose(
            result["growth"].to_numpy(),
            [np.nan, np.nan, np.nan, np.nan, np.log(4.0), np.log(4.0)],
            equal_nan=True,
        )


class NetRatioTest(unittest.TestCase):
    def test_signed_numerator_over_denominator(self) -> None:
        result = net_ratio(
            [np.array([10.0, 20.0]), np.array([1.0, 2.0])],
            [np.array([4.0, 5.0])],
            np.array([2.0, 0.0]),
        )
        # (10 + 1 - 4) / 2 = 3.5; a zero denominator yields NaN like ratio.
        np.testing.assert_allclose(result, [3.5, np.nan], equal_nan=True)

    def test_missing_numerator_term_propagates(self) -> None:
        result = net_ratio(
            [np.array([10.0, np.nan])],
            [np.array([4.0, 1.0])],
            np.array([2.0, 2.0]),
        )
        np.testing.assert_allclose(result, [3.0, np.nan], equal_nan=True)

    def test_registered_feature_builds_gross_profitability(self) -> None:
        index = pd.MultiIndex.from_product(
            [pd.date_range("2024-01-01", periods=1), ["A", "B"]],
            names=["date", "instrument"],
        )
        panel = pd.DataFrame(
            {"sales": [100.0, 50.0], "cogs": [60.0, 40.0], "assets": [200.0, 100.0]},
            index=index,
        )
        specs = OmegaConf.create(
            [
                {
                    "name": "net_ratio",
                    "add": ["sales"],
                    "subtract": ["cogs"],
                    "denominator": "assets",
                    "as": "gross_profitability",
                }
            ]
        )

        result = build_features(specs, panel)

        np.testing.assert_allclose(result["gross_profitability"].to_numpy(), [0.2, 0.1])


class SimpleRatioTest(unittest.TestCase):
    def test_simple_ratio_returns_relative_difference(self) -> None:
        result = simple_ratio(np.array([110.0, 90.0, 5.0]), np.array([100.0, 100.0, 0.0]))
        np.testing.assert_allclose(result, [0.1, -0.1, np.nan], equal_nan=True)

    def test_registered_feature_computes_open_close_return(self) -> None:
        index = pd.MultiIndex.from_product(
            [pd.date_range("2024-01-01", periods=1), ["A", "B"]],
            names=["date", "instrument"],
        )
        panel = pd.DataFrame(
            {"close": [110.0, 90.0], "open": [100.0, 100.0]},
            index=index,
        )
        specs = OmegaConf.create(
            [{"name": "simple_ratio", "numerator": "close", "denominator": "open", "as": "oc"}]
        )

        result = build_features(specs, panel)

        np.testing.assert_allclose(result["oc"].to_numpy(), [0.1, -0.1])


class SimpleDifferenceTest(unittest.TestCase):
    def test_simple_difference_uses_previous_row_denominator(self) -> None:
        result = simple_difference(np.array([110.0, 99.0, 108.0]), np.array([90.0, 100.0, 90.0]))
        # row0 unavailable; 99/90 - 1 = 0.1; 108/100 - 1 = 0.08
        np.testing.assert_allclose(result, [np.nan, 0.1, 0.08], equal_nan=True)

    def test_registered_feature_is_computed_independently_per_entity(self) -> None:
        index = pd.MultiIndex.from_product(
            [pd.date_range("2024-01-01", periods=2), ["A", "B"]],
            names=["date", "instrument"],
        )
        panel = pd.DataFrame(
            {"open": [110.0, 220.0, 99.0, 231.0], "close": [90.0, 200.0, 100.0, 210.0]},
            index=index,
        )
        specs = OmegaConf.create(
            [{"name": "simple_difference", "current": "open", "previous": "close", "as": "co"}]
        )

        result = build_features(specs, panel)

        # A: 99/90 - 1 = 0.1; B: 231/200 - 1 = 0.155
        np.testing.assert_allclose(
            result["co"].to_numpy(),
            [np.nan, np.nan, 0.1, 0.155],
            equal_nan=True,
        )


class PositiveIndicatorTest(unittest.TestCase):
    def test_labels_direction_and_preserves_missing(self) -> None:
        result = positive_indicator(np.array([100.0, 110.0, 99.0, 99.0]))
        # row0 undefined; up -> 1; down -> 0; flat -> 0
        np.testing.assert_allclose(result, [np.nan, 1.0, 0.0, 0.0], equal_nan=True)

    def test_registered_feature_aligns_with_shifted_return(self) -> None:
        index = pd.MultiIndex.from_product(
            [pd.date_range("2024-01-01", periods=3), ["A"]],
            names=["date", "instrument"],
        )
        panel = pd.DataFrame({"close": [100.0, 110.0, 99.0]}, index=index)
        specs = OmegaConf.create(
            [
                {"name": "simple_change", "column": "close", "horizon": 1, "shift": -1, "as": "r"},
                {
                    "name": "positive_indicator",
                    "column": "close",
                    "horizon": 1,
                    "shift": -1,
                    "as": "d",
                },
            ]
        )

        result = build_features(specs, panel)

        # The label is exactly the sign of the identically shifted forward return.
        forward = result["r"].to_numpy()
        direction = result["d"].to_numpy()
        finite = np.isfinite(forward)
        np.testing.assert_array_equal(direction[finite], (forward[finite] > 0).astype(float))


class RollingVolatilityTest(unittest.TestCase):
    def test_matches_pandas_rolling_std_of_log_returns(self) -> None:
        prices = np.array([100.0, 101.0, 99.0, 102.0, 105.0, 103.0])
        expected = pd.Series(_log_returns(prices)).rolling(window=3, min_periods=3).std().to_numpy()

        result = rolling_volatility(prices, window=3)

        np.testing.assert_allclose(result, expected, equal_nan=True)

    def test_full_windows_precede_with_nan_until_history_accrues(self) -> None:
        # Three finite returns are needed, and the first row yields no return.
        prices = np.array([100.0, 110.0, 121.0, 133.1])
        result = rolling_volatility(prices, window=3)

        self.assertTrue(np.isnan(result[:3]).all())
        # Constant log returns imply zero dispersion in the first full window.
        np.testing.assert_allclose(result[3], 0.0, atol=1e-12)

    def test_registered_feature_isolates_entities(self) -> None:
        index = pd.MultiIndex.from_product(
            [pd.date_range("2024-01-01", periods=4), ["A", "B"]],
            names=["date", "instrument"],
        )
        panel = pd.DataFrame(
            {"close": [100.0, 50.0, 110.0, 55.0, 99.0, 60.0, 108.0, 54.0]},
            index=index,
        )
        specs = OmegaConf.create(
            [{"name": "rolling_volatility", "column": "close", "window": 2, "as": "vol"}]
        )

        result = build_features(specs, panel)["vol"].to_numpy()

        a_expected = rolling_volatility(np.array([100.0, 110.0, 99.0, 108.0]), window=2)
        b_expected = rolling_volatility(np.array([50.0, 55.0, 60.0, 54.0]), window=2)
        np.testing.assert_allclose(result[0::2], a_expected, equal_nan=True)
        np.testing.assert_allclose(result[1::2], b_expected, equal_nan=True)


class DownsideDeviationTest(unittest.TestCase):
    def test_penalizes_only_losses(self) -> None:
        prices = np.array([100.0, 90.0, 99.0])
        loss = np.log(0.9)  # the only negative return

        result = downside_deviation(prices, window=2, min_periods=1)

        np.testing.assert_allclose(
            result, [np.nan, abs(loss), np.sqrt(loss**2 / 2)], equal_nan=True
        )

    def test_all_positive_returns_have_zero_downside(self) -> None:
        prices = np.array([100.0, 101.0, 103.0, 106.0])
        result = downside_deviation(prices, window=3, min_periods=1)

        np.testing.assert_allclose(result[1:], 0.0, atol=1e-12)


class RollingSkewnessTest(unittest.TestCase):
    def test_matches_scipy_population_skew(self) -> None:
        from scipy.stats import skew as scipy_skew

        prices = np.array([100.0, 102.0, 101.0, 104.0, 103.0, 107.0, 105.0])
        returns = _log_returns(prices)
        window = 4

        result = rolling_skewness(prices, window=window)

        expected = np.full(len(returns), np.nan)
        for end in range(window - 1, len(returns)):
            window_returns = returns[end - window + 1 : end + 1]
            if np.isfinite(window_returns).all():
                expected[end] = scipy_skew(window_returns, bias=True)
        np.testing.assert_allclose(result, expected, equal_nan=True)


class HighProximityTest(unittest.TestCase):
    def test_value_relative_to_trailing_maximum(self) -> None:
        value = np.array([10.0, 12.0, 11.0])
        high = np.array([10.0, 13.0, 11.0])

        result = high_proximity(value, high, window=2)

        np.testing.assert_allclose(result, [np.nan, 12 / 13, 11 / 13], equal_nan=True)

    def test_registered_feature_isolates_entities(self) -> None:
        index = pd.MultiIndex.from_product(
            [pd.date_range("2024-01-01", periods=3), ["A", "B"]],
            names=["date", "instrument"],
        )
        panel = pd.DataFrame(
            {
                "close": [10.0, 20.0, 12.0, 18.0, 11.0, 22.0],
                "high": [10.0, 21.0, 13.0, 19.0, 11.0, 23.0],
            },
            index=index,
        )
        specs = OmegaConf.create(
            [
                {
                    "name": "high_proximity",
                    "value": "close",
                    "high": "high",
                    "window": 2,
                    "as": "prox",
                }
            ]
        )

        result = build_features(specs, panel)["prox"].to_numpy()

        np.testing.assert_allclose(
            result,
            [np.nan, np.nan, 12 / 13, 18 / 21, 11 / 13, 22 / 23],
            equal_nan=True,
        )


class AmihudIlliquidityTest(unittest.TestCase):
    def test_mean_price_impact_over_window(self) -> None:
        price = np.array([100.0, 110.0, 99.0])
        volume = np.array([10.0, 20.0, 5.0])
        impact_1 = abs(np.log(110 / 100)) / (110.0 * 20.0)
        impact_2 = abs(np.log(99 / 110)) / (99.0 * 5.0)

        result = amihud_illiquidity(price, volume, window=2, min_periods=1)

        np.testing.assert_allclose(
            result, [np.nan, impact_1, (impact_1 + impact_2) / 2], equal_nan=True
        )

    def test_zero_volume_days_are_skipped(self) -> None:
        price = np.array([100.0, 110.0, 99.0])
        volume = np.array([10.0, 0.0, 5.0])
        impact_2 = abs(np.log(99 / 110)) / (99.0 * 5.0)

        result = amihud_illiquidity(price, volume, window=2, min_periods=1)

        # The middle day contributes no observation, so each window averages only day 2.
        np.testing.assert_allclose(result, [np.nan, np.nan, impact_2], equal_nan=True)

    def test_log_returns_natural_log_of_the_raw_ratio(self) -> None:
        price = np.array([100.0, 110.0, 99.0])
        volume = np.array([10.0, 20.0, 5.0])

        raw = amihud_illiquidity(price, volume, window=2, min_periods=1)
        logged = amihud_illiquidity(price, volume, window=2, min_periods=1, log=True)

        finite = np.isfinite(raw)
        np.testing.assert_allclose(logged[finite], np.log(raw[finite]))
        np.testing.assert_array_equal(np.isnan(logged), np.isnan(raw))


if __name__ == "__main__":
    unittest.main()
