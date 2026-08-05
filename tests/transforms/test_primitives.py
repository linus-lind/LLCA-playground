import unittest

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from llca.mappers.features.mapper import build_features
from llca.transforms.primitives import net_ratio, simple_change


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


if __name__ == "__main__":
    unittest.main()
