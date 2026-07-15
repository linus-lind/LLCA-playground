import unittest

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from llca.mappers.features.mapper import build_features
from llca.transforms.primitives import simple_change


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


if __name__ == "__main__":
    unittest.main()
