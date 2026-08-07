"""Score contracts for the non-learning cross-sectional baselines."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from llca.data.modules.masked_panel import MaskedPanel
from llca.models.estimators.baseline import EqualWeightEstimator, InverseVolatilityEstimator
from llca.training.modules.sklearn_config import SklearnTrainingConfig

_FLOOR = 1e-6


def _masked(values: pd.DataFrame) -> MaskedPanel:
    index = values.index
    return MaskedPanel(
        values=values,
        observed=pd.DataFrame(True, index=index, columns=values.columns),
        age=pd.DataFrame(0.0, index=index, columns=values.columns),
        segment=pd.Series(index.get_level_values("entity"), index=index),
    )


def _panels() -> dict[str, MaskedPanel]:
    dates = pd.bdate_range("2021-01-01", periods=3)
    index = pd.MultiIndex.from_product([dates, [1, 2, 3]], names=["date", "entity"])
    # Volatility carries one missing estimate (excluded) and one below-floor value (capped).
    volatility = [
        0.02,
        0.04,
        np.nan,
        0.01,
        1e-9,
        0.05,
        0.03,
        0.03,
        0.03,
    ]
    daily = pd.DataFrame({"f": np.ones(len(index)), "realized_vol_3m": volatility}, index=index)
    loss = pd.DataFrame({"fwd_return": np.zeros(len(index))}, index=index)
    return {"daily_values": _masked(daily), "loss": _masked(loss)}


def _inverse_vol_config() -> DictConfig:
    return OmegaConf.create(
        {
            "name": "inverse-volatility",
            "inputs": {"features": "daily_values"},
            "supervision": {"dataset": "loss", "column": "fwd_return"},
            "volatility": {
                "dataset": "daily_values",
                "column": "realized_vol_3m",
                "floor": _FLOOR,
            },
        }
    )


def _equal_weight_config() -> DictConfig:
    return OmegaConf.create(
        {
            "name": "equal-weight",
            "inputs": {"features": "daily_values"},
            "supervision": {"dataset": "loss", "column": "fwd_return"},
        }
    )


class EqualWeightTest(unittest.TestCase):
    def test_scores_are_uniformly_one(self) -> None:
        estimator = EqualWeightEstimator(_equal_weight_config())
        panels = _panels()
        estimator.fit(panels, training=SklearnTrainingConfig(seed=0, n_jobs=1))
        prediction = estimator.predict(panels)
        self.assertEqual(prediction.kind, "portfolio")
        np.testing.assert_allclose(prediction.values.to_numpy(dtype=float), 1.0)


class InverseVolatilityTest(unittest.TestCase):
    def test_reciprocal_scores_exclude_missing_and_floor_extremes(self) -> None:
        estimator = InverseVolatilityEstimator(_inverse_vol_config())
        panels = _panels()
        estimator.fit(panels, training=SklearnTrainingConfig(seed=0, n_jobs=1))
        prediction = estimator.predict(panels)
        values = prediction.values

        # The missing-volatility row is excluded from the scored universe.
        self.assertEqual(len(values), 8)
        dates = sorted({d for d, _ in panels["daily_values"].values.index})
        self.assertNotIn((dates[0], 3), values.index)

        # A finite positive volatility scores 1/vol; a below-floor value is capped by the floor.
        self.assertAlmostEqual(values.loc[(dates[0], 1)], 1.0 / 0.02)
        self.assertAlmostEqual(values.loc[(dates[1], 2)], 1.0 / _FLOOR)
        self.assertTrue(np.isfinite(values.to_numpy(dtype=float)).all())


if __name__ == "__main__":
    unittest.main()
