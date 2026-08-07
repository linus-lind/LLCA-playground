"""Fit/predict/serialize contract for the single-asset classifier baselines.

Exercises the shared ``SingleAssetClassifierEstimator`` lifecycle through both the logistic
(standardized) and random-forest (unscaled) backends: target-entity restriction, direction
labelling, the signed ``2*P(up)-1`` portfolio score, the single-class fallback, and an exact
serialization round-trip -- all without the Hydra data pipeline.
"""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from llca.models.estimators.logistic_net import LogisticNetEstimator
from llca.models.estimators.random_forest import RandomForestClassifierEstimator
from llca.models.estimators.single_asset_tabular import SingleAssetClassifierEstimator
from llca.training.modules.sklearn_config import SklearnTrainingConfig

_TARGET = 1


def _masked(values: pd.DataFrame) -> object:
    from llca.data.modules.masked_panel import MaskedPanel

    index = values.index
    return MaskedPanel(
        values=values,
        observed=pd.DataFrame(True, index=index, columns=values.columns),
        age=pd.DataFrame(0.0, index=index, columns=values.columns),
        segment=pd.Series(index.get_level_values("entity"), index=index),
    )


def _panels(direction: pd.Series | None = None) -> dict[str, object]:
    dates = pd.bdate_range("2021-01-01", periods=40)
    index = pd.MultiIndex.from_product([dates, [1, 2, 3, 4]], names=["date", "entity"])
    rng = np.random.default_rng(1)
    features = pd.DataFrame(rng.standard_normal((len(index), 2)), index=index, columns=["f0", "f1"])
    fwd_return = pd.Series(rng.standard_normal(len(index)) * 0.02, index=index, name="fwd_return")
    fwd_direction = (fwd_return > 0).astype(float) if direction is None else direction
    loss = pd.DataFrame({"fwd_return": fwd_return, "fwd_direction": fwd_direction})
    return {"daily_values": _masked(features), "loss": _masked(loss)}


def _logistic_config() -> DictConfig:
    return OmegaConf.create(
        {
            "name": "elastic-net",
            "target": {"entity_id": _TARGET},
            "inputs": {"features": "daily_values"},
            "supervision": {"dataset": "loss", "column": "fwd_return"},
            "classification": {"dataset": "loss", "column": "fwd_direction"},
            "l1_ratio": 0.5,
            "C": 1.0,
            "max_iter": 1000,
            "tol": 1e-4,
        }
    )


def _forest_config() -> DictConfig:
    return OmegaConf.create(
        {
            "name": "rf",
            "target": {"entity_id": _TARGET},
            "inputs": {"features": "daily_values"},
            "supervision": {"dataset": "loss", "column": "fwd_return"},
            "classification": {"dataset": "loss", "column": "fwd_direction"},
            "n_estimators": 40,
            "max_depth": 4,
            "min_samples_leaf": 3,
            "max_features": "sqrt",
            "bootstrap": True,
        }
    )


class SingleAssetClassifierContractTest(unittest.TestCase):
    def _backends(self) -> list[tuple[type[SingleAssetClassifierEstimator], DictConfig]]:
        return [
            (LogisticNetEstimator, _logistic_config()),
            (RandomForestClassifierEstimator, _forest_config()),
        ]

    def test_directional_score_and_serialize_roundtrip(self) -> None:
        for cls, config in self._backends():
            with self.subTest(model=config.name):
                estimator = cls(config)
                panels = _panels()
                estimator.fit(panels, training=SklearnTrainingConfig(seed=7, n_jobs=1))

                prediction = estimator.predict(panels)
                values = prediction.values
                self.assertEqual(prediction.kind, "portfolio")
                # Only the target entity's rows are scored.
                self.assertEqual(len(values), 40)
                self.assertTrue((values.index.get_level_values("entity") == _TARGET).all())
                array = values.to_numpy(dtype=float)
                self.assertTrue(np.isfinite(array).all())
                self.assertTrue(((array >= -1.0) & (array <= 1.0)).all())

                with TemporaryDirectory() as directory:
                    bundle = Path(directory) / "model.pkl"
                    estimator._save(bundle)
                    restored = cls.load(bundle)
                pd.testing.assert_series_equal(values, restored.predict(panels).values)

    def test_single_class_fold_falls_back_to_flat(self) -> None:
        dates = pd.bdate_range("2021-01-01", periods=40)
        index = pd.MultiIndex.from_product([dates, [1, 2, 3, 4]], names=["date", "entity"])
        all_up = pd.Series(1.0, index=index, name="fwd_direction")
        for cls, config in self._backends():
            with self.subTest(model=config.name):
                estimator = cls(config)
                estimator.fit(
                    _panels(direction=all_up), training=SklearnTrainingConfig(seed=7, n_jobs=1)
                )
                prediction = estimator.predict(_panels(direction=all_up))
                # A single-class training fold yields P(up)=0.5 -> a flat (zero) position.
                np.testing.assert_allclose(prediction.values.to_numpy(dtype=float), 0.0)

    def test_unfitted_predict_is_rejected(self) -> None:
        estimator = LogisticNetEstimator(_logistic_config())
        with self.assertRaisesRegex(RuntimeError, "not fitted"):
            estimator.predict(_panels())


if __name__ == "__main__":
    unittest.main()
