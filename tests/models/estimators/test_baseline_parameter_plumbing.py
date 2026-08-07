"""Verify RF/EN configuration flows Hydra -> mapper -> estimator -> scikit-learn constructor.

Covers a representative parameter from each category (statistical, ensemble-size, structural,
runtime) and confirms that overriding a value in Hydra actually changes the constructed sklearn
estimator, so no configured knob is silently ignored.
"""

from __future__ import annotations

import unittest
from pathlib import Path
from typing import cast

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig

import llca.mappers.loss  # noqa: F401
import llca.mappers.model  # noqa: F401
from llca.models.estimators.logistic_net import LogisticNetEstimator
from llca.models.estimators.random_forest import RandomForestClassifierEstimator
from llca.training.modules.sklearn_config import SklearnTrainingConfig

_CONFIG_DIR = str(
    (Path(__file__).resolve().parents[3] / "hydra" / "configs" / "training").resolve()
)
_TRAINING = SklearnTrainingConfig(seed=7, n_jobs=1)


def _model_cfg(experiment: str, overrides: list[str]) -> DictConfig:
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=_CONFIG_DIR, version_base=None):
        cfg = compose(config_name="train", overrides=[f"experiment={experiment}", *overrides])
        return cast(DictConfig, cfg.model)


class RandomForestPlumbingTest(unittest.TestCase):
    def test_shipped_baseline_reaches_sklearn(self) -> None:
        estimator = RandomForestClassifierEstimator(
            _model_cfg("rf", []), prediction_kind="portfolio"
        )
        params = estimator._construct(_TRAINING).get_params()
        self.assertEqual(params["n_estimators"], 500)  # ensemble size
        self.assertEqual(params["criterion"], "gini")  # statistical
        self.assertEqual(params["max_depth"], 8)
        self.assertEqual(params["min_samples_split"], 20)
        self.assertEqual(params["min_samples_leaf"], 50)
        self.assertEqual(params["max_features"], "sqrt")
        self.assertIsNone(params["class_weight"])
        self.assertEqual(params["ccp_alpha"], 0.0)
        self.assertIsNone(params["max_samples"])
        self.assertTrue(params["bootstrap"])  # structural
        self.assertFalse(params["oob_score"])
        self.assertEqual(params["random_state"], 7)  # runtime, from the training policy
        self.assertEqual(params["n_jobs"], 1)

    def test_hydra_overrides_change_the_constructed_estimator(self) -> None:
        cfg = _model_cfg(
            "rf",
            [
                "model.max_depth=13",
                "model.criterion=entropy",
                "model.min_samples_split=7",
                "model.ccp_alpha=0.01",
                "model.max_samples=0.7",
                "model.class_weight=balanced_subsample",
                "model.max_features=log2",
            ],
        )
        params = (
            RandomForestClassifierEstimator(cfg, prediction_kind="portfolio")
            ._construct(_TRAINING)
            .get_params()
        )
        self.assertEqual(params["max_depth"], 13)
        self.assertEqual(params["criterion"], "entropy")
        self.assertEqual(params["min_samples_split"], 7)
        self.assertEqual(params["ccp_alpha"], 0.01)
        self.assertEqual(params["max_samples"], 0.7)
        self.assertEqual(params["class_weight"], "balanced_subsample")
        self.assertEqual(params["max_features"], "log2")

    def test_search_candidate_override_reaches_sklearn(self) -> None:
        estimator = RandomForestClassifierEstimator(
            _model_cfg("rf", []),
            prediction_kind="portfolio",
            hyperparameters={"min_samples_leaf": 5, "max_features": 0.5, "max_depth": 20},
        )
        params = estimator._construct(_TRAINING).get_params()
        self.assertEqual(params["min_samples_leaf"], 5)
        self.assertEqual(params["max_features"], 0.5)
        self.assertEqual(params["max_depth"], 20)


class LogisticNetPlumbingTest(unittest.TestCase):
    def test_shipped_baseline_reaches_sklearn(self) -> None:
        estimator = LogisticNetEstimator(_model_cfg("elastic-net", []), prediction_kind="portfolio")
        params = estimator._construct(_TRAINING).get_params()
        self.assertEqual(params["solver"], "saga")  # required for a mixed L1/L2 penalty
        self.assertEqual(params["l1_ratio"], 0.5)  # honored directly (no penalty argument)
        self.assertEqual(params["C"], 1.0)
        self.assertIsNone(params["class_weight"])
        self.assertTrue(params["fit_intercept"])
        self.assertEqual(params["max_iter"], 5000)
        self.assertEqual(params["tol"], 0.0001)
        self.assertEqual(params["random_state"], 7)

    def test_hydra_overrides_change_the_constructed_estimator(self) -> None:
        cfg = _model_cfg(
            "elastic-net",
            [
                "model.C=0.01",
                "model.l1_ratio=0.9",
                "model.class_weight=balanced",
                "model.fit_intercept=false",
            ],
        )
        params = (
            LogisticNetEstimator(cfg, prediction_kind="portfolio")
            ._construct(_TRAINING)
            .get_params()
        )
        self.assertEqual(params["C"], 0.01)
        self.assertEqual(params["l1_ratio"], 0.9)
        self.assertEqual(params["class_weight"], "balanced")
        self.assertFalse(params["fit_intercept"])


if __name__ == "__main__":
    unittest.main()
