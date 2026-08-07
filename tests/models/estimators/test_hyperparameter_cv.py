"""End-to-end inner-CV hyperparameter selection through the real EN/RF estimators."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from llca.loss.portfolio import PortfolioLoss
from llca.models.estimators.logistic_net import LogisticNetEstimator
from llca.models.estimators.random_forest import RandomForestClassifierEstimator
from llca.models.estimators.single_asset_tabular import SingleAssetClassifierEstimator
from llca.training.modules.sklearn_config import SklearnTrainingConfig
from llca.training.tuning import (
    ChoiceDimension,
    HyperparameterSelection,
    InnerCvSettings,
    SearchSettings,
    SearchSpace,
)

_TARGET = 1
_TRAINING = SklearnTrainingConfig(seed=0, n_jobs=1)


def _portfolio_loss() -> PortfolioLoss:
    return PortfolioLoss(
        leverage=1.0,
        risk_aversion=1.0,
        concentration_aversion=0.0,
        execution_fee=0.0001,
        bid_ask_spread=0.0003,
        slippage=0.0002,
        borrow_cost=0.00002,
        normalization="gross",
        return_type="simple",
        common_score_aversion=0.0,
        net_exposure_aversion=0.0,
        net_exposure_tolerance=1.0,
    )


def _masked(values: pd.DataFrame) -> object:
    from llca.data.modules.masked_panel import MaskedPanel

    index = values.index
    return MaskedPanel(
        values=values,
        observed=pd.DataFrame(True, index=index, columns=values.columns),
        age=pd.DataFrame(0.0, index=index, columns=values.columns),
        segment=pd.Series(index.get_level_values("entity"), index=index),
    )


def _panels(periods: int = 220) -> dict[str, object]:
    dates = pd.bdate_range("2019-01-01", periods=periods)
    index = pd.MultiIndex.from_product([dates, [_TARGET]], names=["date", "entity"])
    rng = np.random.default_rng(0)
    features = pd.DataFrame(
        rng.standard_normal((len(index), 3)), index=index, columns=["f0", "f1", "f2"]
    )
    # A weak dependence of direction on a feature, so folds carry both classes.
    signal = 0.4 * features["f0"].to_numpy() + rng.standard_normal(len(index))
    fwd_return = pd.Series(signal * 0.01, index=index, name="fwd_return")
    fwd_direction = (fwd_return > 0).astype(float)
    loss = pd.DataFrame({"fwd_return": fwd_return, "fwd_direction": fwd_direction})
    return {"daily_values": _masked(features), "loss": _masked(loss)}


def _config(name: str, **hyperparameters: object) -> DictConfig:
    base = {
        "name": name,
        "target": {"entity_id": _TARGET},
        "inputs": {"features": "daily_values"},
        "supervision": {"dataset": "loss", "column": "fwd_return"},
        "classification": {"dataset": "loss", "column": "fwd_direction"},
    }
    return OmegaConf.create({**base, **hyperparameters})


def _inner_cv() -> InnerCvSettings:
    return InnerCvSettings(
        train_size=60, val_size=20, step_size=20, purge=1, lookback=0, min_folds=2
    )


def _logistic_selection(enabled: bool) -> HyperparameterSelection:
    return HyperparameterSelection(
        enabled=enabled,
        inner_cv=_inner_cv(),
        search=SearchSettings("grid", 0, 0),
        search_space=SearchSpace(
            (ChoiceDimension("C", (0.01, 1.0, 100.0)), ChoiceDimension("l1_ratio", (0.5,)))
        ),
        baseline={"C": 1.0, "l1_ratio": 0.5},
        standard_error_margin=1.0,
    )


def _forest_selection(enabled: bool) -> HyperparameterSelection:
    return HyperparameterSelection(
        enabled=enabled,
        inner_cv=_inner_cv(),
        search=SearchSettings("random", 4, 3),
        search_space=SearchSpace(
            (ChoiceDimension("min_samples_leaf", (5, 20, 50)), ChoiceDimension("max_depth", (2, 4)))
        ),
        baseline={"min_samples_leaf": 20, "max_depth": 4},
        standard_error_margin=1.0,
    )


class LogisticNetCvTest(unittest.TestCase):
    def _estimator(self, enabled: bool) -> LogisticNetEstimator:
        return LogisticNetEstimator(
            _config("elastic-net", C=1.0, l1_ratio=0.5, max_iter=500, tol=1e-4),
            cv_objective=_portfolio_loss(),
            selection=_logistic_selection(enabled),
        )

    def test_enabled_selection_runs_and_final_model_uses_selected_parameters(self) -> None:
        estimator = self._estimator(enabled=True)
        estimator.fit(_panels(), training=_TRAINING)

        result = estimator._selection_result
        assert result is not None
        self.assertGreaterEqual(result.fold_count, 2)
        self.assertEqual(result.evaluated_candidates, 3)  # 3 x 1 grid
        self.assertIn("C", estimator._hyperparameters)
        # The refitted model carries exactly the selected hyperparameters.
        self.assertEqual(estimator._model.get_params()["C"], estimator._hyperparameters["C"])
        # It is fit on the entire training window, not an inner fold (220 target rows).
        self.assertEqual(int(estimator._model.n_iter_.shape[0]), 1)

        prediction = estimator.predict(_panels())
        self.assertEqual(prediction.kind, "portfolio")
        self.assertEqual(len(prediction.values), 220)

    def test_disabled_selection_uses_baseline(self) -> None:
        estimator = self._estimator(enabled=False)
        estimator.fit(_panels(), training=_TRAINING)
        self.assertIsNone(estimator._selection_result)
        self.assertEqual(estimator._hyperparameters, {"C": 1.0, "l1_ratio": 0.5})
        self.assertEqual(estimator._model.get_params()["C"], 1.0)

    def test_serialization_preserves_predictions(self) -> None:
        estimator = self._estimator(enabled=True)
        panels = _panels()
        estimator.fit(panels, training=_TRAINING)
        prediction = estimator.predict(panels)
        with TemporaryDirectory() as directory:
            bundle = Path(directory) / "model.pkl"
            estimator._save(bundle)
            restored = LogisticNetEstimator.load(bundle)
        pd.testing.assert_series_equal(prediction.values, restored.predict(panels).values)
        self.assertEqual(restored._hyperparameters, estimator._hyperparameters)


class RandomForestCvTest(unittest.TestCase):
    def test_grid_search_evaluates_every_candidate_and_refits_on_full_train(self) -> None:
        selection = HyperparameterSelection(
            enabled=True,
            inner_cv=_inner_cv(),
            search=SearchSettings("grid", 0, 0),
            search_space=SearchSpace(
                (ChoiceDimension("min_samples_leaf", (5, 20)), ChoiceDimension("max_depth", (2, 4)))
            ),
            baseline={"min_samples_leaf": 20, "max_depth": 4},
            standard_error_margin=1.0,
        )
        estimator = RandomForestClassifierEstimator(
            _config(
                "rf",
                n_estimators=25,
                max_depth=4,
                min_samples_leaf=20,
                max_features="sqrt",
                bootstrap=True,
            ),
            cv_objective=_portfolio_loss(),
            selection=selection,
        )
        estimator.fit(_panels(), training=_TRAINING)

        result = estimator._selection_result
        assert result is not None
        self.assertEqual(result.search_method, "grid")
        self.assertEqual(result.evaluated_candidates, 4)  # 2 x 2 exhaustive grid
        self.assertGreaterEqual(result.fold_count, 2)
        params = estimator._model.get_params()
        self.assertEqual(params["n_estimators"], 25)  # fixed, never searched
        self.assertEqual(params["min_samples_leaf"], estimator._hyperparameters["min_samples_leaf"])
        self.assertEqual(len(estimator.predict(_panels()).values), 220)

    def test_random_search_selection_and_fixed_parameters(self) -> None:
        estimator = RandomForestClassifierEstimator(
            _config(
                "rf",
                n_estimators=25,
                max_depth=4,
                min_samples_leaf=20,
                max_features="sqrt",
                bootstrap=True,
            ),
            cv_objective=_portfolio_loss(),
            selection=_forest_selection(enabled=True),
        )
        estimator.fit(_panels(), training=_TRAINING)

        result = estimator._selection_result
        assert result is not None
        self.assertLessEqual(result.evaluated_candidates, 4)  # random, <= n_trials
        params = estimator._model.get_params()
        # Fixed parameters are never searched.
        self.assertEqual(params["n_estimators"], 25)
        self.assertTrue(params["bootstrap"])
        # Tuned parameters match the selection.
        self.assertEqual(params["min_samples_leaf"], estimator._hyperparameters["min_samples_leaf"])
        self.assertEqual(len(estimator.predict(_panels()).values), 220)


class CloneIsolationTest(unittest.TestCase):
    def test_candidate_clone_is_independent_and_non_recursive(self) -> None:
        estimator = LogisticNetEstimator(
            _config("elastic-net", C=1.0, l1_ratio=0.5, max_iter=500, tol=1e-4),
            cv_objective=_portfolio_loss(),
            selection=_logistic_selection(enabled=True),
        )
        clone = estimator._candidate_factory({"C": 0.01, "l1_ratio": 0.5})
        self.assertIsInstance(clone, SingleAssetClassifierEstimator)
        self.assertIsNone(clone._selection)  # will not recurse into selection
        self.assertEqual(dict(clone._hyperparameters), {"C": 0.01, "l1_ratio": 0.5})
        self.assertIsNone(clone._cv_objective)


if __name__ == "__main__":
    unittest.main()
