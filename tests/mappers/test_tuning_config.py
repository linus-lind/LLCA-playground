"""Hydra composition and validation of the hyperparameter-selection group."""

from __future__ import annotations

import unittest
from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig

from llca.mappers import validate_config
from llca.mappers.model.tuning import build_hyperparameter_selection
from llca.mappers.modules.config_validation_error import ConfigValidationError

_CONFIG_DIR = str(
    (Path(__file__).resolve().parents[2] / "hydra" / "configs" / "training").resolve()
)


def _compose(overrides: list[str]) -> DictConfig:
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=_CONFIG_DIR, version_base=None):
        return compose(config_name="train", overrides=overrides)


def _selection_errors(overrides: list[str]) -> list[str]:
    try:
        validate_config(_compose(overrides))
    except ConfigValidationError as exc:
        return [m for m in exc.errors if "hyperparameter_selection" in m or "search_space" in m]
    return []


class TuningCompositionTest(unittest.TestCase):
    def test_elastic_net_composes_enabled_selection(self) -> None:
        cfg = _compose(["experiment=elastic-net"])
        selection = build_hyperparameter_selection(cfg.hyperparameter_selection, cfg.model)
        assert selection is not None
        self.assertTrue(selection.enabled)
        self.assertEqual(set(selection.search_space.names()), {"C", "l1_ratio"})
        self.assertEqual(selection.baseline, {"C": 1.0, "l1_ratio": 0.5})
        self.assertEqual(selection.inner_cv.purge, 1)
        self.assertEqual(selection.search.method, "grid")
        # Two exhaustive axes: 13 log-spaced C values x 7 l1_ratio values = 91 combinations.
        self.assertEqual(selection.search_space.grid_size(), 91)

    def test_random_forest_composes_enabled_selection(self) -> None:
        cfg = _compose(["experiment=rf"])
        selection = build_hyperparameter_selection(cfg.hyperparameter_selection, cfg.model)
        assert selection is not None
        self.assertEqual(
            set(selection.search_space.names()),
            {"max_depth", "min_samples_leaf", "min_samples_split", "max_features"},
        )
        self.assertEqual(selection.baseline["min_samples_leaf"], 50)
        self.assertEqual(selection.baseline["max_features"], "sqrt")
        self.assertEqual(selection.search.method, "grid")
        # Four exhaustive axes: 4 * 4 * 2 * 2 = 64 candidate combinations.
        self.assertEqual(selection.search_space.grid_size(), 64)

    def test_off_preset_disables_selection(self) -> None:
        cfg = _compose(["experiment=elastic-net", "hyperparameter_selection=off"])
        selection = build_hyperparameter_selection(cfg.hyperparameter_selection, cfg.model)
        assert selection is not None
        self.assertFalse(selection.enabled)
        # The baseline is still available, so a disabled model fits its default hyperparameters.
        self.assertEqual(selection.baseline, {"C": 1.0, "l1_ratio": 0.5})


class TuningValidationTest(unittest.TestCase):
    def test_shipped_experiments_have_no_selection_errors(self) -> None:
        self.assertEqual(_selection_errors(["experiment=elastic-net"]), [])
        self.assertEqual(_selection_errors(["experiment=rf"]), [])

    def test_selection_rejected_for_non_tunable_model(self) -> None:
        errors = _selection_errors(
            ["experiment=equal-weight", "hyperparameter_selection=walk-forward"]
        )
        self.assertTrue(
            any("does not support hyperparameter selection" in m for m in errors), errors
        )

    def test_unknown_search_method_is_rejected(self) -> None:
        errors = _selection_errors(
            ["experiment=elastic-net", "hyperparameter_selection.search.method=bayesian"]
        )
        self.assertTrue(any("search.method must be one of" in m for m in errors), errors)

    def test_purge_below_label_horizon_is_rejected(self) -> None:
        errors = _selection_errors(
            ["experiment=elastic-net", "hyperparameter_selection.cv.purge=0"]
        )
        self.assertTrue(
            any("must be >= the supervision label horizon" in m for m in errors), errors
        )

    def test_non_positive_inner_train_size_is_rejected(self) -> None:
        errors = _selection_errors(["experiment=rf", "hyperparameter_selection.cv.train_size=0"])
        self.assertTrue(any("cv.train_size must be positive" in m for m in errors), errors)

    def test_enabled_selection_without_a_loss_is_rejected(self) -> None:
        errors = _selection_errors(["experiment=elastic-net", "loss=none"])
        self.assertTrue(any("requires a loss to score candidates" in m for m in errors), errors)


if __name__ == "__main__":
    unittest.main()
