"""Cross-cutting model x objective x engine compatibility gate (pre-I/O validation)."""

from __future__ import annotations

import unittest
from pathlib import Path

from hydra import compose, initialize_config_dir

from llca.mappers import validate_config
from llca.mappers.modules.config_validation_error import ConfigValidationError

_CONFIG_DIR = str(
    (Path(__file__).resolve().parents[2] / "hydra" / "configs" / "training").resolve()
)


def _compose(overrides: list[str]):
    with initialize_config_dir(config_dir=_CONFIG_DIR, version_base=None):
        return compose(config_name="train", overrides=overrides)


class ModelCompatibilityValidatorTest(unittest.TestCase):
    def _errors(self, overrides: list[str]) -> list[str]:
        try:
            validate_config(_compose(overrides))
        except ConfigValidationError as exc:
            return list(exc.errors)
        return []

    def test_torch_model_rejects_null_objective_before_io(self) -> None:
        # Regression: loss=none previously passed validation and only crashed at
        # build_model, after the full dataset had already been loaded.
        errors = self._errors(["experiment=fmg-ctct-2", "loss=none"])
        self.assertTrue(
            any("requires a differentiable objective" in message for message in errors),
            errors,
        )

    def test_sklearn_baselines_tolerate_null_objective(self) -> None:
        # The sklearn engine does not require a differentiable objective the way torch does.
        # (The rf experiment enables hyperparameter selection, which independently needs a loss
        # to score candidates, so it is disabled here to isolate the engine-tolerance property.)
        self.assertEqual(
            self._errors(["experiment=rf", "loss=none", "hyperparameter_selection=off"]), []
        )
        self.assertEqual(self._errors(["experiment=equal-weight", "loss=none"]), [])

    def test_unsupported_objective_kind_is_rejected(self) -> None:
        errors = self._errors(["experiment=fmg-ctt", "loss=mse"])
        self.assertTrue(
            any("does not support objective kind" in message for message in errors), errors
        )

    def test_unsupported_training_engine_is_rejected(self) -> None:
        errors = self._errors(["experiment=rf", "training=torch"])
        self.assertTrue(
            any("does not support training engine" in message for message in errors), errors
        )
        errors = self._errors(["experiment=fmg-ctt", "training=sklearn"])
        self.assertTrue(
            any("does not support training engine" in message for message in errors), errors
        )

    def test_shipped_presets_validate(self) -> None:
        for preset in (
            "fmg-ctct-2",
            "fmg-ctct-1",
            "fmg-ctt",
            "fmg-clstm",
            "rf",
            "elastic-net",
            "equal-weight",
            "inverse-volatility",
        ):
            self.assertEqual(self._errors([f"experiment={preset}"]), [], preset)


if __name__ == "__main__":
    unittest.main()
