"""Feature transform configuration validation, including trailing-window contracts."""

from __future__ import annotations

import unittest
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from llca.mappers import validate_config
from llca.mappers.features.mapper import feature_registry
from llca.mappers.modules.config_validation_error import ConfigValidationError

_CONFIG_DIR = str(
    (Path(__file__).resolve().parents[2] / "hydra" / "configs" / "training").resolve()
)


def _spec(**fields: object) -> DictConfig:
    spec = OmegaConf.create(fields)
    assert isinstance(spec, DictConfig)
    return spec


class ShippedFeatureConfigTest(unittest.TestCase):
    def test_default_experiment_features_validate(self) -> None:
        # Locks in that every shipped daily_values feature, including the rolling
        # transforms, passes package validation as composed by a real experiment.
        # Only feature-scoped errors are asserted on, so unrelated composition gaps in
        # a bare compose do not mask (or manufacture) a feature regression.
        with initialize_config_dir(config_dir=_CONFIG_DIR, version_base=None):
            cfg = compose(config_name="train", overrides=["experiment=fmg-ctt"])
        try:
            validate_config(cfg)
        except ConfigValidationError as exc:
            errors = list(exc.errors)
        else:
            errors = []
        feature_errors = [message for message in errors if message.startswith("features")]
        self.assertEqual(feature_errors, [])


class RollingWindowValidatorTest(unittest.TestCase):
    def test_valid_spec_reports_no_errors(self) -> None:
        errors = feature_registry.validate(
            "rolling_volatility",
            _spec(name="rolling_volatility", column="close", window=21, min_periods=10),
        )
        self.assertEqual(errors, [])

    def test_window_is_required(self) -> None:
        errors = feature_registry.validate(
            "rolling_volatility", _spec(name="rolling_volatility", column="close")
        )
        self.assertTrue(any("window is required" in message for message in errors), errors)

    def test_non_positive_window_is_rejected(self) -> None:
        errors = feature_registry.validate(
            "amihud_illiquidity",
            _spec(name="amihud_illiquidity", price="close", volume="volume", window=0),
        )
        self.assertTrue(any("window must be positive" in message for message in errors), errors)

    def test_min_periods_cannot_exceed_window(self) -> None:
        errors = feature_registry.validate(
            "high_proximity",
            _spec(name="high_proximity", value="close", high="high", window=21, min_periods=30),
        )
        self.assertTrue(
            any("min_periods must be <= window" in message for message in errors), errors
        )


if __name__ == "__main__":
    unittest.main()
