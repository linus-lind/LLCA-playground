import unittest

from omegaconf import OmegaConf

from llca.mappers.training.config_validator import _validate_training
from llca.mappers.training.mapper import build_training
from llca.training.modules.training_config import AdamWConfig


def _root_config() -> object:
    return OmegaConf.create(
        {
            "training": {
                "name": "torch",
                "seed": 42,
                "deterministic": True,
                "epochs": 2,
                "batch_size": 4,
                "grad_clip": 1.0,
                "device": "cpu",
                "precision": "fp32",
                "gradient_checkpointing": False,
                "optimizer": {
                    "name": "adamw",
                    "learning_rate": 0.001,
                    "weight_decay": 0.0001,
                    "fused": False,
                },
                "early_stopping": {"patience": 1, "min_delta": 0.0},
                "diagnostics": {
                    "interval": 1,
                    "component_gradient_norms": True,
                    "parameter_update_norms": True,
                },
            }
        }
    )


class TrainingMapperTest(unittest.TestCase):
    def test_maps_and_validates_adamw_and_diagnostics(self) -> None:
        root = _root_config()
        self.assertEqual(_validate_training(root), [])  # type: ignore[arg-type]
        training = build_training(root.training)  # type: ignore[attr-defined]
        self.assertIsInstance(training.optimizer, AdamWConfig)
        parameters = training.tracking_parameters()
        self.assertEqual(parameters["training.optimizer.name"], "adamw")
        self.assertEqual(parameters["training.optimizer.weight_decay"], 0.0001)
        self.assertEqual(parameters["training.batch_size"], 4)

    def test_rejects_zero_diagnostic_interval(self) -> None:
        root = _root_config()
        root.training.diagnostics.interval = 0  # type: ignore[attr-defined]
        errors = _validate_training(root)  # type: ignore[arg-type]
        self.assertTrue(any("diagnostics.interval" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
