import unittest

import torch
from omegaconf import OmegaConf

from llca.mappers.loss import build_loss
from llca.training.modules.training_diagnostics import objective_diagnostics


class TargetLossDiagnosticsTest(unittest.TestCase):
    def test_mse_exposes_regression_metrics_without_portfolio_components(self) -> None:
        objective = build_loss(OmegaConf.create({"name": "mse", "reduction": "mean"}))

        output = objective(
            torch.tensor([1.0, 3.0]),
            torch.tensor([2.0, 1.0]),
            torch.tensor([True, True]),
        )
        metrics = objective_diagnostics(output)

        self.assertEqual(
            set(metrics), {"objective/mean_squared_error", "objective/mean_absolute_error"}
        )
        self.assertNotIn("objective/variance", metrics)

    def test_binary_objective_exposes_classification_metrics(self) -> None:
        objective = build_loss(
            OmegaConf.create({"name": "binary-cross-entropy", "reduction": "mean"})
        )

        output = objective(
            torch.tensor([3.0, -3.0]),
            torch.tensor([1.0, 0.0]),
            torch.tensor([True, True]),
        )
        metrics = objective_diagnostics(output)

        self.assertEqual(
            set(metrics),
            {
                "objective/accuracy",
                "objective/positive_probability",
                "objective/positive_rate",
            },
        )


if __name__ == "__main__":
    unittest.main()
