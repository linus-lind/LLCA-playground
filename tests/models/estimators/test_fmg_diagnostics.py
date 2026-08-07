import unittest

import torch

from llca.loss.modules.portfolio_loss_output import PortfolioLossOutput
from llca.models.estimators.fmg.fmg_ctct_2 import score_metrics


class FmgDiagnosticsTest(unittest.TestCase):
    def test_score_saturation_and_portfolio_components_are_detached(self) -> None:
        scores = torch.tensor([[-0.99, 0.50], [0.96, 0.10]], requires_grad=True)
        mask = torch.tensor([[True, True], [True, False]])
        loss = scores.square().mean()
        output = PortfolioLossOutput(
            loss=loss,
            mean_return=loss + 1.0,
            cash_return=loss + 1.5,
            variance=loss + 2.0,
            turnover=loss + 3.0,
            cost=loss + 4.0,
            gross_exposure=loss + 5.0,
            net_exposure=loss + 6.0,
            long_exposure=loss + 7.0,
            short_exposure=loss + 8.0,
            concentration=loss + 9.0,
            common_score_penalty=loss + 10.0,
            net_exposure_penalty=loss + 11.0,
            market_penalty=loss + 12.0,
        )

        metrics = score_metrics(scores, mask, output, 0.95)

        self.assertAlmostEqual(float(metrics["scores/saturation_fraction"]), 2.0 / 3.0)
        self.assertIn("objective/mean_return", metrics)
        self.assertIn("objective/turnover", metrics)
        self.assertIn("objective/market_penalty", metrics)
        self.assertNotIn("objective/loss", metrics)
        self.assertTrue(
            all(
                not value.requires_grad
                for value in metrics.values()
                if isinstance(value, torch.Tensor)
            )
        )


if __name__ == "__main__":
    unittest.main()
