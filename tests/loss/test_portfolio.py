import unittest

import numpy as np
import torch

from llca.loss.portfolio import PortfolioLoss


def _loss(**overrides: object) -> PortfolioLoss:
    arguments = {
        "leverage": 1.0,
        "risk_aversion": 1.0,
        "concentration_aversion": 0.0,
        "execution_fee": 0.0,
        "bid_ask_spread": 0.0,
        "slippage": 0.0,
        "borrow_cost": 0.0,
        "common_score_aversion": 0.0,
        "net_exposure_aversion": 0.0,
        "net_exposure_tolerance": 0.0,
        "normalization": "market_neutral",
        "return_type": "simple",
    }
    arguments.update(overrides)
    return PortfolioLoss(**arguments)  # type: ignore[arg-type]


class PortfolioLossTest(unittest.TestCase):
    def test_rejects_invalid_runtime_tensor_contracts(self) -> None:
        objective = _loss()
        scores = torch.zeros(2, 2)
        returns = torch.zeros_like(scores)

        with self.assertRaisesRegex(ValueError, "boolean"):
            objective(scores, returns, torch.ones_like(scores))
        with self.assertRaisesRegex(ValueError, "at least one valid"):
            objective(scores, returns, torch.tensor([[True, False], [False, False]]))
        scores[0, 0] = torch.nan
        with self.assertRaisesRegex(ValueError, "scores must be finite"):
            objective(scores, returns, torch.ones_like(scores, dtype=torch.bool))

    def test_market_neutral_normalization_masks_demeans_and_scales(self) -> None:
        scores = torch.tensor([[1.0, 2.0, 3.0, 99.0], [5.0, 5.0, 0.0, 0.0]])
        valid = torch.tensor([[True, True, True, False], [True, True, False, False]])

        weights = _loss().normalize_weights(scores, valid)

        torch.testing.assert_close(weights[0], torch.tensor([-0.5, 0.0, 0.5, 0.0]))
        torch.testing.assert_close(weights[1], torch.zeros(4))
        torch.testing.assert_close(weights.sum(dim=1), torch.zeros(2), atol=1e-7, rtol=0.0)
        self.assertAlmostEqual(float(weights[0].abs().sum()), 1.0)

    def test_gross_normalization_supports_directional_single_asset_models(self) -> None:
        weights = _loss(normalization="gross").normalize_weights(
            torch.tensor([[2.0]]), torch.tensor([[True]])
        )
        torch.testing.assert_close(weights, torch.ones(1, 1))

    def test_bounded_normalization_preserves_scores_below_leverage_cap(self) -> None:
        scores = torch.tensor([[0.2, -0.1, 99.0], [0.1, 0.2, 0.3]])
        valid = torch.tensor([[True, True, False], [True, True, True]])

        weights = _loss(normalization="bounded").normalize_weights(scores, valid)

        torch.testing.assert_close(weights[0], torch.tensor([0.2, -0.1, 0.0]))
        torch.testing.assert_close(weights[1], torch.tensor([0.1, 0.2, 0.3]))
        torch.testing.assert_close(weights.abs().sum(dim=1), torch.tensor([0.3, 0.6]))
        torch.testing.assert_close(weights.sum(dim=1), torch.tensor([0.1, 0.6]))

    def test_bounded_normalization_caps_gross_without_forcing_side_balance(self) -> None:
        scores = torch.tensor([[2.0, 1.0, -1.0], [-2.0, -1.0, 0.0]])
        valid = torch.ones_like(scores, dtype=torch.bool)

        weights = _loss(normalization="bounded").normalize_weights(scores, valid)

        torch.testing.assert_close(weights[0], torch.tensor([0.5, 0.25, -0.25]))
        torch.testing.assert_close(weights[1], torch.tensor([-2.0 / 3.0, -1.0 / 3.0, 0.0]))
        torch.testing.assert_close(weights.abs().sum(dim=1), torch.ones(2))
        torch.testing.assert_close(weights.sum(dim=1), torch.tensor([0.5, -1.0]))

    def test_market_neutral_weights_ignore_common_score_offsets(self) -> None:
        objective = _loss()
        valid = torch.ones(1, 3, dtype=torch.bool)
        scores = torch.tensor([[-2.0, 1.0, 4.0]])

        original = objective.normalize_weights(scores, valid)
        shifted = objective.normalize_weights(scores + 100.0, valid)

        torch.testing.assert_close(original, shifted)

    def test_log_returns_are_converted_before_loss_accounting(self) -> None:
        objective = _loss(return_type="log")
        scores = torch.tensor([[1.0, -1.0]])
        log_returns = torch.tensor([[np.log1p(0.1), np.log1p(-0.1)]])

        output = objective(scores, log_returns)

        self.assertAlmostEqual(float(output.mean_return), 0.1, places=6)
        self.assertAlmostEqual(float(output.gross_exposure), 1.0, places=6)
        self.assertAlmostEqual(float(output.net_exposure), 0.0, places=6)
        self.assertAlmostEqual(float(output.long_exposure), 0.5, places=6)
        self.assertAlmostEqual(float(output.short_exposure), 0.5, places=6)

    def test_invalid_simple_return_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "greater than -100%"):
            _loss()(torch.tensor([[1.0, -1.0]]), torch.tensor([[-1.0, 0.0]]))

    def test_common_score_penalty_has_gradient_for_saturated_long_only_weights(self) -> None:
        objective = _loss(
            normalization="bounded",
            common_score_aversion=1.0,
            net_exposure_aversion=0.0,
        )
        scores = torch.tensor([[2.0, 1.0]], requires_grad=True)

        output = objective(scores, torch.zeros_like(scores))
        output.loss.backward()

        self.assertAlmostEqual(float(output.common_score_penalty.detach()), 2.25)
        self.assertAlmostEqual(float(output.market_penalty.detach()), 2.25)
        self.assertIsNotNone(scores.grad)
        self.assertGreater(float(scores.grad.abs().sum()), 0.0)  # type: ignore[union-attr]

    def test_net_exposure_penalty_respects_tolerance_band(self) -> None:
        objective = _loss(
            normalization="bounded",
            net_exposure_aversion=2.0,
            net_exposure_tolerance=0.1,
        )

        output = objective(
            torch.tensor([[0.4, -0.2], [0.2, -0.15]]),
            torch.zeros(2, 2),
        )

        self.assertAlmostEqual(float(output.net_exposure_penalty), 0.005, places=7)
        self.assertAlmostEqual(float(output.market_penalty), 0.01, places=7)


if __name__ == "__main__":
    unittest.main()
