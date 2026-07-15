import unittest

from omegaconf import OmegaConf

from llca.loss.portfolio import PortfolioLoss
from llca.mappers.loss.config_validator import _validate_loss
from llca.mappers.loss.mapper import build_loss, prediction_kind
from llca.mappers.loss.portfolio import _validate_portfolio


def _config() -> object:
    return OmegaConf.create(
        {
            "model": {
                "supervision": {"dataset": "loss", "column": "fwd_return"},
            },
            "features": {
                "loss": [
                    {
                        "name": "log_change",
                        "column": "open",
                        "horizon": 1,
                        "shift": -2,
                        "as": "fwd_return",
                    }
                ]
            },
            "loss": {
                "name": "portfolio",
                "leverage": 1.0,
                "normalization": "market_neutral",
                "return_type": "log",
                "risk_aversion": 1.0,
                "concentration_aversion": 0.0,
                "common_score_aversion": 0.0,
                "net_exposure_aversion": 0.0,
                "net_exposure_tolerance": 0.0,
                "execution_fee": 0.0,
                "bid_ask_spread": 0.0,
                "slippage": 0.0,
                "borrow_cost": 0.0,
            },
        }
    )


class PortfolioLossMapperTest(unittest.TestCase):
    def test_objectives_map_to_canonical_prediction_kinds(self) -> None:
        self.assertEqual(prediction_kind("portfolio"), "portfolio")
        self.assertEqual(prediction_kind("mse"), "regression")
        self.assertEqual(prediction_kind("binary-cross-entropy"), "binary")

    def test_maps_valid_return_and_normalization_contract(self) -> None:
        config = _config()
        self.assertEqual(_validate_portfolio(config.loss), [])  # type: ignore[attr-defined]
        self.assertEqual(_validate_loss(config), [])  # type: ignore[arg-type]

        objective = build_loss(config.loss)  # type: ignore[attr-defined]

        self.assertIsInstance(objective, PortfolioLoss)
        self.assertEqual(objective.normalization, "market_neutral")
        self.assertEqual(objective.return_type, "log")
        self.assertEqual(objective.common_score_aversion, 0.0)
        self.assertEqual(objective.net_exposure_aversion, 0.0)
        self.assertEqual(objective.net_exposure_tolerance, 0.0)

    def test_rejects_target_transform_return_type_mismatch(self) -> None:
        config = _config()
        config.features.loss[0].name = "simple_change"  # type: ignore[attr-defined]

        errors = _validate_loss(config)  # type: ignore[arg-type]

        self.assertTrue(any("conflicts with supervision" in error for error in errors))

    def test_rejects_unknown_normalization(self) -> None:
        config = _config()
        config.loss.normalization = "unknown"  # type: ignore[attr-defined]

        errors = _validate_portfolio(config.loss)  # type: ignore[attr-defined]

        self.assertTrue(any("loss.normalization" in error for error in errors))

    def test_maps_bounded_normalization(self) -> None:
        config = _config()
        config.loss.normalization = "bounded"  # type: ignore[attr-defined]

        self.assertEqual(_validate_portfolio(config.loss), [])  # type: ignore[attr-defined]
        objective = build_loss(config.loss)  # type: ignore[attr-defined]

        self.assertIsInstance(objective, PortfolioLoss)
        self.assertEqual(objective.normalization, "bounded")

    def test_maps_market_exposure_regularization(self) -> None:
        config = _config()
        config.loss.common_score_aversion = 0.003  # type: ignore[attr-defined]
        config.loss.net_exposure_aversion = 0.004  # type: ignore[attr-defined]
        config.loss.net_exposure_tolerance = 0.2  # type: ignore[attr-defined]

        self.assertEqual(_validate_portfolio(config.loss), [])  # type: ignore[attr-defined]
        objective = build_loss(config.loss)  # type: ignore[attr-defined]

        self.assertEqual(objective.common_score_aversion, 0.003)  # type: ignore[union-attr]
        self.assertEqual(objective.net_exposure_aversion, 0.004)  # type: ignore[union-attr]
        self.assertEqual(objective.net_exposure_tolerance, 0.2)  # type: ignore[union-attr]

    def test_rejects_net_exposure_tolerance_above_leverage(self) -> None:
        config = _config()
        config.loss.net_exposure_tolerance = 1.1  # type: ignore[attr-defined]

        errors = _validate_portfolio(config.loss)  # type: ignore[attr-defined]

        self.assertIn("loss.net_exposure_tolerance must be <= loss.leverage", errors)


if __name__ == "__main__":
    unittest.main()
