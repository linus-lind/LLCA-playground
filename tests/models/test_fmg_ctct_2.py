import unittest
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir

from llca.mappers import build_loss, build_model, validate_config
from llca.mappers.model.mapper import model_registry
from llca.models.estimators.fmg import FmgCtct2Estimator
from llca.models.fmg import FmgCtct2
from llca.models.modules.conv_layer import ConvLayer


def _network() -> FmgCtct2:
    return FmgCtct2(
        num_features=2,
        num_context_vars=2,
        model_dim=8,
        feature_embedding_dim=4,
        sequence_length=4,
        cnn_layers=[ConvLayer(2, 2, 1, 0, 0)],
        n_heads=2,
        dropout=0.0,
        score_activation="tanh",
    )


class FmgCtct2NetworkTest(unittest.TestCase):
    def test_every_asset_queries_and_receives_a_score_and_gradient(self) -> None:
        model = _network()
        n_assets = 3
        features = torch.randn(
            n_assets,
            4 + model.buffer_size,
            2,
            requires_grad=True,
        )
        feature_age = torch.zeros_like(features)
        context = torch.randn(n_assets, 2)
        context_age = torch.zeros_like(context)
        shapes: dict[str, tuple[int, ...]] = {}

        def capture(_module: object, args: tuple[torch.Tensor, ...]) -> None:
            shapes["query"] = tuple(args[0].shape)
            shapes["keys"] = tuple(args[1].shape)

        handle = model.cross_sectional_attention.attention.register_forward_pre_hook(capture)
        scores, diagnostics = model(features, feature_age, context, context_age)
        handle.remove()
        scores.sum().backward()

        self.assertEqual(scores.shape, (n_assets,))
        self.assertTrue(bool((scores.detach().abs() <= 1.0).all()))
        self.assertEqual(shapes["query"], (4, n_assets, 8))
        self.assertEqual(shapes["keys"], (4, n_assets, 8))
        self.assertEqual(diagnostics["context"].shape, (n_assets, 2))
        assert features.grad is not None
        self.assertTrue(all(float(features.grad[i].abs().sum()) > 0.0 for i in range(n_assets)))


class FmgCtct2ConfigurationTest(unittest.TestCase):
    def test_canonical_hydra_model_is_registered_and_valid(self) -> None:
        config_dir = str((Path(__file__).resolve().parents[2] / "hydra" / "configs").resolve())
        with initialize_config_dir(config_dir=config_dir, version_base=None):
            cfg = compose(config_name="train")

        self.assertEqual(cfg.model.name, "fmg-ctct-2")
        self.assertEqual(model_registry.validate("fmg-ctct-2", cfg), [])

    def test_regression_objective_changes_native_prediction_semantics(self) -> None:
        config_dir = str((Path(__file__).resolve().parents[2] / "hydra" / "configs").resolve())
        with initialize_config_dir(config_dir=config_dir, version_base=None):
            cfg = compose(
                config_name="train",
                overrides=["experiment=fmg-ctct-2", "loss=mse"],
            )
        validate_config(cfg)
        objective = build_loss(cfg.loss)
        estimator = build_model(
            cfg.model,
            loss=objective,
            loss_config=cfg.loss,
        )()

        self.assertIsInstance(estimator, FmgCtct2Estimator)
        self.assertEqual(estimator._prediction_kind, "regression")


if __name__ == "__main__":
    unittest.main()
