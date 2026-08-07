import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast

import pandas as pd
import torch
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

from llca.data.modules.masked_panel import MaskedPanel
from llca.mappers.model.mapper import model_registry
from llca.models.estimators.fmg import FmgCttEstimator
from llca.models.fmg import FmgCtt
from llca.models.modules.conv_layer import ConvLayer
from llca.models.utils.ewma_standardizer import EwmaStandardizer
from tests.models.test_fmg_ctct_1 import _estimator_config, _objective, _panels, _tail_panels


def _network() -> FmgCtt:
    return FmgCtt(
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


def _perturb_non_target(panels: dict[str, MaskedPanel]) -> dict[str, MaskedPanel]:
    """Change other assets drastically while preserving all target inputs."""
    changed = dict(panels)
    for name in ("features", "context"):
        panel = panels[name]
        values = panel.values.copy()
        non_target = values.index.get_level_values("instrument_id") != 101
        values.loc[non_target, :] += 1_000_000.0
        changed[name] = MaskedPanel(
            values=values,
            observed=panel.observed,
            age=panel.age,
            segment=panel.segment,
        )
    return changed


class FmgCttNetworkTest(unittest.TestCase):
    def test_temporal_series_flows_directly_to_one_bounded_allocation(self) -> None:
        model = _network()
        window = 4 + model.buffer_size
        features = torch.randn(1, window, 2, requires_grad=True)
        feature_age = torch.zeros_like(features)
        context = torch.randn(1, 2)
        context_age = torch.zeros_like(context)

        allocation, diagnostics = model(features, feature_age, context, context_age)
        allocation.sum().backward()

        self.assertFalse(hasattr(model, "cross_sectional_attention"))
        # Option 1: temporal self-attention carries no context pathway.
        self.assertIsNone(model.temporal_attention.grn.context_projection)
        # Context is embedded at feature_embedding_dim (4), not model_dim (8).
        self.assertEqual(model.context_encoder.embedding_dim, 4)
        self.assertEqual(allocation.shape, (1,))
        self.assertLessEqual(abs(float(allocation.detach()[0])), 1.0)
        self.assertEqual(diagnostics["context"].shape, (1, 2))
        assert features.grad is not None
        self.assertGreater(float(features.grad.abs().sum()), 0.0)

    def test_rejects_more_than_the_target_asset(self) -> None:
        model = _network()
        features = torch.randn(2, 4 + model.buffer_size, 2)
        with self.assertRaisesRegex(ValueError, "exactly one asset"):
            model(
                features,
                torch.zeros_like(features),
                torch.randn(2, 2),
                torch.zeros(2, 2),
            )


class FmgCttEstimatorTest(unittest.TestCase):
    def test_other_assets_are_removed_before_scaling_training_and_inference(self) -> None:
        config = cast(DictConfig, _estimator_config())
        config.name = "fmg-ctt"
        estimator = FmgCttEstimator(config, _objective())
        estimator._feature_columns = ["x1", "x2"]
        estimator._context_columns = ["c1", "c2"]
        estimator._model = estimator._build_model()
        estimator._feature_ewma = EwmaStandardizer(
            half_life=3.0, history_buffer=estimator.required_history
        )
        panels = _panels()
        estimator._feature_ewma.fit(panels["features"])

        raw = estimator._windows(panels)
        self.assertTrue((raw.index.get_level_values("instrument_id") == 101).all())
        self.assertEqual(raw.features.values.shape[0], 5)
        self.assertEqual(raw.context[0].shape[0], 3)

        windows = estimator._to_windows(raw, batch_size=2)
        assert windows is not None
        self.assertTrue(all(batch.n_max == 1 for batch in windows.batches))

        output = estimator._forward_batch(windows, windows.batches[0])
        self.assertTrue(bool(torch.isfinite(output.loss).item()))
        output.loss.backward()  # type: ignore[no-untyped-call]

        # predict() sees a tail-only panel; see _tail_panels' docstring for why.
        predict_panels = _tail_panels(panels, estimator.required_history)
        baseline = estimator.predict(predict_panels)
        perturbed = estimator.predict(_perturb_non_target(predict_panels))
        self.assertEqual(baseline.kind, "portfolio")
        assert isinstance(baseline.values, pd.Series)
        assert isinstance(perturbed.values, pd.Series)
        pd.testing.assert_series_equal(baseline.values, perturbed.values)
        self.assertTrue((baseline.values.abs() <= 1.0).all())

        with TemporaryDirectory() as tmp:
            bundle = Path(tmp) / "fmg-ctt.pt"
            estimator._save(bundle)
            restored = FmgCttEstimator.load(bundle, torch.device("cpu"))
            restored_prediction = restored.predict(predict_panels)
        assert isinstance(restored_prediction.values, pd.Series)
        pd.testing.assert_series_equal(restored_prediction.values, baseline.values)


class FmgCttConfigurationTest(unittest.TestCase):
    def test_hydra_configuration_is_registered_and_valid(self) -> None:
        config_dir = str(
            (Path(__file__).resolve().parents[2] / "hydra" / "configs" / "training").resolve()
        )
        with initialize_config_dir(config_dir=config_dir, version_base=None):
            cfg = compose(
                config_name="train",
                overrides=[
                    "experiment=fmg-ctt",
                    "model.target.entity_id=14593",
                ],
            )

        self.assertTrue(model_registry.is_registered("fmg-ctt"))
        self.assertEqual(model_registry.validate("fmg-ctt", cfg), [])
        self.assertEqual(cfg.loss.normalization, "bounded")
        self.assertEqual(cfg.loss.leverage, 1.0)
        self.assertEqual(cfg.model.score_activation, "tanh")

    def test_missing_target_and_cross_sectional_loss_are_rejected(self) -> None:
        config_dir = str(
            (Path(__file__).resolve().parents[2] / "hydra" / "configs" / "training").resolve()
        )
        with initialize_config_dir(config_dir=config_dir, version_base=None):
            cfg = compose(
                config_name="train",
                overrides=["experiment=fmg-ctt", "model.target.entity_id=null"],
            )
        cfg.loss.normalization = "market_neutral"

        errors = model_registry.validate("fmg-ctt", cfg)

        self.assertIn("model.target.entity_id is required", errors)
        self.assertIn("fmg-ctt requires loss.normalization 'bounded'", errors)


if __name__ == "__main__":
    unittest.main()
