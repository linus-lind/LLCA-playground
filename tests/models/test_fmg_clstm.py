import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast

import pandas as pd
import torch
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf
from torch import nn

from llca.mappers.model.mapper import model_registry
from llca.models.estimators.fmg import FmgClstmEstimator
from llca.models.fmg import FmgClstm
from llca.models.modules.conv_layer import ConvLayer
from llca.models.utils.ewma_standardizer import EwmaStandardizer
from tests.models.test_fmg_ctct_1 import _estimator_config, _objective, _panels, _tail_panels
from tests.models.test_fmg_ctt import _perturb_non_target


def _network() -> FmgClstm:
    return FmgClstm(
        num_features=2,
        num_context_vars=2,
        model_dim=8,
        feature_embedding_dim=4,
        cnn_layers=[ConvLayer(2, 2, 1, 0, 0)],
        lstm_num_layers=2,
        lstm_recurrent_dropout=0.2,
        lstm_output_dropout=0.3,
        lstm_bias=True,
        lstm_bidirectional=False,
        dropout=0.0,
        score_activation="tanh",
    )


def _config() -> DictConfig:
    config = cast(DictConfig, _estimator_config())
    config.name = "fmg-clstm"
    del config.transformer
    config.lstm = OmegaConf.create(
        {
            "num_layers": 2,
            "recurrent_dropout": 0.1,
            "output_dropout": 0.1,
            "bias": True,
            "bidirectional": False,
        }
    )
    return config


class FmgClstmNetworkTest(unittest.TestCase):
    def test_lstm_dimensions_terminal_dropout_and_gate_follow_hydra_contract(self) -> None:
        model = _network()
        window = 4 + model.buffer_size
        features = torch.randn(1, window, 2, requires_grad=True)
        calls = {"gate": 0, "grn": 0}

        def gate_hook(_module: object, _args: object, _output: object) -> None:
            calls["gate"] += 1

        def grn_hook(_module: object, _args: object, _output: object) -> None:
            calls["grn"] += 1

        gate_handle = model.temporal_lstm.gate_add_norm.register_forward_hook(gate_hook)
        grn_handle = model.temporal_lstm.grn.register_forward_hook(grn_hook)
        allocation, diagnostics = model(
            features,
            torch.zeros_like(features),
            torch.randn(1, 2),
            torch.zeros(1, 2),
        )
        gate_handle.remove()
        grn_handle.remove()
        allocation.sum().backward()

        lstm = model.temporal_lstm.lstm
        self.assertEqual(lstm.input_size, 8)
        self.assertEqual(lstm.hidden_size, 8)
        self.assertEqual(lstm.num_layers, 2)
        self.assertAlmostEqual(lstm.dropout, 0.2)
        self.assertTrue(lstm.bias)
        self.assertFalse(lstm.bidirectional)
        self.assertIsInstance(model.temporal_lstm.gate_add_norm.dropout, nn.Dropout)
        output_dropout = cast(nn.Dropout, model.temporal_lstm.gate_add_norm.dropout)
        self.assertAlmostEqual(output_dropout.p, 0.3)
        self.assertEqual(calls, {"gate": 1, "grn": 1})
        self.assertFalse(hasattr(model, "temporal_attention"))
        self.assertFalse(hasattr(model, "aggregation"))
        self.assertFalse(hasattr(model, "cross_sectional_attention"))
        # Context is embedded at feature_embedding_dim (4), not model_dim (8).
        self.assertEqual(model.context_encoder.embedding_dim, 4)
        self.assertEqual(allocation.shape, (1,))
        self.assertLessEqual(abs(float(allocation.detach()[0])), 1.0)
        self.assertEqual(diagnostics["context"].shape, (1, 2))
        assert features.grad is not None
        self.assertGreater(float(features.grad.abs().sum()), 0.0)

    def test_rejects_more_than_one_asset(self) -> None:
        model = _network()
        features = torch.randn(2, 4 + model.buffer_size, 2)
        with self.assertRaisesRegex(ValueError, "exactly one asset"):
            model(
                features,
                torch.zeros_like(features),
                torch.randn(2, 2),
                torch.zeros(2, 2),
            )


class FmgClstmEstimatorTest(unittest.TestCase):
    def test_target_only_allocation_training_inference_and_bundle_roundtrip(self) -> None:
        estimator = FmgClstmEstimator(_config(), _objective())
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
        windows = estimator._to_windows(raw, batch_size=2)
        assert windows is not None

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
            bundle = Path(tmp) / "fmg-clstm.pt"
            estimator._save(bundle)
            restored = FmgClstmEstimator.load(bundle, torch.device("cpu"))
            restored_prediction = restored.predict(predict_panels)
        assert isinstance(restored_prediction.values, pd.Series)
        pd.testing.assert_series_equal(restored_prediction.values, baseline.values)


class FmgClstmConfigurationTest(unittest.TestCase):
    def test_hydra_configuration_is_registered_and_valid(self) -> None:
        config_dir = str(
            (Path(__file__).resolve().parents[2] / "hydra" / "configs" / "training").resolve()
        )
        with initialize_config_dir(config_dir=config_dir, version_base=None):
            cfg = compose(
                config_name="train",
                overrides=[
                    "experiment=fmg-clstm",
                    "model.target.entity_id=14593",
                ],
            )

        self.assertTrue(model_registry.is_registered("fmg-clstm"))
        self.assertEqual(model_registry.validate("fmg-clstm", cfg), [])
        self.assertNotIn("transformer", cfg.model)
        self.assertEqual(cfg.model.lstm.num_layers, 3)
        self.assertEqual(cfg.model.lstm.recurrent_dropout, 0.1)
        self.assertEqual(cfg.model.lstm.output_dropout, 0.1)

    def test_rejects_invalid_recurrent_and_dimensional_configuration(self) -> None:
        config_dir = str(
            (Path(__file__).resolve().parents[2] / "hydra" / "configs" / "training").resolve()
        )
        with initialize_config_dir(config_dir=config_dir, version_base=None):
            cfg = compose(
                config_name="train",
                overrides=[
                    "experiment=fmg-clstm",
                    "model.target.entity_id=14593",
                ],
            )
        cfg.model.lstm.num_layers = 1
        cfg.model.lstm.recurrent_dropout = 0.1
        cfg.model.lstm.bidirectional = True

        errors = model_registry.validate("fmg-clstm", cfg)

        self.assertIn("model.lstm.recurrent_dropout must be 0.0 when num_layers is 1", errors)
        self.assertIn(
            "model.lstm.bidirectional must be false to preserve the model.d_model output width",
            errors,
        )


if __name__ == "__main__":
    unittest.main()
