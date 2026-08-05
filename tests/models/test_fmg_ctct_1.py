import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from llca.data.modules.masked_panel import MaskedPanel
from llca.loss.portfolio import PortfolioLoss
from llca.mappers.model.mapper import model_registry
from llca.models.estimators.fmg import FmgCtct1Estimator
from llca.models.fmg import FmgCtct1
from llca.models.modules.conv_layer import ConvLayer
from llca.models.utils.standardizer import Standardizer


def _network() -> FmgCtct1:
    return FmgCtct1(
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


def _objective() -> PortfolioLoss:
    return PortfolioLoss(
        leverage=1.0,
        normalization="bounded",
        return_type="simple",
        risk_aversion=0.5,
        concentration_aversion=0.0,
        execution_fee=0.0,
        bid_ask_spread=0.0,
        slippage=0.0,
        borrow_cost=0.0,
        common_score_aversion=0.0,
        net_exposure_aversion=0.0,
        net_exposure_tolerance=0.0,
    )


def _estimator_config() -> object:
    return OmegaConf.create(
        {
            "name": "fmg-ctct-1",
            "d_model": 8,
            "feature_embedding_dim": 4,
            "dropout": 0.0,
            "sequence_length": 2,
            "score_activation": "tanh",
            "target": {"entity_id": 101},
            "diagnostics": {"score_saturation_threshold": 0.95},
            "inputs": {"features": "features", "context": ["context"]},
            "cnn": {"layers": [{"out_channels": 2, "kernel_size": [2, 1], "padding": [0, 0]}]},
            "transformer": {"n_heads": 2},
            "supervision": {"dataset": "loss", "column": "fwd_return"},
        }
    )


def _panel(values: pd.DataFrame, observed: pd.DataFrame, segment: pd.Series) -> MaskedPanel:
    return MaskedPanel(
        values=values,
        observed=observed,
        age=pd.DataFrame(0, index=values.index, columns=values.columns),
        segment=segment,
    )


def _panels() -> dict[str, MaskedPanel]:
    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    index = pd.MultiIndex.from_product([dates, [101, 202]], names=["date", "instrument_id"])
    entity = index.get_level_values("instrument_id")
    segment = pd.Series(np.where(entity == 101, 0, 1), index=index)
    features = pd.DataFrame(
        {
            "x1": np.linspace(1.0, 2.0, len(index)),
            "x2": np.linspace(-1.0, 1.0, len(index)),
        },
        index=index,
    )
    context = pd.DataFrame(
        {
            "c1": np.where(entity == 101, 1.0, -1.0),
            "c2": np.arange(len(index), dtype=float),
        },
        index=index,
    )
    returns = pd.DataFrame({"fwd_return": np.where(entity == 101, 0.01, np.nan)}, index=index)
    return {
        "features": _panel(
            features,
            pd.DataFrame(True, index=index, columns=features.columns),
            segment,
        ),
        "context": _panel(
            context,
            pd.DataFrame(True, index=index, columns=context.columns),
            segment,
        ),
        "loss": _panel(
            returns,
            pd.DataFrame({"fwd_return": np.asarray(entity == 101, dtype=bool)}, index=index),
            segment,
        ),
    }


class FmgCtct1NetworkTest(unittest.TestCase):
    def test_only_target_queries_while_all_assets_receive_gradient(self) -> None:
        model = _network()
        n_assets = 3
        window = 4 + model.buffer_size
        features = torch.randn(n_assets, window, 2, requires_grad=True)
        feature_age = torch.zeros_like(features)
        context = torch.randn(n_assets, 2)
        context_age = torch.zeros_like(context)
        shapes: dict[str, tuple[int, ...]] = {}

        def capture(_module: object, args: tuple[torch.Tensor, ...]) -> None:
            shapes["query"] = tuple(args[0].shape)
            shapes["keys"] = tuple(args[1].shape)

        handle = model.cross_sectional_attention.attention.register_forward_pre_hook(capture)
        allocation, diagnostics = model(
            features,
            feature_age,
            context,
            context_age,
            torch.tensor([1]),
        )
        handle.remove()
        allocation.sum().backward()

        self.assertEqual(allocation.shape, (1,))
        self.assertLessEqual(abs(float(allocation.detach()[0])), 1.0)
        self.assertEqual(shapes["query"], (4, 1, 8))
        self.assertEqual(shapes["keys"], (4, n_assets, 8))
        self.assertEqual(diagnostics["context"].shape, (n_assets, 2))
        assert features.grad is not None
        self.assertTrue(all(float(features.grad[i].abs().sum()) > 0.0 for i in range(n_assets)))

    def test_rejects_invalid_target_position(self) -> None:
        model = _network()
        features = torch.randn(2, 4 + model.buffer_size, 2)
        with self.assertRaisesRegex(IndexError, "outside"):
            model(
                features,
                torch.zeros_like(features),
                torch.randn(2, 2),
                torch.zeros(2, 2),
                torch.tensor([2]),
            )


class FmgCtct1EstimatorTest(unittest.TestCase):
    def test_context_assets_do_not_require_supervision_and_output_is_allocation(self) -> None:
        estimator = FmgCtct1Estimator(_estimator_config(), _objective())  # type: ignore[arg-type]
        estimator._feature_columns = ["x1", "x2"]
        estimator._context_columns = ["c1", "c2"]
        estimator._model = estimator._build_model()
        panels = _panels()

        raw = estimator._windows(panels)
        counts = raw.index.to_frame(index=False).groupby("date").size()
        self.assertTrue((counts == 2).all())
        context_only = np.asarray(raw.index.get_level_values("instrument_id") != 101, dtype=bool)
        self.assertTrue(torch.isnan(raw.supervision[torch.from_numpy(context_only)]).all())

        estimator._feature_scaler = Standardizer.fit(raw.features.values)
        estimator._context_scaler = Standardizer.fit(raw.context[0])
        windows = estimator._to_windows(raw, batch_size=2)
        assert windows is not None
        output = estimator._forward_batch(windows, windows.batches[0])
        self.assertTrue(bool(torch.isfinite(output.loss).item()))
        output.loss.backward()  # type: ignore[no-untyped-call]
        self.assertIn("allocations/mean", output.diagnostic_metrics())

        prediction = estimator.predict(panels)
        self.assertEqual(prediction.kind, "portfolio")
        self.assertEqual(prediction.values.name, "weight")
        self.assertTrue((prediction.index.get_level_values("instrument_id") == 101).all())
        self.assertTrue((prediction.values.abs() <= 1.0).all())

        with TemporaryDirectory() as tmp:
            bundle = Path(tmp) / "fmg-ctct-1.pt"
            estimator._save(bundle)
            restored = FmgCtct1Estimator.load(bundle, torch.device("cpu"))
            restored_prediction = restored.predict(panels)
        assert isinstance(prediction.values, pd.Series)
        assert isinstance(restored_prediction.values, pd.Series)
        pd.testing.assert_series_equal(restored_prediction.values, prediction.values)


class FmgCtct1ConfigurationTest(unittest.TestCase):
    def test_hydra_configuration_is_registered_and_valid(self) -> None:
        config_dir = str(
            (Path(__file__).resolve().parents[2] / "hydra" / "configs" / "training").resolve()
        )
        with initialize_config_dir(config_dir=config_dir, version_base=None):
            cfg = compose(
                config_name="train",
                overrides=[
                    "experiment=fmg-ctct-1",
                    "model.target.entity_id=14593",
                ],
            )

        self.assertTrue(model_registry.is_registered("fmg-ctct-1"))
        self.assertEqual(model_registry.validate("fmg-ctct-1", cfg), [])
        self.assertEqual(cfg.loss.normalization, "bounded")
        self.assertEqual(cfg.loss.leverage, 1.0)
        self.assertEqual(cfg.model.score_activation, "tanh")

    def test_missing_target_and_incompatible_normalization_are_rejected(self) -> None:
        config_dir = str(
            (Path(__file__).resolve().parents[2] / "hydra" / "configs" / "training").resolve()
        )
        with initialize_config_dir(config_dir=config_dir, version_base=None):
            cfg = compose(
                config_name="train",
                overrides=["experiment=fmg-ctct-1", "model.target.entity_id=null"],
            )
        cfg.loss.normalization = "gross"

        errors = model_registry.validate("fmg-ctct-1", cfg)

        self.assertIn("model.target.entity_id is required", errors)
        self.assertIn("fmg-ctct-1 requires loss.normalization 'bounded'", errors)


if __name__ == "__main__":
    unittest.main()
