import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast

import numpy as np
import pandas as pd
import torch
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from llca.data.modules.masked_panel import MaskedPanel
from llca.loss.portfolio import PortfolioLoss
from llca.mappers import build_loss, build_model, validate_config
from llca.mappers.model.mapper import model_registry
from llca.models.estimators.fmg import FmgCtct2Estimator
from llca.models.fmg import FmgCtct2
from llca.models.modules.conv_layer import ConvLayer
from llca.models.utils.ewma_standardizer import EwmaStandardizer


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
        # Option 1: cross-sectional attention carries no context pathway.
        self.assertIsNone(model.cross_sectional_attention.grn.context_projection)
        self.assertEqual(shapes["query"], (4, n_assets, 8))
        self.assertEqual(shapes["keys"], (4, n_assets, 8))
        self.assertEqual(diagnostics["context"].shape, (n_assets, 2))
        assert features.grad is not None
        self.assertTrue(all(float(features.grad[i].abs().sum()) > 0.0 for i in range(n_assets)))


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
            "name": "fmg-ctct-2",
            "d_model": 8,
            "feature_embedding_dim": 4,
            "dropout": 0.0,
            "sequence_length": 2,
            "score_activation": "tanh",
            "diagnostics": {"score_saturation_threshold": 0.95},
            "inputs": {"features": "features", "context": ["context"]},
            "standardization": {"half_life": 3.0},
            "cnn": {"layers": [{"out_channels": 2, "kernel_size": [2, 1], "padding": [0, 0]}]},
            "transformer": {"n_heads": 2},
            "supervision": {"dataset": "loss", "column": "fwd_return"},
        }
    )


def _panel(values: pd.DataFrame, segment: pd.Series) -> MaskedPanel:
    return MaskedPanel(
        values=values,
        observed=pd.DataFrame(True, index=values.index, columns=values.columns),
        age=pd.DataFrame(0, index=values.index, columns=values.columns),
        segment=segment,
    )


def _panels() -> dict[str, MaskedPanel]:
    """Two entities on clearly different feature scales, both cross-sectionally scored."""
    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    index = pd.MultiIndex.from_product([dates, [101, 202]], names=["date", "instrument_id"])
    entity = index.get_level_values("instrument_id")
    segment = pd.Series(np.where(entity == 101, 0, 1), index=index)
    # Assign clearly entity-specific series (small scale for 101, large scale for 202).
    low = np.array([1.0, 1.1, 0.9, 1.2, 1.05])
    high = np.array([1000.0, 900.0, 1100.0, 800.0, 1050.0])
    x1 = np.empty(len(index))
    x1[np.asarray(entity == 101)] = low
    x1[np.asarray(entity == 202)] = high
    features = pd.DataFrame({"x1": x1, "x2": np.linspace(-1.0, 1.0, len(index))}, index=index)
    context = pd.DataFrame(
        {
            "c1": np.where(entity == 101, 1.0, -1.0),
            "c2": np.arange(len(index), dtype=float),
        },
        index=index,
    )
    returns = pd.DataFrame({"fwd_return": np.full(len(index), 0.01)}, index=index)
    return {
        "features": _panel(features, segment),
        "context": _panel(context, segment),
        "loss": _panel(returns, segment),
    }


def _slice_dates(panels: dict[str, MaskedPanel], dates: pd.DatetimeIndex) -> dict[str, MaskedPanel]:
    keep_by_name = {
        name: panel.values.index.get_level_values("date").isin(dates)
        for name, panel in panels.items()
    }
    return {name: panel.slice_rows(keep_by_name[name]) for name, panel in panels.items()}


def _xs_frame(values: pd.DataFrame, entity: int) -> pd.DataFrame:
    selected = values.xs(entity, level="instrument_id")
    assert isinstance(selected, pd.DataFrame)
    return selected[["x1", "x2"]]


def _fit_through_train(train_panels: dict[str, MaskedPanel]) -> FmgCtct2Estimator:
    """Build, fit, and causally advance one estimator through ``train_panels``."""
    estimator = FmgCtct2Estimator(_estimator_config(), _objective())  # type: ignore[arg-type]
    estimator._feature_columns = ["x1", "x2"]
    estimator._context_columns = ["c1", "c2"]
    estimator._model = estimator._build_model()
    estimator._feature_ewma = EwmaStandardizer(
        half_life=3.0, history_buffer=estimator.required_history
    )
    estimator._feature_ewma.fit(train_panels["features"])
    estimator._windows(train_panels)
    return estimator


def _train_predict_split(
    panels: dict[str, MaskedPanel],
) -> tuple[dict[str, MaskedPanel], dict[str, MaskedPanel]]:
    all_dates = pd.DatetimeIndex(
        panels["features"].values.index.get_level_values("date").unique().sort_values()
    )
    # sequence_length=2, buffer_size=1 -> a window needs 3 rows. Train covers the first 3
    # dates; predict's panel starts 1 date (= buffer_size) earlier than train ends, so it
    # exercises the real overlap between a causal history prefix and already-advanced rows,
    # exactly like this pipeline's val-tail/test-lookback-prefix overlap.
    train_dates, predict_dates = all_dates[:3], all_dates[2:]
    return _slice_dates(panels, train_dates), _slice_dates(panels, predict_dates)


class FmgCtct2EstimatorTest(unittest.TestCase):
    def test_fit_predict_serialize_round_trip(self) -> None:
        train_panels, predict_panels = _train_predict_split(_panels())

        estimator = _fit_through_train(train_panels)
        raw = estimator._windows(train_panels)
        self.assertGreater(len(raw.index), 0)
        windows = estimator._to_windows(raw, batch_size=2)
        assert windows is not None
        output = estimator._forward_batch(windows, windows.batches[0])
        self.assertTrue(bool(torch.isfinite(output.loss).item()))
        output.loss.backward()  # type: ignore[no-untyped-call]

        with TemporaryDirectory() as tmp:
            bundle = Path(tmp) / "fmg-ctct-2.pt"
            estimator._save(bundle)
            restored = FmgCtct2Estimator.load(bundle, torch.device("cpu"))

        # Save happens before either estimator has ever seen predict_panels, so each is
        # doing a genuinely fresh (non-replay) continuation from the identical saved state.
        baseline = estimator.predict(predict_panels)
        restored_prediction = restored.predict(predict_panels)

        self.assertEqual(baseline.kind, "portfolio")
        assert isinstance(baseline.values, pd.Series)
        assert isinstance(restored_prediction.values, pd.Series)
        self.assertTrue((baseline.values.abs() <= 1.0).all())
        pd.testing.assert_series_equal(restored_prediction.values, baseline.values)

    def test_entity_isolation_at_the_estimator_level(self) -> None:
        """Perturbing one entity's raw predict-time history must not move another entity's
        *normalized inputs*, now that normalization is per-entity rather than pooled across
        the cross-section.

        This intentionally checks ``_combined``'s normalized values, not ctct-2's final
        score: ctct-2's cross-sectional attention deliberately mixes entities together, so
        its *score* legitimately depends on the whole cross-section by architecture, not
        just its own EWMA-standardized history. Entity isolation is a normalizer property.
        """
        panels = _panels()
        train_panels, predict_panels = _train_predict_split(panels)

        perturbed_features = predict_panels["features"].values.copy()
        perturbed_features.loc[
            perturbed_features.index.get_level_values("instrument_id") == 202, "x1"
        ] += 5_000_000.0
        perturbed_panels = dict(predict_panels)
        perturbed_panels["features"] = MaskedPanel(
            values=perturbed_features,
            observed=predict_panels["features"].observed,
            age=predict_panels["features"].age,
            segment=predict_panels["features"].segment,
        )
        # Independently-fit estimators: within one instance, re-normalizing already-seen
        # dates is a replay that ignores the (here, perturbed) input by design, which would
        # make a single-estimator comparison vacuous.
        baseline_estimator = _fit_through_train(train_panels)
        perturbed_estimator = _fit_through_train(train_panels)

        baseline_combined = baseline_estimator._combined(predict_panels)
        perturbed_combined = perturbed_estimator._combined(perturbed_panels)

        baseline_101 = _xs_frame(baseline_combined.values, 101)
        perturbed_101 = _xs_frame(perturbed_combined.values, 101)
        perturbed_202 = _xs_frame(perturbed_combined.values, 202)
        baseline_202 = _xs_frame(baseline_combined.values, 202)

        pd.testing.assert_frame_equal(baseline_101, perturbed_101)
        self.assertFalse(perturbed_202.equals(baseline_202))


def _estimator_config_with_risk_free() -> object:
    config = cast(DictConfig, _estimator_config())
    config.risk_free = {"dataset": "risk_free", "column": "rf"}
    return config


def _with_risk_free(
    panels: dict[str, MaskedPanel],
    rate_by_date: dict[pd.Timestamp, float],
    *,
    observed: bool = True,
) -> dict[str, MaskedPanel]:
    """Attach a date-level risk-free panel broadcast across entities, like the aligned pipeline."""
    index = panels["loss"].values.index
    dates = index.get_level_values("date")
    values = pd.DataFrame({"rf": [rate_by_date[date] for date in dates]}, index=index)
    observed_frame = pd.DataFrame(observed, index=index, columns=["rf"])
    enriched = dict(panels)
    enriched["risk_free"] = MaskedPanel(
        values=values,
        observed=observed_frame,
        age=pd.DataFrame(0, index=index, columns=["rf"]),
        segment=panels["loss"].segment,
    )
    return enriched


def _fit_with_risk_free(train_panels: dict[str, MaskedPanel]) -> FmgCtct2Estimator:
    estimator = FmgCtct2Estimator(_estimator_config_with_risk_free(), _objective())  # type: ignore[arg-type]
    estimator._feature_columns = ["x1", "x2"]
    estimator._context_columns = ["c1", "c2"]
    estimator._model = estimator._build_model()
    estimator._feature_ewma = EwmaStandardizer(
        half_life=3.0, history_buffer=estimator.required_history
    )
    estimator._feature_ewma.fit(train_panels["features"])
    return estimator


class FmgCtct2RiskFreeTest(unittest.TestCase):
    def _rates(self) -> dict[pd.Timestamp, float]:
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        return {date: 0.001 * (position + 1) for position, date in enumerate(dates)}

    def test_risk_free_reaches_objective_aligned_per_date(self) -> None:
        rates = self._rates()
        train_panels, _ = _train_predict_split(_with_risk_free(_panels(), rates))
        estimator = _fit_with_risk_free(train_panels)

        raw = estimator._windows(train_panels)
        assert raw.risk_free is not None
        expected = np.array([rates[date] for date in raw.index.get_level_values("date")])
        np.testing.assert_allclose(raw.risk_free.numpy(), expected, rtol=0, atol=1e-7)

        windows = estimator._to_windows(raw, batch_size=2)
        assert windows is not None and windows.risk_free is not None
        output = estimator._forward_batch(windows, windows.batches[0])
        self.assertTrue(bool(torch.isfinite(output.loss).item()))
        output.loss.backward()  # type: ignore[no-untyped-call]

    def test_missing_risk_free_on_a_scored_date_raises(self) -> None:
        rates = self._rates()
        panels = _with_risk_free(_panels(), rates, observed=False)
        train_panels, _ = _train_predict_split(panels)
        estimator = _fit_with_risk_free(train_panels)
        with self.assertRaisesRegex(ValueError, "risk-free .* is missing"):
            estimator._windows(train_panels)


class FmgCtct2ConfigurationTest(unittest.TestCase):
    def test_canonical_hydra_model_is_registered_and_valid(self) -> None:
        config_dir = str(
            (Path(__file__).resolve().parents[2] / "hydra" / "configs" / "training").resolve()
        )
        with initialize_config_dir(config_dir=config_dir, version_base=None):
            cfg = compose(config_name="train", overrides=["experiment=fmg-ctct-2"])

        self.assertEqual(cfg.model.name, "fmg-ctct-2")
        self.assertEqual(model_registry.validate("fmg-ctct-2", cfg), [])

    def test_regression_objective_changes_native_prediction_semantics(self) -> None:
        config_dir = str(
            (Path(__file__).resolve().parents[2] / "hydra" / "configs" / "training").resolve()
        )
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
        assert isinstance(estimator, FmgCtct2Estimator)  # narrow Estimator[Any] for type checking
        self.assertEqual(estimator._prediction_kind, "regression")


if __name__ == "__main__":
    unittest.main()
