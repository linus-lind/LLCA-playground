"""Composed-config audit of every shipped experiment's split and nested-CV geometry.

These tests resolve the real Hydra experiments and assert the split contract each model ends up
with: a purge of one observation (the current label horizon), a model-derived input lookback, and
-- because the splitter is end-anchored -- identical scored test windows across models that share
the outer geometry regardless of how much history each attaches.
"""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig

import llca.mappers.loss  # noqa: F401  (register loss builders)
import llca.mappers.model  # noqa: F401  (register model builders/validators)
from llca.data.modules.masked_panel import MaskedPanel, MaskedPanels
from llca.mappers import build_loss, build_model, build_split, validate_config

_CONFIG_DIR = str(
    (Path(__file__).resolve().parents[2] / "hydra" / "configs" / "training").resolve()
)

_FMG_EXPERIMENTS = ("fmg-clstm", "fmg-ctt", "fmg-ctct-1", "fmg-ctct-2")
_BASELINE_EXPERIMENTS = ("elastic-net", "rf", "equal-weight", "inverse-volatility")
_TUNED_EXPERIMENTS = ("elastic-net", "rf")


def _compose(overrides: list[str]) -> DictConfig:
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=_CONFIG_DIR, version_base=None):
        return compose(config_name="train", overrides=overrides)


def _required_lookback(cfg: DictConfig) -> int:
    loss_cfg = cfg.get("loss") if cfg.get("loss") and cfg.loss.get("name") else None
    objective = build_loss(loss_cfg) if loss_cfg is not None else None
    factory = build_model(
        cfg.model,
        loss=objective,
        loss_config=loss_cfg,
        hyperparameter_selection=cfg.get("hyperparameter_selection"),
    )
    return int(factory().required_lookback)


def _panel(n_dates: int) -> MaskedPanels:
    dates = pd.bdate_range("2005-01-01", periods=n_dates)
    index = pd.MultiIndex.from_product([dates, [1]], names=["date", "entity"])
    values = pd.DataFrame(np.arange(len(index), dtype=float), index=index, columns=["f"])
    return {
        "daily_values": MaskedPanel(
            values=values,
            observed=pd.DataFrame(True, index=index, columns=["f"]),
            age=pd.DataFrame(0.0, index=index, columns=["f"]),
            segment=pd.Series(index.get_level_values("entity"), index=index),
        )
    }


class SplitCompositionMatrixTest(unittest.TestCase):
    def test_every_experiment_purges_one_observation(self) -> None:
        for experiment in _FMG_EXPERIMENTS + _BASELINE_EXPERIMENTS:
            cfg = _compose([f"experiment={experiment}"])
            self.assertEqual(cfg.split.purge_size, 1, experiment)
            self.assertIsNone(cfg.split.get("lookback"), experiment)
            validate_config(cfg)  # raises ConfigValidationError on any error

    def test_temporal_models_derive_full_receptive_lookback(self) -> None:
        # sequence_length (252) + CNN buffer (8) - 1 = 259 prior observations.
        for experiment in _FMG_EXPERIMENTS:
            cfg = _compose([f"experiment={experiment}"])
            self.assertEqual(_required_lookback(cfg), 259, experiment)

    def test_point_in_time_baselines_need_no_lookback(self) -> None:
        for experiment in _BASELINE_EXPERIMENTS:
            cfg = _compose([f"experiment={experiment}"])
            self.assertEqual(_required_lookback(cfg), 0, experiment)

    def test_tuned_baselines_resolve_to_a_nonempty_grid(self) -> None:
        for experiment in _TUNED_EXPERIMENTS:
            cfg = _compose([f"experiment={experiment}"])
            self.assertTrue(cfg.hyperparameter_selection.enabled, experiment)
            self.assertEqual(cfg.hyperparameter_selection.search.method, "grid", experiment)
            self.assertGreater(len(cfg.model.search_space), 0, experiment)
            self.assertEqual(cfg.hyperparameter_selection.cv.purge, 1, experiment)


class EndAnchoredComparabilityTest(unittest.TestCase):
    """FMG, elastic-net, and random forest evaluate the same scored dates despite differing
    lookback, because end-anchoring fixes the scored windows to the newest observations."""

    def _scored_windows(self, experiment: str, panel: MaskedPanels) -> tuple[pd.Timestamp, ...]:
        overrides = [
            f"experiment={experiment}",
            "split.train_size=60",
            "split.val_size=12",
            "split.test_size=15",
        ]
        cfg = _compose(overrides)
        splitter = build_split(cfg.split, default_lookback=_required_lookback(cfg))
        fold, _, _ = next(iter(splitter.split(panel, "daily_values")))
        return (
            fold.train_start,
            fold.train_end,
            fold.val_start,
            fold.val_end,
            fold.test_start,
            fold.test_end,
        )

    def test_shared_geometry_yields_identical_scored_windows(self) -> None:
        panel = _panel(160)
        fmg = self._scored_windows("fmg-clstm", panel)
        elastic = self._scored_windows("elastic-net", panel)
        forest = self._scored_windows("rf", panel)
        self.assertEqual(fmg, elastic)
        self.assertEqual(fmg, forest)
        # And the shared test window ends on the newest available observation.
        newest = pd.bdate_range("2005-01-01", periods=160)[-1]
        self.assertEqual(fmg[-1], newest)


if __name__ == "__main__":
    unittest.main()
