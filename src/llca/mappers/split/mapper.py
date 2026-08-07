"""Build temporal split strategies independently from training execution."""

from __future__ import annotations

from typing import Any

from omegaconf import DictConfig

from llca.mappers.modules.registry import Registry
from llca.splitting.single import SingleSplitter
from llca.splitting.splitter import Splitter
from llca.splitting.walk_forward import WalkForwardSplitter

splitter_registry: Registry[Splitter[Any]] = Registry("splitter")


def _resolve_lookback(cfg: DictConfig, default_lookback: int) -> int:
    """Use an explicit ``split.lookback`` when set, else the model's required input history.

    Leaving ``lookback`` unset lets the split default to the estimator's ``required_lookback``
    so temporal models attach exactly their causal window minus one and point-in-time models
    attach none, without a hand-maintained per-model constant. Because the split is
    end-anchored, this affects only history depth, never the scored evaluation dates.
    """
    explicit = cfg.get("lookback")
    return default_lookback if explicit is None else int(explicit)


@splitter_registry.register("walk_forward")
def _build_walk_forward(cfg: DictConfig, *, default_lookback: int = 0) -> WalkForwardSplitter:
    """Map validated rolling-window lengths to a walk-forward strategy."""
    return WalkForwardSplitter(
        train_size=cfg.train_size,
        val_size=cfg.val_size,
        test_size=cfg.test_size,
        purge_size=cfg.purge_size,
        step_size=cfg.step_size,
        lookback=_resolve_lookback(cfg, default_lookback),
    )


@splitter_registry.register("single_split")
def _build_single_split(cfg: DictConfig, *, default_lookback: int = 0) -> SingleSplitter:
    """Map validated window lengths to a single chronological strategy."""
    return SingleSplitter(
        train_size=cfg.train_size,
        val_size=cfg.val_size,
        test_size=cfg.test_size,
        purge_size=cfg.purge_size,
        lookback=_resolve_lookback(cfg, default_lookback),
    )


def build_split(cfg: DictConfig, *, default_lookback: int = 0) -> Splitter[Any]:
    """Construct the registered temporal split strategy from validated settings.

    ``default_lookback`` is the input history to attach when ``split.lookback`` is unset,
    supplied by the caller from the estimator's ``required_lookback``.
    """
    return splitter_registry.build(str(cfg.name), cfg, default_lookback=default_lookback)
