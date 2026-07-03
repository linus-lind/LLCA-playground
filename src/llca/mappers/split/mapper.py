"""Build temporal split strategies independently from training execution."""

from omegaconf import DictConfig

from llca.mappers.modules.registry import Registry
from llca.splitting.single import SingleSplitter
from llca.splitting.splitter import Splitter
from llca.splitting.walk_forward import WalkForwardSplitter

splitter_registry: Registry[Splitter] = Registry("splitter")


@splitter_registry.register("walk_forward")
def _build_walk_forward(cfg: DictConfig) -> WalkForwardSplitter:
    """Map validated rolling-window lengths to a walk-forward strategy."""
    return WalkForwardSplitter(
        train_size=cfg.train_size,
        val_size=cfg.val_size,
        test_size=cfg.test_size,
        purge_size=cfg.purge_size,
        step_size=cfg.step_size,
        lookback=cfg.lookback,
    )


@splitter_registry.register("single_split")
def _build_single_split(cfg: DictConfig) -> SingleSplitter:
    """Map validated window lengths to a single chronological strategy."""
    return SingleSplitter(
        train_size=cfg.train_size,
        val_size=cfg.val_size,
        test_size=cfg.test_size,
        purge_size=cfg.purge_size,
        lookback=cfg.lookback,
    )


def build_split(cfg: DictConfig) -> Splitter:
    """Construct the registered temporal split strategy from validated settings."""
    return splitter_registry.build(str(cfg.name), cfg)
