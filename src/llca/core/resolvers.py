from __future__ import annotations

from omegaconf import OmegaConf


def _negate(value: int | float) -> int | float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError("the 'neg' resolver requires an integer or floating-point value")
    return -value


def register_resolvers() -> None:
    """Register project-wide OmegaConf resolvers before Hydra composes a job."""
    if not OmegaConf.has_resolver("neg"):
        OmegaConf.register_new_resolver("neg", _negate)
