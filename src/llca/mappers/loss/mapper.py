from omegaconf import DictConfig
from torch import nn

from llca.mappers.modules.registry import Registry

loss_registry: Registry[nn.Module] = Registry("loss")


def build_loss(cfg: DictConfig) -> nn.Module:
    """Construct a registered differentiable objective from validated Hydra settings."""
    return loss_registry.build(cfg.name, cfg)
