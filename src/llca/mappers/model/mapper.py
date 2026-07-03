from collections.abc import Callable

from omegaconf import DictConfig
from torch import nn

from llca.mappers.modules.registry import Registry
from llca.models.estimators.estimator import Estimator

EstimatorFactory = Callable[[], Estimator]
model_registry: Registry[EstimatorFactory] = Registry("model")


def build_model(cfg: DictConfig, *, loss: nn.Module | None = None) -> EstimatorFactory:
    """Build a fresh-estimator factory from a registered model configuration."""
    return model_registry.build(cfg.name, cfg, loss=loss)
