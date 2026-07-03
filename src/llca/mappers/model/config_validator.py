from omegaconf import DictConfig

from llca.mappers.config_validation import register_validator
from llca.mappers.model.mapper import model_registry


@register_validator
def _validate_model(cfg: DictConfig) -> list[str]:
    name = cfg.model.name
    if not model_registry.is_registered(name):
        return [f"model.name '{name}' is not registered"]
    return model_registry.validate(name, cfg)
