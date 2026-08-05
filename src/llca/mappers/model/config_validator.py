from __future__ import annotations

from omegaconf import DictConfig

from llca.mappers.config_validation import register_validator
from llca.mappers.loss.mapper import objective_kind
from llca.mappers.model.mapper import model_capabilities, model_registry
from llca.pipeline.contracts import TrainingEngine


@register_validator
def _validate_model(cfg: DictConfig) -> list[str]:
    name = cfg.model.name
    if not model_registry.is_registered(name):
        return [f"model.name '{name}' is not registered"]
    errors = model_registry.validate(name, cfg)
    capabilities = model_capabilities(str(name))

    loss = cfg.get("loss")
    loss_name = loss.get("name") if isinstance(loss, DictConfig) else None
    if isinstance(loss_name, str):
        try:
            kind = objective_kind(loss_name)
        except KeyError:
            pass
        else:
            if kind not in capabilities.objective_kinds:
                errors.append(
                    f"model '{name}' does not support objective kind '{kind}'; supported: "
                    f"{sorted(capabilities.objective_kinds)}"
                )

    training = cfg.get("training")
    engine_name = training.get("name") if isinstance(training, DictConfig) else None
    if isinstance(engine_name, str):
        try:
            engine = TrainingEngine(engine_name)
        except ValueError:
            pass
        else:
            if engine not in capabilities.training_engines:
                errors.append(
                    f"model '{name}' does not support training engine '{engine}'; supported: "
                    f"{sorted(capabilities.training_engines)}"
                )
    return errors
