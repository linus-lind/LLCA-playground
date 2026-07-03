"""Validation for model-independent training runtime configuration."""

import torch
from omegaconf import DictConfig

from llca.mappers.config_validation import ConfigField, check_fields, register_validator
from llca.mappers.training.mapper import optimizer_registry

_TRAINING_FIELDS = [
    ConfigField("seed", "int", minimum=0, maximum=2**32),
    ConfigField("deterministic", "bool"),
    ConfigField("epochs", "int", positive=True),
    ConfigField("batch_size", "int", positive=True),
    ConfigField("grad_clip", "number", positive=True),
    ConfigField("device", "str"),
    ConfigField("precision", "str"),
    ConfigField("gradient_checkpointing", "bool"),
    ConfigField("optimizer", "mapping"),
    ConfigField("early_stopping", "mapping"),
    ConfigField("diagnostics", "mapping"),
]
_ADAM_FIELDS = [
    ConfigField("learning_rate", "number", positive=True),
    ConfigField("weight_decay", "number", minimum=0.0),
    ConfigField("fused", "bool"),
]
_EARLY_STOPPING_FIELDS = [
    ConfigField("patience", "int", minimum=0),
    ConfigField("min_delta", "number", minimum=0.0),
]
_DIAGNOSTIC_FIELDS = [
    ConfigField("interval", "int", positive=True),
    ConfigField("component_gradient_norms", "bool"),
    ConfigField("parameter_update_norms", "bool"),
]
_PRECISIONS = ("bf16", "fp32")


@optimizer_registry.register_validator("adam")
@optimizer_registry.register_validator("adamw")
def _validate_adam_family(cfg: DictConfig) -> list[str]:
    return check_fields(cfg, "training.optimizer", _ADAM_FIELDS)


@register_validator
def _validate_training(cfg: DictConfig) -> list[str]:
    """Validate generic execution settings and delegate optimizer-specific fields."""
    training = cfg.get("training")
    if not isinstance(training, DictConfig):
        return ["training must be a mapping"]

    errors = check_fields(training, "training", _TRAINING_FIELDS)
    precision = training.get("precision")
    if isinstance(precision, str) and precision not in _PRECISIONS:
        errors.append(f"training.precision '{precision}' must be one of {list(_PRECISIONS)}")

    device = training.get("device")
    if isinstance(device, str) and device != "auto":
        try:
            torch.device(device)
        except (RuntimeError, ValueError):
            errors.append(f"training.device '{device}' must be 'auto' or a valid PyTorch device")

    optimizer = training.get("optimizer")
    if isinstance(optimizer, DictConfig):
        errors.extend(check_fields(optimizer, "training.optimizer", [ConfigField("name", "str")]))
        name = optimizer.get("name")
        if isinstance(name, str):
            if not optimizer_registry.is_registered(name):
                errors.append(
                    f"training.optimizer.name '{name}' is not registered; "
                    f"available: {optimizer_registry.available()}"
                )
            else:
                errors.extend(optimizer_registry.validate(name, optimizer))

    early_stopping = training.get("early_stopping")
    if isinstance(early_stopping, DictConfig):
        errors.extend(
            check_fields(early_stopping, "training.early_stopping", _EARLY_STOPPING_FIELDS)
        )
    diagnostics = training.get("diagnostics")
    if isinstance(diagnostics, DictConfig):
        errors.extend(
            check_fields(
                diagnostics,
                "training.diagnostics",
                _DIAGNOSTIC_FIELDS,
            )
        )
    return errors
