"""Validation for temporal train/validation/test split strategies."""

from __future__ import annotations

from omegaconf import DictConfig

from llca.mappers.config_validation import ConfigField, check_fields, register_validator
from llca.mappers.split.mapper import splitter_registry
from llca.mappers.supervision import supervision_forward_horizon

_SPLIT_FIELDS = [
    ConfigField("train_size", "int", positive=True),
    ConfigField("val_size", "int", positive=True),
    ConfigField("test_size", "int", positive=True),
    ConfigField("purge_size", "int", minimum=0),
    ConfigField("lookback", "int", minimum=0, required=False),
]
_STEP_FIELD = ConfigField("step_size", "int", positive=True)


@splitter_registry.register_validator("walk_forward")
def _validate_walk_forward(cfg: DictConfig) -> list[str]:
    return check_fields(cfg, "split", [*_SPLIT_FIELDS, _STEP_FIELD])


@splitter_registry.register_validator("single_split")
def _validate_single_split(cfg: DictConfig) -> list[str]:
    return check_fields(cfg, "split", _SPLIT_FIELDS)


@register_validator
def _validate_split(cfg: DictConfig) -> list[str]:
    """Validate the selected split name and delegate its strategy-specific fields."""
    split = cfg.get("split")
    if not isinstance(split, DictConfig):
        return ["split must be a mapping"]

    errors: list[str] = []
    name = split.get("name")
    if not isinstance(name, str):
        errors.extend(check_fields(split, "split", [ConfigField("name", "str")]))
    elif not splitter_registry.is_registered(name):
        errors.append(
            f"split.name '{name}' is not registered; available: {splitter_registry.available()}"
        )
    else:
        errors.extend(splitter_registry.validate(name, split))
    errors.extend(_validate_purge_horizon(split, cfg))
    return errors


def _validate_purge_horizon(split: DictConfig, cfg: DictConfig) -> list[str]:
    """Require the purge to cover the supervision label's forward horizon.

    Purge guards against the last training label realizing inside the next scored window, so it
    must be at least the target's forward horizon regardless of the model's sequence length or
    feature windows. The current one-step close-to-close return needs a purge of ``1``; a future
    multi-step target raises the requirement without any splitter change.
    """
    horizon = supervision_forward_horizon(cfg)
    purge = split.get("purge_size")
    if (
        horizon is not None
        and isinstance(purge, int)
        and not isinstance(purge, bool)
        and purge < horizon
    ):
        return [
            f"split.purge_size ({purge}) must be >= the supervision label forward horizon "
            f"({horizon}) to prevent train/validation/test target leakage"
        ]
    return []
