"""Validation for training-run recovery selection."""

from omegaconf import DictConfig

from llca.mappers.config_validation import ConfigField, check_fields, register_validator

_MODES = ("off", "list", "auto", "explicit")
_FIELDS = [
    ConfigField("mode", "str"),
    ConfigField("run_id", "str", required=False),
    ConfigField("allow_source_mismatch", "bool"),
]


@register_validator
def _validate_recovery(cfg: DictConfig) -> list[str]:
    recovery = cfg.get("recovery")
    if not isinstance(recovery, DictConfig):
        return ["recovery must be a mapping"]
    errors = check_fields(recovery, "recovery", _FIELDS)
    mode = recovery.get("mode")
    run_id = recovery.get("run_id")
    if isinstance(mode, str) and mode not in _MODES:
        errors.append(f"recovery.mode '{mode}' must be one of {list(_MODES)}")
    if mode == "explicit" and not isinstance(run_id, str):
        errors.append("recovery.run_id is required when recovery.mode is 'explicit'")
    if mode in {"off", "list", "auto"} and run_id is not None:
        errors.append(f"recovery.run_id must be null when recovery.mode is '{mode}'")
    return errors
