"""Map Hydra recovery policy into a typed runtime configuration."""

from typing import cast

from omegaconf import DictConfig

from llca.training.modules.recovery_config import RecoveryConfig, RecoveryMode

_MODES = ("off", "list", "auto", "explicit")


def build_recovery(cfg: DictConfig) -> RecoveryConfig:
    """Build a recovery policy and reject unsafe ambiguous combinations early."""
    mode = cfg.get("mode")
    if not isinstance(mode, str) or mode not in _MODES:
        raise ValueError(f"recovery.mode must be one of {list(_MODES)}, got {mode!r}")
    run_id = cfg.get("run_id")
    if run_id is not None and not isinstance(run_id, str):
        raise ValueError("recovery.run_id must be a string or null")
    if mode == "explicit" and not run_id:
        raise ValueError("recovery.run_id is required when recovery.mode is 'explicit'")
    if mode != "explicit" and run_id is not None:
        raise ValueError(f"recovery.run_id must be null when recovery.mode is '{mode}'")
    allow_source_mismatch = cfg.get("allow_source_mismatch")
    if not isinstance(allow_source_mismatch, bool):
        raise ValueError("recovery.allow_source_mismatch must be a boolean")
    return RecoveryConfig(
        mode=cast(RecoveryMode, mode),
        run_id=run_id,
        allow_source_mismatch=allow_source_mismatch,
    )
