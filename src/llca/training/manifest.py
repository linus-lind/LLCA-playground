"""Build minimal, versioned manifests for reproducible training runs."""

from __future__ import annotations

import base64
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from omegaconf import DictConfig, OmegaConf

from llca.core.paths import PROJECT_ROOT
from llca.training.recovery import source_fingerprint

TRAINING_MANIFEST_SCHEMA_VERSION = 1
INVOCATION_MANIFEST_SCHEMA_VERSION = 1
SOURCE_SNAPSHOT_SCHEMA_VERSION = 1

_TRAINING_SECTIONS = (
    "experiment_name",
    "data",
    "preprocessing",
    "features",
    "masking",
    "loss",
    "model",
    "training",
    "split",
)


def _resolved_mapping(config: Mapping[str, Any] | DictConfig) -> dict[str, Any]:
    if isinstance(config, DictConfig):
        return cast(dict[str, Any], OmegaConf.to_container(config, resolve=True))
    return dict(config)


def build_training_manifest(config: Mapping[str, Any] | DictConfig) -> dict[str, Any]:
    """Select only state that defines or reproduces a trained model.

    Analytics presentation, recovery selection, tracking locations, and Hydra runtime
    internals are invocation concerns and deliberately excluded from the model contract.
    """
    resolved = _resolved_mapping(config)
    missing = [section for section in _TRAINING_SECTIONS if section not in resolved]
    if missing:
        raise ValueError(f"training configuration is missing manifest sections: {missing}")
    return {
        "schema_version": TRAINING_MANIFEST_SCHEMA_VERSION,
        **{section: resolved[section] for section in _TRAINING_SECTIONS},
    }


def build_invocation_manifest(
    *,
    task_overrides: Sequence[str],
    config_choices: Mapping[str, str],
) -> dict[str, Any]:
    """Record user intent in addition to the fully resolved training manifest."""
    return {
        "schema_version": INVOCATION_MANIFEST_SCHEMA_VERSION,
        "task_overrides": list(task_overrides),
        "config_choices": dict(sorted(config_choices.items())),
    }


def build_source_snapshot(source_root: Path | None = None) -> dict[str, Any]:
    """Archive exact package sources so dirty worktrees remain reconstructable."""
    root = source_root or PROJECT_ROOT / "src" / "llca"
    files = {
        path.relative_to(root).as_posix(): base64.b64encode(path.read_bytes()).decode("ascii")
        for path in sorted(root.rglob("*.py"))
    }
    return {
        "schema_version": SOURCE_SNAPSHOT_SCHEMA_VERSION,
        "source_sha256": source_fingerprint(root),
        "content_encoding": "base64",
        "files": files,
    }
