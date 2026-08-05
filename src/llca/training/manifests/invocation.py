"""Hydra invocation evidence kept separate from the model contract."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

INVOCATION_MANIFEST_SCHEMA_VERSION = 1


def build_invocation_manifest(
    *,
    task_overrides: Sequence[str],
    config_choices: Mapping[str, str],
) -> dict[str, Any]:
    """Record user overrides and selected Hydra config groups."""
    return {
        "schema_version": INVOCATION_MANIFEST_SCHEMA_VERSION,
        "task_overrides": list(task_overrides),
        "config_choices": dict(sorted(config_choices.items())),
    }
