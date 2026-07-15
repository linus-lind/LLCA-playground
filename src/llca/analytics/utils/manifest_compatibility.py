"""Read-only canonicalization for registered manifests used by analytics."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from llca.data.versioning import DATA_MANIFEST_SCHEMA_VERSION, validate_data_manifest
from llca.training.manifests.training import (
    TRAINING_MANIFEST_SCHEMA_VERSION,
    validate_training_manifest,
)

_VERSION_ONLY_SCHEMA = 2
_TRAINING_FIELDS = {
    "schema_version",
    "experiment_name",
    "data",
    "preprocessing",
    "features",
    "masking",
    "loss",
    "model",
    "training",
    "split",
}
_DATA_FIELDS = {"schema_version", "plan", "sources", "datasets"}


def _canonicalize_version_only_revision(
    value: object,
    current_version: int,
    expected_fields: set[str],
) -> object:
    """Map the retired version-only revision only when its complete shape is known."""
    if not isinstance(value, dict) or value.get("schema_version") != _VERSION_ONLY_SCHEMA:
        return value
    if set(value) != expected_fields:
        return value
    canonical = deepcopy(value)
    canonical["schema_version"] = current_version
    return canonical


def canonical_training_manifest(value: object) -> dict[str, Any]:
    """Validate a registered training manifest under the current analytics contract."""
    canonical = _canonicalize_version_only_revision(
        value,
        TRAINING_MANIFEST_SCHEMA_VERSION,
        _TRAINING_FIELDS,
    )
    return validate_training_manifest(canonical)


def canonical_data_manifest(value: object) -> dict[str, Any]:
    """Validate a registered data manifest under the current analytics contract."""
    canonical = _canonicalize_version_only_revision(
        value,
        DATA_MANIFEST_SCHEMA_VERSION,
        _DATA_FIELDS,
    )
    return validate_data_manifest(canonical)
