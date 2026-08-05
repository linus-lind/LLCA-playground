"""Shared provenance and audit-document contracts used by training and analytics."""

from llca.core.provenance.environment import build_environment_manifest
from llca.core.provenance.source import (
    SOURCE_FINGERPRINT_TAG,
    build_source_snapshot,
    source_fingerprint,
)
from llca.core.provenance.training_manifest import (
    TRAINING_MANIFEST_SCHEMA_VERSION,
    build_training_manifest,
    validate_training_manifest,
)

__all__ = [
    "SOURCE_FINGERPRINT_TAG",
    "TRAINING_MANIFEST_SCHEMA_VERSION",
    "build_environment_manifest",
    "build_source_snapshot",
    "build_training_manifest",
    "source_fingerprint",
    "validate_training_manifest",
]
