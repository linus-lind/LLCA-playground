TRAINING_MANIFEST_ARTIFACT = "pipeline/training_manifest.json"
"""Canonical, resolved training contract attached to every model-producing run."""

DATA_MANIFEST_ARTIFACT = "pipeline/data_manifest.json"
"""Restorable raw-data pointers and processed-panel fingerprints for one run."""

INVOCATION_MANIFEST_ARTIFACT = "pipeline/hydra_invocation.json"
"""Hydra choices and task overrides that produced the resolved training contract."""

SOURCE_SNAPSHOT_ARTIFACT = "pipeline/source_snapshot.json"
"""Exact executable Python source files used by a model-producing run."""

LEGACY_PIPELINE_CONFIG_ARTIFACT = "pipeline/config.json"
"""Pre-manifest artifact path retained exclusively for loading existing runs."""

PIPELINE_CONFIG_ARTIFACT = TRAINING_MANIFEST_ARTIFACT
"""Backward-compatible symbol for the current canonical training artifact path."""
