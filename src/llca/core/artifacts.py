TRAINING_MANIFEST_ARTIFACT = "pipeline/training_manifest.json"
"""Canonical, resolved training contract attached to every model-producing run."""

DATA_MANIFEST_ARTIFACT = "pipeline/data_manifest.json"
"""Restorable raw-data pointers and processed-panel fingerprints for one run."""

INVOCATION_MANIFEST_ARTIFACT = "pipeline/hydra_invocation.json"
"""Hydra choices and task overrides that produced the resolved training contract."""

SOURCE_SNAPSHOT_ARTIFACT = "pipeline/source_snapshot.json"
"""Exact executable Python source files used by a model-producing run."""

ENVIRONMENT_MANIFEST_ARTIFACT = "pipeline/environment_manifest.json"
"""Python, package, operating-system, and accelerator versions used by a run."""
