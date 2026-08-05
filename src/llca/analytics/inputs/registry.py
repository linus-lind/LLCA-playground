"""Read-only access to registered models and their canonical training manifests."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from omegaconf import OmegaConf

from llca.analytics.modules.analytics_config import RegisteredModelConfig
from llca.analytics.modules.registered_model import RegisteredModelMetadata
from llca.core.artifacts import (
    DATA_MANIFEST_ARTIFACT,
    TRAINING_MANIFEST_ARTIFACT,
)
from llca.core.provenance.training_manifest import (
    TRAINING_MANIFEST_SCHEMA_VERSION,
    validate_training_manifest,
)
from llca.data.versioning import (
    DATA_MANIFEST_SCHEMA_VERSION,
    DataVersioningError,
    validate_data_manifest,
)
from llca.models.estimators.estimator import Estimator

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
    """Upgrade a legacy version-only manifest revision to the current schema version.

    Only rewrites the schema version when ``value`` is exactly the retired revision and carries
    precisely ``expected_fields``; anything else is returned untouched so unknown shapes are left
    for the validator to reject.
    """
    if not isinstance(value, dict) or value.get("schema_version") != _VERSION_ONLY_SCHEMA:
        return value
    if set(value) != expected_fields:
        return value
    canonical = deepcopy(value)
    canonical["schema_version"] = current_version
    return canonical


def canonical_training_manifest(value: object) -> dict[str, Any]:
    """Upgrade and validate a stored training manifest, returning its canonical form."""
    canonical = _canonicalize_version_only_revision(
        value,
        TRAINING_MANIFEST_SCHEMA_VERSION,
        _TRAINING_FIELDS,
    )
    return validate_training_manifest(canonical)


def canonical_data_manifest(value: object) -> dict[str, Any]:
    """Upgrade and validate a stored data manifest, returning its canonical form."""
    canonical = _canonicalize_version_only_revision(
        value,
        DATA_MANIFEST_SCHEMA_VERSION,
        _DATA_FIELDS,
    )
    return validate_data_manifest(canonical)


def _required_date_tag(tags: dict[str, str], name: str) -> pd.Timestamp:
    """Parse the timestamp stored in registry tag ``name``.

    Raises ``ValueError`` when the tag is missing or does not parse to a valid date.
    """
    value = tags.get(name)
    if value is None:
        raise ValueError(f"registered model version is missing required '{name}' tag")
    parsed = pd.Timestamp(value)
    if pd.isna(parsed):
        raise ValueError(f"registered model tag '{name}' is not a valid date: {value!r}")
    return parsed


def get_registered_model_metadata(
    config: RegisteredModelConfig,
    tracking_uri: str,
) -> RegisteredModelMetadata:
    """Fetch a registered model version's metadata without loading the model itself.

    Reads the version's tags and downloads and validates its training and data manifests,
    returning them with the resolved test window and model URI. Raises ``ValueError`` if a
    manifest is invalid or a required date tag is missing. Kept lightweight so the comparison
    window can be decided before any weights are loaded.
    """
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()
    version = client.get_model_version(config.name, str(config.version))
    tags = dict(version.tags or {})
    try:
        local_config = client.download_artifacts(str(version.run_id), TRAINING_MANIFEST_ARTIFACT)
        loaded_config = json.loads(Path(local_config).read_text(encoding="utf-8"))
        pipeline_config = OmegaConf.create(canonical_training_manifest(loaded_config))
    except (MlflowException, FileNotFoundError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(
            f"registered model {config.name}/{config.version} has an invalid canonical "
            f"training manifest: {exc}"
        ) from exc
    try:
        local_manifest = client.download_artifacts(str(version.run_id), DATA_MANIFEST_ARTIFACT)
        loaded_manifest = json.loads(Path(local_manifest).read_text(encoding="utf-8"))
        data_manifest = canonical_data_manifest(loaded_manifest)
    except (
        MlflowException,
        FileNotFoundError,
        json.JSONDecodeError,
        DataVersioningError,
    ) as exc:
        raise ValueError(
            f"registered model {config.name}/{config.version} has an invalid canonical "
            f"data manifest: {exc}"
        ) from exc
    return RegisteredModelMetadata(
        config=config,
        run_id=str(version.run_id),
        model_uri=f"models:/{config.name}/{config.version}",
        test_start=_required_date_tag(tags, "test_start"),
        test_end=_required_date_tag(tags, "test_end"),
        pipeline_config=pipeline_config,
        data_manifest=data_manifest,
    )


def load_registered_estimator(
    metadata: RegisteredModelMetadata,
    device_name: str,
) -> Estimator[Any]:
    """Load a registered model's estimator from MLflow and place it on ``device_name``.

    Unwraps the PyFunc model to recover the LLCA estimator and sets its inference device. Raises
    ``ValueError`` if the model was not logged through the Estimator/PyFunc contract.
    """
    loaded_model = mlflow.pyfunc.load_model(metadata.model_uri)
    python_model = loaded_model.unwrap_python_model()
    estimator = getattr(python_model, "estimator", None)
    if not isinstance(estimator, Estimator):
        raise ValueError(
            f"registered model {metadata.model_uri} was not logged through the LLCA "
            "Estimator/Pyfunc contract"
        )
    estimator.set_inference_device(device_name)
    return estimator
