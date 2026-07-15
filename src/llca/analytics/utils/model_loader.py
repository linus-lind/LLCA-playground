from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from omegaconf import OmegaConf

from llca.analytics.utils.config import RegisteredModelConfig
from llca.analytics.utils.manifest_compatibility import (
    canonical_data_manifest,
    canonical_training_manifest,
)
from llca.analytics.utils.registered_model_metadata import RegisteredModelMetadata
from llca.core.artifacts import (
    DATA_MANIFEST_ARTIFACT,
    TRAINING_MANIFEST_ARTIFACT,
)
from llca.data.versioning import DataVersioningError
from llca.models.estimators.estimator import Estimator


def _required_date_tag(tags: dict[str, str], name: str) -> pd.Timestamp:
    """Parse a required registry date tag and reject missing or invalid metadata."""
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
    """Read model-version tags cheaply before deciding the common comparison window."""
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
    """Load any estimator logged through LLCA's generic MLflow PyFunc contract.

    Models are intentionally loaded one at a time by the comparison runner so several
    registry versions do not accumulate their neural-network weights on the GPU.
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
