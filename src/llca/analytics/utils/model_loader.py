from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import mlflow
import pandas as pd
import torch
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from omegaconf import DictConfig, OmegaConf

from llca.analytics.utils.config import RegisteredModelConfig
from llca.analytics.utils.registered_model_metadata import RegisteredModelMetadata
from llca.core.artifacts import (
    LEGACY_PIPELINE_CONFIG_ARTIFACT,
    TRAINING_MANIFEST_ARTIFACT,
)
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
    fallback_config: DictConfig | None = None,
) -> RegisteredModelMetadata:
    """Read model-version tags cheaply before deciding the common comparison window."""
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()
    version = client.get_model_version(config.name, str(config.version))
    tags = dict(version.tags or {})
    configured_artifact = tags.get("training_manifest_artifact") or tags.get(
        "pipeline_config_artifact"
    )
    artifacts = tuple(
        dict.fromkeys(
            filter(
                None,
                (
                    configured_artifact,
                    TRAINING_MANIFEST_ARTIFACT,
                    LEGACY_PIPELINE_CONFIG_ARTIFACT,
                ),
            )
        )
    )
    pipeline_config: DictConfig | None = None
    for artifact in artifacts:
        try:
            local_config = client.download_artifacts(str(version.run_id), artifact)
            pipeline_config = cast(
                DictConfig,
                OmegaConf.create(json.loads(Path(local_config).read_text(encoding="utf-8"))),
            )
            break
        except (MlflowException, FileNotFoundError, json.JSONDecodeError):
            continue
    if pipeline_config is None:
        if fallback_config is None:
            raise ValueError(
                f"registered model {config.name}/{config.version} has no readable "
                "training manifest artifact"
            ) from None
        pipeline_config = cast(
            DictConfig,
            OmegaConf.create(OmegaConf.to_container(fallback_config, resolve=True)),
        )
    assert pipeline_config is not None
    return RegisteredModelMetadata(
        config=config,
        run_id=str(version.run_id),
        model_uri=f"models:/{config.name}/{config.version}",
        test_start=_required_date_tag(tags, "test_start"),
        test_end=_required_date_tag(tags, "test_end"),
        pipeline_config=pipeline_config,
    )


def load_registered_estimator(
    metadata: RegisteredModelMetadata,
    device_name: str,
) -> Estimator:
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
    resolved_device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu") if device_name == "auto" else device_name
    )
    estimator.to_device(resolved_device)
    return estimator
