"""MLflow logging contract for immutable run manifests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mlflow

from llca.core.artifacts import (
    DATA_MANIFEST_ARTIFACT,
    ENVIRONMENT_MANIFEST_ARTIFACT,
    INVOCATION_MANIFEST_ARTIFACT,
    SOURCE_SNAPSHOT_ARTIFACT,
    TRAINING_MANIFEST_ARTIFACT,
)


@dataclass(frozen=True, slots=True)
class RunManifests:
    """Immutable model, data, invocation, source, and environment evidence for one run."""

    training: dict[str, Any]
    data: dict[str, Any] | None = None
    invocation: dict[str, Any] | None = None
    source: dict[str, Any] | None = None
    environment: dict[str, Any] | None = None

    def log(
        self,
        *,
        include_invocation: bool = True,
        include_source: bool = True,
        include_environment: bool = True,
    ) -> None:
        """Attach canonical JSON documents to the currently active MLflow run."""
        mlflow.log_dict(self.training, TRAINING_MANIFEST_ARTIFACT)
        if self.data is not None:
            mlflow.log_dict(self.data, DATA_MANIFEST_ARTIFACT)
        if include_invocation and self.invocation is not None:
            mlflow.log_dict(self.invocation, INVOCATION_MANIFEST_ARTIFACT)
        if include_source and self.source is not None:
            mlflow.log_dict(self.source, SOURCE_SNAPSHOT_ARTIFACT)
        if include_environment and self.environment is not None:
            mlflow.log_dict(self.environment, ENVIRONMENT_MANIFEST_ARTIFACT)
