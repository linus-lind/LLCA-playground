from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from uuid import uuid4

import mlflow
import torch

_LATEST = "latest.pt"
_BEST = "best.pt"
_ARTIFACT_PATH = "checkpoints"
_REQUIRED_TRAINING_KEYS = (
    "model_state_dict",
    "optimizer_state_dict",
    "optimizer_name",
    "epoch",
    "best_val",
    "best_state",
    "epochs_without_improvement",
)


class CheckpointValidationError(ValueError):
    """Raised when a checkpoint cannot represent a resumable training state."""


def validate_training_checkpoint(
    payload: object, *, source: str | Path | None = None
) -> dict[str, Any]:
    """Validate the model-independent portion of a resumable checkpoint schema."""
    label = f"checkpoint '{source}'" if source is not None else "checkpoint"
    if not isinstance(payload, dict):
        raise CheckpointValidationError(f"{label} must contain a mapping")
    missing = [key for key in _REQUIRED_TRAINING_KEYS if key not in payload]
    if missing:
        raise CheckpointValidationError(f"{label} is missing required keys: {missing}")
    if not isinstance(payload["model_state_dict"], dict):
        raise CheckpointValidationError(f"{label} model_state_dict must be a mapping")
    if not isinstance(payload["optimizer_state_dict"], dict):
        raise CheckpointValidationError(f"{label} optimizer_state_dict must be a mapping")
    optimizer_name = payload["optimizer_name"]
    if not isinstance(optimizer_name, str) or not optimizer_name:
        raise CheckpointValidationError(f"{label} optimizer_name must be a non-empty string")
    epoch = payload["epoch"]
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
        raise CheckpointValidationError(f"{label} epoch must be a non-negative integer")
    without_improvement = payload["epochs_without_improvement"]
    if (
        isinstance(without_improvement, bool)
        or not isinstance(without_improvement, int)
        or without_improvement < 0
    ):
        raise CheckpointValidationError(
            f"{label} epochs_without_improvement must be a non-negative integer"
        )
    if payload["best_state"] is not None and not isinstance(payload["best_state"], dict):
        raise CheckpointValidationError(f"{label} best_state must be a mapping or null")
    try:
        float(payload["best_val"])
    except (TypeError, ValueError) as exc:
        raise CheckpointValidationError(f"{label} best_val must be numeric") from exc
    return payload


class Checkpointer:
    """Persist latest and best resumable training states in a run-specific directory.

    Checkpoints are local PyTorch payloads and may optionally be mirrored as MLflow
    artifacts when a run is active. Their schema is owned by the estimator's checkpoint
    state factory, keeping this service independent of model architecture.
    """

    def __init__(self, directory: str | Path, *, log_to_mlflow: bool = True) -> None:
        self._directory = Path(directory)
        self._directory.mkdir(parents=True, exist_ok=True)
        self._log_to_mlflow = log_to_mlflow

    @property
    def directory(self) -> Path:
        return self._directory

    @property
    def latest_path(self) -> Path:
        return self._directory / _LATEST

    @property
    def best_path(self) -> Path:
        return self._directory / _BEST

    def has_latest(self) -> bool:
        return self.latest_path.exists()

    def save_latest(self, payload: dict[str, Any]) -> Path:
        return self._write(self.latest_path, payload)

    def save_best(self, payload: dict[str, Any]) -> Path:
        return self._write(self.best_path, payload)

    def load_latest(self, *, map_location: str | torch.device) -> dict[str, Any]:
        """Load and validate the latest state, failing rather than silently restarting."""
        if not self.latest_path.exists():
            raise FileNotFoundError(f"resume checkpoint does not exist: {self.latest_path}")
        payload = torch.load(self.latest_path, map_location=map_location, weights_only=False)
        return validate_training_checkpoint(payload, source=self.latest_path)

    def _write(self, path: Path, payload: dict[str, Any]) -> Path:
        """Atomically replace one checkpoint and mirror it to the active MLflow run."""
        temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        try:
            with temporary.open("wb") as handle:
                torch.save(payload, handle)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        if self._log_to_mlflow and mlflow.active_run() is not None:
            mlflow.log_artifact(str(path), artifact_path=_ARTIFACT_PATH)
        return path
