"""Discover, validate, and materialize resumable MLflow training runs."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import socket
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

import mlflow
import torch
from mlflow import MlflowClient
from mlflow.entities import Run
from mlflow.exceptions import MlflowException
from omegaconf import DictConfig, OmegaConf

from llca.core.artifacts import (
    DATA_MANIFEST_ARTIFACT,
    LEGACY_PIPELINE_CONFIG_ARTIFACT,
    TRAINING_MANIFEST_ARTIFACT,
)
from llca.core.paths import CHECKPOINTS_DIR, PROJECT_ROOT
from llca.training.checkpointer import (
    checkpoint_optimizer_name,
    validate_training_checkpoint,
)
from llca.training.modules.recovery_config import RecoveryConfig

RUN_KIND_TAG = "llca.run_kind"
RUN_KIND_PARENT = "cross_validation"
RUN_KIND_FOLD = "fold"
RUN_PHASE_TAG = "llca.run_phase"
RUN_PHASE_PREPARED = "prepared"
RUN_PHASE_TRAINING = "training"
RUN_PHASE_TRAINED = "trained"
RUN_PHASE_MODEL_LOGGED = "model_logged"
RUN_PHASE_REGISTERED = "registered"
RUN_PHASE_COMPLETED = "completed"
PIPELINE_FINGERPRINT_TAG = "llca.pipeline_sha256"
SOURCE_FINGERPRINT_TAG = "llca.source_sha256"
LOGGED_MODEL_URI_TAG = "llca.logged_model_uri"
REGISTERED_MODEL_VERSION_TAG = "llca.registered_model_version"

_LATEST_ARTIFACT = "checkpoints/latest.pt"
_DATA_TAG_PREFIXES = (
    "llca.data_manifest_sha256",
    "raw_data_md5_",
    "processed_data_md5_",
    "raw_data_sha256_",
    "raw_data_dvc_",
    "processed_data_sha256_",
)


class RecoveryError(RuntimeError):
    """Raised when an interrupted run cannot be selected or resumed safely."""


@dataclass(frozen=True, slots=True)
class ResumeCandidate:
    """Cheap MLflow metadata describing one unfinished fold run."""

    run_id: str
    parent_run_id: str
    status: str
    phase: str
    fold_index: int | None
    completed_epochs: int | None
    best_validation_loss: float | None
    start_time: int
    checkpoint_available: bool
    config_available: bool
    issues: tuple[str, ...]

    @property
    def resumable(self) -> bool:
        return self.checkpoint_available and self.config_available and not self.issues


@dataclass(frozen=True, slots=True)
class RecoverySelection:
    """One explicitly resolved fold and its containing cross-validation run."""

    candidate: ResumeCandidate
    pipeline_config: dict[str, Any]

    @property
    def run_id(self) -> str:
        return self.candidate.run_id

    @property
    def parent_run_id(self) -> str:
        return self.candidate.parent_run_id


def pipeline_fingerprint(config: Mapping[str, Any] | DictConfig) -> str:
    """Hash the resolved training pipeline while excluding the recovery invocation."""
    if isinstance(config, DictConfig):
        raw = cast(dict[str, Any], OmegaConf.to_container(config, resolve=True))
    else:
        raw = dict(config)
    canonical = dict(raw)
    canonical.pop("recovery", None)
    encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def source_fingerprint(source_root: Path | None = None) -> str:
    """Hash executable package sources to reject silent code changes on future resumes."""
    root = source_root or PROJECT_ROOT / "src" / "llca"
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def configuration_differences(
    current: Mapping[str, Any] | DictConfig,
    stored: Mapping[str, Any] | DictConfig,
) -> tuple[str, ...]:
    """Describe current values ignored in favor of the selected run configuration."""

    def plain(value: Mapping[str, Any] | DictConfig) -> dict[str, Any]:
        if isinstance(value, DictConfig):
            return cast(dict[str, Any], OmegaConf.to_container(value, resolve=True))
        return dict(value)

    left = plain(current)
    right = plain(stored)
    if "schema_version" in right:
        left = {
            key: (right[key] if key == "schema_version" else left[key])
            for key in right
            if key == "schema_version" or key in left
        }
    for ignored in ("recovery", "mlflow_tracking_uri"):
        left.pop(ignored, None)
        right.pop(ignored, None)

    differences: list[str] = []

    def walk(a: object, b: object, prefix: str) -> None:
        if isinstance(a, dict) and isinstance(b, dict):
            for key in sorted(set(a) | set(b)):
                path = f"{prefix}.{key}" if prefix else str(key)
                if key not in a:
                    differences.append(f"{path}: current=<missing>, stored={b[key]!r}")
                elif key not in b:
                    differences.append(f"{path}: current={a[key]!r}, stored=<missing>")
                else:
                    walk(a[key], b[key], path)
        elif a != b:
            differences.append(f"{prefix}: current={a!r}, stored={b!r}")

    walk(left, right, "")
    return tuple(differences)


class RunLock:
    """Hold a non-blocking OS lock so two processes cannot resume one fold concurrently."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._handle: Any | None = None

    def __enter__(self) -> RunLock:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.touch(exist_ok=True)
        handle = self._path.open("r+b")
        if self._path.stat().st_size == 0:
            handle.write(b"\0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(  # type: ignore[attr-defined]
                    handle.fileno(),
                    fcntl.LOCK_EX | fcntl.LOCK_NB,  # type: ignore[attr-defined]
                )
        except OSError as exc:
            handle.close()
            raise RecoveryError(
                f"run lock '{self._path}' is already held; another process may be training"
            ) from exc
        owner = f"pid={os.getpid()} host={socket.gethostname()}".encode()
        handle.seek(1)
        handle.truncate()
        handle.write(owner)
        handle.flush()
        self._handle = handle
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        handle = self._handle
        if handle is None:
            return
        handle.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(  # type: ignore[attr-defined]
                handle.fileno(), fcntl.LOCK_UN  # type: ignore[attr-defined]
            )
        handle.close()
        self._handle = None


class RecoveryService:
    """Resolve recovery policy against MLflow without touching training data."""

    def __init__(
        self,
        tracking_uri: str,
        experiment_name: str,
        *,
        checkpoint_root: str | Path = CHECKPOINTS_DIR,
    ) -> None:
        mlflow.set_tracking_uri(tracking_uri)
        self._client = MlflowClient()
        self._experiment_name = experiment_name
        self._checkpoint_root = Path(checkpoint_root)

    def list_candidates(self) -> tuple[ResumeCandidate, ...]:
        """Return every unfinished fold, including non-resumable entries with reasons."""
        experiment = self._client.get_experiment_by_name(self._experiment_name)
        if experiment is None:
            return ()
        runs = self._client.search_runs(
            [experiment.experiment_id],
            max_results=1000,
            order_by=["attributes.start_time DESC"],
        )
        candidates = [
            candidate
            for run in runs
            if (candidate := self._candidate_for_run(run)) is not None
        ]
        return tuple(candidates)

    def resolve(self, config: RecoveryConfig) -> RecoverySelection | None:
        """Apply off/list/auto/explicit selection semantics and load stored configuration."""
        if config.mode == "off":
            return None
        candidates = self.list_candidates()
        if config.mode == "list":
            print(self.format_candidates(candidates))
            return None
        if config.mode == "auto":
            eligible = tuple(candidate for candidate in candidates if candidate.resumable)
            if len(eligible) != 1:
                table = self.format_candidates(candidates)
                raise RecoveryError(
                    "recovery.mode='auto' requires exactly one resumable run, "
                    f"found {len(eligible)}\n{table}"
                )
            candidate = eligible[0]
        else:
            assert config.run_id is not None
            candidate = self._resolve_explicit(config.run_id, candidates)
        if not candidate.resumable:
            details = "; ".join(candidate.issues) or "required artifacts are unavailable"
            raise RecoveryError(f"run {candidate.run_id} is not resumable: {details}")
        stored = self._load_pipeline_config(candidate.run_id)
        return RecoverySelection(candidate=candidate, pipeline_config=stored)

    def effective_config(
        self,
        current: DictConfig,
        selection: RecoverySelection,
        recovery: RecoveryConfig,
        tracking_uri: str,
    ) -> DictConfig:
        """Rebuild the pipeline from stored values while retaining invocation-only policy."""
        payload = dict(selection.pipeline_config)
        payload["mlflow_tracking_uri"] = tracking_uri
        payload["recovery"] = {
            "mode": recovery.mode,
            "run_id": recovery.run_id,
            "allow_source_mismatch": recovery.allow_source_mismatch,
        }
        effective = OmegaConf.create(payload)
        differences = configuration_differences(current, selection.pipeline_config)
        if differences:
            print("Stored run configuration overrides current Hydra values:")
            for difference in differences[:20]:
                print(f"  - {difference}")
            if len(differences) > 20:
                print(f"  - ... {len(differences) - 20} additional differences")
        return effective

    def preflight_checkpoint(self, selection: RecoverySelection) -> dict[str, Any]:
        """Materialize and deeply validate the selected training checkpoint on CPU."""
        path = self._materialize_artifact(selection.run_id, _LATEST_ARTIFACT)
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        payload = validate_training_checkpoint(checkpoint, source=path)
        model_config = selection.pipeline_config.get("model")
        if payload.get("config") != model_config:
            raise RecoveryError(
                "checkpoint model configuration differs from the training manifest; "
                "the artifacts do not describe one reproducible run"
            )
        training_config = selection.pipeline_config.get("training")
        expected_optimizer = None
        if isinstance(training_config, dict):
            optimizer = training_config.get("optimizer")
            if isinstance(optimizer, dict):
                expected_optimizer = optimizer.get("name")
        actual_optimizer = checkpoint_optimizer_name(payload)
        if expected_optimizer is not None and actual_optimizer != expected_optimizer:
            raise RecoveryError(
                f"checkpoint optimizer {actual_optimizer!r} differs from stored "
                f"configuration {expected_optimizer!r}"
            )
        stored_fingerprint = self._client.get_run(selection.run_id).data.tags.get(
            PIPELINE_FINGERPRINT_TAG
        )
        actual_fingerprint = pipeline_fingerprint(selection.pipeline_config)
        if stored_fingerprint is not None and stored_fingerprint != actual_fingerprint:
            raise RecoveryError("stored pipeline fingerprint does not match training manifest")
        data_fingerprint = self._client.get_run(selection.run_id).data.tags.get(
            "llca.data_manifest_sha256"
        )
        if data_fingerprint is not None:
            try:
                data_path = self._client.download_artifacts(
                    selection.run_id, DATA_MANIFEST_ARTIFACT
                )
                data_manifest = json.loads(Path(data_path).read_text(encoding="utf-8"))
                encoded = json.dumps(
                    data_manifest,
                    sort_keys=True,
                    separators=(",", ":"),
                    default=str,
                ).encode("utf-8")
            except (MlflowException, FileNotFoundError, json.JSONDecodeError) as exc:
                raise RecoveryError("run has no readable data manifest") from exc
            if hashlib.sha256(encoded).hexdigest() != data_fingerprint:
                raise RecoveryError("stored data manifest fingerprint does not match artifact")
        return payload

    def validate_provenance(
        self,
        selection: RecoverySelection,
        current_tags: Mapping[str, str],
        *,
        allow_source_mismatch: bool,
    ) -> None:
        """Require unchanged data and, by default, unchanged source provenance."""
        stored = self._client.get_run(selection.run_id).data.tags
        errors: list[str] = []
        for key, value in current_tags.items():
            if key.startswith(_DATA_TAG_PREFIXES):
                previous = stored.get(key)
                if previous is not None and previous != value:
                    errors.append(f"{key}: stored={previous!r}, current={value!r}")
        for key in ("git_commit", SOURCE_FINGERPRINT_TAG):
            previous = stored.get(key)
            current = current_tags.get(key)
            if previous is not None and current is not None and previous != current:
                if not allow_source_mismatch:
                    errors.append(f"{key}: stored={previous!r}, current={current!r}")
        if errors:
            raise RecoveryError(
                "selected run is incompatible with current data/source provenance:\n  - "
                + "\n  - ".join(errors)
            )

    @staticmethod
    def format_candidates(candidates: tuple[ResumeCandidate, ...]) -> str:
        """Render stable, copy-friendly candidate rows for terminal selection."""
        if not candidates:
            return "No unfinished fold runs found."
        header = (
            "run_id                           status     phase         fold  epochs  "
            "best_val       resumable  issues"
        )
        rows = [header]
        for item in candidates:
            fold = "-" if item.fold_index is None else str(item.fold_index)
            epochs = "-" if item.completed_epochs is None else str(item.completed_epochs)
            best = (
                "-"
                if item.best_validation_loss is None
                else f"{item.best_validation_loss:.8g}"
            )
            issues = "; ".join(item.issues) or "-"
            rows.append(
                f"{item.run_id:<32} {item.status:<10} {item.phase:<13} "
                f"{fold:<5} {epochs:<7} {best:<14} {str(item.resumable):<10} {issues}"
            )
        return "\n".join(rows)

    def _resolve_explicit(
        self, run_id: str, candidates: tuple[ResumeCandidate, ...]
    ) -> ResumeCandidate:
        direct = next((candidate for candidate in candidates if candidate.run_id == run_id), None)
        if direct is not None:
            return direct
        try:
            run = self._client.get_run(run_id)
        except MlflowException as exc:
            raise RecoveryError(f"MLflow run {run_id!r} does not exist") from exc
        experiment = self._client.get_experiment_by_name(self._experiment_name)
        if experiment is None or run.info.experiment_id != experiment.experiment_id:
            raise RecoveryError(
                f"run {run_id} does not belong to experiment {self._experiment_name!r}"
            )
        if self._is_fold(run):
            candidate = self._candidate_for_run(run)
            if candidate is None:
                raise RecoveryError(f"fold run {run_id} is already completed")
            return candidate
        children = tuple(candidate for candidate in candidates if candidate.parent_run_id == run_id)
        eligible = tuple(candidate for candidate in children if candidate.resumable)
        if len(eligible) != 1:
            raise RecoveryError(
                f"parent run {run_id} must have exactly one resumable child, found "
                f"{len(eligible)}\n{self.format_candidates(children)}"
            )
        return eligible[0]

    def _candidate_for_run(self, run: Run) -> ResumeCandidate | None:
        if not self._is_fold(run):
            return None
        tags = run.data.tags
        phase = tags.get(RUN_PHASE_TAG, "legacy")
        if phase == RUN_PHASE_COMPLETED or (phase == "legacy" and run.info.status == "FINISHED"):
            return None
        parent_run_id = tags.get("mlflow.parentRunId")
        if not parent_run_id:
            return None
        checkpoint_available = (
            self._checkpoint_root.joinpath(run.info.run_id, "latest.pt").is_file()
            or self._artifact_exists(run.info.run_id, _LATEST_ARTIFACT)
        )
        config_available = any(
            self._artifact_exists(run.info.run_id, artifact)
            for artifact in (
                TRAINING_MANIFEST_ARTIFACT,
                LEGACY_PIPELINE_CONFIG_ARTIFACT,
            )
        )
        issues: list[str] = []
        if not checkpoint_available:
            issues.append("missing latest checkpoint")
        if not config_available:
            issues.append("missing pipeline config")
        fold_value = run.data.params.get("fold_index")
        try:
            fold_index = int(fold_value) if fold_value is not None else None
        except ValueError:
            fold_index = None
            issues.append("invalid fold_index")
        epoch_metric = run.data.metrics.get("progress/epoch")
        completed_epochs = int(epoch_metric) if epoch_metric is not None else None
        best_metric = self._best_metric(run.info.run_id, "epoch/val_loss")
        return ResumeCandidate(
            run_id=run.info.run_id,
            parent_run_id=parent_run_id,
            status=run.info.status,
            phase=phase,
            fold_index=fold_index,
            completed_epochs=completed_epochs,
            best_validation_loss=best_metric,
            start_time=int(run.info.start_time or 0),
            checkpoint_available=checkpoint_available,
            config_available=config_available,
            issues=tuple(issues),
        )

    @staticmethod
    def _is_fold(run: Run) -> bool:
        kind = run.data.tags.get(RUN_KIND_TAG)
        name = run.info.run_name or ""
        return kind == RUN_KIND_FOLD or (
            kind is None and name.startswith("fold_") and "mlflow.parentRunId" in run.data.tags
        )

    def _artifact_exists(self, run_id: str, artifact_path: str) -> bool:
        parent, name = artifact_path.rsplit("/", maxsplit=1)
        try:
            return any(item.path == artifact_path and not item.is_dir for item in self._client.list_artifacts(run_id, parent))
        except (MlflowException, OSError):
            return False

    def _load_pipeline_config(self, run_id: str) -> dict[str, Any]:
        last_error: Exception | None = None
        for artifact in (TRAINING_MANIFEST_ARTIFACT, LEGACY_PIPELINE_CONFIG_ARTIFACT):
            try:
                path = self._client.download_artifacts(run_id, artifact)
                value = json.loads(Path(path).read_text(encoding="utf-8"))
            except (MlflowException, FileNotFoundError, json.JSONDecodeError) as exc:
                last_error = exc
                continue
            if not isinstance(value, dict):
                raise RecoveryError(
                    f"run {run_id} training manifest must be a JSON object"
                )
            return cast(dict[str, Any], value)
        raise RecoveryError(f"run {run_id} has no readable training manifest") from last_error

    def _materialize_artifact(self, run_id: str, artifact_path: str) -> Path:
        target = self._checkpoint_root / run_id / Path(artifact_path).name
        if target.is_file():
            return target
        try:
            downloaded = Path(self._client.download_artifacts(run_id, artifact_path))
        except (MlflowException, OSError) as exc:
            raise RecoveryError(f"cannot download {artifact_path} for run {run_id}") from exc
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.{uuid4().hex}.download")
        try:
            shutil.copyfile(downloaded, temporary)
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
        return target

    def _best_metric(self, run_id: str, key: str) -> float | None:
        try:
            history = self._client.get_metric_history(run_id, key)
        except MlflowException:
            return None
        return min((float(metric.value) for metric in history), default=None)
