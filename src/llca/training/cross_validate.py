"""Chronological fold orchestration with resumable, idempotent MLflow lifecycle."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, cast

import mlflow
from mlflow import MlflowClient
from mlflow.entities import Run
from omegaconf import DictConfig, OmegaConf

from llca.core.artifacts import TRAINING_MANIFEST_ARTIFACT
from llca.core.paths import CHECKPOINTS_DIR
from llca.data.modules.masked_panel import MaskedPanels
from llca.data.versioning import DATA_MANIFEST_FINGERPRINT_TAG, data_manifest_fingerprint
from llca.models.estimators.estimator import Estimator
from llca.splitting.fold import Fold
from llca.splitting.splitter import Splitter
from llca.training.modules.training_config import TrainingConfig
from llca.training.recovery import (
    LOGGED_MODEL_URI_TAG,
    PIPELINE_FINGERPRINT_TAG,
    REGISTERED_MODEL_VERSION_TAG,
    RUN_KIND_FOLD,
    RUN_KIND_PARENT,
    RUN_KIND_TAG,
    RUN_PHASE_COMPLETED,
    RUN_PHASE_MODEL_LOGGED,
    RUN_PHASE_PREPARED,
    RUN_PHASE_REGISTERED,
    RUN_PHASE_TAG,
    RUN_PHASE_TRAINED,
    RUN_PHASE_TRAINING,
    RecoveryError,
    RecoverySelection,
    RunLock,
    pipeline_fingerprint,
)
from llca.training.run_manifests import RunManifests
from llca.training.training_tracker import MlflowTrainingTracker


def _fold_parameters(fold: Fold) -> dict[str, int | str]:
    return {
        "fold_index": fold.index,
        "train_start": fold.train_start.date().isoformat(),
        "train_end": fold.train_end.date().isoformat(),
        "val_start": fold.val_start.date().isoformat(),
        "val_end": fold.val_end.date().isoformat(),
        "test_start": fold.test_start.date().isoformat(),
        "test_end": fold.test_end.date().isoformat(),
    }


def _log_fold_window(fold: Fold) -> None:
    """Log immutable evaluation boundaries for one newly created fold run."""
    mlflow.log_params(_fold_parameters(fold))


def _assert_fold_window(run: Run, fold: Fold) -> None:
    """Reject data/split changes that map a resumed run to different date boundaries."""
    expected = {name: str(value) for name, value in _fold_parameters(fold).items()}
    mismatches = [
        f"{name}: stored={run.data.params.get(name)!r}, current={value!r}"
        for name, value in expected.items()
        if run.data.params.get(name) != value
    ]
    if mismatches:
        raise RecoveryError(
            f"fold boundaries for run {run.info.run_id} changed:\n  - "
            + "\n  - ".join(mismatches)
        )


def _set_phase(client: MlflowClient, run_id: str, phase: str) -> None:
    client.set_tag(run_id, RUN_PHASE_TAG, phase)


def _find_logged_model_uri(
    client: MlflowClient, experiment_id: str, run_id: str
) -> str | None:
    """Recover a model logged immediately before a process died prior to phase tagging."""
    matches = [
        model
        for model in client.search_logged_models([experiment_id], max_results=1000)
        if model.source_run_id == run_id and str(model.status).endswith("READY")
    ]
    if len(matches) > 1:
        raise RecoveryError(f"run {run_id} owns multiple ready logged models")
    return str(matches[0].model_uri) if matches else None


def _find_registered_version(
    client: MlflowClient, registry_model_name: str, run_id: str
) -> str | None:
    """Return an existing registry version so retrying registration stays idempotent."""
    versions = client.search_model_versions(f"name = '{registry_model_name}'")
    matches = [version for version in versions if str(version.run_id) == run_id]
    if len(matches) > 1:
        raise RecoveryError(
            f"run {run_id} is already attached to multiple versions of {registry_model_name}"
        )
    return str(matches[0].version) if matches else None


def _register_model(
    model_uri: str,
    registry_model_name: str,
    fold: Fold,
    run_tags: Mapping[str, str] | None,
) -> str:
    """Register a fitted fold model with immutable evaluation and provenance tags."""
    version_tags = dict(run_tags or {})
    version_tags["fold_index"] = str(fold.index)
    version_tags["test_start"] = fold.test_start.date().isoformat()
    version_tags["test_end"] = fold.test_end.date().isoformat()
    version_tags["pipeline_config_artifact"] = TRAINING_MANIFEST_ARTIFACT
    version_tags["training_manifest_artifact"] = TRAINING_MANIFEST_ARTIFACT
    version = mlflow.register_model(model_uri, registry_model_name, tags=version_tags)
    return str(version.version)


def _execute_fold(
    run: Run,
    fold: Fold,
    train: MaskedPanels,
    val: MaskedPanels,
    estimator_factory: Callable[[], Estimator],
    training: TrainingConfig,
    run_tags: Mapping[str, str] | None,
    registry_model_name: str | None,
    manifests: RunManifests,
    *,
    resume: bool,
) -> None:
    """Train or recover one fold and complete logging/registration exactly once."""
    client = MlflowClient()
    run_id = run.info.run_id
    if resume:
        _assert_fold_window(run, fold)
    else:
        _log_fold_window(fold)
        mlflow.log_params(training.tracking_parameters())
    manifests.log(include_invocation=not resume, include_source=not resume)

    registered_version = (
        _find_registered_version(client, registry_model_name, run_id)
        if registry_model_name is not None
        else None
    )
    if registered_version is not None:
        client.set_tag(run_id, REGISTERED_MODEL_VERSION_TAG, registered_version)
        _set_phase(client, run_id, RUN_PHASE_COMPLETED)
        return

    latest = client.get_run(run_id)
    model_uri = latest.data.tags.get(LOGGED_MODEL_URI_TAG)
    if model_uri is None:
        model_uri = _find_logged_model_uri(client, run.info.experiment_id, run_id)
        if model_uri is not None:
            client.set_tag(run_id, LOGGED_MODEL_URI_TAG, model_uri)
            _set_phase(client, run_id, RUN_PHASE_MODEL_LOGGED)

    if model_uri is None:
        estimator = estimator_factory()
        _set_phase(client, run_id, RUN_PHASE_TRAINING)
        estimator.fit(
            train=train,
            training=training,
            val=val,
            tracker=MlflowTrainingTracker(
                run_id,
                batch_log_interval=training.diagnostics.interval,
            ),
            checkpoint_dir=CHECKPOINTS_DIR / run_id,
            resume=resume,
        )
        _set_phase(client, run_id, RUN_PHASE_TRAINED)
        model_uri = estimator.log_model()
        client.set_tag(run_id, LOGGED_MODEL_URI_TAG, model_uri)
        _set_phase(client, run_id, RUN_PHASE_MODEL_LOGGED)

    if registry_model_name is not None:
        registered_version = _find_registered_version(client, registry_model_name, run_id)
        if registered_version is None:
            registered_version = _register_model(
                model_uri,
                registry_model_name,
                fold,
                run_tags,
            )
        client.set_tag(run_id, REGISTERED_MODEL_VERSION_TAG, registered_version)
        _set_phase(client, run_id, RUN_PHASE_REGISTERED)
    _set_phase(client, run_id, RUN_PHASE_COMPLETED)


def _run_fold(
    fold: Fold,
    train: MaskedPanels,
    val: MaskedPanels,
    estimator_factory: Callable[[], Estimator],
    training: TrainingConfig,
    parent_run_id: str,
    run_tags: Mapping[str, str] | None,
    registry_model_name: str | None,
    manifests: RunManifests,
    pipeline_hash: str,
    *,
    resume_run_id: str | None = None,
) -> None:
    """Open a new or existing nested run and execute one idempotent fold lifecycle."""
    if resume_run_id is not None:
        lock_path = CHECKPOINTS_DIR / resume_run_id / ".run.lock"
        with RunLock(lock_path):
            with mlflow.start_run(run_id=resume_run_id, nested=True) as active:
                client = MlflowClient()
                client.set_tag(active.info.run_id, PIPELINE_FINGERPRINT_TAG, pipeline_hash)
                run = client.get_run(active.info.run_id)
                _execute_fold(
                    run,
                    fold,
                    train,
                    val,
                    estimator_factory,
                    training,
                    run_tags,
                    registry_model_name,
                    manifests,
                    resume=True,
                )
        return

    tags = dict(run_tags or {})
    tags |= {
        "mlflow.parentRunId": parent_run_id,
        RUN_KIND_TAG: RUN_KIND_FOLD,
        RUN_PHASE_TAG: RUN_PHASE_PREPARED,
        PIPELINE_FINGERPRINT_TAG: pipeline_hash,
    }
    with mlflow.start_run(run_name=f"fold_{fold.index}", nested=True, tags=tags) as active:
        lock_path = CHECKPOINTS_DIR / active.info.run_id / ".run.lock"
        with RunLock(lock_path):
            run = MlflowClient().get_run(active.info.run_id)
            _execute_fold(
                run,
                fold,
                train,
                val,
                estimator_factory,
                training,
                run_tags,
                registry_model_name,
                manifests,
                resume=False,
            )


def _children_by_fold(client: MlflowClient, parent: Run) -> dict[int, Run]:
    """Index existing child runs and reject duplicate fold ownership."""
    runs = client.search_runs([parent.info.experiment_id], max_results=1000)
    children = [
        run
        for run in runs
        if run.data.tags.get("mlflow.parentRunId") == parent.info.run_id
        and (
            run.data.tags.get(RUN_KIND_TAG) == RUN_KIND_FOLD
            or (run.info.run_name or "").startswith("fold_")
        )
    ]
    by_fold: dict[int, Run] = {}
    for run in children:
        value = run.data.params.get("fold_index")
        if value is None:
            raise RecoveryError(f"child run {run.info.run_id} is missing fold_index")
        try:
            fold_index = int(value)
        except ValueError as exc:
            raise RecoveryError(
                f"child run {run.info.run_id} has invalid fold_index {value!r}"
            ) from exc
        if fold_index in by_fold:
            raise RecoveryError(
                f"parent run {parent.info.run_id} has multiple children for fold {fold_index}"
            )
        by_fold[fold_index] = run
    return by_fold


def _run_completed(run: Run) -> bool:
    phase = run.data.tags.get(RUN_PHASE_TAG)
    return phase == RUN_PHASE_COMPLETED or (phase is None and run.info.status == "FINISHED")


def _execute_cross_validation(
    parent: Run,
    panels: MaskedPanels,
    splitter: Splitter,
    estimator_factory: Callable[[], Estimator],
    training: TrainingConfig,
    primary_dataset: str,
    run_tags: Mapping[str, str] | None,
    registry_model_name: str | None,
    manifests: RunManifests,
    pipeline_hash: str,
    recovery: RecoverySelection | None,
) -> None:
    client = MlflowClient()
    parent_id = parent.info.run_id
    _set_phase(client, parent_id, RUN_PHASE_TRAINING)
    children = _children_by_fold(client, parent)
    selected_used = False

    for fold, train, val in splitter.split(panels, primary_dataset):
        existing = children.get(fold.index)
        if existing is not None and _run_completed(existing):
            continue
        resume_run_id: str | None = None
        if existing is not None:
            if recovery is None or existing.info.run_id != recovery.run_id:
                raise RecoveryError(
                    f"fold {fold.index} has unfinished run {existing.info.run_id}; "
                    "select that run explicitly"
                )
            resume_run_id = existing.info.run_id
            selected_used = True
        elif recovery is not None and recovery.candidate.fold_index == fold.index:
            raise RecoveryError(
                f"selected run {recovery.run_id} is not a child of parent {parent_id}"
            )

        _run_fold(
            fold,
            train,
            val,
            estimator_factory,
            training,
            parent_id,
            run_tags,
            registry_model_name,
            manifests,
            pipeline_hash,
            resume_run_id=resume_run_id,
        )

    if recovery is not None and not selected_used:
        raise RecoveryError(
            f"selected run {recovery.run_id} does not map to any fold in the stored pipeline"
        )
    _set_phase(client, parent_id, RUN_PHASE_COMPLETED)


def cross_validate(
    panels: MaskedPanels,
    splitter: Splitter,
    estimator_factory: Callable[[], Estimator],
    training: TrainingConfig,
    mlflow_tracking_uri: str,
    mlflow_experiment_name: str,
    *,
    primary_dataset: str,
    run_tags: Mapping[str, str] | None = None,
    registry_model_name: str | None = None,
    pipeline_config: DictConfig | Mapping[str, Any],
    data_manifest: Mapping[str, Any] | None = None,
    invocation_manifest: Mapping[str, Any] | None = None,
    source_manifest: Mapping[str, Any] | None = None,
    recovery: RecoverySelection | None = None,
) -> None:
    """Execute new folds or continue one selected interrupted cross-validation job."""
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    experiment = mlflow.set_experiment(mlflow_experiment_name)
    if isinstance(pipeline_config, DictConfig):
        resolved_config = cast(
            dict[str, Any], OmegaConf.to_container(pipeline_config, resolve=True)
        )
    else:
        resolved_config = dict(pipeline_config)
    manifests = RunManifests(
        training=resolved_config,
        data=dict(data_manifest) if data_manifest is not None else None,
        invocation=dict(invocation_manifest) if invocation_manifest is not None else None,
        source=dict(source_manifest) if source_manifest is not None else None,
    )
    effective_run_tags = dict(run_tags or {})
    if data_manifest is not None:
        effective_run_tags[DATA_MANIFEST_FINGERPRINT_TAG] = data_manifest_fingerprint(
            data_manifest
        )
    pipeline_hash = pipeline_fingerprint(resolved_config)

    if recovery is not None:
        with mlflow.start_run(run_id=recovery.parent_run_id) as active:
            parent = MlflowClient().get_run(active.info.run_id)
            manifests.log(include_invocation=False, include_source=False)
            MlflowClient().set_tag(parent.info.run_id, PIPELINE_FINGERPRINT_TAG, pipeline_hash)
            _execute_cross_validation(
                parent,
                panels,
                splitter,
                estimator_factory,
                training,
                primary_dataset,
                effective_run_tags,
                registry_model_name,
                manifests,
                pipeline_hash,
                recovery,
            )
        return

    parent_tags = dict(effective_run_tags)
    parent_tags |= {
        RUN_KIND_TAG: RUN_KIND_PARENT,
        RUN_PHASE_TAG: RUN_PHASE_PREPARED,
        PIPELINE_FINGERPRINT_TAG: pipeline_hash,
    }
    with mlflow.start_run(
        experiment_id=experiment.experiment_id,
        run_name="cross_validation",
        tags=parent_tags,
    ) as active:
        parent = MlflowClient().get_run(active.info.run_id)
        manifests.log()
        _execute_cross_validation(
            parent,
            panels,
            splitter,
            estimator_factory,
            training,
            primary_dataset,
            effective_run_tags,
            registry_model_name,
            manifests,
            pipeline_hash,
            None,
        )
