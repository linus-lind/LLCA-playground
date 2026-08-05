from __future__ import annotations

from pathlib import Path

import hydra
from dotenv import load_dotenv
from hydra.core.hydra_config import HydraConfig
from hydra.types import RunMode
from omegaconf import DictConfig

from llca.core.paths import PROJECT_ROOT, chdir_to_project_root
from llca.core.provenance.environment import build_environment_manifest
from llca.core.provenance.source import SOURCE_FINGERPRINT_TAG, build_source_snapshot
from llca.core.provenance.training_manifest import build_training_manifest
from llca.core.resolvers import register_resolvers
from llca.data.versioning import provenance_tags
from llca.mappers import (
    build_loss,
    build_model,
    build_recovery,
    build_split,
    build_training,
    validate_config,
)
from llca.mappers.loss.mapper import objective_kind
from llca.mappers.model.mapper import model_capabilities
from llca.pipeline.preparation import prepare_training_data
from llca.training.engine.execution import execute_training
from llca.training.manifests import build_invocation_manifest
from llca.training.recovery import RecoverySelection, RecoveryService
from llca.utils.git import git_commit, git_dirty

register_resolvers()
load_dotenv(PROJECT_ROOT / ".env")

_CONFIG_PATH = (
    "../configs/training"
    if (Path(__file__).resolve().parents[1] / "configs").is_dir()
    else "../../../hydra/configs/training"
)


def _selected(cfg: DictConfig, group: str) -> DictConfig | None:
    section = cfg.get(group)
    return section if section is not None and section.get("name") is not None else None


@hydra.main(config_path=_CONFIG_PATH, config_name="train", version_base=None)
def main(cfg: DictConfig) -> None:
    """Build and execute the configured data-to-training pipeline.

    Configuration is validated before I/O. Named datasets pass through preprocessing,
    feature construction, alignment and masking; registered mappers then construct the
    objective, estimator factory, training policy, and temporal splitter. Raw/processed
    data hashes and the Git commit are attached to execution-plan runs for provenance.
    """
    recovery_config = build_recovery(cfg.recovery)
    recovery_service: RecoveryService | None = None
    recovery: RecoverySelection | None = None
    pipeline_config: DictConfig | dict[str, object] = cfg
    if recovery_config.mode != "off":
        tracking_uri = str(cfg.mlflow_tracking_uri)
        recovery_service = RecoveryService(tracking_uri, str(cfg.experiment_name))
        recovery = recovery_service.resolve(recovery_config)
        if recovery_config.mode == "list":
            return
        if recovery is None:
            raise RuntimeError("recovery selection did not return a resumable run")
        cfg = recovery_service.effective_config(
            cfg,
            recovery,
            recovery_config,
            tracking_uri,
        )
        pipeline_config = recovery.pipeline_config
        checkpoint = recovery_service.preflight_checkpoint(recovery)
        print(
            f"Resuming fold run {recovery.run_id} from completed epoch "
            f"{int(checkpoint['epoch']) + 1}."
        )

    validate_config(cfg)

    capabilities = model_capabilities(str(cfg.model.name))
    requirements = capabilities.resolve_data(cfg.model)
    prepared = prepare_training_data(cfg, requirements, data_view=capabilities.data_view)

    loss_cfg = _selected(cfg, "loss")
    objective = build_loss(loss_cfg) if loss_cfg is not None else None

    estimator_factory = build_model(cfg.model, loss=objective, loss_config=loss_cfg)

    training = build_training(cfg.training)
    splitter = build_split(cfg.split)

    data_manifest = prepared.data_manifest
    run_tags = provenance_tags(data_manifest)
    run_tags |= {
        "llca.model": str(cfg.model.name),
        "llca.training_engine": str(training.engine),
        "llca.data_view": capabilities.data_view,
    }
    if loss_cfg is not None:
        run_tags["llca.objective"] = str(loss_cfg.name)
        run_tags["llca.objective_kind"] = str(objective_kind(str(loss_cfg.name)))
    source_manifest = build_source_snapshot()
    environment_manifest = build_environment_manifest()

    commit = git_commit()
    if commit is not None:
        run_tags["git_commit"] = commit
    dirty = git_dirty()
    if dirty is not None:
        run_tags["git_dirty"] = str(dirty).lower()
    run_tags[SOURCE_FINGERPRINT_TAG] = str(source_manifest["source_sha256"])

    if recovery is not None:
        if recovery_service is None:
            raise RuntimeError("recovery service is unavailable for the selected run")
        recovery_service.validate_provenance(
            recovery,
            run_tags,
            allow_source_mismatch=recovery_config.allow_source_mismatch,
        )

    registry_model_name = cfg.experiment_name
    hydra_config = HydraConfig.get()
    invocation_manifest = build_invocation_manifest(
        task_overrides=hydra_config.overrides.task,
        config_choices=hydra_config.runtime.choices,
    )
    if hydra_config.mode == RunMode.MULTIRUN:
        registry_model_name = f"{cfg.experiment_name}_{hydra_config.job.num}"

    if recovery is None:
        pipeline_config = build_training_manifest(cfg)

    execute_training(
        prepared.data,
        splitter,
        estimator_factory,
        training,
        mlflow_tracking_uri=cfg.mlflow_tracking_uri,
        mlflow_experiment_name=cfg.experiment_name,
        primary_dataset=prepared.plan.primary_dataset,
        run_tags=run_tags,
        registry_model_name=registry_model_name,
        pipeline_config=pipeline_config,
        data_manifest=data_manifest,
        invocation_manifest=invocation_manifest,
        source_manifest=source_manifest,
        environment_manifest=environment_manifest,
        recovery=recovery,
    )


if __name__ == "__main__":
    chdir_to_project_root()
    main()
