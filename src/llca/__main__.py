import hydra
from dotenv import load_dotenv
from hydra.core.hydra_config import HydraConfig
from hydra.types import RunMode
from omegaconf import DictConfig

from llca.core.paths import PROJECT_ROOT, chdir_to_project_root
from llca.core.resolvers import register_resolvers
from llca.data.masking import align_and_mask
from llca.data.versioning import archive_raw_sources, build_data_manifest, provenance_tags
from llca.mappers import (
    build_datasets,
    build_feature_panels,
    build_loss,
    build_masking,
    build_model,
    build_recovery,
    build_split,
    build_training,
    data_source_path,
    validate_config,
)
from llca.mappers.preprocessing import build_preprocessing
from llca.training.cross_validate import cross_validate
from llca.training.manifest import (
    build_invocation_manifest,
    build_source_snapshot,
    build_training_manifest,
)
from llca.training.recovery import (
    SOURCE_FINGERPRINT_TAG,
    RecoverySelection,
    RecoveryService,
)
from llca.utils.utils import git_commit, git_dirty

register_resolvers()
load_dotenv(PROJECT_ROOT / ".env")


def _selected(cfg: DictConfig, group: str) -> DictConfig | None:
    section = cfg.get(group)
    return section if section is not None and section.get("name") is not None else None


@hydra.main(config_path="../../hydra/configs", config_name="train", version_base=None)
def main(cfg: DictConfig) -> None:
    """Build and execute the configured data-to-training pipeline.

    Configuration is validated before I/O. Named datasets pass through preprocessing,
    feature construction, alignment and masking; registered mappers then construct the
    objective, estimator factory, training policy, and temporal splitter. Raw/processed
    data hashes and the Git commit are attached to cross-validation runs for provenance.
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
        assert recovery is not None
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

    logical_sources = {
        str(name): data_source_path(spec) for name, spec in cfg.data.datasets.items()
    }
    archived_sources = archive_raw_sources(logical_sources)

    datasets = build_datasets(cfg.get("data"))
    datasets = build_preprocessing(cfg.get("preprocessing"), datasets)
    feature_panels = build_feature_panels(cfg.get("features"), datasets)
    membership = build_masking(cfg.get("masking"))
    panels = align_and_mask(datasets, feature_panels, str(cfg.model.inputs.features), membership)

    loss_cfg = _selected(cfg, "loss")
    objective = build_loss(loss_cfg) if loss_cfg is not None else None

    estimator_factory = build_model(cfg.model, loss=objective)

    training = build_training(cfg.training)
    splitter = build_split(cfg.split)

    data_manifest = build_data_manifest(
        logical_sources,
        feature_panels,
        archived_sources=archived_sources,
    )
    run_tags = provenance_tags(data_manifest)
    source_manifest = build_source_snapshot()

    commit = git_commit()
    if commit is not None:
        run_tags["git_commit"] = commit
    dirty = git_dirty()
    if dirty is not None:
        run_tags["git_dirty"] = str(dirty).lower()
    run_tags[SOURCE_FINGERPRINT_TAG] = str(source_manifest["source_sha256"])

    if recovery is not None:
        assert recovery_service is not None
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

    cross_validate(
        panels,
        splitter,
        estimator_factory,
        training,
        mlflow_tracking_uri=cfg.mlflow_tracking_uri,
        mlflow_experiment_name=cfg.experiment_name,
        primary_dataset=cfg.model.inputs.features,
        run_tags=run_tags,
        registry_model_name=registry_model_name,
        pipeline_config=pipeline_config,
        data_manifest=data_manifest,
        invocation_manifest=invocation_manifest,
        source_manifest=source_manifest,
        recovery=recovery,
    )


if __name__ == "__main__":
    chdir_to_project_root()
    main()
