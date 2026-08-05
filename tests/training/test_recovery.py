import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import mlflow
import torch
from omegaconf import OmegaConf

from llca.core.artifacts import DATA_MANIFEST_ARTIFACT, TRAINING_MANIFEST_ARTIFACT
from llca.core.provenance.source import SOURCE_FINGERPRINT_TAG
from llca.data.versioning import DATA_MANIFEST_FINGERPRINT_TAG, data_manifest_fingerprint
from llca.mappers.recovery.config_validator import _validate_recovery
from llca.mappers.recovery.mapper import build_recovery
from llca.training.engine.checkpointer import (
    Checkpointer,
    CheckpointValidationError,
    validate_training_checkpoint,
)
from llca.training.modules.recovery_config import RecoveryConfig
from llca.training.recovery import (
    PIPELINE_FINGERPRINT_TAG,
    RUN_KIND_FOLD,
    RUN_KIND_PARENT,
    RUN_KIND_TAG,
    RUN_PHASE_TAG,
    RUN_PHASE_TRAINING,
    RecoveryError,
    RecoveryService,
    RunLock,
    pipeline_fingerprint,
)


def _pipeline() -> dict[str, object]:
    return {
        "schema_version": 1,
        "experiment_name": "recovery-test",
        "data": {},
        "preprocessing": {},
        "features": {},
        "masking": {},
        "loss": {"name": "portfolio"},
        "model": {"name": "example", "width": 4},
        "training": {"epochs": 5, "optimizer": {"name": "adamw"}},
        "split": {"name": "single"},
    }


def _data_manifest() -> dict[str, object]:
    return {"schema_version": 1, "plan": {}, "sources": {}, "datasets": {}}


def _checkpoint() -> dict[str, object]:
    return {
        "config": _pipeline()["model"],
        "model_state_dict": {},
        "optimizer_state_dict": {"param_groups": []},
        "optimizer_name": "adamw",
        "epoch": 1,
        "best_val": 0.25,
        "best_state": {},
        "epochs_without_improvement": 0,
    }


class RecoveryConfigTest(unittest.TestCase):
    def test_maps_explicit_selection(self) -> None:
        root = OmegaConf.create(
            {
                "recovery": {
                    "mode": "explicit",
                    "run_id": "abc",
                    "allow_source_mismatch": False,
                }
            }
        )
        self.assertEqual(_validate_recovery(root), [])
        config = build_recovery(root.recovery)
        self.assertEqual(config.run_id, "abc")

    def test_rejects_ambiguous_run_id(self) -> None:
        root = OmegaConf.create(
            {
                "recovery": {
                    "mode": "auto",
                    "run_id": "abc",
                    "allow_source_mismatch": False,
                }
            }
        )
        self.assertTrue(any("must be null" in error for error in _validate_recovery(root)))
        with self.assertRaisesRegex(ValueError, "must be null"):
            build_recovery(root.recovery)


class CheckpointerRecoveryTest(unittest.TestCase):
    def test_atomic_checkpoint_is_loadable_and_leaves_no_temporary_file(self) -> None:
        with TemporaryDirectory() as directory:
            checkpointer = Checkpointer(directory, log_to_mlflow=False)
            checkpointer.save_latest(_checkpoint())
            loaded = checkpointer.load_latest(map_location="cpu")
            self.assertEqual(loaded["epoch"], 1)
            self.assertEqual([path.name for path in Path(directory).iterdir()], ["latest.pt"])

    def test_checkpoint_schema_reports_missing_fields(self) -> None:
        with self.assertRaisesRegex(CheckpointValidationError, "missing required keys"):
            validate_training_checkpoint({"epoch": 1})

    def test_run_lock_rejects_concurrent_owner(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "run.lock"
            with RunLock(path):
                with self.assertRaisesRegex(RecoveryError, "already held"):
                    with RunLock(path):
                        self.fail("second lock acquisition must not succeed")


@patch.dict(os.environ, {"MLFLOW_ALLOW_FILE_STORE": "true"})
class RecoveryServiceTest(unittest.TestCase):
    def _create_interrupted_fold(
        self, tracking_uri: str, checkpoint_root: Path, *, name: str = "fold_1"
    ) -> tuple[str, str]:
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment("recovery-test")
        with mlflow.start_run(
            run_name="cross_validation",
            tags={RUN_KIND_TAG: RUN_KIND_PARENT, RUN_PHASE_TAG: RUN_PHASE_TRAINING},
        ) as parent:
            parent_id = parent.info.run_id
            with mlflow.start_run(
                run_name=name,
                nested=True,
                tags={
                    RUN_KIND_TAG: RUN_KIND_FOLD,
                    RUN_PHASE_TAG: RUN_PHASE_TRAINING,
                    PIPELINE_FINGERPRINT_TAG: pipeline_fingerprint(_pipeline()),
                    DATA_MANIFEST_FINGERPRINT_TAG: data_manifest_fingerprint(_data_manifest()),
                    SOURCE_FINGERPRINT_TAG: "source-hash",
                    "raw_data_sha256_prices": "old-hash",
                },
            ) as child:
                child_id = child.info.run_id
                mlflow.log_param("fold_index", int(name.rsplit("_", maxsplit=1)[1]))
                mlflow.log_dict(_pipeline(), TRAINING_MANIFEST_ARTIFACT)
                mlflow.log_dict(_data_manifest(), DATA_MANIFEST_ARTIFACT)
        directory = checkpoint_root / child_id
        directory.mkdir(parents=True)
        torch.save(_checkpoint(), directory / "latest.pt")
        return parent_id, child_id

    def test_auto_selects_only_resumable_run_and_validates_checkpoint(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            tracking_uri = (root / "mlruns").resolve().as_uri()
            checkpoint_root = root / "checkpoints"
            parent_id, child_id = self._create_interrupted_fold(tracking_uri, checkpoint_root)
            service = RecoveryService(
                tracking_uri,
                "recovery-test",
                checkpoint_root=checkpoint_root,
            )
            selection = service.resolve(RecoveryConfig("auto", None, False))
            assert selection is not None
            self.assertEqual(selection.run_id, child_id)
            self.assertEqual(selection.parent_run_id, parent_id)
            self.assertEqual(service.preflight_checkpoint(selection)["epoch"], 1)

    def test_explicit_parent_selects_its_single_interrupted_fold(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            tracking_uri = (root / "mlruns").resolve().as_uri()
            checkpoint_root = root / "checkpoints"
            parent_id, child_id = self._create_interrupted_fold(tracking_uri, checkpoint_root)
            service = RecoveryService(
                tracking_uri,
                "recovery-test",
                checkpoint_root=checkpoint_root,
            )
            selection = service.resolve(RecoveryConfig("explicit", parent_id, False))
            assert selection is not None
            self.assertEqual(selection.run_id, child_id)

    def test_provenance_rejects_changed_data(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            tracking_uri = (root / "mlruns").resolve().as_uri()
            checkpoint_root = root / "checkpoints"
            _, child_id = self._create_interrupted_fold(tracking_uri, checkpoint_root)
            service = RecoveryService(
                tracking_uri,
                "recovery-test",
                checkpoint_root=checkpoint_root,
            )
            selection = service.resolve(RecoveryConfig("explicit", child_id, False))
            assert selection is not None
            with self.assertRaisesRegex(RecoveryError, "raw_data_sha256_prices"):
                service.validate_provenance(
                    selection,
                    {
                        "raw_data_sha256_prices": "new-hash",
                        SOURCE_FINGERPRINT_TAG: "source-hash",
                    },
                    allow_source_mismatch=False,
                )

    def test_auto_rejects_multiple_resumable_runs_and_lists_ids(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            tracking_uri = (root / "mlruns").resolve().as_uri()
            checkpoint_root = root / "checkpoints"
            _, first = self._create_interrupted_fold(tracking_uri, checkpoint_root)
            _, second = self._create_interrupted_fold(tracking_uri, checkpoint_root, name="fold_2")
            service = RecoveryService(
                tracking_uri,
                "recovery-test",
                checkpoint_root=checkpoint_root,
            )
            with self.assertRaises(RecoveryError) as raised:
                service.resolve(RecoveryConfig("auto", None, False))
            message = str(raised.exception)
            self.assertIn(first, message)
            self.assertIn(second, message)


if __name__ == "__main__":
    unittest.main()
