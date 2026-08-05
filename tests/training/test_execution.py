import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

import pandas as pd
import torch
from mlflow import MlflowClient
from omegaconf import OmegaConf

from llca.core.artifacts import ENVIRONMENT_MANIFEST_ARTIFACT
from llca.core.provenance.source import SOURCE_FINGERPRINT_TAG
from llca.splitting.fold import Fold
from llca.training.engine.execution import execute_training
from llca.training.modules.recovery_config import RecoveryConfig
from llca.training.modules.training_config import (
    AdamWConfig,
    EarlyStoppingConfig,
    TrainingConfig,
    TrainingDiagnosticsConfig,
)
from llca.training.recovery import (
    RUN_PHASE_COMPLETED,
    RUN_PHASE_TAG,
    RecoveryService,
)


def _training() -> TrainingConfig:
    return TrainingConfig(
        seed=42,
        deterministic=True,
        epochs=2,
        batch_size=1,
        grad_clip=1.0,
        device="cpu",
        precision="fp32",
        gradient_checkpointing=False,
        optimizer=AdamWConfig(0.001, 0.0, False),
        early_stopping=EarlyStoppingConfig(2, 0.0),
        diagnostics=TrainingDiagnosticsConfig(1, False, False),
    )


def _pipeline() -> dict[str, object]:
    return {
        "schema_version": 1,
        "experiment_name": "cross-validation-recovery",
        "data": {},
        "preprocessing": {},
        "features": {},
        "masking": {},
        "loss": {"name": "portfolio"},
        "model": {"name": "example", "width": 4},
        "training": {"epochs": 2, "optimizer": {"name": "adamw"}},
        "split": {"name": "single"},
    }


def _checkpoint() -> dict[str, object]:
    return {
        "config": _pipeline()["model"],
        "model_state_dict": {},
        "optimizer_state_dict": {"param_groups": []},
        "optimizer_name": "adamw",
        "epoch": 0,
        "best_val": 0.25,
        "best_state": {},
        "epochs_without_improvement": 0,
    }


def _data_manifest() -> dict[str, object]:
    return {"schema_version": 1, "plan": {}, "sources": {}, "datasets": {}}


class _OneFoldSplitter:
    def __init__(self) -> None:
        self.fold = Fold(
            index=1,
            train_start=pd.Timestamp("2020-01-01"),
            train_end=pd.Timestamp("2020-06-30"),
            val_start=pd.Timestamp("2020-07-01"),
            val_end=pd.Timestamp("2020-09-30"),
            test_start=pd.Timestamp("2020-10-01"),
            test_end=pd.Timestamp("2020-12-31"),
        )

    @property
    def name(self) -> str:
        return "single"

    def split(self, panels: object, primary: str) -> object:
        del panels, primary
        yield self.fold, {}, {}


@patch.dict(os.environ, {"MLFLOW_ALLOW_FILE_STORE": "true"})
class TrainingExecutionRecoveryTest(unittest.TestCase):
    def test_interrupted_fold_resumes_same_runs_and_completes_lifecycle(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            tracking_uri = (root / "mlruns").resolve().as_uri()
            checkpoints = root / "checkpoints"
            pipeline = OmegaConf.create(_pipeline())
            splitter = _OneFoldSplitter()

            interrupted = MagicMock()

            def fail_after_checkpoint(**kwargs: object) -> None:
                checkpoint_dir = Path(str(kwargs["checkpoint_dir"]))
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                torch.save(_checkpoint(), checkpoint_dir / "latest.pt")
                raise RuntimeError("simulated process interruption")

            interrupted.fit.side_effect = fail_after_checkpoint
            with patch("llca.training.engine.execution.CHECKPOINTS_DIR", checkpoints):
                with self.assertRaisesRegex(RuntimeError, "simulated"):
                    execute_training(
                        {},
                        splitter,
                        lambda: interrupted,
                        _training(),
                        tracking_uri,
                        "cross-validation-recovery",
                        primary_dataset="features",
                        run_tags={SOURCE_FINGERPRINT_TAG: "source-hash"},
                        registry_model_name=None,
                        pipeline_config=pipeline,
                        data_manifest=_data_manifest(),
                        invocation_manifest={"schema_version": 1, "task_overrides": []},
                        source_manifest={"schema_version": 1, "files": {}},
                        environment_manifest={"schema_version": 1, "python": {}},
                    )

            service = RecoveryService(
                tracking_uri,
                "cross-validation-recovery",
                checkpoint_root=checkpoints,
            )
            selection = service.resolve(RecoveryConfig("auto", None, False))
            assert selection is not None
            service.preflight_checkpoint(selection)

            recovered = MagicMock()
            recovered.log_model.return_value = "models:/recovered"
            with patch("llca.training.engine.execution.CHECKPOINTS_DIR", checkpoints):
                execute_training(
                    {},
                    splitter,
                    lambda: recovered,
                    _training(),
                    tracking_uri,
                    "cross-validation-recovery",
                    primary_dataset="features",
                    run_tags={SOURCE_FINGERPRINT_TAG: "source-hash"},
                    registry_model_name=None,
                    pipeline_config=selection.pipeline_config,
                    data_manifest=_data_manifest(),
                    recovery=selection,
                )

            self.assertTrue(recovered.fit.call_args.kwargs["resume"])
            client = MlflowClient(tracking_uri=tracking_uri)
            parent = client.get_run(selection.parent_run_id)
            child = client.get_run(selection.run_id)
            self.assertEqual(parent.info.status, "FINISHED")
            self.assertEqual(child.info.status, "FINISHED")
            self.assertEqual(parent.data.tags[RUN_PHASE_TAG], RUN_PHASE_COMPLETED)
            self.assertEqual(child.data.tags[RUN_PHASE_TAG], RUN_PHASE_COMPLETED)
            pipeline_artifacts = {
                artifact.path for artifact in client.list_artifacts(selection.run_id, "pipeline")
            }
            self.assertIn(ENVIRONMENT_MANIFEST_ARTIFACT, pipeline_artifacts)


if __name__ == "__main__":
    unittest.main()
