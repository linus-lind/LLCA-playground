import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast
from unittest.mock import MagicMock

import torch
from torch import Tensor, nn

from llca.training.checkpointer import Checkpointer
from llca.training.modules.tracking import TrainingTracker
from llca.training.modules.training_config import (
    AdamWConfig,
    EarlyStoppingConfig,
    TrainingConfig,
    TrainingDiagnosticsConfig,
)
from llca.training.modules.training_diagnostics import (
    PanelBatchMetadata,
    TrainingBatchOutput,
)
from llca.training.modules.training_task import TrainingTask
from llca.training.trainer import Trainer


class _TwoComponentModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.first = nn.Linear(1, 1, bias=False)
        self.second = nn.Linear(1, 1, bias=False)
        nn.init.ones_(self.first.weight)
        nn.init.ones_(self.second.weight)

    def forward(self, values: Tensor) -> Tensor:
        return cast(Tensor, self.second(self.first(values)))


def _config() -> TrainingConfig:
    return TrainingConfig(
        seed=42,
        deterministic=True,
        epochs=1,
        batch_size=1,
        grad_clip=0.1,
        device="cpu",
        precision="fp32",
        gradient_checkpointing=False,
        optimizer=AdamWConfig(
            learning_rate=0.01,
            weight_decay=0.001,
            fused=False,
        ),
        early_stopping=EarlyStoppingConfig(patience=1, min_delta=0.0),
        diagnostics=TrainingDiagnosticsConfig(
            interval=1,
            component_gradient_norms=True,
            parameter_update_norms=True,
        ),
    )


class TrainerDiagnosticsTest(unittest.TestCase):
    def test_logs_actual_clipping_updates_components_and_optimizer_lr(self) -> None:
        model = _TwoComponentModel()
        tracker_mock = MagicMock(spec=TrainingTracker)
        tracker = cast(TrainingTracker, tracker_mock)

        def loss_for_batch(_: int) -> TrainingBatchOutput:
            scores = model(torch.tensor([[10.0]]))
            return TrainingBatchOutput(
                loss=scores.square().mean(),
                metrics={"scores/mean": scores.detach().mean()},
            )

        trainer = Trainer(
            config=_config(),
            task=TrainingTask(
                model=model,
                batches=[0],
                train_step=loss_for_batch,
                batch_metadata=lambda _batch, index: PanelBatchMetadata(
                    index=index + 1,
                    observations=4,
                    start_date="2024-01-02",
                    end_date="2024-01-02",
                    dates=1,
                    entities=4,
                ),
            ),
            tracker=tracker,
        )
        trainer.fit()

        tracker_mock.begin.assert_called_once()
        call = tracker_mock.on_batch_end.call_args
        diagnostics = call.kwargs["diagnostics"]
        self.assertGreater(diagnostics["gradient/norm_before_clip"], 0.1)
        self.assertAlmostEqual(diagnostics["gradient/norm_after_clip"], 0.1, places=5)
        self.assertLess(diagnostics["gradient/clip_factor"], 1.0)
        self.assertEqual(diagnostics["gradient/clipped"], 1.0)
        self.assertGreater(diagnostics["parameters/update_norm"], 0.0)
        self.assertGreater(diagnostics["parameters/update_to_parameter_ratio"], 0.0)
        self.assertIn("gradient/components_pre_clip/first", diagnostics)
        self.assertIn("gradient/components_pre_clip/second", diagnostics)
        self.assertEqual(diagnostics["optimizer/learning_rate"], 0.01)
        self.assertEqual(diagnostics["batch/date_count"], 1.0)
        self.assertEqual(diagnostics["batch/entity_union_count"], 4.0)
        self.assertEqual(call.kwargs["observations"], 4)

    def test_adamw_config_builds_decoupled_optimizer(self) -> None:
        model = nn.Linear(1, 1)
        optimizer = _config().optimizer.build(model.parameters(), torch.device("cpu"))
        self.assertIsInstance(optimizer, torch.optim.AdamW)

    def test_model_diagnostics_are_lazy_between_logging_steps(self) -> None:
        model = nn.Linear(1, 1)
        diagnostic_calls = 0

        def diagnostics() -> dict[str, float]:
            nonlocal diagnostic_calls
            diagnostic_calls += 1
            return {"model/example": 1.0}

        def loss_for_batch(_: int) -> TrainingBatchOutput:
            prediction = model(torch.ones(1, 1))
            return TrainingBatchOutput(
                loss=prediction.square().mean(),
                metrics_factory=diagnostics,
            )

        config = replace(
            _config(),
            diagnostics=replace(_config().diagnostics, interval=2),
        )
        Trainer(
            config=config,
            task=TrainingTask(model=model, batches=[0], train_step=loss_for_batch),
        ).fit()

        self.assertEqual(diagnostic_calls, 0)

    def test_resume_requires_an_existing_checkpoint(self) -> None:
        model = nn.Linear(1, 1)
        with TemporaryDirectory() as directory:
            trainer = Trainer(
                config=_config(),
                task=TrainingTask(
                    model=model,
                    batches=[0],
                    train_step=lambda _: model(torch.ones(1, 1)).square().mean(),
                ),
                checkpoint_dir=directory,
                checkpoint_state=lambda *_: {},
            )
            with self.assertRaisesRegex(FileNotFoundError, "resume checkpoint"):
                trainer.fit(resume=True)

    def test_terminal_resume_restores_best_state_without_an_extra_step(self) -> None:
        model = nn.Linear(1, 1, bias=False)
        nn.init.constant_(model.weight, 1.0)
        config = _config()
        optimizer = config.optimizer.build(model.parameters(), torch.device("cpu"))
        current_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
        best_state = {name: torch.full_like(value, 2.0) for name, value in current_state.items()}
        calls = 0

        def train_step(_: int) -> Tensor:
            nonlocal calls
            calls += 1
            return model(torch.ones(1, 1)).square().mean()

        with TemporaryDirectory() as directory:
            Checkpointer(Path(directory), log_to_mlflow=False).save_latest(
                {
                    "model_state_dict": current_state,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "optimizer_name": "adamw",
                    "epoch": 0,
                    "best_val": 0.1,
                    "best_state": best_state,
                    "epochs_without_improvement": 0,
                }
            )
            Trainer(
                config=config,
                task=TrainingTask(model=model, batches=[0], train_step=train_step),
                checkpoint_dir=directory,
                checkpoint_state=lambda *_: {},
            ).fit(resume=True)

        self.assertEqual(calls, 0)
        self.assertTrue(torch.equal(model.weight.detach(), torch.full_like(model.weight, 2.0)))


if __name__ == "__main__":
    unittest.main()
