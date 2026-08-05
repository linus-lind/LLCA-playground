"""Reusable PyTorch persistence, device, checkpoint, and training lifecycle."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Self

import torch
from torch import Tensor

from llca.data.modules.masked_panel import MaskedPanels
from llca.models.estimators.estimator import Estimator
from llca.models.estimators.objective_output import objective_loss
from llca.training.engine.reproducibility import capture_rng_state
from llca.training.modules.tracking import TrainingTracker
from llca.training.modules.training_config import TrainingConfig
from llca.training.modules.training_policy import TrainingPolicy
from llca.training.modules.training_task import TrainingTask


class TorchEstimator[DataT](Estimator[DataT], ABC):
    """Reusable torch bundle, device, objective, and checkpoint implementation."""

    @abstractmethod
    def to_device(self, device: torch.device) -> None:
        """Move torch inference state without rebuilding the model."""

    def set_inference_device(self, device: str) -> None:
        resolved = torch.device(
            ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
        )
        self.to_device(resolved)

    @abstractmethod
    def _inference_payload(self) -> dict[str, Any]:
        """Everything required to reconstruct the torch estimator for inference."""

    @classmethod
    @abstractmethod
    def _from_payload(cls, payload: dict[str, Any], map_location: torch.device) -> Self:
        """Construct a bare estimator from a payload, before state is restored."""

    @abstractmethod
    def _restore(self, payload: dict[str, Any]) -> None:
        """Load model weights and preprocessing state from a payload in place."""

    def _save(self, path: str | Path) -> None:
        """Serialize the subclass-defined torch inference bundle to ``path``."""
        torch.save(self._inference_payload(), Path(path))

    @classmethod
    def load(cls, path: str | Path, device: str | torch.device = "auto") -> Self:
        """Reconstruct a torch estimator on the requested or best available device."""
        map_location = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device == "auto"
            else torch.device(device)
        )
        payload = torch.load(Path(path), map_location=map_location, weights_only=False)
        estimator = cls._from_payload(payload, map_location)
        estimator._restore(payload)
        return estimator

    @staticmethod
    def _loss_value(output: object) -> Tensor:
        """Unwrap a differentiable scalar from a bare or structured objective result."""
        return objective_loss(output)

    def _training_state(
        self,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        best_val: float,
        best_state: dict[str, Tensor] | None,
        epochs_without_improvement: int,
    ) -> dict[str, Any]:
        """Extend inference state with everything required for deterministic resume."""
        state = self._inference_payload()
        state["optimizer_state_dict"] = optimizer.state_dict()
        state["optimizer_name"] = type(optimizer).__name__.lower()
        state["epoch"] = epoch
        state["best_val"] = best_val
        state["best_state"] = best_state
        state["epochs_without_improvement"] = epochs_without_improvement
        state["rng_state"] = capture_rng_state()
        return state


class TorchTrainableEstimator[BatchT](TorchEstimator[MaskedPanels], ABC):
    """Execute a model-specific ``TrainingTask`` through the shared torch trainer."""

    @abstractmethod
    def _build_training_task(
        self,
        train: MaskedPanels,
        val: MaskedPanels | None,
        training: TrainingConfig,
        device: torch.device,
    ) -> TrainingTask[BatchT]:
        """Prepare fitted state, batches, and train/validation steps."""

    def fit(
        self,
        train: MaskedPanels,
        *,
        training: TrainingPolicy,
        val: MaskedPanels | None = None,
        tracker: TrainingTracker | None = None,
        checkpoint_dir: str | Path | None = None,
        resume: bool = False,
    ) -> None:
        """Build a model task and execute it through the shared trainer."""
        from llca.training.engine.trainer import Trainer

        if not isinstance(training, TrainingConfig):
            raise TypeError(
                f"{self._MODEL_NAME} requires the 'torch' training engine, got '{training.engine}'"
            )
        device = training.prepare()
        task = self._build_training_task(train, val, training, device)
        trainer = Trainer(
            config=training,
            task=task,
            tracker=tracker,
            checkpoint_dir=checkpoint_dir,
            checkpoint_state=self._training_state,
        )
        trainer.fit(resume=resume)
