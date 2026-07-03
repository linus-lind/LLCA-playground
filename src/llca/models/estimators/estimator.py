from __future__ import annotations

import os
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar, Self

import mlflow
import torch
from torch import Tensor

from llca.data.modules.masked_panel import MaskedPanels
from llca.models.estimators.evaluation_spec import EvaluationSpec
from llca.models.estimators.prediction import PredictionOutput
from llca.training.modules.tracking import TrainingTracker
from llca.training.modules.training_config import TrainingConfig
from llca.training.modules.training_diagnostics import objective_loss
from llca.training.modules.training_task import TrainingTask
from llca.training.reproducibility import capture_rng_state


def _llca_package_dir() -> str:
    """Return the installed package root included with serialized MLflow models."""
    import llca

    return str(Path(llca.__file__).resolve().parent)


class Estimator(ABC):
    """Define the common lifecycle for trainable and serializable pipeline models.

    Subclasses translate generic ``MaskedPanels`` into model inputs and return a typed
    ``PredictionOutput``. Persistence has two levels: inference bundles contain only the
    state required by ``predict``; resumable training states additionally contain the
    optimizer, progress counters, best weights, and random-number-generator state.
    """

    _MODEL_NAME: ClassVar[str]
    _BUNDLE_ARTIFACT: ClassVar[str]
    _BUNDLE_FILENAME: ClassVar[str]

    @abstractmethod
    def fit(
        self,
        train: MaskedPanels,
        *,
        training: TrainingConfig,
        val: MaskedPanels | None = None,
        tracker: TrainingTracker | None = None,
        checkpoint_dir: str | Path | None = None,
        resume: bool = False,
    ) -> None:
        """Fit the estimator from training panels and optional validation data."""
        ...

    @abstractmethod
    def predict(self, test: MaskedPanels) -> PredictionOutput:
        """Return native model outputs indexed like the predictable test observations."""
        ...

    @property
    @abstractmethod
    def evaluation_spec(self) -> EvaluationSpec:
        """Return model-independent dataset roles needed by the analytics pipeline."""

    @property
    def required_history(self) -> int:
        """Number of prior dates inference needs before its first reported prediction."""
        return 0

    @abstractmethod
    def to_device(self, device: torch.device) -> None:
        """Move inference state to a caller-selected device without rebuilding the model."""

    @abstractmethod
    def _inference_payload(self) -> dict[str, Any]:
        """Everything required to reconstruct the estimator for inference."""

    @classmethod
    @abstractmethod
    def _from_payload(cls, payload: dict[str, Any], map_location: torch.device) -> Self:
        """Construct a bare estimator from a payload, before state is restored."""

    @abstractmethod
    def _restore(self, payload: dict[str, Any]) -> None:
        """Load model weights and preprocessing state from a payload in place."""

    def _save(self, path: str | Path) -> None:
        """Serialize the subclass-defined inference bundle to ``path``."""
        torch.save(self._inference_payload(), Path(path))

    @classmethod
    def load(cls, path: str | Path, map_location: torch.device) -> Self:
        """Reconstruct an estimator and restore its inference state on ``map_location``."""
        payload = torch.load(Path(path), map_location=map_location, weights_only=False)
        estimator = cls._from_payload(payload, map_location)
        estimator._restore(payload)
        return estimator

    def log_model(self) -> str:
        """Log the inference bundle as an MLflow pyfunc model and return its URI."""
        from llca.models.pyfunc import Pyfunc

        with tempfile.TemporaryDirectory() as tmp:
            bundle = os.path.join(tmp, self._BUNDLE_FILENAME)
            self._save(bundle)
            info = mlflow.pyfunc.log_model(
                name=self._MODEL_NAME,
                python_model=Pyfunc(type(self), self._BUNDLE_ARTIFACT),
                artifacts={self._BUNDLE_ARTIFACT: bundle},
                code_paths=[_llca_package_dir()],
            )
        return str(info.model_uri)

    @staticmethod
    def _loss_value(output: object) -> Tensor:
        """Loss modules may return either a bare scalar `Tensor` or a dataclass with a
        `.loss` field plus auxiliary diagnostics (e.g. `PortfolioLossOutput` — turnover,
        variance, ... logged separately but not backpropagated). This unwraps either form
        to the single tensor `.backward()` should be called on.
        """
        return objective_loss(output)

    def _training_state(
        self,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        best_val: float,
        best_state: dict[str, Tensor] | None,
        epochs_without_improvement: int,
    ) -> dict[str, Any]:
        """Extend the inference bundle with all state required for deterministic resume."""
        state = self._inference_payload()
        state["optimizer_state_dict"] = optimizer.state_dict()
        state["optimizer_name"] = type(optimizer).__name__.lower()
        state["epoch"] = epoch
        state["best_val"] = best_val
        state["best_state"] = best_state
        state["epochs_without_improvement"] = epochs_without_improvement
        state["rng_state"] = capture_rng_state()
        return state


class TrainableEstimator[BatchT](Estimator, ABC):
    """Provide the reusable fit lifecycle around a model-specific ``TrainingTask``.

    Subclasses prepare data and define forward/objective behavior in one task factory.
    Device resolution, optimizer execution, tracking, checkpointing, early stopping, and
    exact resume remain shared by every trainable estimator.
    """

    @abstractmethod
    def _build_training_task(
        self,
        train: MaskedPanels,
        val: MaskedPanels | None,
        training: TrainingConfig,
        device: torch.device,
    ) -> TrainingTask[BatchT]:
        """Prepare fitted preprocessing state, model, batches, and train/validation steps."""

    def fit(
        self,
        train: MaskedPanels,
        *,
        training: TrainingConfig,
        val: MaskedPanels | None = None,
        tracker: TrainingTracker | None = None,
        checkpoint_dir: str | Path | None = None,
        resume: bool = False,
    ) -> None:
        """Build a model-specific task and execute it through the shared trainer."""
        from llca.training.trainer import Trainer

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
