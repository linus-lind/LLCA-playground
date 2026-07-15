"""Backend-neutral estimator lifecycle and MLflow persistence boundary."""

from __future__ import annotations

import os
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar, Self

import mlflow

from llca.models.estimators.evaluation_spec import EvaluationSpec
from llca.models.estimators.prediction import PredictionOutput
from llca.training.modules.tracking import TrainingTracker
from llca.training.modules.training_policy import TrainingPolicy


def _llca_package_dir() -> str:
    """Return the installed package root included with serialized MLflow models."""
    import llca

    return str(Path(llca.__file__).resolve().parent)


class Estimator[DataT](ABC):
    """Backend-neutral lifecycle for trainable and serializable pipeline models.

    Concrete implementations may use PyTorch, scikit-learn, native libraries, or an
    external engine. The orchestrator depends only on fitting, prediction, persistence,
    evaluation metadata, and an optional inference-device hook.
    """

    _MODEL_NAME: ClassVar[str]
    _BUNDLE_ARTIFACT: ClassVar[str]
    _BUNDLE_FILENAME: ClassVar[str]

    @abstractmethod
    def fit(
        self,
        train: DataT,
        *,
        training: TrainingPolicy,
        val: DataT | None = None,
        tracker: TrainingTracker | None = None,
        checkpoint_dir: str | Path | None = None,
        resume: bool = False,
    ) -> None:
        """Fit the estimator from training data and optional validation data."""
        ...

    @abstractmethod
    def predict(self, test: DataT) -> PredictionOutput:
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

    def set_inference_device(self, device: str) -> None:
        """Apply an optional backend-specific inference device selection."""
        del device

    @abstractmethod
    def _save(self, path: str | Path) -> None:
        """Write the complete inference bundle to ``path``."""

    @classmethod
    @abstractmethod
    def load(cls, path: str | Path, device: str = "auto") -> Self:
        """Restore an inference bundle using a backend-defined device interpretation."""

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
