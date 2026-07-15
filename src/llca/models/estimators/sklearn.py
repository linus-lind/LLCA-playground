"""Reusable lifecycle for picklable scikit-learn-compatible estimators."""

from __future__ import annotations

import pickle
from abc import ABC, abstractmethod
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Self, cast

from llca.models.estimators.estimator import Estimator
from llca.training.modules.sklearn_config import SklearnTrainingConfig
from llca.training.modules.tracking import TrainingTracker
from llca.training.modules.training_policy import TrainingPolicy


class SklearnEstimator[DataT](Estimator[DataT], ABC):
    """Provide policy validation, scalar tracking, and backend-neutral persistence.

    Subclasses own conversion from their selected data view to arrays, estimator
    construction, task-specific fit metrics, and prediction semantics. The base is also
    suitable for libraries exposing a scikit-learn-style fit/predict API.
    """

    @abstractmethod
    def _fit_backend(
        self,
        train: DataT,
        val: DataT | None,
        training: SklearnTrainingConfig,
    ) -> Mapping[str, float]:
        """Fit backend state and return only metrics meaningful to this objective."""

    @abstractmethod
    def _inference_payload(self) -> dict[str, Any]:
        """Return all fitted state required for inference."""

    @classmethod
    @abstractmethod
    def _from_payload(cls, payload: dict[str, Any]) -> Self:
        """Construct an estimator shell from serialized state."""

    @abstractmethod
    def _restore(self, payload: dict[str, Any]) -> None:
        """Restore fitted backend and preprocessing state in place."""

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
        """Fit a classical estimator once and emit its declared scalar diagnostics."""
        del checkpoint_dir
        if not isinstance(training, SklearnTrainingConfig):
            raise TypeError(
                f"{self._MODEL_NAME} requires the 'sklearn' training engine, "
                f"got '{training.engine}'"
            )
        if resume:
            raise NotImplementedError(
                f"{self._MODEL_NAME} does not implement incremental checkpoint recovery"
            )
        if tracker is not None:
            tracker.begin(total_epochs=1, steps_per_epoch=1)
        metrics = self._fit_backend(train, val, training)
        if tracker is not None:
            tracker.on_epoch_end(0, metrics=metrics)

    def _save(self, path: str | Path) -> None:
        """Serialize the trusted fitted payload used by the MLflow model artifact."""
        with Path(path).open("wb") as stream:
            pickle.dump(self._inference_payload(), stream, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str | Path, device: str = "auto") -> Self:
        """Restore a trusted MLflow bundle; classical estimators ignore device choice."""
        del device
        with Path(path).open("rb") as stream:
            payload = pickle.load(stream)  # noqa: S301 - trusted MLflow artifact
        if not isinstance(payload, dict):
            raise TypeError("sklearn estimator bundle must contain a mapping payload")
        typed_payload = cast(dict[str, Any], payload)
        estimator = cls._from_payload(typed_payload)
        estimator._restore(typed_payload)
        return estimator
