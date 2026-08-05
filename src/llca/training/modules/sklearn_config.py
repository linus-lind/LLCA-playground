"""Training policy for estimators fitted by scikit-learn-compatible engines."""

from __future__ import annotations

from dataclasses import dataclass

from llca.pipeline.contracts import TrainingEngine


@dataclass(frozen=True, slots=True)
class SklearnTrainingConfig:
    """Backend settings shared by random forests, boosting, and linear estimators."""

    seed: int
    n_jobs: int
    log_interval: int = 1

    @property
    def engine(self) -> TrainingEngine:
        return TrainingEngine.SKLEARN

    @property
    def tracking_interval(self) -> int:
        return self.log_interval

    def tracking_parameters(self) -> dict[str, str | int | float | bool]:
        return {
            "training.engine": self.engine,
            "training.seed": self.seed,
            "training.n_jobs": self.n_jobs,
        }
