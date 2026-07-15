"""Backend-neutral contract for estimator fitting policies."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol, runtime_checkable

from llca.pipeline.contracts import TrainingEngine

type TrackingParameter = str | int | float | bool


@runtime_checkable
class TrainingPolicy(Protocol):
    """Configuration consumed by one registered estimator training engine."""

    @property
    def engine(self) -> TrainingEngine:
        """Identify the compatible execution implementation."""

    @property
    def tracking_interval(self) -> int:
        """Return the preferred progress-event interval for the tracking backend."""

    def tracking_parameters(self) -> Mapping[str, TrackingParameter]:
        """Return stable scalar parameters suitable for experiment comparison."""
