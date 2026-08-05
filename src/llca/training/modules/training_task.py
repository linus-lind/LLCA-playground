from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from torch import nn

from llca.models.estimators.objective_output import BatchMetadata, ObjectiveResult


@dataclass(frozen=True, slots=True)
class TrainingTask[BatchT]:
    """Bind model-specific data and steps to the model-independent optimization loop.

    Implementing a new estimator requires constructing this adapter: the trainer then owns
    optimization, precision, diagnostics, tracking, early stopping, and checkpoints.
    """

    model: nn.Module
    batches: Sequence[BatchT]
    train_step: Callable[[BatchT], ObjectiveResult]
    validation_step: Callable[[], float] | None = None
    batch_metadata: Callable[[BatchT, int], BatchMetadata] | None = None
    gradient_group: Callable[[str], str] | None = None
