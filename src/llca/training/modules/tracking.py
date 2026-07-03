from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol

from llca.training.modules.training_diagnostics import BatchMetadata


class TrainingTracker(Protocol):
    """Backend-independent event contract consumed by the reusable trainer."""

    def begin(
        self,
        total_epochs: int,
        steps_per_epoch: int | None = None,
        *,
        start_step: int = 0,
        batch_manifest: list[BatchMetadata] | None = None,
    ) -> None:
        """Start or resume tracking for one optimization session."""

    def on_batch_end(
        self,
        loss: float,
        *,
        grad_norm: float | None = None,
        clipped: bool = False,
        observations: int | None = None,
        diagnostics: Mapping[str, float] | None = None,
    ) -> None:
        """Record one completed optimizer step."""

    def on_epoch_end(
        self,
        epoch: int,
        *,
        metrics: Mapping[str, float],
        learning_rate: float | None = None,
    ) -> None:
        """Record one completed epoch."""
