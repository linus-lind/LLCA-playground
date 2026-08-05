"""Runtime policy for discovering and resuming interrupted training runs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

type RecoveryMode = Literal["off", "list", "auto", "explicit"]


@dataclass(frozen=True, slots=True)
class RecoveryConfig:
    """Select how the training entry point handles unfinished MLflow runs."""

    mode: RecoveryMode
    run_id: str | None
    allow_source_mismatch: bool
