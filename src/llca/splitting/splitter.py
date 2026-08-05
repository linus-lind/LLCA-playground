from __future__ import annotations

from collections.abc import Iterator
from typing import Protocol

from llca.splitting.fold import Fold


class Splitter[DataT](Protocol):
    """Define a date-based split strategy for synchronized named panels."""

    @property
    def name(self) -> str:
        """Stable execution-strategy name used for tracking and manifests."""
        ...

    def split(self, data: DataT, primary: str) -> Iterator[tuple[Fold, DataT, DataT]]:
        """Yield fold metadata plus training and validation panels."""
        ...
