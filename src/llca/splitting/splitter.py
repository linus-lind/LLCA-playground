from collections.abc import Iterator
from typing import Protocol

from llca.data.modules.masked_panel import MaskedPanels
from llca.splitting.fold import Fold


class Splitter(Protocol):
    """Define a date-based split strategy for synchronized named panels."""

    def split(
        self, panels: MaskedPanels, primary: str
    ) -> Iterator[tuple[Fold, MaskedPanels, MaskedPanels]]:
        """Yield fold metadata plus training and validation panels."""
        ...
