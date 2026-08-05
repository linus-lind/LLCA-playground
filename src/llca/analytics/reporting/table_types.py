from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd

type NumberFormat = Literal["decimal", "percent", "integer", "pvalue"]


@dataclass(frozen=True, slots=True)
class PublicationTable:
    """Hold one numeric or textual table together with paper-facing metadata."""

    name: str
    title: str
    caption: str
    frame: pd.DataFrame
    row_formats: tuple[NumberFormat, ...] = ()
    column_formats: tuple[NumberFormat, ...] = ()
    cell_p_values: pd.DataFrame | None = None
    panels: tuple[PublicationTable, ...] = ()
    layout_columns: int = 1


@dataclass(frozen=True, slots=True)
class PublicationReport:
    """Describe the report directory and every generated table artifact."""

    directory: Path
    artifacts: dict[str, tuple[Path, ...]]
