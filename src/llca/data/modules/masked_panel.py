from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class MaskedPanel:
    """Store aligned values together with observation age and continuity metadata.

    ``values``, ``observed``, and ``age`` share shape ``[R, F]`` and identical axes.
    ``observed`` marks new source observations; ``age`` is zero there, increases while a
    value is carried forward, and is ``-1`` before first availability. ``segment[R]``
    prevents temporal operations from crossing discontinuities.
    """

    values: pd.DataFrame
    observed: pd.DataFrame
    age: pd.DataFrame
    segment: pd.Series

    def __post_init__(self) -> None:
        """Reject ambiguous or misaligned panel state at its construction boundary."""
        if not self.values.index.is_unique:
            raise ValueError("MaskedPanel values index must be unique")
        if not self.values.columns.is_unique:
            raise ValueError("MaskedPanel values columns must be unique")
        for name, frame in (("observed", self.observed), ("age", self.age)):
            if not frame.index.equals(self.values.index):
                raise ValueError(f"MaskedPanel {name} index must equal values index")
            if not frame.columns.equals(self.values.columns):
                raise ValueError(f"MaskedPanel {name} columns must equal values columns")
        if not self.segment.index.equals(self.values.index):
            raise ValueError("MaskedPanel segment index must equal values index")

    @property
    def columns(self) -> list[str]:
        return [str(column) for column in self.values.columns]

    def slice_rows(self, mask: pd.Series | np.ndarray) -> MaskedPanel:
        """Apply one row mask consistently to every component of the panel contract."""
        return MaskedPanel(
            values=self.values[mask],
            observed=self.observed[mask],
            age=self.age[mask],
            segment=self.segment[mask],
        )


MaskedPanels = dict[str, MaskedPanel]
