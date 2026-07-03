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
