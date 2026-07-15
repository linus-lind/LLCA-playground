"""Provide ragged panel batching and lazy device transfer for sequence estimators."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import torch
from torch import Tensor


@dataclass(frozen=True, slots=True)
class DateSlice:
    """Map one date's flat observation rows into a batch-wide entity axis.

    ``rows[K]`` and ``cols[K]`` are parallel vectors. The former selects source
    observations; the latter gives their positions in the dense ``N_max`` entity axis
    shared by all dates in the batch.
    """

    rows: Tensor
    cols: Tensor


@dataclass(frozen=True, slots=True)
class Batch:
    """Describe consecutive dates packed into a dense ``[D_batch, N_max]`` layout.

    ``N_max`` is the union of entities appearing anywhere in the block, not a per-date
    count. Individual ``DateSlice`` objects identify which positions are valid.
    """

    n_max: int
    dates: list[DateSlice]
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    observations: int


@dataclass(frozen=True, slots=True)
class Field:
    """Store aligned point-in-time values and ages for indexed device retrieval.

    ``values`` and ``age`` have shape ``[R, C]``. ``rows`` selects along ``R`` and moves
    only the selected tensors to the configured execution device.
    """

    values: Tensor
    age: Tensor
    device: torch.device

    def rows(self, rows: Tensor) -> tuple[Tensor, Tensor]:
        """Gather the requested observation rows and transfer them to the execution device."""
        return self.values[rows].to(self.device), self.age[rows].to(self.device)


@dataclass(frozen=True, slots=True)
class Window:
    """Represent overlapping causal windows by compact CPU buffers and start offsets.

    ``values`` and ``age`` remain ``[R_raw, F]`` rather than being expanded for every
    window. Retrieval gathers ``[K, W, F]`` for ``K`` requested target rows and transfers
    only that batch to the execution device, bounding device memory by batch size.
    """

    values: Tensor
    age: Tensor
    starts: Tensor
    window: int
    device: torch.device

    def rows(self, rows: Tensor) -> tuple[Tensor, Tensor]:
        """Gather causal slices ``[K, W, F]`` from compact start offsets."""
        starts = self.starts[rows]
        idx = starts.unsqueeze(1) + torch.arange(self.window).unsqueeze(0)
        return self.values[idx].to(self.device), self.age[idx].to(self.device)


def build_batches(index: pd.Index, batch_size: int) -> list[Batch]:
    """Group flat rows into consecutive date blocks with a shared entity layout.

    ``batch_size`` counts dates, not observations. Within each block, every distinct
    entity receives one column, and each date stores parallel source-row and destination-
    column indices. A plain date index is treated as a single-entity panel.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not index.is_unique:
        raise ValueError("batch index must not contain duplicate date/entity rows")
    dates = index.get_level_values(0)
    instruments = index.get_level_values(1) if index.nlevels > 1 else pd.Index([0] * len(index))

    by_date: dict[object, list[tuple[int, object]]] = {}
    for position in range(len(index)):
        by_date.setdefault(dates[position], []).append((position, instruments[position]))
    ordered_dates = dates.unique().sort_values()

    batches: list[Batch] = []
    for start in range(0, len(ordered_dates), batch_size):
        block = ordered_dates[start : start + batch_size]
        union: dict[object, int] = {}
        for date in block:
            for _, instrument in by_date[date]:
                union.setdefault(instrument, len(union))
        slices = [
            DateSlice(
                rows=torch.tensor([position for position, _ in by_date[date]], dtype=torch.long),
                cols=torch.tensor(
                    [union[instrument] for _, instrument in by_date[date]], dtype=torch.long
                ),
            )
            for date in block
        ]
        batches.append(
            Batch(
                n_max=len(union),
                dates=slices,
                start_date=pd.Timestamp(block[0]),
                end_date=pd.Timestamp(block[-1]),
                observations=sum(len(date_slice.rows) for date_slice in slices),
            )
        )
    return batches
