"""Build compact causal sequence references from indexed panel observations."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from llca.data.index_spec import time_level
from llca.data.modules.masked_panel import MaskedPanel


@dataclass(frozen=True, slots=True)
class WindowedTensor:
    """Reference overlapping windows without duplicating their underlying observations.

    ``values`` and ``age`` are compact ``[R_raw, F]`` buffers. ``starts[R]`` contains one
    offset per valid target; a target's history is the following ``W = window`` rows.
    Construction guarantees that a window remains within one segment and contiguous part
    of the global time calendar.
    """

    values: Tensor
    age: Tensor
    starts: Tensor
    window: int


@dataclass(frozen=True, slots=True)
class SequenceInput:
    """Declare a named column group as causal-history or point-in-time input.

    Windowed inputs are represented by ``[R_raw, F]`` buffers plus offsets; non-windowed
    inputs are aligned directly to valid targets as ``[R, F]`` tensors.
    """

    name: str
    columns: Sequence[str]
    windowed: bool = True


_PointTensor = tuple[Tensor, Tensor]
_Tensors = dict[str, WindowedTensor | _PointTensor]


def _segment_runs(
    segment: pd.Series, time_positions: np.ndarray, window: int
) -> Iterator[np.ndarray]:
    """Yield segment-local, calendar-contiguous row runs long enough for one window.

    Positions are ordered against the global observed calendar. Any gap greater than one
    calendar position splits a segment, preventing histories from crossing missing spans
    or discontinuous entity regimes.
    """
    positions = np.arange(len(segment))
    for _, group in pd.Series(positions).groupby(segment.to_numpy(), sort=False):
        rows = group.to_numpy()
        order = np.argsort(time_positions[rows], kind="stable")
        rows = rows[order]
        breaks = np.flatnonzero(np.diff(time_positions[rows]) > 1) + 1
        for run in np.split(rows, breaks):
            if len(run) >= window:
                yield run


def _point(frame: pd.DataFrame, index: pd.Index, columns: list[str]) -> Tensor:
    """Align point-in-time columns to target rows as a float tensor ``[R, F]``."""
    values = frame.reindex(index)[columns].to_numpy(dtype=np.float32)
    return torch.from_numpy(values)


def _concat_index(chunks: list[pd.Index], template: pd.Index) -> pd.Index:
    """Concatenate index chunks while preserving the template type for empty output."""
    if not chunks:
        return template[:0]
    return chunks[0].append(chunks[1:]) if len(chunks) > 1 else chunks[0]


def build_sequences(
    masked: MaskedPanel,
    inputs: Sequence[SequenceInput],
    sequence_length: int,
    buffer_size: int,
) -> tuple[_Tensors, pd.Index]:
    """Construct reusable causal windows and aligned point-in-time inputs.

    ``W = sequence_length + buffer_size``. All windowed column groups share offsets
    because valid placement depends only on segment and calendar continuity; their compact
    buffers remain separate views by requested columns. The returned index has ``R`` rows,
    matches every ``starts`` vector and point input, and identifies the final observation
    of each window.
    """
    windowed_inputs = [item for item in inputs if item.windowed]
    point_inputs = [item for item in inputs if not item.windowed]
    if not windowed_inputs:
        raise ValueError("build_sequences requires at least one windowed input")

    window = sequence_length + buffer_size
    window_columns = [column for item in windowed_inputs for column in item.columns]
    time = time_level(masked.values)
    dates = pd.DatetimeIndex(masked.values.index.get_level_values(time))
    calendar = dates.unique().sort_values()
    time_positions = calendar.get_indexer(dates)

    values_frame = masked.values[window_columns].to_numpy(dtype=np.float32)
    age_frame = masked.age[window_columns].to_numpy(dtype=np.float32)

    flat_chunks: list[np.ndarray] = []
    flat_age_chunks: list[np.ndarray] = []
    start_chunks: list[np.ndarray] = []
    index_chunks: list[pd.Index] = []
    offset = 0

    for run in _segment_runs(masked.segment, time_positions, window):
        flat_chunks.append(values_frame[run])
        flat_age_chunks.append(age_frame[run])
        start_chunks.append(offset + np.arange(len(run) - window + 1, dtype=np.int64))
        index_chunks.append(masked.values.index[run[window - 1 :]])
        offset += len(run)

    index = _concat_index(index_chunks, masked.values.index)
    tensors: _Tensors = {}

    if not flat_chunks:
        empty_flat = torch.empty((0, len(window_columns)), dtype=torch.float32)
        empty_starts = torch.empty(0, dtype=torch.long)
        col_offset = 0
        for item in windowed_inputs:
            width = len(item.columns)
            sl = slice(col_offset, col_offset + width)
            tensors[item.name] = WindowedTensor(
                empty_flat[:, sl], empty_flat[:, sl], empty_starts, window
            )
            col_offset += width
        for item in point_inputs:
            empty = torch.empty((0, len(item.columns)), dtype=torch.float32)
            tensors[item.name] = (empty, empty.clone())
        return tensors, index

    flat_values = torch.from_numpy(np.concatenate(flat_chunks, axis=0))
    flat_age = torch.from_numpy(np.concatenate(flat_age_chunks, axis=0))
    starts = torch.from_numpy(np.concatenate(start_chunks, axis=0))

    col_offset = 0
    for item in windowed_inputs:
        width = len(item.columns)
        sl = slice(col_offset, col_offset + width)
        tensors[item.name] = WindowedTensor(
            values=flat_values[:, sl].contiguous(),
            age=flat_age[:, sl].contiguous(),
            starts=starts,
            window=window,
        )
        col_offset += width

    for item in point_inputs:
        columns = list(item.columns)
        tensors[item.name] = (
            _point(masked.values, index, columns),
            _point(masked.age, index, columns),
        )

    return tensors, index
