from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class SinusoidalPositionalEncoding(nn.Module):
    """Add fixed sinusoidal positions to an ordered sequence.

    For input ``[..., S, D]``, positions vary along ``S`` and broadcast across all earlier
    dimensions. The precomputed ``[max_len, D]`` encoding is registered as a buffer, so it
    follows device moves and checkpoints without receiving gradients.
    """

    pe: Tensor

    def __init__(self, d_model: int, max_len: int) -> None:
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].size(1)])
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """Add encodings for the current sequence length, validating ``S <= max_len``."""
        seq_len = x.size(-2)
        if seq_len > self.pe.size(0):
            raise ValueError(
                f"sequence length {seq_len} exceeds positional encoding max_len {self.pe.size(0)}"
            )
        return x + self.pe[:seq_len]
