from __future__ import annotations

import torch
from torch import Tensor, nn


class ContinuousVariableEncoder(nn.Module):
    """Encode continuous variables independently while accounting for data staleness.

    Each of the ``V`` variables owns a projection from a scalar to an ``E``-dimensional
    embedding and a learned non-negative decay rate. ``x`` and ``age`` have shape
    ``[..., V]``; non-negative ages attenuate stale values before projection, while a
    negative age selects a learned missing-value embedding. The result has shape
    ``[..., V, E]`` and preserves all leading sample or time dimensions.
    """

    def __init__(self, num_vars: int, embedding_dim: int) -> None:
        super().__init__()
        self.num_vars = num_vars
        self.embedding_dim = embedding_dim
        self.projections = nn.ModuleList(nn.Linear(1, embedding_dim) for _ in range(num_vars))
        self.decay = nn.Parameter(torch.zeros(num_vars))
        self.missing = nn.Parameter(torch.randn(num_vars, embedding_dim) * 0.02)

    def forward(self, x: Tensor, age: Tensor) -> Tensor:
        """Return freshness-adjusted embeddings with shape ``[..., V, E]``."""
        available = age >= 0
        safe_x = torch.where(available, x, torch.zeros_like(x))
        gamma = torch.exp(-torch.nn.functional.softplus(self.decay) * age.clamp(min=0.0))
        decayed = gamma * safe_x
        projected = torch.stack(
            [projection(decayed[..., i : i + 1]) for i, projection in enumerate(self.projections)],
            dim=-2,
        )
        return torch.where(available.unsqueeze(-1), projected, self.missing)
