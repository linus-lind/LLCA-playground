import torch
from torch import Tensor, nn


class TemporalAggregation(nn.Module):
    """Collapse a temporal sequence using attention from its most recent state.

    A projected final timestep forms one query per sample. Its similarity to every state
    produces normalized temporal weights, reducing ``z[N, T, D]`` to ``[N, D]`` without
    introducing a separate learned query vector.
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.query_projection = nn.Linear(d_model, d_model, bias=False)

    def forward(self, z: Tensor) -> Tensor:
        """Return one attention-weighted embedding per sample."""
        query = self.query_projection(z[:, -1, :])
        scores = torch.einsum("btd,bd->bt", z, query)
        weights = torch.softmax(scores, dim=-1)
        return torch.einsum("bt,btd->bd", weights, z)
