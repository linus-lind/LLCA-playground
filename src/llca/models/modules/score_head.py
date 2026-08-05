from __future__ import annotations

from typing import cast

from torch import Tensor, nn

_ACTIVATIONS: dict[str, type[nn.Module]] = {
    "identity": nn.Identity,
    "tanh": nn.Tanh,
    "softsign": nn.Softsign,
}


class ScoreHead(nn.Module):
    """Map entity embeddings ``[..., D]`` to scalar model outputs ``[...]``.

    The optional activation controls the native score range. The head does not normalize
    values across entities; allocation or probability normalization belongs to a
    downstream objective or output adapter.
    """

    def __init__(self, d_model: int, activation: str = "identity") -> None:
        super().__init__()
        if activation not in _ACTIVATIONS:
            raise ValueError(
                f"unknown score activation '{activation}', available: {sorted(_ACTIVATIONS)}"
            )
        self.linear = nn.Linear(d_model, 1)
        self.activation = _ACTIVATIONS[activation]()

    def forward(self, embedding: Tensor) -> Tensor:
        """Project the final embedding axis to one scalar and remove that axis."""
        return cast(Tensor, self.activation(self.linear(embedding)).squeeze(-1))
