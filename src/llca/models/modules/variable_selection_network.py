import torch
from torch import Tensor, nn

from llca.models.modules.gated_residual_network import GatedResidualNetwork


class VariableSelectionNetwork(nn.Module):
    """Reduce a variable axis with learned, context-dependent selection weights.

    Input embeddings have shape ``[..., V, E]``. A gating network produces ``V``
    softmax weights per leading position, while independent residual networks transform
    each variable embedding. Their weighted sum has shape ``[..., E]``; the returned
    weights have shape ``[..., V]`` and can be used for diagnostics.
    """

    def __init__(
        self,
        num_vars: int,
        embedding_dim: int,
        context_size: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_vars = num_vars
        self.embedding_dim = embedding_dim
        self.weight_grn = GatedResidualNetwork(
            input_size=num_vars * embedding_dim,
            hidden_size=embedding_dim,
            output_size=num_vars,
            context_size=context_size,
            dropout=dropout,
        )
        self.variable_grns = nn.ModuleList(
            GatedResidualNetwork(
                input_size=embedding_dim,
                hidden_size=embedding_dim,
                output_size=embedding_dim,
                dropout=dropout,
            )
            for _ in range(num_vars)
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(
        self, embeddings: Tensor, age: Tensor | None = None, context: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        """Select variables, excluding unavailable inputs without producing invalid softmaxes.

        ``age`` has shape ``[..., V]`` and negative entries mask individual variables.
        If every variable at a position is unavailable, the mask is relaxed for that
        position so the learned logits remain finite instead of applying softmax to only
        negative infinity values.
        """
        flattened = embeddings.flatten(start_dim=-2)
        logits = self.weight_grn(flattened, context)
        if age is not None:
            available = age >= 0
            keep = available | ~available.any(dim=-1, keepdim=True)
            logits = logits.masked_fill(~keep, float("-inf"))
        weights: Tensor = self.softmax(logits)
        processed = torch.stack(
            [grn(embeddings[..., i, :]) for i, grn in enumerate(self.variable_grns)],
            dim=-2,
        )
        combined: Tensor = (processed * weights.unsqueeze(-1)).sum(dim=-2)
        return combined, weights
