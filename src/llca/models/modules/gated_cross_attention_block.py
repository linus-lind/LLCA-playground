from __future__ import annotations

from typing import cast

from torch import Tensor, nn

from llca.models.modules.gate_add_norm import GateAddNorm
from llca.models.modules.gated_residual_network import GatedResidualNetwork


class GatedCrossAttentionBlock(nn.Module):
    """Fuse query tokens with a separate context sequence through gated attention.

    ``query[B, Q, D]`` supplies only the representations for which outputs are required;
    ``key_value[B, S, D]`` supplies the full information set they may attend to. Keeping
    these roles separate avoids computing unused query outputs in target-specific models
    while preserving gradients through every attended context token. Static context
    enrichment is performed upstream in the local encoder, so this block carries no
    context pathway.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.gate_add_norm = GateAddNorm(d_model, d_model, dropout)
        self.grn = GatedResidualNetwork(d_model, d_model, dropout, d_model)

    def forward(
        self,
        query: Tensor,
        key_value: Tensor,
        key_padding_mask: Tensor | None = None,
    ) -> Tensor:
        """Return one refined output for every query token.

        ``key_padding_mask[B, S]`` follows PyTorch's convention: ``True`` excludes a
        context key. The unchanged query is the residual, so the block can fall back to
        the target representation when cross-sectional context is unhelpful.
        """
        attended, _ = self.attention(
            query,
            key_value,
            key_value,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        gated = self.gate_add_norm(attended, query)
        return cast(Tensor, self.grn(gated))
