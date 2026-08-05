from __future__ import annotations

from typing import cast

from torch import Tensor, nn

from llca.models.modules.gate_add_norm import GateAddNorm
from llca.models.modules.gated_residual_network import GatedResidualNetwork


class GatedAttentionBlock(nn.Module):
    """Combine self-attention with a gated residual and context-conditioned refinement.

    The block accepts ``x[B, S, D]`` and is agnostic to what batch ``B`` and sequence
    ``S`` represent. Optional positional encoding is applied before attention when the
    sequence is ordered. Attention weights are not materialized, allowing PyTorch to use
    memory-efficient attention kernels where supported.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float,
        *,
        context_size: int | None = None,
        positional_encoding: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.positional_encoding = positional_encoding
        self.input_norm = nn.LayerNorm(d_model) if positional_encoding is not None else None
        self.attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.gate_add_norm = GateAddNorm(d_model, d_model, dropout)
        self.grn = GatedResidualNetwork(d_model, d_model, d_model, context_size, dropout)

    def forward(
        self,
        x: Tensor,
        context: Tensor | None = None,
        key_padding_mask: Tensor | None = None,
    ) -> Tensor:
        """Return attended embeddings with shape ``[B, S, D]``.

        ``key_padding_mask[B, S]`` follows PyTorch's convention: ``True`` excludes a key.
        ``context[B, C]`` is broadcast over the sequence before the residual network.
        """
        y = x
        if self.positional_encoding is not None and self.input_norm is not None:
            y = self.input_norm(self.positional_encoding(x))

        attended, _ = self.attention(y, y, y, key_padding_mask=key_padding_mask, need_weights=False)
        gated = self.gate_add_norm(attended, y)

        broadcast_context = None
        if context is not None:
            broadcast_context = context.unsqueeze(1).expand(*gated.shape[:-1], context.size(-1))

        return cast(Tensor, self.grn(gated, broadcast_context))
