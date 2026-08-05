from __future__ import annotations

from torch import Tensor

from llca.models.fmg.base import FmgTemporalModel
from llca.models.modules.conv_layer import ConvLayer
from llca.models.modules.gated_cross_attention_block import GatedCrossAttentionBlock


class FmgCtct1(FmgTemporalModel):
    """Allocate one target asset using the full cross-section as predictive context.

    All instruments receive the shared local and temporal encoding. At every historical
    timestep, only the configured target representation acts as a query while every valid
    instrument remains available as a key/value. The resulting single universe-embedded
    target series is aggregated once and mapped to one signed allocation.
    """

    def __init__(
        self,
        num_features: int,
        num_context_vars: int,
        model_dim: int,
        feature_embedding_dim: int,
        sequence_length: int,
        cnn_layers: list[ConvLayer],
        n_heads: int,
        dropout: float,
        score_activation: str,
    ) -> None:
        super().__init__(
            num_features=num_features,
            num_context_vars=num_context_vars,
            model_dim=model_dim,
            feature_embedding_dim=feature_embedding_dim,
            sequence_length=sequence_length,
            cnn_layers=cnn_layers,
            n_heads=n_heads,
            dropout=dropout,
            score_activation=score_activation,
        )
        self.cross_sectional_attention = GatedCrossAttentionBlock(
            model_dim, n_heads, dropout=dropout
        )

    def forward(
        self,
        features: Tensor,
        feature_age: Tensor,
        context: Tensor,
        context_age: Tensor,
        target_index: Tensor,
        *,
        stock_mask: Tensor | None = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Return one target allocation and the shared selection diagnostics.

        Input rows are the current date's instruments. ``target_index`` is a scalar row
        position resolved from the stable entity identifier by the estimator; it is never
        inferred from cross-sectional ordering.
        """
        if target_index.numel() != 1:
            raise ValueError("target_index must contain exactly one row position")
        if target_index.dtype.is_floating_point or target_index.dtype.is_complex:
            raise TypeError("target_index must use an integer tensor dtype")
        if bool(((target_index < 0) | (target_index >= features.size(0))).any().item()):
            raise IndexError("target_index is outside the current cross-section")
        if stock_mask is not None and not bool(stock_mask[target_index].all().item()):
            raise ValueError("the target asset must be valid in stock_mask")

        local, selection_weights = self._encode_local(features, feature_age, context, context_age)
        cross_input = local.transpose(0, 1)  # [T, N, D]
        query = local.index_select(0, target_index.reshape(1)).transpose(0, 1)  # [T, 1, D]

        key_padding_mask = None
        if stock_mask is not None:
            key_padding_mask = (~stock_mask).unsqueeze(0).expand(cross_input.size(0), -1)
        target_universal = self.cross_sectional_attention(
            query,
            cross_input,
            key_padding_mask=key_padding_mask,
        ).transpose(0, 1)  # [1, T, D]

        target_embedding = self.aggregation(target_universal)
        allocation = self.head(target_embedding)
        return allocation, selection_weights
