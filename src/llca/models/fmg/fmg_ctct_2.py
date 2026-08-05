"""Full-universe FMG-CTCT-2 allocation network."""

from __future__ import annotations

from torch import Tensor

from llca.models.fmg.base import FmgTemporalModel
from llca.models.modules.conv_layer import ConvLayer
from llca.models.modules.gated_attention_block import GatedAttentionBlock


class FmgCtct2(FmgTemporalModel):
    """Produce cross-sectional scores from local histories and point-in-time context.

    One forward call represents a single prediction date with ``N`` instruments. Local
    encoding and temporal attention operate on ``[N, T, D]`` independently per instrument.
    Cross-sectional attention then exchanges information between instruments at each
    historical date before temporal aggregation produces one score per asset.
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
        self.cross_sectional_attention = GatedAttentionBlock(
            model_dim, n_heads, dropout=dropout, positional_encoding=None
        )

    def forward(
        self,
        features: Tensor,
        feature_age: Tensor,
        context: Tensor,
        context_age: Tensor,
        *,
        stock_mask: Tensor | None = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Score one cross-section and expose variable-selection diagnostics."""
        local, selection_weights = self._encode_local(features, feature_age, context, context_age)

        cross_input = local.transpose(0, 1)
        key_padding_mask = None
        if stock_mask is not None:
            key_padding_mask = (~stock_mask).unsqueeze(0).expand(cross_input.size(0), -1)
        cross_output = self.cross_sectional_attention(
            cross_input, key_padding_mask=key_padding_mask
        )
        universal = cross_output.transpose(0, 1)

        stock_embedding = self.aggregation(universal)
        scores = self.head(stock_embedding)
        return scores, selection_weights
