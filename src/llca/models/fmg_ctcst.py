from torch import Tensor, nn

from llca.models.modules.context_encoder import ContextEncoder
from llca.models.modules.conv_layer import ConvLayer
from llca.models.modules.gated_attention_block import GatedAttentionBlock
from llca.models.modules.local_feature_encoder import LocalFeatureEncoder
from llca.models.modules.score_head import ScoreHead
from llca.models.modules.sinusoidal_positional_encoding import SinusoidalPositionalEncoding
from llca.models.modules.temporal_aggregation import TemporalAggregation

_SELECTION_CONTEXT = "feature_selection"
_GRN_CONTEXT = "pre_transformer"


class FmgLocalModel(nn.Module):
    """Share context, local-feature encoding, and the final head across FMG variants.

    Raw features include ``B = buffer_size`` additional left-context rows for the causal
    convolution. ``feature_embedding_dim`` is used only while the ``F`` raw variables are
    separate; all subsequent modules use ``D = model_dim``. Subclasses choose their
    temporal processor and whether any cross-sectional computation is present.
    """

    def __init__(
        self,
        num_features: int,
        num_context_vars: int,
        model_dim: int,
        feature_embedding_dim: int,
        cnn_layers: list[ConvLayer],
        dropout: float,
        score_activation: str,
    ) -> None:
        super().__init__()
        self.selection_context_name = _SELECTION_CONTEXT
        self.grn_context_name = _GRN_CONTEXT

        self.context_encoder = ContextEncoder(
            num_context_vars, model_dim, (_SELECTION_CONTEXT, _GRN_CONTEXT), dropout=dropout
        )
        self.feature_encoder = LocalFeatureEncoder(
            num_features=num_features,
            model_dim=model_dim,
            feature_embedding_dim=feature_embedding_dim,
            cnn_layers=cnn_layers,
            selection_context_size=model_dim,
            grn_context_size=model_dim,
            dropout=dropout,
        )
        self.head = ScoreHead(model_dim, score_activation)

    @property
    def buffer_size(self) -> int:
        """Return the additional feature-history rows required by the local encoder."""
        return self.feature_encoder.buffer_size

    def _encode_features(
        self,
        features: Tensor,
        feature_age: Tensor,
        context: Tensor,
        context_age: Tensor,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Build the CNN-enhanced feature sequence before temporal processing."""
        contexts, context_weights = self.context_encoder(context, context_age)
        encoded, feature_weights = self.feature_encoder(
            features,
            feature_age,
            selection_context=contexts[self.selection_context_name],
            grn_context=contexts[self.grn_context_name],
        )
        selection_weights = {"context": context_weights, "feature": feature_weights}
        return encoded, selection_weights


class FmgTemporalModel(FmgLocalModel):
    """Add gated temporal self-attention and learned temporal aggregation."""

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
            cnn_layers=cnn_layers,
            dropout=dropout,
            score_activation=score_activation,
        )
        self.temporal_attention = GatedAttentionBlock(
            model_dim,
            n_heads,
            dropout=dropout,
            positional_encoding=SinusoidalPositionalEncoding(model_dim, max_len=sequence_length),
        )
        self.aggregation = TemporalAggregation(model_dim)

    def _encode_local(
        self,
        features: Tensor,
        feature_age: Tensor,
        context: Tensor,
        context_age: Tensor,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Build temporally contextualized representations for transformer variants."""
        encoded, selection_weights = self._encode_features(
            features, feature_age, context, context_age
        )
        return self.temporal_attention(encoded), selection_weights


class FmgCtcst(FmgTemporalModel):
    """Produce cross-sectional scores from local histories and point-in-time context.

    One forward call represents a single prediction date with ``N`` instruments. Local
    encoding and temporal attention operate on ``[N, T, D]`` independently per instrument.
    Transposing to ``[T, N, D]`` then makes instruments the attention sequence, allowing
    information exchange within each historical date. Temporal aggregation reduces the
    result to ``[N, D]`` and the score head returns one unnormalized score per instrument.
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
        """Score one cross-section and expose variable-selection diagnostics.

        ``features`` and ``feature_age`` are ``[N, T + B, F]``; ``context`` and
        ``context_age`` are ``[N, C]``. ``stock_mask[N]`` is ``True`` for valid entities
        and prevents padded entities from acting as attention keys. Returns scores ``[N]``
        plus context weights ``[N, C]`` and feature weights ``[N, T + B, F]``.
        """
        local, selection_weights = self._encode_local(
            features, feature_age, context, context_age
        )

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
