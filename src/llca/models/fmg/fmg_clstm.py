from __future__ import annotations

from torch import Tensor

from llca.models.fmg.base import FmgLocalModel
from llca.models.modules.conv_layer import ConvLayer
from llca.models.modules.temporal_lstm import TemporalLSTM


class FmgClstm(FmgLocalModel):
    """Allocate one target asset from a CNN-encoded history processed by an LSTM."""

    def __init__(
        self,
        num_features: int,
        num_context_vars: int,
        model_dim: int,
        feature_embedding_dim: int,
        cnn_layers: list[ConvLayer],
        lstm_num_layers: int,
        lstm_recurrent_dropout: float,
        lstm_output_dropout: float,
        lstm_bias: bool,
        lstm_bidirectional: bool,
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
        self.temporal_lstm = TemporalLSTM(
            model_dim=model_dim,
            num_layers=lstm_num_layers,
            recurrent_dropout=lstm_recurrent_dropout,
            output_dropout=lstm_output_dropout,
            bias=lstm_bias,
            bidirectional=lstm_bidirectional,
        )

    def forward(
        self,
        features: Tensor,
        feature_age: Tensor,
        context: Tensor,
        context_age: Tensor,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Map the sole target sequence directly from final LSTM state to allocation."""
        if features.size(0) != 1:
            raise ValueError(
                f"fmg-clstm expects exactly one asset per forward call, got {features.size(0)}"
            )
        encoded, selection_weights = self._encode_features(
            features, feature_age, context, context_age
        )
        target_embedding = self.temporal_lstm(encoded)
        allocation = self.head(target_embedding)
        return allocation, selection_weights
