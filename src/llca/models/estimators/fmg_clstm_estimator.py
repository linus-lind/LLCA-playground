"""Bind the target-only data path to the recurrent FMG-CLSTM model."""

from typing import cast

from torch import Tensor

from llca.models.estimators.fmg_ctcst_estimator import _conv_layer
from llca.models.estimators.fmg_ctt_estimator import FmgCttEstimator
from llca.models.fmg_clstm import FmgClstm
from llca.models.fmg_ctcst import FmgLocalModel


class FmgClstmEstimator(FmgCttEstimator):
    """Train and serve one direct allocation using only the target's LSTM state."""

    _MODEL_NAME = "fmg-clstm"
    _BUNDLE_ARTIFACT = "fmg-clstm_bundle"
    _BUNDLE_FILENAME = "fmg-clstm.pt"

    def _build_model(self) -> FmgLocalModel:
        """Construct the configured LSTM after target input widths are known."""
        cnn_layers = [_conv_layer(layer) for layer in self._config.cnn.layers]
        lstm = self._config.lstm
        return FmgClstm(
            num_features=len(self._feature_columns),
            num_context_vars=len(self._context_columns),
            model_dim=int(self._config.d_model),
            feature_embedding_dim=int(self._config.feature_embedding_dim),
            cnn_layers=cnn_layers,
            lstm_num_layers=int(lstm.num_layers),
            lstm_recurrent_dropout=float(lstm.recurrent_dropout),
            lstm_output_dropout=float(lstm.output_dropout),
            lstm_bias=bool(lstm.bias),
            lstm_bidirectional=bool(lstm.bidirectional),
            dropout=float(self._config.dropout),
            score_activation=str(self._config.score_activation),
        )

    def _allocate(
        self,
        features: Tensor,
        feature_age: Tensor,
        context: Tensor,
        context_age: Tensor,
        target_index: Tensor,
    ) -> Tensor:
        assert isinstance(self._model, FmgClstm)
        if target_index.numel() != 1 or int(target_index.item()) != 0:
            raise ValueError("fmg-clstm target must be the sole row at position zero")
        allocation, _ = self._model(features, feature_age, context, context_age)
        return cast(Tensor, allocation.float())
