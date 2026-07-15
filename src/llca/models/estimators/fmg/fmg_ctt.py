"""Train and serve the cross-section-free FMG-CTT allocation model."""

from __future__ import annotations

from typing import cast

from torch import Tensor

from llca.models.estimators.fmg.base import conv_layer_from_config
from llca.models.estimators.fmg.single_asset import FmgTargetOnlyEstimator
from llca.models.fmg import FmgCtt
from llca.models.fmg.base import FmgLocalModel


class FmgCttEstimator(FmgTargetOnlyEstimator):
    """Restrict all inputs to one configured entity before fitting or inference."""

    _MODEL_NAME = "fmg-ctt"
    _BUNDLE_ARTIFACT = "fmg-ctt_bundle"
    _BUNDLE_FILENAME = "fmg-ctt.pt"

    def _build_model(self) -> FmgLocalModel:
        """Construct the temporal-only network after target input widths are known."""
        transformer = self._config.transformer
        cnn_layers = [conv_layer_from_config(layer) for layer in self._config.cnn.layers]
        return FmgCtt(
            num_features=len(self._feature_columns),
            num_context_vars=len(self._context_columns),
            model_dim=int(self._config.d_model),
            feature_embedding_dim=int(self._config.feature_embedding_dim),
            sequence_length=int(self._config.sequence_length),
            cnn_layers=cnn_layers,
            n_heads=int(transformer.n_heads),
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
        if not isinstance(self._model, FmgCtt):
            raise RuntimeError(f"{self._MODEL_NAME} is not fitted")
        if target_index.numel() != 1 or int(target_index.item()) != 0:
            raise ValueError("fmg-ctt target must be the sole row at position zero")
        allocation, _ = self._model(features, feature_age, context, context_age)
        return cast(Tensor, allocation.float())
