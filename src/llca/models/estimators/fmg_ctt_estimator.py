"""Train and serve the cross-section-free FMG-CTT allocation model."""

from __future__ import annotations

from typing import cast

import pandas as pd
import torch
from torch import Tensor

from llca.data.modules.masked_panel import MaskedPanels
from llca.models.estimators.fmg_ctcst_estimator import (
    FmgCtcstEstimator,
    _conv_layer,
    _Raw,
)
from llca.models.estimators.fmg_ctct_1_estimator import FmgCtct1Estimator
from llca.models.estimators.prediction import PredictionOutput
from llca.models.fmg_ctt import FmgCtt
from llca.models.utils.sequences import WindowedTensor, build_sequences


class FmgCttEstimator(FmgCtct1Estimator):
    """Restrict all inputs to one configured entity before fitting or inference."""

    _MODEL_NAME = "fmg-ctt"
    _BUNDLE_ARTIFACT = "fmg-ctt_bundle"
    _BUNDLE_FILENAME = "fmg-ctt.pt"

    def _build_model(self) -> FmgCtt:
        """Construct the temporal-only network after target input widths are known."""
        transformer = self._config.transformer
        cnn_layers = [_conv_layer(layer) for layer in self._config.cnn.layers]
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

    def _windows(self, split: MaskedPanels) -> _Raw:
        """Build training windows after physically removing all other entities."""
        target_split = self._target_only_split(split)
        raw = FmgCtcstEstimator._windows(self, target_split)
        if len(raw.index) == 0:
            raise ValueError(
                f"target entity {self._target_entity_id} has no constructible sequence "
                f"with observed finite supervision in "
                f"'{self._supervision_dataset}.{self._supervision_column}'"
            )
        return raw

    def _allocate(
        self,
        features: Tensor,
        feature_age: Tensor,
        context: Tensor,
        context_age: Tensor,
        target_index: Tensor,
    ) -> Tensor:
        assert isinstance(self._model, FmgCtt)
        if target_index.numel() != 1 or int(target_index.item()) != 0:
            raise ValueError("fmg-ctt target must be the sole row at position zero")
        allocation, _ = self._model(features, feature_age, context, context_age)
        return cast(Tensor, allocation.float())

    @torch.inference_mode()
    def predict(self, test: MaskedPanels) -> PredictionOutput:
        """Return target allocations without constructing any non-target sequence."""
        assert (
            isinstance(self._model, FmgCtt)
            and self._feature_scaler is not None
            and self._context_scaler is not None
        )
        self._model.eval()
        target_test = self._target_only_split(test)
        tensors, raw_index = build_sequences(
            self._combined(target_test),
            self._inputs(),
            self._sequence_length,
            self._model.buffer_size,
        )
        index = cast(pd.MultiIndex, raw_index)
        if len(index) == 0:
            raise ValueError(
                f"target entity {self._target_entity_id} has no constructible test sequence"
            )

        features_raw = cast(WindowedTensor, tensors["features"])
        context_val, context_age = cast(tuple[Tensor, Tensor], tensors["context"])
        features = self._windowed_field(features_raw, self._feature_scaler)
        context = self._field((context_val, context_age), self._context_scaler)

        values: list[float] = []
        for position in range(len(index)):
            rows = torch.tensor([position], dtype=torch.long)
            target_index = torch.zeros(1, dtype=torch.long, device=self._device)
            allocation = self._allocate(
                *features.rows(rows),
                *context.rows(rows),
                target_index,
            )
            values.append(float(allocation[0].cpu().item()))

        return PredictionOutput(
            kind="allocation",
            values=pd.Series(values, index=index, name="weight"),
        )
