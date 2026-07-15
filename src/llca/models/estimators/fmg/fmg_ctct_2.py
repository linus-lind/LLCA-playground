"""Training and inference adapter for the full-universe FMG-CTCT-2 model."""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from llca.data.modules.masked_panel import MaskedPanels
from llca.models.estimators.fmg.base import (
    FmgEstimator,
    PreparedWindows,
    conv_layer_from_config,
)
from llca.models.estimators.prediction import PredictionOutput
from llca.models.fmg import FmgCtct2
from llca.models.fmg.base import FmgLocalModel
from llca.models.utils.batching import Batch
from llca.models.utils.sequences import WindowedTensor, build_sequences
from llca.training.modules.training_diagnostics import (
    TrainingBatchOutput,
    objective_diagnostics,
    tensor_distribution_diagnostics,
)


def score_metrics(
    scores: Tensor,
    mask: Tensor,
    loss_output: object,
    saturation_threshold: float,
) -> dict[str, float | Tensor]:
    """Combine score distributions with objective-provided scalar diagnostics."""
    return tensor_distribution_diagnostics(
        scores,
        namespace="scores",
        mask=mask,
        saturation_threshold=saturation_threshold,
    ) | objective_diagnostics(loss_output)


class FmgCtct2Estimator(FmgEstimator):
    """Train and serve one score for every asset in a full cross-section."""

    _MODEL_NAME = "fmg-ctct-2"
    _BUNDLE_ARTIFACT = "fmg-ctct-2_bundle"
    _BUNDLE_FILENAME = "fmg-ctct-2.pt"

    def _build_model(self) -> FmgLocalModel:
        transformer = self._config.transformer
        return FmgCtct2(
            num_features=len(self._feature_columns),
            num_context_vars=len(self._context_columns),
            model_dim=int(self._config.d_model),
            feature_embedding_dim=int(self._config.feature_embedding_dim),
            sequence_length=int(self._sequence_length),
            cnn_layers=[conv_layer_from_config(layer) for layer in self._config.cnn.layers],
            n_heads=int(transformer.n_heads),
            dropout=float(self._config.dropout),
            score_activation=str(self._config.score_activation),
        )

    def _model_forward(
        self,
        features: Tensor,
        feature_age: Tensor,
        context: Tensor,
        context_age: Tensor,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        model = self._require_model()
        if not isinstance(model, FmgCtct2):
            raise RuntimeError(f"{self._MODEL_NAME} has incompatible model state")
        return cast(
            tuple[Tensor, dict[str, Tensor]],
            model(features, feature_age, context, context_age),
        )

    def _score(
        self, features: Tensor, feature_age: Tensor, context: Tensor, context_age: Tensor
    ) -> Tensor:
        day_scores, _ = self._model_forward(features, feature_age, context, context_age)
        return day_scores.float()

    def _forward_batch(self, windows: PreparedWindows, batch: Batch) -> TrainingBatchOutput:
        """Score a date block and evaluate its joint portfolio objective."""
        objective = self._loss
        if objective is None:
            raise RuntimeError(f"{self._MODEL_NAME} objective is unavailable during training")
        n_dates = len(batch.dates)
        scores = torch.zeros(n_dates, batch.n_max, device=self._device)
        supervision = torch.zeros(n_dates, batch.n_max, device=self._device)
        mask = torch.zeros(n_dates, batch.n_max, dtype=torch.bool, device=self._device)

        for position, date_slice in enumerate(batch.dates):
            rows = date_slice.rows
            cols = date_slice.cols.to(self._device)
            features, feature_age = windows.features.rows(rows)
            context, context_age = windows.context.rows(rows)
            if torch.is_grad_enabled() and self._gradient_checkpointing:
                day_scores = checkpoint(
                    self._score,
                    features,
                    feature_age,
                    context,
                    context_age,
                    use_reentrant=False,
                )
            else:
                day_scores = self._score(features, feature_age, context, context_age)
            scores[position, cols] = day_scores
            supervision[position, cols] = windows.supervision[rows.to(self._device)]
            mask[position, cols] = True

        loss_output = objective(scores.float(), supervision.float(), mask)
        loss = self._loss_value(loss_output)
        return TrainingBatchOutput(
            loss=loss,
            metrics_factory=lambda: score_metrics(
                scores, mask, loss_output, self._score_saturation_threshold
            ),
        )

    @torch.inference_mode()
    def predict(self, test: MaskedPanels) -> PredictionOutput:
        """Return one native objective score per constructible asset/date row."""
        model = self._require_model()
        feature_scaler, context_scaler = self._require_scalers()
        if not isinstance(model, FmgCtct2):
            raise RuntimeError(f"{self._MODEL_NAME} has incompatible model state")
        model.eval()
        tensors, index = build_sequences(
            self._combined(test), self._inputs(), self._sequence_length, model.buffer_size
        )
        features_raw = cast(WindowedTensor, tensors["features"])
        context_val, context_age = cast(tuple[Tensor, Tensor], tensors["context"])
        features = self._windowed_field(features_raw, feature_scaler)
        context = self._field((context_val, context_age), context_scaler)

        dates = index.get_level_values(0)
        scores = np.zeros(len(index), dtype=float)
        for date in dates.unique():
            positions = np.flatnonzero(dates == date)
            rows = torch.from_numpy(positions).long()
            day_scores, _ = self._model_forward(*features.rows(rows), *context.rows(rows))
            scores[positions] = day_scores.float().cpu().numpy()

        return PredictionOutput(
            kind=self._prediction_kind,
            values=pd.Series(scores, index=index, name="score"),
        )
