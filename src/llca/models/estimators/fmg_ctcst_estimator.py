"""Adapt panel data, training services, and persistence to the FMG-CTCST network."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from llca.data.modules.masked_panel import MaskedPanel, MaskedPanels
from llca.models.estimators.estimator import TrainableEstimator
from llca.models.estimators.evaluation_spec import EvaluationSpec
from llca.models.estimators.prediction import PredictionOutput
from llca.models.fmg_ctcst import FmgCtcst, FmgTemporalModel
from llca.models.modules.conv_layer import ConvLayer
from llca.models.utils.batching import Batch, Field, Window, build_batches
from llca.models.utils.sequences import SequenceInput, WindowedTensor, build_sequences
from llca.models.utils.standardizer import Standardizer
from llca.training.modules.training_config import TrainingConfig
from llca.training.modules.training_diagnostics import (
    PanelBatchMetadata,
    TrainingBatchOutput,
    objective_diagnostics,
    tensor_distribution_diagnostics,
)
from llca.training.modules.training_task import TrainingTask

_BUNDLE_FORMAT_VERSION = 2


def _conv_layer(layer: DictConfig) -> ConvLayer:
    """Translate one validated Hydra convolution entry into the model value object."""
    kernel_height, kernel_width = int(layer.kernel_size[0]), int(layer.kernel_size[1])
    pad_height, pad_width = int(layer.padding[0]), int(layer.padding[1])
    return ConvLayer(int(layer.out_channels), kernel_height, kernel_width, pad_height, pad_width)


def _training_metrics(
    scores: Tensor,
    mask: Tensor,
    loss_output: object,
    saturation_threshold: float,
) -> dict[str, float | Tensor]:
    """Combine reusable tensor summaries with objective-provided diagnostics.

    ``scores`` and ``mask`` have shape ``[D_batch, N_max]``. Only valid positions enter
    distribution statistics; structured loss components are exposed under the objective
    namespace without retaining their autograd graphs.
    """
    return tensor_distribution_diagnostics(
        scores,
        namespace="scores",
        mask=mask,
        saturation_threshold=saturation_threshold,
    ) | objective_diagnostics(loss_output)


@dataclass(frozen=True, slots=True)
class _Raw:
    """Hold usable targets and lazy sequence references before scaling and batching.

    The target-level ``starts``, context, supervision, and index contain ``R`` usable
    prediction rows. Feature values and ages remain compact unfiltered ``[R_raw, F]``
    history buffers because filtered target rows may still be required as past inputs.
    """

    features: WindowedTensor
    context: tuple[Tensor, Tensor]
    supervision: Tensor
    index: pd.MultiIndex


@dataclass(frozen=True, slots=True)
class _Windows:
    """Hold scaled lazy inputs, device targets, and precomputed date-block batches."""

    features: Window
    context: Field
    supervision: Tensor
    index: pd.MultiIndex
    batches: list[Batch]


class FmgCtcstEstimator(TrainableEstimator[Batch]):
    """Integrate FMG-CTCST with the generic panel training and inference pipeline.

    The estimator builds causal feature sequences and point-in-time context from named
    datasets, fits split-specific standardizers, and groups usable targets into date
    blocks for portfolio objectives. Within a block, each date is scored independently;
    outputs are then packed into ``[D_batch, N_max]`` tensors because the objective may
    couple dates through turnover or risk. Prediction preserves the native
    ``(date, instrument)`` index and returns unnormalized ranking scores.
    """

    _MODEL_NAME = "fmg-ctct-2"
    _BUNDLE_ARTIFACT = "fmg-ctct-2_bundle"
    _BUNDLE_FILENAME = "fmg-ctct-2.pt"

    def __init__(
        self,
        config: DictConfig,
        loss: nn.Module | None,
        device: torch.device | None = None,
    ) -> None:
        self._config = config
        self._loss = loss
        context = config.inputs.context
        self._feature_dataset_name = str(config.inputs.features)
        self._context_dataset_names = (
            [str(context)] if isinstance(context, str) else [str(name) for name in context]
        )
        self._supervision_dataset = str(config.supervision.dataset)
        self._supervision_column = str(config.supervision.column)

        self._device = device if device is not None else torch.device("cpu")
        self._gradient_checkpointing = False
        self._score_saturation_threshold = 0.95
        self._sequence_length = config.sequence_length
        self._feature_columns: list[str] = []
        self._context_columns: list[str] = []
        self._model: FmgTemporalModel | None = None
        self._feature_scaler: Standardizer | None = None
        self._context_scaler: Standardizer | None = None

    def _build_model(self) -> FmgTemporalModel:
        """Construct the network after input column counts are known from the data split."""
        transformer = self._config.transformer
        cnn_layers = [_conv_layer(layer) for layer in self._config.cnn.layers]
        feature_embedding_dim = self._config.feature_embedding_dim
        return FmgCtcst(
            num_features=len(self._feature_columns),
            num_context_vars=len(self._context_columns),
            model_dim=int(self._config.d_model),
            feature_embedding_dim=int(feature_embedding_dim),
            sequence_length=int(self._sequence_length),
            cnn_layers=cnn_layers,
            n_heads=int(transformer.n_heads),
            dropout=float(self._config.dropout),
            score_activation=str(self._config.score_activation),
        )

    def _inputs(self) -> list[SequenceInput]:
        return [
            SequenceInput("features", self._feature_columns, windowed=True),
            SequenceInput("context", self._context_columns, windowed=False),
        ]

    def _combined(self, split: MaskedPanels) -> MaskedPanel:
        """Align configured feature and context datasets into one sequence-building panel.

        Values, observation masks, and ages are concatenated column-wise on their shared
        panel index. Segments come from the feature dataset because they define which rows
        may form a continuous temporal sequence.
        """
        names = [self._feature_dataset_name, *self._context_dataset_names]
        parts = [split[name] for name in names]
        return MaskedPanel(
            values=pd.concat([part.values for part in parts], axis=1),
            observed=pd.concat([part.observed for part in parts], axis=1),
            age=pd.concat([part.age for part in parts], axis=1),
            segment=split[self._feature_dataset_name].segment,
        )

    def _supervision(self, split: MaskedPanels, index: pd.MultiIndex) -> tuple[Tensor, Tensor]:
        """Align target values and availability to the predictable row index."""
        panel = split[self._supervision_dataset]
        values = panel.values[self._supervision_column].reindex(index).to_numpy(dtype=float)
        observed = (
            panel.observed[self._supervision_column]
            .reindex(index)
            .fillna(False)
            .to_numpy(dtype=bool)
        )
        return torch.from_numpy(values).float(), torch.from_numpy(observed)

    def _windows(self, split: MaskedPanels) -> _Raw:
        """Build lazy causal inputs and retain only rows with usable supervision.

        Filtering applies to target references, point-in-time context, and the returned
        index. The compact feature-history buffers are deliberately left intact because a
        row unusable as a target can still be required by a later causal window.
        """
        assert self._model is not None
        tensors, raw_index = build_sequences(
            self._combined(split), self._inputs(), self._sequence_length, self._model.buffer_size
        )
        index = cast(pd.MultiIndex, raw_index)
        supervision, observed = self._supervision(split, index)
        keep = observed & torch.isfinite(supervision)
        keep_np = keep.numpy()

        features_raw = cast(WindowedTensor, tensors["features"])
        context_values, context_age = cast(tuple[Tensor, Tensor], tensors["context"])

        return _Raw(
            features=WindowedTensor(
                values=features_raw.values,
                age=features_raw.age,
                starts=features_raw.starts[keep],
                window=features_raw.window,
            ),
            context=(context_values[keep], context_age[keep]),
            supervision=supervision[keep],
            index=index[keep_np],
        )

    def _to_windows(self, raw: _Raw, batch_size: int) -> _Windows | None:
        """Scale raw inputs, assign transfer devices, and group target rows by date."""
        assert self._feature_scaler is not None and self._context_scaler is not None
        if len(raw.index) == 0:
            return None
        return _Windows(
            features=self._windowed_field(raw.features, self._feature_scaler),
            context=self._field(raw.context, self._context_scaler),
            supervision=raw.supervision.to(self._device),
            index=raw.index,
            batches=build_batches(raw.index, batch_size),
        )

    def _field(self, pair: tuple[Tensor, Tensor], scaler: Standardizer) -> Field:
        """Scale a point-in-time ``[R, C]`` pair and attach lazy device transfer."""
        values, age = pair
        return Field(
            values=scaler.transform(values),
            age=age,
            device=self._device,
        )

    def _windowed_field(self, raw: WindowedTensor, scaler: Standardizer) -> Window:
        """Scale compact ``[R_raw, F]`` buffers while retaining lazy window offsets."""
        return Window(
            values=scaler.transform(raw.values),
            age=raw.age,
            starts=raw.starts,
            window=raw.window,
            device=self._device,
        )

    def _model_forward(
        self,
        features: Tensor,
        feature_age: Tensor,
        context: Tensor,
        context_age: Tensor,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Run one date-level network call inside the trainer's precision context."""
        assert isinstance(self._model, FmgCtcst)
        return cast(
            tuple[Tensor, dict[str, Tensor]],
            self._model(features, feature_age, context, context_age),
        )

    def _score(
        self, features: Tensor, feature_age: Tensor, context: Tensor, context_age: Tensor
    ) -> Tensor:
        assert self._model is not None
        day_scores, _ = self._model_forward(features, feature_age, context, context_age)
        return day_scores.float()

    def _forward_batch(self, windows: _Windows, batch: Batch) -> TrainingBatchOutput:
        """Score a date block and evaluate its joint objective.

        Each date-level cross-section is scattered into shared ``[D_batch, N_max]`` score,
        supervision, and validity tensors. The objective is evaluated once after all dates
        so it can model temporal terms such as turnover. Consequently, all date-level
        graphs remain live until backward; optional checkpointing trades recomputation for
        lower activation memory.
        """
        assert self._model is not None
        objective = self._loss
        if objective is None:
            raise RuntimeError("FMG-CTCST objective is unavailable during training")
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
            metrics_factory=lambda: _training_metrics(
                scores, mask, loss_output, self._score_saturation_threshold
            ),
        )

    @torch.inference_mode()
    def _evaluate(self, windows: _Windows) -> float:
        """Return the unweighted mean objective across precomputed validation batches."""
        assert self._model is not None
        self._model.eval()
        total = 0.0
        for batch in windows.batches:
            total += float(self._forward_batch(windows, batch).loss.item())
        return total / max(len(windows.batches), 1)

    def _build_training_task(
        self,
        train: MaskedPanels,
        val: MaskedPanels | None,
        training: TrainingConfig,
        device: torch.device,
    ) -> TrainingTask[Batch]:
        """Prepare FMG-specific panel data and steps for the reusable trainer.

        Column contracts are inferred from the training split before model construction.
        Standardizers are fitted only on training inputs; validation uses those same
        statistics. Optimization and execution policy remain outside the estimator.
        """
        if self._loss is None:
            raise ValueError("FMG-CTCST training requires an objective")
        self._device = device
        self._gradient_checkpointing = training.gradient_checkpointing
        self._score_saturation_threshold = float(
            self._config.diagnostics.score_saturation_threshold
        )
        self._feature_columns = list(train[self._feature_dataset_name].columns)
        self._context_columns = [
            column for name in self._context_dataset_names for column in train[name].columns
        ]
        self._model = self._build_model().to(self._device)
        self._loss = self._loss.to(self._device)

        raw = self._windows(train)
        self._feature_scaler = Standardizer.fit(raw.features.values)
        self._context_scaler = Standardizer.fit(raw.context[0])
        train_windows = self._to_windows(raw, training.batch_size)
        assert train_windows is not None, "training split produced no usable sequences"
        val_windows = (
            self._to_windows(self._windows(val), training.batch_size) if val is not None else None
        )

        return TrainingTask(
            model=self._model,
            batches=train_windows.batches,
            train_step=lambda batch: self._forward_batch(train_windows, batch),
            validation_step=(
                (lambda: self._evaluate(val_windows)) if val_windows is not None else None
            ),
            batch_metadata=lambda batch, index: PanelBatchMetadata(
                index=index + 1,
                observations=batch.observations,
                start_date=batch.start_date.date().isoformat(),
                end_date=batch.end_date.date().isoformat(),
                dates=len(batch.dates),
                entities=batch.n_max,
            ),
        )

    @torch.inference_mode()
    def predict(self, test: MaskedPanels) -> PredictionOutput:
        """Predict every constructible test sequence as an unnormalized ranking score.

        Inference does not require targets and scores each date using its exact instrument
        count, without training-only date-block padding. The returned Series is indexed by
        the final row of each causal window; portfolio normalization remains downstream.
        """
        assert (
            isinstance(self._model, FmgCtcst)
            and self._feature_scaler is not None
            and self._context_scaler is not None
        )
        self._model.eval()
        tensors, index = build_sequences(
            self._combined(test), self._inputs(), self._sequence_length, self._model.buffer_size
        )
        features_raw = cast(WindowedTensor, tensors["features"])
        context_val, context_age = cast(tuple[Tensor, Tensor], tensors["context"])
        features = self._windowed_field(features_raw, self._feature_scaler)
        context = self._field((context_val, context_age), self._context_scaler)

        dates = index.get_level_values(0)
        scores = np.zeros(len(index), dtype=float)
        for date in dates.unique():
            positions = np.flatnonzero(dates == date)
            rows = torch.from_numpy(positions).long()
            day_scores, _ = self._model_forward(*features.rows(rows), *context.rows(rows))
            day_scores = day_scores.float()
            scores[positions] = day_scores.cpu().numpy()

        return PredictionOutput(
            kind="ranking",
            values=pd.Series(scores, index=index, name="score"),
        )

    @property
    def required_history(self) -> int:
        """Cover the temporal feature window plus rows consumed by causal CNN kernels."""
        buffer = self._model.buffer_size if self._model is not None else 0
        return int(self._sequence_length) + int(buffer)

    @property
    def evaluation_spec(self) -> EvaluationSpec:
        """Expose panel roles without leaking the FMG configuration into analytics."""
        return EvaluationSpec(
            primary_dataset=self._feature_dataset_name,
            supervision_dataset=self._supervision_dataset,
            supervision_column=self._supervision_column,
        )

    def to_device(self, device: torch.device) -> None:
        """Move the fitted network and objective while keeping CPU-backed input scalers."""
        self._device = device
        if self._model is not None:
            self._model = self._model.to(device)
        if self._loss is not None:
            self._loss = self._loss.to(device)

    def _inference_payload(self) -> dict[str, Any]:
        """Package architecture, weights, column order, and scalers for inference."""
        assert self._model is not None
        assert self._feature_scaler is not None and self._context_scaler is not None
        return {
            "format_version": _BUNDLE_FORMAT_VERSION,
            "config": OmegaConf.to_container(self._config, resolve=True),
            "feature_columns": list(self._feature_columns),
            "context_columns": list(self._context_columns),
            "model_state_dict": self._model.state_dict(),
            "feature_scaler": self._feature_scaler.state_dict(),
            "context_scaler": self._context_scaler.state_dict(),
        }

    @classmethod
    def _from_payload(
        cls, payload: dict[str, Any], map_location: torch.device
    ) -> FmgCtcstEstimator:
        """Recreate the unfitted estimator shell from serialized configuration."""
        config = cast(DictConfig, OmegaConf.create(payload["config"]))
        return cls(config=config, loss=payload.get("loss"), device=map_location)

    def _restore(self, payload: dict[str, Any]) -> None:
        """Rebuild the network and restore fitted inference state from a bundle."""
        self._feature_columns = list(payload["feature_columns"])
        self._context_columns = list(payload["context_columns"])
        self._model = self._build_model().to(self._device)
        self._model.load_state_dict(payload["model_state_dict"])
        self._feature_scaler = Standardizer.from_state_dict(payload["feature_scaler"], device="cpu")
        self._context_scaler = Standardizer.from_state_dict(payload["context_scaler"], device="cpu")
        if self._loss is not None:
            self._loss = self._loss.to(self._device)
