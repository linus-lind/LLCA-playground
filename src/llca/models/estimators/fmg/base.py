"""Shared panel, training, scaling, and persistence lifecycle for FMG estimators."""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, Self, cast

import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from torch import Tensor, nn

from llca.data.modules.masked_panel import MaskedPanel, MaskedPanels
from llca.models.estimators.evaluation_spec import EvaluationSpec
from llca.models.estimators.prediction import (
    PredictionKind,
    prediction_kind_from_bundle,
    validate_prediction_kind,
)
from llca.models.estimators.torch import TorchTrainableEstimator
from llca.models.fmg.base import FmgLocalModel
from llca.models.modules.conv_layer import ConvLayer
from llca.models.utils.batching import Batch, Field, Window, build_batches
from llca.models.utils.sequences import SequenceInput, WindowedTensor, build_sequences
from llca.models.utils.standardizer import Standardizer
from llca.training.modules.training_config import TrainingConfig
from llca.training.modules.training_diagnostics import (
    PanelBatchMetadata,
    TrainingBatchOutput,
)
from llca.training.modules.training_task import TrainingTask

_BUNDLE_FORMAT_VERSION = 1
_VERSION_ONLY_BUNDLE_FORMAT = 2
_BUNDLE_FIELDS = {
    "format_version",
    "config",
    "feature_columns",
    "context_columns",
    "model_state_dict",
    "feature_scaler",
    "context_scaler",
    "prediction_kind",
}


def _validate_bundle_format(payload: dict[str, Any], model_name: str) -> None:
    """Accept the retired version-only revision only for the exact known payload shape."""
    version = payload.get("format_version")
    if version == _BUNDLE_FORMAT_VERSION:
        return
    if version == _VERSION_ONLY_BUNDLE_FORMAT and set(payload) == _BUNDLE_FIELDS:
        return
    raise ValueError(
        f"unsupported {model_name} bundle format {version!r}; expected {_BUNDLE_FORMAT_VERSION}"
    )


def conv_layer_from_config(layer: DictConfig) -> ConvLayer:
    """Translate one validated Hydra convolution entry into the model value object."""
    kernel_height, kernel_width = int(layer.kernel_size[0]), int(layer.kernel_size[1])
    pad_height, pad_width = int(layer.padding[0]), int(layer.padding[1])
    return ConvLayer(int(layer.out_channels), kernel_height, kernel_width, pad_height, pad_width)


@dataclass(frozen=True, slots=True)
class RawWindows:
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
class PreparedWindows:
    """Hold scaled lazy inputs, device targets, and precomputed date-block batches."""

    features: Window
    context: Field
    supervision: Tensor
    index: pd.MultiIndex
    batches: list[Batch]


class FmgEstimator(TorchTrainableEstimator[Batch]):
    """Provide model-independent FMG panel preparation and fitted-state persistence.

    The estimator builds causal feature sequences and point-in-time context from named
    datasets, fits split-specific standardizers, and groups usable targets into date
    blocks. Concrete variants own only network construction, batch objectives, and native
    prediction behavior.
    """

    def __init__(
        self,
        config: DictConfig,
        loss: nn.Module | None,
        device: torch.device | None = None,
        prediction_kind: PredictionKind | None = None,
    ) -> None:
        self._config = config
        self._loss = loss
        self._prediction_kind = validate_prediction_kind(
            prediction_kind
            or ("portfolio" if callable(getattr(loss, "normalize_weights", None)) else "regression")
        )
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
        self._model: FmgLocalModel | None = None
        self._feature_scaler: Standardizer | None = None
        self._context_scaler: Standardizer | None = None

    @abstractmethod
    def _build_model(self) -> FmgLocalModel:
        """Construct the concrete network after input column counts are known."""
        raise NotImplementedError

    def _inputs(self) -> list[SequenceInput]:
        return [
            SequenceInput("features", self._feature_columns, windowed=True),
            SequenceInput("context", self._context_columns, windowed=False),
        ]

    def _require_model(self) -> FmgLocalModel:
        """Return fitted model state with a stable runtime error outside its lifecycle."""
        if self._model is None:
            raise RuntimeError(f"{self._MODEL_NAME} is not fitted")
        return self._model

    def _require_scalers(self) -> tuple[Standardizer, Standardizer]:
        """Return fitted preprocessing state or reject premature inference/serialization."""
        if self._feature_scaler is None or self._context_scaler is None:
            raise RuntimeError(f"{self._MODEL_NAME} preprocessing state is not fitted")
        return self._feature_scaler, self._context_scaler

    def _combined(self, split: MaskedPanels) -> MaskedPanel:
        """Align configured feature and context datasets into one sequence-building panel.

        Values, observation masks, and ages are concatenated column-wise on their shared
        panel index. Segments come from the feature dataset because they define which rows
        may form a continuous temporal sequence.
        """
        names = [self._feature_dataset_name, *self._context_dataset_names]
        parts = [split[name] for name in names]
        primary_index = parts[0].values.index
        misaligned = [
            name
            for name, part in zip(names, parts, strict=True)
            if not part.values.index.equals(primary_index)
        ]
        if misaligned:
            raise ValueError(
                f"model inputs must share the '{self._feature_dataset_name}' row index; "
                f"misaligned datasets: {misaligned}"
            )
        columns = [str(column) for part in parts for column in part.values.columns]
        duplicates = sorted({column for column in columns if columns.count(column) > 1})
        if duplicates:
            raise ValueError(f"model input columns must be unique across datasets: {duplicates}")
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

    def _windows(self, split: MaskedPanels) -> RawWindows:
        """Build lazy causal inputs and retain only rows with usable supervision.

        Filtering applies to target references, point-in-time context, and the returned
        index. The compact feature-history buffers are deliberately left intact because a
        row unusable as a target can still be required by a later causal window.
        """
        model = self._require_model()
        tensors, raw_index = build_sequences(
            self._combined(split), self._inputs(), self._sequence_length, model.buffer_size
        )
        index = cast(pd.MultiIndex, raw_index)
        supervision, observed = self._supervision(split, index)
        keep = observed & torch.isfinite(supervision)
        keep_np = keep.numpy()

        features_raw = cast(WindowedTensor, tensors["features"])
        context_values, context_age = cast(tuple[Tensor, Tensor], tensors["context"])

        return RawWindows(
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

    def _to_windows(self, raw: RawWindows, batch_size: int) -> PreparedWindows | None:
        """Scale raw inputs, assign transfer devices, and group target rows by date."""
        feature_scaler, context_scaler = self._require_scalers()
        if len(raw.index) == 0:
            return None
        return PreparedWindows(
            features=self._windowed_field(raw.features, feature_scaler),
            context=self._field(raw.context, context_scaler),
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

    @abstractmethod
    def _forward_batch(self, windows: PreparedWindows, batch: Batch) -> TrainingBatchOutput:
        """Evaluate one chronological batch through the concrete FMG objective."""
        raise NotImplementedError

    @torch.inference_mode()
    def _evaluate(self, windows: PreparedWindows) -> float:
        """Return the unweighted mean objective across precomputed validation batches."""
        model = self._require_model()
        model.eval()
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
            raise ValueError(f"{self._MODEL_NAME} training requires an objective")
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
        if len(raw.index) == 0:
            raise ValueError("training split produced no usable sequences")
        self._feature_scaler = Standardizer.fit(raw.features.values)
        self._context_scaler = Standardizer.fit(raw.context[0])
        train_windows = self._to_windows(raw, training.batch_size)
        if train_windows is None:
            raise RuntimeError("training windows disappeared after preprocessing")
        val_windows = None
        if val is not None:
            val_raw = self._windows(val)
            validation_dates = len(val_raw.index.get_level_values(0).unique())
            val_windows = self._to_windows(val_raw, max(validation_dates, 1))
            if val_windows is None:
                raise ValueError("validation split produced no usable sequences")

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
        model = self._require_model()
        feature_scaler, context_scaler = self._require_scalers()
        return {
            "format_version": _BUNDLE_FORMAT_VERSION,
            "config": OmegaConf.to_container(self._config, resolve=True),
            "feature_columns": list(self._feature_columns),
            "context_columns": list(self._context_columns),
            "model_state_dict": model.state_dict(),
            "feature_scaler": feature_scaler.state_dict(),
            "context_scaler": context_scaler.state_dict(),
            "prediction_kind": self._prediction_kind,
        }

    @classmethod
    def _from_payload(cls, payload: dict[str, Any], map_location: torch.device) -> Self:
        """Recreate the unfitted estimator shell from serialized configuration."""
        _validate_bundle_format(payload, cls._MODEL_NAME)
        config = cast(DictConfig, OmegaConf.create(payload["config"]))
        return cls(
            config=config,
            loss=None,
            device=map_location,
            prediction_kind=prediction_kind_from_bundle(payload["prediction_kind"]),
        )

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
