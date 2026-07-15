"""Adapt target-specific panel data and portfolio training to FMG-CTCT-1."""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from llca.data.modules.masked_panel import MaskedPanels
from llca.models.estimators.fmg_ctcst_estimator import (
    FmgCtcstEstimator,
    _conv_layer,
    _Raw,
    _Windows,
)
from llca.models.estimators.prediction import PredictionOutput
from llca.models.fmg_ctct_1 import FmgCtct1
from llca.models.utils.batching import Batch
from llca.models.utils.sequences import WindowedTensor, build_sequences
from llca.training.modules.training_diagnostics import (
    TrainingBatchOutput,
    objective_diagnostics,
    tensor_distribution_diagnostics,
)


def _allocation_metrics(
    allocations: Tensor,
    mask: Tensor,
    loss_output: object,
    saturation_threshold: float,
) -> dict[str, float | Tensor]:
    """Expose allocation distributions separately from objective diagnostics."""
    return tensor_distribution_diagnostics(
        allocations,
        namespace="allocations",
        mask=mask,
        saturation_threshold=saturation_threshold,
    ) | objective_diagnostics(loss_output)


class FmgCtct1Estimator(FmgCtcstEstimator):
    """Train one target allocation while retaining every asset as attention context.

    The configured target is resolved against the second level of the canonical panel
    index (``instrument_id`` in the current data configuration). Target identity is never
    inferred from row ordering. Training dates require one constructible target sequence
    and one observed target return; non-target rows need inputs only and remain available
    as cross-sectional keys and values.
    """

    _MODEL_NAME = "fmg-ctct-1"
    _BUNDLE_ARTIFACT = "fmg-ctct-1_bundle"
    _BUNDLE_FILENAME = "fmg-ctct-1.pt"

    def __init__(
        self,
        config: DictConfig,
        loss: nn.Module | None,
        device: torch.device | None = None,
    ) -> None:
        super().__init__(config=config, loss=loss, device=device)
        self._target_entity_id = int(config.target.entity_id)

    def _build_model(self) -> FmgCtct1:
        """Construct the target-query network after input widths are known."""
        transformer = self._config.transformer
        cnn_layers = [_conv_layer(layer) for layer in self._config.cnn.layers]
        return FmgCtct1(
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

    @staticmethod
    def _entity_values(index: pd.Index) -> pd.Index:
        if not isinstance(index, pd.MultiIndex) or index.nlevels < 2:
            raise ValueError("fmg-ctct-1 requires a (date, entity) MultiIndex")
        return index.get_level_values(1)

    def _target_position(self, index: pd.Index) -> int:
        """Resolve exactly one target row inside a single-date cross-section."""
        entities = self._entity_values(index)
        positions = np.flatnonzero(np.asarray(entities == self._target_entity_id))
        if len(positions) != 1:
            date = index.get_level_values(0)[0] if len(index) else "<empty>"
            raise ValueError(
                f"target entity {self._target_entity_id} must occur exactly once on {date}; "
                f"found {len(positions)} rows"
            )
        return int(positions[0])

    def _windows(self, split: MaskedPanels) -> _Raw:
        """Keep full context cross-sections only on dates with a usable target label."""
        assert self._model is not None
        tensors, raw_index = build_sequences(
            self._combined(split), self._inputs(), self._sequence_length, self._model.buffer_size
        )
        index = cast(pd.MultiIndex, raw_index)
        entities = self._entity_values(index)
        is_target_np = np.asarray(entities == self._target_entity_id, dtype=bool)

        source_index = split[self._feature_dataset_name].values.index
        source_entities = self._entity_values(source_index)
        if not bool(np.asarray(source_entities == self._target_entity_id).any()):
            raise ValueError(
                f"target entity {self._target_entity_id} is absent from dataset "
                f"'{self._feature_dataset_name}'"
            )
        if not is_target_np.any():
            raise ValueError(
                f"target entity {self._target_entity_id} has no constructible sequence; "
                "check its active history and model.sequence_length"
            )

        target_index = index[is_target_np]
        target_dates = target_index.get_level_values(0)
        if target_dates.duplicated().any():
            duplicate = target_dates[target_dates.duplicated()][0]
            raise ValueError(
                f"target entity {self._target_entity_id} occurs more than once on {duplicate}"
            )

        supervision, observed = self._supervision(split, index)
        is_target = torch.from_numpy(is_target_np)
        valid_target = is_target & observed & torch.isfinite(supervision)
        if not bool(valid_target.any().item()):
            raise ValueError(
                f"target entity {self._target_entity_id} has no observed finite supervision "
                f"in '{self._supervision_dataset}.{self._supervision_column}'"
            )

        valid_dates = index.get_level_values(0)[valid_target.numpy()].unique()
        keep_np = np.asarray(index.get_level_values(0).isin(valid_dates), dtype=bool)
        keep = torch.from_numpy(keep_np)

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

    def _allocate(
        self,
        features: Tensor,
        feature_age: Tensor,
        context: Tensor,
        context_age: Tensor,
        target_index: Tensor,
    ) -> Tensor:
        assert isinstance(self._model, FmgCtct1)
        allocation, _ = self._model(
            features,
            feature_age,
            context,
            context_age,
            target_index,
        )
        return cast(Tensor, allocation.float())

    def _forward_batch(self, windows: _Windows, batch: Batch) -> TrainingBatchOutput:
        """Evaluate one chronological date block as a single-asset portfolio."""
        assert isinstance(self._model, FmgCtct1)
        objective = self._loss
        if objective is None:
            raise RuntimeError("FMG-CTCT-1 objective is unavailable during training")

        n_dates = len(batch.dates)
        allocations = torch.zeros(n_dates, 1, device=self._device)
        supervision = torch.zeros(n_dates, 1, device=self._device)
        mask = torch.ones(n_dates, 1, dtype=torch.bool, device=self._device)

        for position, date_slice in enumerate(batch.dates):
            rows = date_slice.rows
            day_index = windows.index[rows.numpy()]
            target_position = self._target_position(day_index)
            target_index = torch.tensor([target_position], dtype=torch.long, device=self._device)
            features, feature_age = windows.features.rows(rows)
            context, context_age = windows.context.rows(rows)
            if torch.is_grad_enabled() and self._gradient_checkpointing:
                day_allocation = checkpoint(
                    self._allocate,
                    features,
                    feature_age,
                    context,
                    context_age,
                    target_index,
                    use_reentrant=False,
                )
            else:
                day_allocation = self._allocate(
                    features, feature_age, context, context_age, target_index
                )

            target_row = rows[target_position].to(self._device)
            allocations[position, 0] = day_allocation[0]
            supervision[position, 0] = windows.supervision[target_row]

        loss_output = objective(allocations.float(), supervision.float(), mask)
        loss = self._loss_value(loss_output)
        return TrainingBatchOutput(
            loss=loss,
            metrics_factory=lambda: _allocation_metrics(
                allocations,
                mask,
                loss_output,
                self._score_saturation_threshold,
            ),
        )

    @torch.inference_mode()
    def predict(self, test: MaskedPanels) -> PredictionOutput:
        """Return one final signed allocation per constructible target date."""
        assert (
            isinstance(self._model, FmgCtct1)
            and self._feature_scaler is not None
            and self._context_scaler is not None
        )
        self._model.eval()
        tensors, raw_index = build_sequences(
            self._combined(test), self._inputs(), self._sequence_length, self._model.buffer_size
        )
        index = cast(pd.MultiIndex, raw_index)
        features_raw = cast(WindowedTensor, tensors["features"])
        context_val, context_age = cast(tuple[Tensor, Tensor], tensors["context"])
        features = self._windowed_field(features_raw, self._feature_scaler)
        context = self._field((context_val, context_age), self._context_scaler)

        dates = index.get_level_values(0)
        values: list[float] = []
        target_rows: list[int] = []
        for date in dates.unique().sort_values():
            positions = np.flatnonzero(np.asarray(dates == date))
            day_index = index[positions]
            matches = np.flatnonzero(
                np.asarray(self._entity_values(day_index) == self._target_entity_id)
            )
            if len(matches) == 0:
                continue
            target_position = self._target_position(day_index)
            rows = torch.from_numpy(positions).long()
            target_index = torch.tensor([target_position], dtype=torch.long, device=self._device)
            allocation = self._allocate(
                *features.rows(rows),
                *context.rows(rows),
                target_index,
            )
            values.append(float(allocation[0].cpu().item()))
            target_rows.append(int(positions[target_position]))

        if not values:
            raise ValueError(
                f"target entity {self._target_entity_id} has no constructible test sequence"
            )
        output_index = index[np.asarray(target_rows, dtype=int)]
        return PredictionOutput(
            kind="allocation",
            values=pd.Series(values, index=output_index, name="weight"),
        )
