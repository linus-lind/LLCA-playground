"""Shared data contracts for direct single-asset FMG allocation estimators."""

from __future__ import annotations

from abc import abstractmethod
from typing import cast

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from llca.data.modules.masked_panel import MaskedPanels
from llca.models.estimators.fmg.base import (
    FmgEstimator,
    PreparedWindows,
    RawWindows,
)
from llca.models.estimators.objective_output import (
    TrainingBatchOutput,
    objective_diagnostics,
    tensor_distribution_diagnostics,
)
from llca.models.estimators.prediction import PredictionKind, PredictionOutput
from llca.models.utils.batching import Batch
from llca.models.utils.sequences import WindowedTensor, build_sequences


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


class FmgSingleAssetEstimator(FmgEstimator):
    """Provide target identity and direct-allocation training for FMG variants.

    The target is resolved from the entity level of the canonical panel index. Concrete
    estimators decide whether non-target entities remain available as model context or
    are removed before sequence construction.
    """

    def __init__(
        self,
        config: DictConfig,
        loss: nn.Module | None,
        device: torch.device | None = None,
        prediction_kind: PredictionKind | None = None,
    ) -> None:
        if prediction_kind not in (None, "portfolio"):
            raise ValueError("single-asset allocation estimators require portfolio predictions")
        super().__init__(
            config=config,
            loss=loss,
            device=device,
            prediction_kind="portfolio",
        )
        self._target_entity_id = int(self._config.target.entity_id)

    def _entity_values(self, index: pd.Index) -> pd.Index:
        if not isinstance(index, pd.MultiIndex) or index.nlevels < 2:
            raise ValueError(f"{self._MODEL_NAME} requires a (date, entity) MultiIndex")
        return index.get_level_values(1)

    def _target_only_split(self, split: MaskedPanels) -> MaskedPanels:
        """Remove every non-target entity before sequence construction and scaling."""
        source_index = split[self._feature_dataset_name].values.index
        source_entities = self._entity_values(source_index)
        keep = np.asarray(source_entities == self._target_entity_id, dtype=bool)
        if not keep.any():
            raise ValueError(
                f"target entity {self._target_entity_id} is absent from dataset "
                f"'{self._feature_dataset_name}'"
            )
        target_dates = source_index[keep].get_level_values(0)
        if target_dates.duplicated().any():
            duplicate = target_dates[target_dates.duplicated()][0]
            raise ValueError(
                f"target entity {self._target_entity_id} occurs more than once on {duplicate}"
            )

        target_split: MaskedPanels = {}
        for name, panel in split.items():
            if not panel.values.index.equals(source_index):
                raise ValueError(
                    f"dataset '{name}' is not aligned with '{self._feature_dataset_name}'"
                )
            target_split[name] = panel.slice_rows(keep)
        return target_split

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

    @abstractmethod
    def _allocate(
        self,
        features: Tensor,
        feature_age: Tensor,
        context: Tensor,
        context_age: Tensor,
        target_index: Tensor,
    ) -> Tensor:
        """Return the configured target's signed allocation for one date."""
        raise NotImplementedError

    def _forward_batch(self, windows: PreparedWindows, batch: Batch) -> TrainingBatchOutput:
        """Evaluate one chronological date block as a single-asset portfolio."""
        if self._model is None:
            raise RuntimeError(f"{self._MODEL_NAME} is not fitted")
        objective = self._loss
        if objective is None:
            raise RuntimeError(f"{self._MODEL_NAME} objective is unavailable during training")

        n_dates = len(batch.dates)
        allocations = torch.zeros(n_dates, 1, device=self._device)
        supervision = torch.zeros(n_dates, 1, device=self._device)
        mask = torch.ones(n_dates, 1, dtype=torch.bool, device=self._device)
        risk_free = (
            torch.zeros(n_dates, device=self._device) if windows.risk_free is not None else None
        )

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
            if risk_free is not None:
                assert windows.risk_free is not None
                risk_free[position] = windows.risk_free[target_row]

        if risk_free is None:
            loss_output = objective(allocations.float(), supervision.float(), mask)
        else:
            loss_output = objective(
                allocations.float(), supervision.float(), mask, risk_free=risk_free
            )
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


class FmgTargetOnlyEstimator(FmgSingleAssetEstimator):
    """Share the data path used by models that never encode non-target assets."""

    def _windows(self, split: MaskedPanels) -> RawWindows:
        """Build training windows after physically removing all other entities."""
        target_split = self._target_only_split(split)
        raw = FmgEstimator._windows(self, target_split)
        if len(raw.index) == 0:
            raise ValueError(
                f"target entity {self._target_entity_id} has no constructible sequence "
                f"with observed finite supervision in "
                f"'{self._supervision_dataset}.{self._supervision_column}'"
            )
        return raw

    @torch.inference_mode()
    def predict(self, test: MaskedPanels) -> PredictionOutput:
        """Return target allocations without constructing any non-target sequence."""
        if self._model is None or self._feature_ewma is None:
            raise RuntimeError(f"{self._MODEL_NAME} is not fitted")
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
        features = self._windowed_field(features_raw)
        context = self._field((context_val, context_age))

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
            kind="portfolio",
            values=pd.Series(values, index=index, name="weight"),
        )
