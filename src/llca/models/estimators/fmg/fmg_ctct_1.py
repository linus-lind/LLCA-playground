"""Adapt target-specific panel data and portfolio training to FMG-CTCT-1."""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from llca.data.modules.masked_panel import MaskedPanels
from llca.models.estimators.fmg.base import (
    RawWindows,
    conv_layer_from_config,
)
from llca.models.estimators.fmg.single_asset import FmgSingleAssetEstimator
from llca.models.estimators.prediction import PredictionOutput
from llca.models.fmg import FmgCtct1
from llca.models.fmg.base import FmgLocalModel
from llca.models.utils.sequences import WindowedTensor, build_sequences


class FmgCtct1Estimator(FmgSingleAssetEstimator):
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

    def _build_model(self) -> FmgLocalModel:
        """Construct the target-query network after input widths are known."""
        transformer = self._config.transformer
        cnn_layers = [conv_layer_from_config(layer) for layer in self._config.cnn.layers]
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

    def _windows(self, split: MaskedPanels) -> RawWindows:
        """Keep full context cross-sections only on dates with a usable target label."""
        if self._model is None:
            raise RuntimeError(f"{self._MODEL_NAME} is not fitted")
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

    def _allocate(
        self,
        features: Tensor,
        feature_age: Tensor,
        context: Tensor,
        context_age: Tensor,
        target_index: Tensor,
    ) -> Tensor:
        if not isinstance(self._model, FmgCtct1):
            raise RuntimeError(f"{self._MODEL_NAME} is not fitted")
        allocation, _ = self._model(
            features,
            feature_age,
            context,
            context_age,
            target_index,
        )
        return cast(Tensor, allocation.float())

    @torch.inference_mode()
    def predict(self, test: MaskedPanels) -> PredictionOutput:
        """Return one final signed allocation per constructible target date."""
        if (
            not isinstance(self._model, FmgCtct1)
            or self._feature_scaler is None
            or self._context_scaler is None
        ):
            raise RuntimeError(f"{self._MODEL_NAME} is not fitted")
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
            kind="portfolio",
            values=pd.Series(values, index=output_index, name="weight"),
        )
