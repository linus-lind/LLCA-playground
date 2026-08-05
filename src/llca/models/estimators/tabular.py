"""Shared aligned-panel access for cross-sectional scikit-learn and baseline models."""

from __future__ import annotations

import pandas as pd
from omegaconf import DictConfig

from llca.data.modules.masked_panel import MaskedPanels
from llca.models.estimators.evaluation_spec import EvaluationSpec
from llca.models.estimators.prediction import PredictionKind, validate_prediction_kind
from llca.models.estimators.sklearn import SklearnEstimator


class TabularEstimator(SklearnEstimator[MaskedPanels]):
    """Cross-sectional estimator over the aligned-panel view of point-in-time features.

    Concrete estimators consume same-date feature rows without a temporal window, so no
    prior history is required. Feature and context datasets are concatenated column-wise on
    the shared ``(date, entity)`` index and the supervision binding names the aligned target.
    Subclasses own fitting, prediction, and serialization; this base owns only panel access.
    """

    def __init__(self, config: DictConfig, prediction_kind: PredictionKind = "portfolio") -> None:
        self._config = config
        self._prediction_kind = validate_prediction_kind(prediction_kind)
        inputs = config.inputs
        self._feature_dataset = str(inputs.features)
        context = inputs.get("context")
        if context is None:
            self._context_datasets: list[str] = []
        elif isinstance(context, str):
            self._context_datasets = [context]
        else:
            self._context_datasets = [str(name) for name in context]
        self._supervision_dataset = str(config.supervision.dataset)
        self._supervision_column = str(config.supervision.column)

    @property
    def evaluation_spec(self) -> EvaluationSpec:
        """Expose panel roles without leaking the model configuration into analytics."""
        return EvaluationSpec(
            primary_dataset=self._feature_dataset,
            supervision_dataset=self._supervision_dataset,
            supervision_column=self._supervision_column,
        )

    @property
    def required_history(self) -> int:
        """Cross-sectional models read same-date features and need no prior history."""
        return 0

    def _live_rows(self, split: MaskedPanels) -> pd.Series:
        """Rows whose primary feature dataset carries at least one available value."""
        primary = split[self._feature_dataset].values
        return primary.notna().any(axis=1)

    def _feature_frame(self, split: MaskedPanels) -> pd.DataFrame:
        """Concatenate feature and context columns aligned to the primary panel index."""
        names = [self._feature_dataset, *self._context_datasets]
        primary_index = split[names[0]].values.index
        frames: list[pd.DataFrame] = []
        for name in names:
            values = split[name].values
            if not values.index.equals(primary_index):
                values = values.reindex(primary_index)
            frames.append(values)
        combined = pd.concat(frames, axis=1)
        combined.columns = [str(column) for column in combined.columns]
        if combined.columns.has_duplicates:
            duplicates = sorted(combined.columns[combined.columns.duplicated()].unique())
            raise ValueError(f"model input columns must be unique across datasets: {duplicates}")
        return combined

    def _supervision_series(self, split: MaskedPanels) -> pd.Series:
        """Return the aligned target with unobserved entries set to NaN."""
        panel = split[self._supervision_dataset]
        observed = panel.observed[self._supervision_column].fillna(False).astype(bool)
        return panel.values[self._supervision_column].where(observed)
