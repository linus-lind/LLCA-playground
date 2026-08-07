from __future__ import annotations

from abc import abstractmethod
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Self, cast

import mlflow
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn

from llca.data.index_spec import time_level
from llca.data.modules.masked_panel import MaskedPanels
from llca.models.estimators.objective_output import objective_loss
from llca.models.estimators.objective_panel import pack_objective_panel
from llca.models.estimators.prediction import (
    PredictionKind,
    PredictionOutput,
    prediction_kind_from_bundle,
)
from llca.models.estimators.tabular import TabularEstimator
from llca.training.modules.sklearn_config import SklearnTrainingConfig
from llca.training.modules.tracking import TrainingTracker
from llca.training.modules.training_policy import TrainingPolicy
from llca.training.tuning import (
    HyperparameterSelection,
    HyperparameterSelectionResult,
    ParameterValue,
    select_hyperparameters,
)

_STD_FLOOR = 1e-8


_SELECTION_ARTIFACT = "hyperparameter_selection.json"


class SingleAssetClassifierEstimator(TabularEstimator):
    _STANDARDIZE: bool = False

    def __init__(
        self,
        config: DictConfig,
        prediction_kind: PredictionKind = "portfolio",
        *,
        cv_objective: nn.Module | None = None,
        selection: HyperparameterSelection | None = None,
        hyperparameters: Mapping[str, ParameterValue] | None = None,
    ) -> None:
        super().__init__(config, prediction_kind)
        self._target_entity_id = int(config.target.entity_id)
        classification = config.classification
        self._label_dataset = str(classification.dataset)
        self._label_column = str(classification.column)
        risk_free = config.get("risk_free")
        self._risk_free_dataset = None if risk_free is None else str(risk_free.dataset)
        self._risk_free_column = None if risk_free is None else str(risk_free.column)
        self._risk_free_by_date: pd.Series | None = None
        self._cv_objective = cv_objective
        self._selection = selection
        self._hyperparameters: dict[str, ParameterValue] = dict(hyperparameters or {})
        self._selection_result: HyperparameterSelectionResult | None = None
        self._model: Any | None = None
        self._feature_columns: list[str] = []
        self._means: pd.Series | None = None
        self._center: np.ndarray | None = None
        self._scale: np.ndarray | None = None

    @abstractmethod
    def _construct(self, training: SklearnTrainingConfig) -> Any:
        """Return an unfitted scikit-learn classifier exposing ``predict_proba``."""

    def _tuned(self, name: str, default: ParameterValue = None) -> ParameterValue:
        """Return a hyperparameter, preferring a selected/candidate value over the config default.

        A missing or null configuration entry falls back to ``default``, so an optional scikit-learn
        argument left out of the config resolves to that estimator's documented default rather than
        ``None``. Callers that treat null as a meaningful value (e.g. unlimited ``max_depth``) omit
        ``default`` and handle ``None`` explicitly.
        """
        if name in self._hyperparameters:
            return self._hyperparameters[name]
        value = cast(ParameterValue, self._config.get(name))
        return default if value is None else value

    def _tuned_float(self, name: str, default: float | None = None) -> float:
        return float(cast(float, self._tuned(name, default)))

    def _tuned_int(self, name: str) -> int:
        return int(cast(int, self._tuned(name)))

    def _tuned_bool(self, name: str) -> bool:
        return bool(cast(bool, self._tuned(name)))

    def _tuned_number(self, name: str, default: int | float | None = None) -> int | float:
        """Return a numeric hyperparameter preserving int-vs-float (e.g. count vs fraction)."""
        value = self._tuned(name, default)
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise TypeError(f"hyperparameter '{name}' must be numeric, got {value!r}")
        return value

    def _optional_int(self, name: str) -> int | None:
        """Return ``None`` for an unset/none hyperparameter, else an int (e.g. ``max_depth``)."""
        value = self._tuned(name)
        return None if value is None else int(cast(int, value))

    def _optional_number(self, name: str) -> int | float | None:
        """Return ``None`` for an unset hyperparameter, else a number (e.g. ``max_samples``)."""
        value = self._tuned(name)
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise TypeError(f"hyperparameter '{name}' must be numeric or null, got {value!r}")
        return value

    def _target_mask(self, index: pd.Index) -> np.ndarray:
        """Boolean mask selecting the configured target entity's rows of ``index``."""
        if not isinstance(index, pd.MultiIndex) or index.nlevels < 2:
            raise ValueError(f"{self._MODEL_NAME} requires a (date, entity) MultiIndex")
        entities = index.get_level_values(1)
        return np.asarray(entities == self._target_entity_id, dtype=bool)

    def _feature_frame(self, split: MaskedPanels) -> pd.DataFrame:
        frame = super()._feature_frame(split)
        return frame.loc[self._target_mask(frame.index)]

    def _live_rows(self, split: MaskedPanels) -> pd.Series:
        live = super()._live_rows(split)
        return live.loc[self._target_mask(live.index)]

    def _supervision_series(self, split: MaskedPanels) -> pd.Series:
        series = super()._supervision_series(split)
        return series.loc[self._target_mask(series.index)]

    def _label_series(self, split: MaskedPanels) -> pd.Series:
        """Return the target entity's observed classification label as a float ``{0, 1}`` series."""
        panel = split[self._label_dataset]
        observed = panel.observed[self._label_column].fillna(False).astype(bool)
        series = panel.values[self._label_column].where(observed)
        return series.loc[self._target_mask(series.index)]

    def _prepare_design(self, frame: pd.DataFrame, *, fit: bool) -> np.ndarray:
        """Impute with training means and, when the family requires it, standardize."""
        assert self._means is not None
        filled = cast(
            np.ndarray,
            frame.fillna(self._means.reindex(frame.columns)).fillna(0.0).to_numpy(dtype=float),
        )
        if not self._STANDARDIZE:
            return filled
        if fit:
            self._center = filled.mean(axis=0)
            self._scale = np.maximum(filled.std(axis=0), _STD_FLOOR)
        assert self._center is not None and self._scale is not None
        return cast(np.ndarray, (filled - self._center) / self._scale)

    def fit(
        self,
        train: MaskedPanels,
        *,
        training: TrainingPolicy,
        val: MaskedPanels | None = None,
        tracker: TrainingTracker | None = None,
        checkpoint_dir: str | Path | None = None,
        resume: bool = False,
    ) -> None:
        """Resolve hyperparameters (selecting by inner CV when enabled), then fit on full train."""
        self._resolve_hyperparameters(train, training)
        super().fit(
            train=train,
            training=training,
            val=val,
            tracker=tracker,
            checkpoint_dir=checkpoint_dir,
            resume=resume,
        )

    def _candidate_factory(self, parameters: Mapping[str, ParameterValue]) -> Self:
        """Build a fresh, self-contained estimator carrying one candidate's hyperparameters."""
        return type(self)(
            config=self._config,
            prediction_kind=self._prediction_kind,
            cv_objective=None,
            selection=None,
            hyperparameters=parameters,
        )

    def _resolve_risk_free_by_date(self, split: MaskedPanels) -> pd.Series | None:
        """Collapse the bound date-level risk-free return to one observed rate per date.

        Returns ``None`` when no risk-free dataset is configured. The aligned panel broadcasts
        the rate across the (single) target entity, so taking the first observed value per date
        recovers the per-date series the objective funds residual cash with.
        """
        if self._risk_free_dataset is None or self._risk_free_column is None:
            return None
        panel = split[self._risk_free_dataset]
        observed = panel.observed[self._risk_free_column].fillna(False).astype(bool)
        series = panel.values[self._risk_free_column].where(observed)
        return series.groupby(level=time_level(series)).first()

    def _fold_objective(self, scores: pd.Series, returns: pd.Series) -> float:
        """Score one inner fold's validation predictions through the deployment objective.

        A portfolio CV objective is funded with the same per-date risk-free rate the deployed
        model uses, so residual cash (which, under gross single-asset normalization, is ``2`` on
        a short and ``0`` on a long) earns the risk-free rate during selection exactly as it will
        in training and evaluation.
        """
        if self._cv_objective is None:
            raise RuntimeError(
                f"{self._MODEL_NAME} hyperparameter selection requires a CV objective"
            )
        score_tensor, return_tensor, mask, dates = pack_objective_panel(scores, returns)
        risk_free_tensor = self._fold_risk_free(dates)
        with torch.inference_mode():
            if risk_free_tensor is None:
                output = self._cv_objective(score_tensor, return_tensor, mask)
            else:
                output = self._cv_objective(
                    score_tensor, return_tensor, mask, risk_free=risk_free_tensor
                )
        return float(objective_loss(output).item())

    def _fold_risk_free(self, dates: pd.Index) -> torch.Tensor | None:
        """Align the resolved per-date risk-free rate to a fold's scored dates for the objective.

        Returns ``None`` for a non-portfolio CV objective (which takes no risk-free rate). Raises
        if a scored date lacks a rate, so a plumbing gap fails loudly rather than dropping cash.
        """
        if self._risk_free_by_date is None or not callable(
            getattr(self._cv_objective, "normalize_weights", None)
        ):
            return None
        aligned = self._risk_free_by_date.reindex(dates)
        if aligned.isna().any():
            raise ValueError(
                f"{self._MODEL_NAME} risk-free rate is missing on a cross-validation scored date"
            )
        return torch.from_numpy(aligned.to_numpy(dtype=np.float32))

    def _resolve_hyperparameters(self, train: MaskedPanels, training: TrainingPolicy) -> None:
        """Fix the hyperparameters used by the final fit, running inner-CV selection if enabled."""
        if self._selection is None:
            return  # a candidate clone already carries its hyperparameters
        if not self._selection.enabled:
            self._hyperparameters = dict(self._selection.baseline)
            return
        self._risk_free_by_date = self._resolve_risk_free_by_date(train)
        result = select_hyperparameters(
            train=train,
            primary=self._feature_dataset,
            selection=self._selection,
            candidate_factory=self._candidate_factory,
            realized_returns=self._supervision_series,
            fold_objective=self._fold_objective,
            training=training,
        )
        self._hyperparameters = dict(result.selected_parameters)
        self._selection_result = result
        self._log_selection_result(result)

    def _log_selection_result(self, result: HyperparameterSelectionResult) -> None:
        """Record selection provenance on the active MLflow run, if any."""
        if mlflow.active_run() is None:
            return
        parameters: dict[str, str] = {
            "hyperparameter_selection/search_method": result.search_method,
            "hyperparameter_selection/selected_is_baseline": str(result.selected_is_baseline),
        }
        parameters |= {
            f"hyperparameter_selection/baseline/{name}": str(value)
            for name, value in result.baseline_parameters.items()
        }
        parameters |= {
            f"hyperparameter_selection/selected/{name}": str(value)
            for name, value in result.selected_parameters.items()
        }
        mlflow.log_params(parameters)
        mlflow.log_metrics(result.summary_metrics())
        mlflow.log_dict(result.to_dict(), _SELECTION_ARTIFACT)

    def _fit_backend(
        self,
        train: MaskedPanels,
        val: MaskedPanels | None,
        training: SklearnTrainingConfig,
    ) -> Mapping[str, float]:
        del val
        features = self._feature_frame(train)
        label = self._label_series(train).reindex(features.index)
        usable = (
            label.notna().to_numpy(dtype=bool)
            & np.isfinite(label.to_numpy(dtype=float))
            & features.notna().any(axis=1).to_numpy(dtype=bool)
        )
        if not usable.any():
            raise ValueError(
                f"{self._MODEL_NAME} target entity {self._target_entity_id} produced no usable rows"
            )
        design = features.loc[usable]
        label_values = label.loc[usable].to_numpy(dtype=float)
        if not np.isin(label_values, (0.0, 1.0)).all():
            raise ValueError(
                f"{self._MODEL_NAME} classification label "
                f"'{self._label_dataset}.{self._label_column}' must be a binary {{0, 1}} indicator; "
                "derive the direction label in the feature pipeline (e.g. positive_indicator)"
            )
        outcomes = label_values.astype(int)
        self._feature_columns = list(design.columns)
        self._means = design.mean()
        matrix = self._prepare_design(design, fit=True)
        positive_rate = float(outcomes.mean())
        if np.unique(outcomes).size < 2:
            # A single-class fold cannot train a classifier; predict a flat position instead.
            self._model = None
            return {
                "train_observations": float(len(outcomes)),
                "train_positive_rate": positive_rate,
                "single_class_fallback": 1.0,
            }
        self._model = self._construct(training)
        self._model.fit(matrix, outcomes)
        return {
            "train_observations": float(len(outcomes)),
            "train_positive_rate": positive_rate,
            "train_accuracy": float(self._model.score(matrix, outcomes)),
        }

    def _positive_probability(self, features: pd.DataFrame) -> np.ndarray:
        """Return ``P(up)`` per row, or a constant 0.5 for a single-class fallback fold."""
        if self._model is None:
            return np.full(len(features), 0.5, dtype=float)
        proba = self._model.predict_proba(self._prepare_design(features, fit=False))
        classes = list(self._model.classes_)
        positive = classes.index(1) if 1 in classes else len(classes) - 1
        return np.asarray(proba[:, positive], dtype=float)

    def predict(self, test: MaskedPanels) -> PredictionOutput:
        if self._means is None:
            raise RuntimeError(f"{self._MODEL_NAME} is not fitted")
        features = self._feature_frame(test).reindex(columns=self._feature_columns)
        live = self._live_rows(test).to_numpy(dtype=bool)
        score = pd.Series(
            2.0 * self._positive_probability(features) - 1.0, index=features.index, name="score"
        )
        return PredictionOutput(kind=self._prediction_kind, values=score[live].astype(float))

    def _inference_payload(self) -> dict[str, Any]:
        if self._means is None:
            raise RuntimeError(f"{self._MODEL_NAME} is not fitted")
        return {
            "config": OmegaConf.to_container(self._config, resolve=True),
            "prediction_kind": self._prediction_kind,
            "hyperparameters": dict(self._hyperparameters),
            "feature_columns": list(self._feature_columns),
            "means": {str(column): float(value) for column, value in self._means.items()},
            "center": None if self._center is None else self._center.tolist(),
            "scale": None if self._scale is None else self._scale.tolist(),
            "model": self._model,
        }

    @classmethod
    def _from_payload(cls, payload: dict[str, Any]) -> Self:
        return cls(
            config=OmegaConf.create(payload["config"]),
            prediction_kind=prediction_kind_from_bundle(payload["prediction_kind"]),
        )

    def _restore(self, payload: dict[str, Any]) -> None:
        self._hyperparameters = dict(payload.get("hyperparameters", {}))
        self._feature_columns = list(payload["feature_columns"])
        self._means = pd.Series(payload["means"], dtype=float)
        center = payload.get("center")
        scale = payload.get("scale")
        self._center = None if center is None else np.asarray(center, dtype=float)
        self._scale = None if scale is None else np.asarray(scale, dtype=float)
        self._model = payload["model"]
