"""Cross-sectional random-forest regressor scored through the portfolio objective."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Self

import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from sklearn.ensemble import RandomForestRegressor  # type: ignore[import-untyped]

from llca.data.modules.masked_panel import MaskedPanels
from llca.models.estimators.prediction import (
    PredictionKind,
    PredictionOutput,
    prediction_kind_from_bundle,
)
from llca.models.estimators.tabular import TabularEstimator
from llca.training.modules.sklearn_config import SklearnTrainingConfig


class RandomForestEstimator(TabularEstimator):
    """Fit a random forest to the aligned target and emit its regression as a score.

    Training predicts the supervision return cross-sectionally with a random forest; the
    resulting per-item forecast is used as a portfolio score, so the analytics comparison
    forms weights from it with the same objective normalization as any other model. Missing
    features are imputed with per-column training means learned once at fit time.
    """

    _MODEL_NAME = "rf"
    _BUNDLE_ARTIFACT = "rf_bundle"
    _BUNDLE_FILENAME = "rf.pkl"

    def __init__(self, config: DictConfig, prediction_kind: PredictionKind = "portfolio") -> None:
        super().__init__(config, prediction_kind)
        self._model: RandomForestRegressor | None = None
        self._feature_columns: list[str] = []
        self._means: pd.Series | None = None

    def _require_model(self) -> RandomForestRegressor:
        if self._model is None or self._means is None:
            raise RuntimeError(f"{self._MODEL_NAME} is not fitted")
        return self._model

    def _impute(self, frame: pd.DataFrame) -> np.ndarray:
        """Fill missing features with learned column means, then any empty column with zero."""
        assert self._means is not None
        filled = frame.fillna(self._means.reindex(frame.columns)).fillna(0.0)
        return filled.to_numpy(dtype=float)

    def _fit_backend(
        self,
        train: MaskedPanels,
        val: MaskedPanels | None,
        training: SklearnTrainingConfig,
    ) -> Mapping[str, float]:
        del val
        features = self._feature_frame(train)
        target = self._supervision_series(train).reindex(features.index)
        finite = np.isfinite(target.to_numpy(dtype=float))
        usable = (
            target.notna().to_numpy(dtype=bool)
            & finite
            & features.notna().any(axis=1).to_numpy(dtype=bool)
        )
        if not usable.any():
            raise ValueError(f"{self._MODEL_NAME} training split produced no usable rows")
        design = features.loc[usable]
        outcomes = target.loc[usable].to_numpy(dtype=float)
        self._feature_columns = list(design.columns)
        self._means = design.mean()
        params = self._config
        self._model = RandomForestRegressor(
            n_estimators=int(params.n_estimators),
            max_depth=(None if params.get("max_depth") is None else int(params.max_depth)),
            min_samples_leaf=int(params.get("min_samples_leaf", 1)),
            max_features=params.get("max_features", 1.0),
            bootstrap=bool(params.get("bootstrap", True)),
            random_state=int(training.seed),
            n_jobs=int(training.n_jobs),
        )
        self._model.fit(self._impute(design), outcomes)
        return {
            "train_observations": float(len(outcomes)),
            "train_r2": float(self._model.score(self._impute(design), outcomes)),
        }

    def predict(self, test: MaskedPanels) -> PredictionOutput:
        model = self._require_model()
        features = self._feature_frame(test).reindex(columns=self._feature_columns)
        live = self._live_rows(test).to_numpy(dtype=bool)
        scores = pd.Series(
            model.predict(self._impute(features)), index=features.index, name="score"
        )
        return PredictionOutput(kind=self._prediction_kind, values=scores[live].astype(float))

    def _inference_payload(self) -> dict[str, Any]:
        model = self._require_model()
        assert self._means is not None
        return {
            "config": OmegaConf.to_container(self._config, resolve=True),
            "prediction_kind": self._prediction_kind,
            "feature_columns": list(self._feature_columns),
            "means": {str(column): float(value) for column, value in self._means.items()},
            "model": model,
        }

    @classmethod
    def _from_payload(cls, payload: dict[str, Any]) -> Self:
        return cls(
            config=OmegaConf.create(payload["config"]),
            prediction_kind=prediction_kind_from_bundle(payload["prediction_kind"]),
        )

    def _restore(self, payload: dict[str, Any]) -> None:
        self._feature_columns = list(payload["feature_columns"])
        self._means = pd.Series(payload["means"], dtype=float)
        self._model = payload["model"]
