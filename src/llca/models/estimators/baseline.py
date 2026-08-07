"""Rule-based sanity baselines that emit deterministic portfolio scores without learning."""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Mapping
from typing import Any, Self

import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from llca.data.modules.masked_panel import MaskedPanels
from llca.models.estimators.prediction import (
    PredictionKind,
    PredictionOutput,
    prediction_kind_from_bundle,
)
from llca.models.estimators.tabular import TabularEstimator
from llca.training.modules.sklearn_config import SklearnTrainingConfig


class BaselineEstimator(TabularEstimator):
    def __init__(self, config: DictConfig, prediction_kind: PredictionKind = "portfolio") -> None:
        super().__init__(config, prediction_kind)
        self._seed = 0

    @abstractmethod
    def _scores(self, split: MaskedPanels, live_index: pd.Index) -> pd.Series:
        """Return one score per live ``(date, entity)`` row, using ``split`` when needed."""

    def _fit_backend(
        self,
        train: MaskedPanels,
        val: MaskedPanels | None,
        training: SklearnTrainingConfig,
    ) -> Mapping[str, float]:
        del train, val
        self._seed = int(training.seed)
        return {}

    def predict(self, test: MaskedPanels) -> PredictionOutput:
        live = self._live_rows(test)
        index = live.index[live.to_numpy(dtype=bool)]
        if len(index) == 0:
            raise ValueError(f"{self._MODEL_NAME} found no live universe rows to score")
        return PredictionOutput(
            kind=self._prediction_kind, values=self._scores(test, index).astype(float)
        )

    def _inference_payload(self) -> dict[str, Any]:
        return {
            "config": OmegaConf.to_container(self._config, resolve=True),
            "prediction_kind": self._prediction_kind,
            "seed": int(self._seed),
        }

    @classmethod
    def _from_payload(cls, payload: dict[str, Any]) -> Self:
        return cls(
            config=OmegaConf.create(payload["config"]),
            prediction_kind=prediction_kind_from_bundle(payload["prediction_kind"]),
        )

    def _restore(self, payload: dict[str, Any]) -> None:
        self._seed = int(payload.get("seed", 0))


class EqualWeightEstimator(BaselineEstimator):
    """Assign one identical positive score so normalization yields 1/N long-only weights."""

    _MODEL_NAME = "equal-weight"
    _BUNDLE_ARTIFACT = "equal-weight_bundle"
    _BUNDLE_FILENAME = "equal-weight.pkl"

    def _scores(self, split: MaskedPanels, live_index: pd.Index) -> pd.Series:
        del split
        return pd.Series(1.0, index=live_index, name="score")


class InverseVolatilityEstimator(BaselineEstimator):
    """Weight each live asset by the reciprocal of a configured realized-volatility feature.

    The score ``1 / max(volatility, floor)`` is normalized cross-sectionally by the objective,
    yielding a long-only inverse-volatility portfolio. Only rows carrying a
    finite positive volatility are eligible, so warmup rows without a volatility estimate are
    excluded from the universe rather than assigned an undefined weight.
    """

    _MODEL_NAME = "inverse-volatility"
    _BUNDLE_ARTIFACT = "inverse-volatility_bundle"
    _BUNDLE_FILENAME = "inverse-volatility.pkl"

    def __init__(self, config: DictConfig, prediction_kind: PredictionKind = "portfolio") -> None:
        super().__init__(config, prediction_kind)
        volatility = config.volatility
        self._volatility_dataset = str(volatility.dataset)
        self._volatility_column = str(volatility.column)
        self._volatility_floor = float(volatility.get("floor", 1e-6))

    def _volatility(self, split: MaskedPanels) -> pd.Series:
        return split[self._volatility_dataset].values[self._volatility_column]

    def _live_rows(self, split: MaskedPanels) -> pd.Series:
        live = super()._live_rows(split)
        volatility = self._volatility(split).reindex(live.index).to_numpy(dtype=float)
        eligible = pd.Series(np.isfinite(volatility) & (volatility > 0.0), index=live.index)
        return live & eligible

    def _scores(self, split: MaskedPanels, live_index: pd.Index) -> pd.Series:
        volatility = self._volatility(split).reindex(live_index).to_numpy(dtype=float)
        inverse = 1.0 / np.maximum(volatility, self._volatility_floor)
        return pd.Series(inverse, index=live_index, name="score")
