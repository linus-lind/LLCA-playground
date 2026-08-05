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
    """Non-learning baseline whose fit only records the reproducibility seed.

    The score is a deterministic function of the live ``(date, entity)`` universe, so the
    analytics comparison constructs weights from it with the same objective normalization as
    any trained model. Concrete baselines implement ``_scores`` over the live universe.
    """

    def __init__(self, config: DictConfig, prediction_kind: PredictionKind = "portfolio") -> None:
        super().__init__(config, prediction_kind)
        self._seed = 0

    @abstractmethod
    def _scores(self, live_index: pd.Index) -> pd.Series:
        """Return one score per live ``(date, entity)`` row."""

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
            kind=self._prediction_kind, values=self._scores(index).astype(float)
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

    def _scores(self, live_index: pd.Index) -> pd.Series:
        return pd.Series(1.0, index=live_index, name="score")


class RandomLongShortEstimator(BaselineEstimator):
    """Assign seeded standard-normal scores so normalization yields a random long-short book."""

    _MODEL_NAME = "random-long-short"
    _BUNDLE_ARTIFACT = "random-long-short_bundle"
    _BUNDLE_FILENAME = "random-long-short.pkl"

    def _scores(self, live_index: pd.Index) -> pd.Series:
        generator = np.random.default_rng(self._seed)
        return pd.Series(generator.standard_normal(len(live_index)), index=live_index, name="score")
