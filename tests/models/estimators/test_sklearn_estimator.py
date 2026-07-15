from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock

import pandas as pd

from llca.models.estimators.evaluation_spec import EvaluationSpec
from llca.models.estimators.prediction import PredictionOutput
from llca.models.estimators.sklearn import SklearnEstimator
from llca.training.modules.sklearn_config import SklearnTrainingConfig


class _MeanEstimator(SklearnEstimator[pd.DataFrame]):
    _MODEL_NAME = "mean"
    _BUNDLE_ARTIFACT = "mean_bundle"
    _BUNDLE_FILENAME = "mean.pkl"

    def __init__(self) -> None:
        self.mean: float | None = None

    def _fit_backend(
        self,
        train: pd.DataFrame,
        val: pd.DataFrame | None,
        training: SklearnTrainingConfig,
    ) -> dict[str, float]:
        del val, training
        self.mean = float(train["target"].mean())
        return {"objective/mean_squared_error": 0.0}

    def predict(self, test: pd.DataFrame) -> PredictionOutput:
        if self.mean is None:
            raise RuntimeError("mean estimator is not fitted")
        return PredictionOutput(
            kind="regression",
            values=pd.Series(self.mean, index=test.index, name="prediction"),
        )

    @property
    def evaluation_spec(self) -> EvaluationSpec:
        return EvaluationSpec("values", "values", "target", objective_layout="rows")

    def _inference_payload(self) -> dict[str, object]:
        return {"mean": self.mean}

    @classmethod
    def _from_payload(cls, payload: dict[str, object]) -> _MeanEstimator:
        del payload
        return cls()

    def _restore(self, payload: dict[str, object]) -> None:
        self.mean = float(payload["mean"])


class SklearnEstimatorTest(unittest.TestCase):
    def test_backend_policy_tracking_prediction_and_bundle_roundtrip(self) -> None:
        estimator = _MeanEstimator()
        tracker = MagicMock()
        data = pd.DataFrame(
            {"target": [1.0, 3.0]},
            index=pd.date_range("2024-01-01", periods=2, name="date"),
        )
        training = SklearnTrainingConfig(seed=7, n_jobs=1)

        estimator.fit(data, training=training, tracker=tracker)

        tracker.begin.assert_called_once_with(total_epochs=1, steps_per_epoch=1)
        tracker.on_epoch_end.assert_called_once_with(
            0,
            metrics={"objective/mean_squared_error": 0.0},
        )
        self.assertEqual(estimator.predict(data).values.tolist(), [2.0, 2.0])

        with TemporaryDirectory() as directory:
            bundle = Path(directory) / "model.pkl"
            estimator._save(bundle)
            restored = _MeanEstimator.load(bundle)
        self.assertEqual(restored.predict(data).values.tolist(), [2.0, 2.0])

    def test_resume_requires_plugin_specific_incremental_support(self) -> None:
        estimator = _MeanEstimator()
        data = pd.DataFrame({"target": [1.0]})
        with self.assertRaisesRegex(NotImplementedError, "checkpoint recovery"):
            estimator.fit(
                data,
                training=SklearnTrainingConfig(seed=7, n_jobs=1),
                resume=True,
            )


if __name__ == "__main__":
    unittest.main()
