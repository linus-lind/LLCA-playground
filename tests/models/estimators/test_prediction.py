import unittest

import numpy as np
import pandas as pd

from llca.models.estimators.prediction import (
    PREDICTION_KINDS,
    PredictionOutput,
    prediction_kind_from_bundle,
)


class PredictionOutputTest(unittest.TestCase):
    def test_contract_exposes_exactly_four_prediction_kinds(self) -> None:
        self.assertEqual(
            PREDICTION_KINDS,
            ("portfolio", "regression", "binary", "multiclass"),
        )

    def test_rejects_removed_prediction_kinds(self) -> None:
        values = pd.Series([0.1], index=pd.Index([0]))
        for removed in ("ranking", "allocation", "classification"):
            with (
                self.subTest(kind=removed),
                self.assertRaisesRegex(ValueError, "unsupported prediction kind"),
            ):
                PredictionOutput(kind=removed, values=values)  # type: ignore[arg-type]

    def test_registered_portfolio_bundle_labels_are_canonicalized(self) -> None:
        self.assertEqual(prediction_kind_from_bundle("ranking"), "portfolio")
        self.assertEqual(prediction_kind_from_bundle("allocation"), "portfolio")
        with self.assertRaisesRegex(ValueError, "unsupported prediction kind"):
            prediction_kind_from_bundle("classification")

    def test_binary_and_multiclass_shapes_are_explicit(self) -> None:
        index = pd.Index([0, 1])
        binary = pd.Series([0.2, 0.8], index=index)
        self.assertEqual(
            PredictionOutput(kind="binary", values=binary, probabilities=binary).kind,
            "binary",
        )

        classes = ["down", "flat", "up"]
        probabilities = pd.DataFrame(
            np.asarray([[0.8, 0.1, 0.1], [0.1, 0.2, 0.7]]),
            index=index,
            columns=classes,
        )
        self.assertEqual(
            PredictionOutput(
                kind="multiclass",
                values=probabilities,
                probabilities=probabilities,
            ).kind,
            "multiclass",
        )


if __name__ == "__main__":
    unittest.main()
