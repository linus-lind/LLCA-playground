import unittest

import numpy as np
import pandas as pd

from llca.analytics.utils.data import restrict_to_test_period, test_window_with_history
from llca.data.modules.masked_panel import MaskedPanel
from llca.models.estimators.prediction import PredictionOutput


def _panel(dates: pd.DatetimeIndex) -> MaskedPanel:
    index = pd.MultiIndex.from_product([dates, [1]], names=["date", "instrument"])
    values = pd.DataFrame({"x": np.arange(len(index), dtype=float)}, index=index)
    return MaskedPanel(
        values=values,
        observed=pd.DataFrame(True, index=index, columns=["x"]),
        age=pd.DataFrame(0, index=index, columns=["x"]),
        segment=pd.Series(0, index=index),
    )


class EvaluationDataTest(unittest.TestCase):
    def test_test_window_includes_only_required_history(self) -> None:
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        panels = {"features": _panel(dates)}

        selected = test_window_with_history(
            panels,
            "features",
            dates[5],
            dates[8],
            lookback=3,
        )

        selected_dates = selected["features"].values.index.get_level_values("date").unique()
        self.assertEqual(selected_dates[0], dates[2])
        self.assertEqual(selected_dates[-1], dates[8])

    def test_restrict_predictions_removes_history_rows(self) -> None:
        dates = pd.date_range("2024-01-01", periods=6, freq="D")
        index = pd.MultiIndex.from_product([dates, [1]], names=["date", "instrument"])
        predictions = PredictionOutput(
            kind="portfolio",
            values=pd.Series(np.arange(len(index)), index=index, name="score"),
        )

        selected = restrict_to_test_period(predictions, dates[2], dates[4])

        self.assertEqual(len(selected.values), 3)
        self.assertEqual(selected.index.get_level_values("date").min(), dates[2])
        self.assertEqual(selected.index.get_level_values("date").max(), dates[4])


if __name__ == "__main__":
    unittest.main()
