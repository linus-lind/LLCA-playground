from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from llca.analytics.factors.panel import (
    _characteristic_frame,
    _return_series,
    prepare_ipca_panel,
)
from llca.analytics.modules.factor_settings import IpcaSettings
from llca.data.modules.masked_panel import MaskedPanel
from llca.pipeline.contracts import DataPlan
from llca.pipeline.preparation import PreparedAnalysisData


def _settings(**overrides: object) -> IpcaSettings:
    values: dict[str, object] = {
        "enabled": True,
        "n_factors": 1,
        "aligning_dataset": "asset_returns",
        "returns_dataset": "asset_returns",
        "return_column": "fwd_return",
        "return_type": "simple",
        "realization_lag": 2,
        "excess_returns": True,
        "characteristics_dataset": "firm_characteristics",
        "min_characteristic_coverage": 0.5,
        "default_max_age": 126,
        "column_max_age": {},
    }
    values.update(overrides)
    return IpcaSettings(**values)  # type: ignore[arg-type]


def _index(dates: pd.DatetimeIndex, entities: tuple[int, ...] = (1,)) -> pd.MultiIndex:
    return pd.MultiIndex.from_product([dates, list(entities)], names=["date", "instrument_id"])


def _returns_panel(index: pd.MultiIndex) -> MaskedPanel:
    return MaskedPanel(
        values=pd.DataFrame({"fwd_return": 0.01}, index=index),
        observed=pd.DataFrame({"fwd_return": True}, index=index),
        age=pd.DataFrame({"fwd_return": 0}, index=index),
        segment=pd.Series(0, index=index),
    )


def _prepared(panels: dict[str, MaskedPanel]) -> PreparedAnalysisData:
    return PreparedAnalysisData(
        data=panels,
        processed_datasets={},
        feature_panels={},
        plan=DataPlan(primary_dataset="asset_returns", datasets={}, csv_chunk_size=1),
        logical_sources={},
        data_manifest={},
    )


class IpcaPanelPreparationTest(unittest.TestCase):
    def test_panel_reads_returns_and_all_characteristics_from_masked_panels(self) -> None:
        dates = pd.bdate_range("2024-01-02", periods=3)
        index = _index(dates)
        char_panel = MaskedPanel(
            values=pd.DataFrame(
                {"quality": [1.0, 1.0, 2.0], "value": [0.5, 0.5, 0.5]}, index=index
            ),
            observed=pd.DataFrame(
                {"quality": [True, False, True], "value": [True, False, False]}, index=index
            ),
            age=pd.DataFrame({"quality": [0, 1, 0], "value": [0, 1, 2]}, index=index),
            segment=pd.Series(0, index=index),
        )
        prepared = _prepared(
            {"asset_returns": _returns_panel(index), "firm_characteristics": char_panel}
        )

        result = prepare_ipca_panel(
            _settings(), pd.Series(0.0, index=dates), prepared, start=dates[0], end=dates[-1]
        )

        # Every feature output is used, read straight from the shared masked panel: no column
        # selection, no re-alignment, and no reset of a value carried across a fresh report.
        self.assertEqual(list(result.characteristics.columns), ["quality", "value"])
        self.assertEqual(result.characteristics["quality"].tolist(), [1.0, 1.0, 2.0])
        self.assertEqual(result.characteristic_ages["value"].tolist(), [0, 1, 2])
        self.assertEqual(result.diagnostics["reference_entities"], 1)
        self.assertEqual(result.diagnostics["grid_rows"], 3)
        # Excess return: 0.01 gross less a zero risk-free rate.
        np.testing.assert_allclose(result.returns.to_numpy(), 0.01)

    def test_panel_restricts_to_the_evaluation_window(self) -> None:
        dates = pd.bdate_range("2024-01-02", periods=5)
        index = _index(dates)
        char_panel = MaskedPanel(
            values=pd.DataFrame({"quality": np.arange(5, dtype=float)}, index=index),
            observed=pd.DataFrame({"quality": True}, index=index),
            age=pd.DataFrame({"quality": 0}, index=index),
            segment=pd.Series(0, index=index),
        )
        prepared = _prepared(
            {"asset_returns": _returns_panel(index), "firm_characteristics": char_panel}
        )

        result = prepare_ipca_panel(
            _settings(), pd.Series(0.0, index=dates), prepared, start=dates[1], end=dates[3]
        )

        window = pd.DatetimeIndex(result.characteristics.index.get_level_values("date")).unique()
        self.assertEqual(window.tolist(), dates[1:4].tolist())
        self.assertEqual(result.characteristics["quality"].tolist(), [1.0, 2.0, 3.0])

    def test_returns_must_be_fresh_observed_finite_and_are_made_excess(self) -> None:
        dates = pd.bdate_range("2024-01-02", periods=4)
        index = _index(dates)
        panel = MaskedPanel(
            values=pd.DataFrame({"fwd_return": [0.10, 0.20, 0.20, np.inf]}, index=index),
            observed=pd.DataFrame({"fwd_return": [True, True, False, True]}, index=index),
            age=pd.DataFrame({"fwd_return": [0, 0, 1, 0]}, index=index),
            segment=pd.Series(0, index=index),
        )
        # The missing third-day quote must not use the stale carried second-day rate.
        risk_free = pd.Series([0.01, 0.03, 0.04], index=dates[[0, 2, 3]])

        response, diagnostics = _return_series(
            {"asset_returns": panel}, _settings(), risk_free, index
        )

        self.assertAlmostEqual(float(response.iloc[0]), 0.09)
        self.assertAlmostEqual(float(response.iloc[1]), 0.19)
        self.assertTrue(response.iloc[2:].isna().all())
        self.assertEqual(diagnostics["observed_finite_return_rows"], 2)

    def test_characteristic_frame_returns_all_columns_with_age(self) -> None:
        dates = pd.bdate_range("2024-01-02", periods=3)
        index = _index(dates)
        char_panel = MaskedPanel(
            values=pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [0.0, 0.0, 0.0]}, index=index),
            observed=pd.DataFrame({"a": True, "b": True}, index=index),
            age=pd.DataFrame({"a": [0, 1, 0], "b": [0, 0, 0]}, index=index),
            segment=pd.Series(0, index=index),
        )

        values, ages, diagnostics = _characteristic_frame(
            {"firm_characteristics": char_panel}, _settings(), index
        )

        self.assertEqual(list(values.columns), ["a", "b"])
        self.assertEqual(ages["a"].tolist(), [0, 1, 0])
        self.assertIn("characteristic_non_missing_fraction", diagnostics)


if __name__ == "__main__":
    unittest.main()
