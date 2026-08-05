from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from llca.analytics.inputs.preparation import prepare_factor_inputs
from llca.pipeline.contracts import DataPlan, EntityScope
from llca.pipeline.preparation import PreparedAnalysisData


def _config() -> object:
    return OmegaConf.create(
        {
            "analytics": {
                "risk_free": {"dataset": "ff", "column": "rf"},
                "factor_analysis": {
                    "enabled": True,
                    "aligning_dataset": "asset_returns",
                    "factors": {
                        "dataset": "ff",
                        "ff6": ["market"],
                        "market": "market",
                    },
                    "spanning": {
                        "dataset": "ff",
                        "portfolios": ["market"],
                        "scale": 1.0,
                        "excess": False,
                    },
                    "ipca": {
                        "enabled": True,
                        "n_factors": 1,
                        "inputs": {
                            "returns": {
                                "dataset": "asset_returns",
                                "column": "fwd_return",
                                "return_type": "simple",
                                "realization_lag": 2,
                                "excess": True,
                            },
                            "characteristics": {"dataset": "firm_characteristics"},
                        },
                        "min_characteristic_coverage": 0.5,
                        "max_age": {"default": 126, "columns": {}},
                    },
                    "timing": {
                        "instruments": {"dataset": "macro", "columns": ["state"]},
                        "market_squared": True,
                        "conditional_alpha": True,
                    },
                    "rolling_beta_window": 3,
                },
            }
        }
    )


class FactorDataPreparationTest(unittest.TestCase):
    def test_one_pipeline_call_prepares_union_and_native_features_are_not_shifted(self) -> None:
        dates = pd.bdate_range("2024-01-02", periods=5)
        # The feature pipeline has already put realized t+2 returns on decision labels t.
        ff = pd.DataFrame(
            {
                "rf": np.array([0.002, 0.003, 0.004]),
                "market": np.array([0.02, 0.03, 0.04]),
            },
            index=dates[:3],
        )
        macro = pd.DataFrame({"state": np.array([0.0, 1.0, np.nan, 3.0, 4.0])}, index=dates)
        entity_index = pd.MultiIndex.from_product([dates, [1]], names=["date", "instrument_id"])
        asset_returns = pd.DataFrame({"fwd_return": 0.01}, index=entity_index)
        characteristics = pd.DataFrame({"quality": 1.0}, index=entity_index)
        panels = {
            "ff": ff,
            "macro": macro,
            "asset_returns": asset_returns,
            "firm_characteristics": characteristics,
        }
        prepared = PreparedAnalysisData(
            data={},
            processed_datasets=panels,
            feature_panels=panels,
            plan=DataPlan(primary_dataset="asset_returns", datasets={}, csv_chunk_size=1),
            logical_sources={},
            data_manifest={"plan": "shared"},
        )
        cfg = _config()

        with patch(
            "llca.analytics.inputs.preparation.prepare_analysis_data",
            return_value=prepared,
        ) as prepare:
            inputs = prepare_factor_inputs(cfg)  # type: ignore[arg-type]

        prepare.assert_called_once()
        requirements = prepare.call_args.args[1]
        self.assertEqual(requirements.primary_dataset, "asset_returns")
        self.assertEqual(
            {requirement.name for requirement in requirements.datasets},
            {"ff", "macro", "asset_returns", "firm_characteristics"},
        )
        self.assertTrue(
            all(
                requirement.entity_scope is EntityScope.UNIVERSE
                for requirement in requirements.datasets
            )
        )
        self.assertEqual(prepare.call_args.kwargs["data_view"], "aligned_panel")
        self.assertIs(inputs.prepared, prepared)
        self.assertEqual(inputs.risk_free.index.tolist(), dates[:3].tolist())
        np.testing.assert_allclose(inputs.risk_free.to_numpy(), ff["rf"].to_numpy())
        assert inputs.sources is not None
        self.assertTrue(inputs.sources.ff6.equals(ff[["market"]]))
        self.assertTrue(inputs.sources.timing_instruments.equals(macro))
        self.assertEqual(inputs.sources.timing_instrument_lag, 1)

    def test_disabled_factor_analysis_prepares_only_risk_free_features(self) -> None:
        dates = pd.bdate_range("2024-01-02", periods=2)
        frame = pd.DataFrame({"rf": [0.001, 0.002]}, index=dates)
        prepared = PreparedAnalysisData(
            data={"ff": frame},
            processed_datasets={"ff": frame},
            feature_panels={"ff": frame},
            plan=DataPlan(primary_dataset="ff", datasets={}, csv_chunk_size=1),
            logical_sources={},
            data_manifest={},
        )
        cfg = OmegaConf.create(
            {
                "analytics": {
                    "risk_free": {"dataset": "ff", "column": "rf"},
                    "factor_analysis": {"enabled": False},
                }
            }
        )

        with patch(
            "llca.analytics.inputs.preparation.prepare_analysis_data",
            return_value=prepared,
        ) as prepare:
            inputs = prepare_factor_inputs(cfg)

        requirements = prepare.call_args.args[1]
        self.assertEqual([item.name for item in requirements.datasets], ["ff"])
        self.assertEqual(prepare.call_args.kwargs["data_view"], "independent")
        self.assertIsNone(inputs.sources)
        self.assertTrue(inputs.risk_free.equals(frame["rf"].rename("risk_free")))


if __name__ == "__main__":
    unittest.main()
