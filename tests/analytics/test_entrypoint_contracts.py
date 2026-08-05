from __future__ import annotations

import unittest
from types import SimpleNamespace
from typing import cast
from unittest.mock import patch

import pandas as pd
from omegaconf import OmegaConf

from llca.analytics.candidates import (
    assert_portfolio_accounting_contract,
    assert_realization_lag_contract,
)
from llca.analytics.factors.report import estimate_ipca_for_report
from llca.analytics.modules.analytics_config import RegisteredModelConfig
from llca.analytics.modules.factor_settings import FactorSources
from llca.analytics.modules.registered_model import RegisteredModelMetadata
from llca.pipeline.preparation import PreparedAnalysisData


def _metadata(label: str, **loss_overrides: object) -> RegisteredModelMetadata:
    loss: dict[str, object] = {
        "name": "portfolio",
        "return_type": "log",
        "normalization": "bounded",
        "leverage": 1.0,
        "execution_fee": 0.0001,
        "bid_ask_spread": 0.0003,
        "slippage": 0.0002,
        "borrow_cost": 0.00002,
        "risk_aversion": 1.0,
        "concentration_aversion": 0.0,
    }
    loss.update(loss_overrides)
    return RegisteredModelMetadata(
        config=RegisteredModelConfig(name=label, version=1, label=label),
        run_id=f"{label}-run",
        model_uri=f"models:/{label}/1",
        test_start=pd.Timestamp("2024-01-01"),
        test_end=pd.Timestamp("2024-12-31"),
        pipeline_config=OmegaConf.create({"loss": loss}),
        data_manifest={},
    )


def _lag_metadata(label: str, shift: int | None) -> RegisteredModelMetadata:
    """Build metadata whose archived training features encode a supervision-return shift."""
    spec: dict[str, object] = {"name": "log_change", "column": "open", "as": "fwd_return"}
    if shift is not None:
        spec["shift"] = shift
    pipeline = {
        "model": {"supervision": {"dataset": "loss", "column": "fwd_return"}},
        "features": {"loss": [spec]},
    }
    return RegisteredModelMetadata(
        config=RegisteredModelConfig(name=label, version=1, label=label),
        run_id=f"{label}-run",
        model_uri=f"models:/{label}/1",
        test_start=pd.Timestamp("2024-01-01"),
        test_end=pd.Timestamp("2024-12-31"),
        pipeline_config=OmegaConf.create(pipeline),
        data_manifest={},
    )


class RealizationLagContractTest(unittest.TestCase):
    def test_accepts_matching_trained_and_analytics_lag(self) -> None:
        models = (_lag_metadata("a", shift=-2), _lag_metadata("b", shift=-2))
        assert_realization_lag_contract(models, 2)

    def test_absent_shift_is_a_zero_lag(self) -> None:
        assert_realization_lag_contract((_lag_metadata("a", shift=None),), 0)

    def test_rejects_trained_lag_that_differs_from_analytics(self) -> None:
        models = (_lag_metadata("a", shift=-2), _lag_metadata("b", shift=-1))
        with self.assertRaisesRegex(ValueError, "b: training=1, analytics=2"):
            assert_realization_lag_contract(models, 2)

    def test_rejects_undeterminable_trained_lag(self) -> None:
        meta = RegisteredModelMetadata(
            config=RegisteredModelConfig(name="a", version=1, label="a"),
            run_id="a-run",
            model_uri="models:/a/1",
            test_start=pd.Timestamp("2024-01-01"),
            test_end=pd.Timestamp("2024-12-31"),
            pipeline_config=OmegaConf.create({}),
            data_manifest={},
        )
        with self.assertRaisesRegex(ValueError, "could not be determined"):
            assert_realization_lag_contract((meta,), 2)


class PortfolioAccountingContractTest(unittest.TestCase):
    def test_accepts_shared_accounting_with_model_specific_risk_preferences(self) -> None:
        first = _metadata("first", risk_aversion=1.0, concentration_aversion=0.0)
        second = _metadata("second", risk_aversion=4.0, concentration_aversion=0.2)

        assert_portfolio_accounting_contract((first, second), "log")

    def test_accepts_different_normalization_and_leverage_mandates(self) -> None:
        first = _metadata("first", normalization="bounded", leverage=1.0)
        second = _metadata("second", normalization="market_neutral", leverage=2.0)

        assert_portfolio_accounting_contract((first, second), "log")

    def test_rejects_training_return_type_different_from_analytics(self) -> None:
        model = _metadata("model", return_type="simple")

        with self.assertRaisesRegex(ValueError, "training=simple, analytics=log"):
            assert_portfolio_accounting_contract((model,), "log")

    def test_rejects_different_realized_accounting_settings(self) -> None:
        first = _metadata("first")
        second = _metadata("second", slippage=0.001, borrow_cost=0.0005)

        with self.assertRaisesRegex(ValueError, "slippage.*borrow_cost"):
            assert_portfolio_accounting_contract((first, second), "log")

    def test_enabled_ipca_primary_failure_propagates(self) -> None:
        sources = cast(
            FactorSources,
            SimpleNamespace(ipca=SimpleNamespace(enabled=True)),
        )
        with patch(
            "llca.analytics.factors.report.prepare_ipca_panel",
            side_effect=ValueError("no viable IPCA cross-section"),
        ):
            with self.assertRaisesRegex(ValueError, "no viable IPCA cross-section"):
                estimate_ipca_for_report(
                    sources,
                    pd.Series(dtype=float),
                    cast(PreparedAnalysisData, object()),
                    start=pd.Timestamp("2024-01-01"),
                    end=pd.Timestamp("2024-12-31"),
                )


if __name__ == "__main__":
    unittest.main()
