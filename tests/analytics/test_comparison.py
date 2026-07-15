import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast
from unittest.mock import patch

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from llca.analytics.audit import build_analytics_manifest
from llca.analytics.comparison import (
    ModelEvaluationResult,
    build_comparison,
)
from llca.analytics.comparison_plots import plot_comparison
from llca.analytics.evaluation import evaluate_predictions
from llca.analytics.reporting import export_publication_report
from llca.analytics.utils.config import ModelEvaluationConfig, RegisteredModelConfig
from llca.analytics.utils.registered_model_metadata import RegisteredModelMetadata
from llca.data.modules.masked_panel import MaskedPanel
from llca.loss.portfolio import PortfolioLoss
from llca.models.estimators.prediction import PredictionOutput


def _config() -> ModelEvaluationConfig:
    return ModelEvaluationConfig(
        models=(
            RegisteredModelConfig("model", 1, "first"),
            RegisteredModelConfig("model", 2, "second"),
        ),
        device="cpu",
        annualization_periods=252,
        return_type="simple",
        signal_buckets=2,
        probability_bins=2,
        classification_threshold=0.5,
        target_threshold=0.0,
        risk_free_rate=0.0,
        minimum_acceptable_return=0.0,
        var_levels=(0.95,),
        rolling_window=2,
        signal_decay_periods=(0, 1),
        active_weight_threshold=0.0001,
        include_initial_trade=True,
        show_plots=False,
        evaluation_end=None,
    )


def _objective() -> PortfolioLoss:
    return PortfolioLoss(
        leverage=1.0,
        risk_aversion=1.0,
        concentration_aversion=0.0,
        execution_fee=0.0001,
        bid_ask_spread=0.0002,
        slippage=0.0001,
        borrow_cost=0.00001,
    )


class ComparisonEvaluationTest(unittest.TestCase):
    def test_models_share_items_tables_and_overlay_plots(self) -> None:
        dates = pd.date_range("2024-01-01", periods=5)
        index = pd.MultiIndex.from_product(
            [dates, ["A", "B", "C", "D"]], names=["date", "instrument"]
        )
        target = pd.Series(np.tile([-0.02, -0.01, 0.01, 0.02], 5), index=index)
        supervision = MaskedPanel(
            values=target.rename("return").to_frame(),
            observed=pd.DataFrame(True, index=index, columns=["return"]),
            age=pd.DataFrame(0, index=index, columns=["return"]),
            segment=pd.Series(np.arange(len(index)), index=index),
        )
        scores = (target, -target)
        results = []
        for model, score in zip(_config().models, scores, strict=True):
            evaluation = evaluate_predictions(
                PredictionOutput(kind="portfolio", values=score.rename("score")),
                supervision,
                "return",
                _objective(),
                _config(),
            )
            metadata = RegisteredModelMetadata(
                config=model,
                run_id=f"run-{model.version}",
                model_uri=f"models:/model/{model.version}",
                test_start=dates[0],
                test_end=dates[-1],
                pipeline_config=OmegaConf.create({}),
                data_manifest={"schema_version": 1, "plan": {}, "sources": {}, "datasets": {}},
            )
            results.append(ModelEvaluationResult(metadata, evaluation))

        comparison = build_comparison(
            tuple(results),
            start=dates[0],
            end=dates[-1],
            common_index=index,
        )

        self.assertEqual(list(comparison.signal_metrics.index), ["first", "second"])
        self.assertEqual(list(comparison.portfolio_metrics.index), ["first", "second"])
        self.assertEqual(len(comparison.common_index), len(index))
        self.assertGreater(
            cast(float, comparison.signal_metrics.loc["first", "mean_daily_rank_ic"]),
            0.0,
        )
        self.assertLess(
            cast(float, comparison.signal_metrics.loc["second", "mean_daily_rank_ic"]),
            0.0,
        )
        with patch("matplotlib.pyplot.show") as show:
            plot_comparison(comparison)
        show.assert_called_once()

        with TemporaryDirectory() as temporary_directory:
            report_config = replace(
                _config(),
                output_dir=Path(temporary_directory),
                table_formats=("csv", "tex", "pdf", "png"),
                table_dpi=72,
            )
            with patch("builtins.print") as console:
                report = export_publication_report(comparison, report_config)
            console.assert_not_called()
            self.assertIn("signal_metrics", report.artifacts)
            self.assertIn("portfolio_performance", report.artifacts)
            self.assertIn("signal_buckets", report.artifacts)
            self.assertIn("yearly_returns", report.artifacts)
            self.assertIn("asset_attribution", report.artifacts)
            self.assertTrue(
                all(
                    path.exists() and path.stat().st_size > 0
                    for paths in report.artifacts.values()
                    for path in paths
                )
            )
            manifest = build_analytics_manifest(
                OmegaConf.create({"analytics": {"return_type": "simple"}}),
                tuple(result.metadata for result in results),
                comparison,
                report,
            )
            files = manifest["report"]["files"]
            self.assertEqual(len(files), sum(map(len, report.artifacts.values())))
            self.assertTrue(all(len(item["sha256"]) == 64 for item in files))


if __name__ == "__main__":
    unittest.main()
