import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast
from unittest.mock import patch

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from omegaconf import OmegaConf

from llca.analytics.audit import build_analytics_manifest
from llca.analytics.comparison import (
    ModelEvaluationResult,
    build_comparison,
    build_model_confidence_summary,
    build_model_significance_frame,
    evaluate_comparison_inference,
)
from llca.analytics.evaluation import evaluate_predictions
from llca.analytics.modules.analytics_config import ModelEvaluationConfig, RegisteredModelConfig
from llca.analytics.modules.registered_model import RegisteredModelMetadata
from llca.analytics.reporting import export_publication_report
from llca.analytics.reporting.figures import build_report_figures
from llca.analytics.reporting.statistics_figures import build_statistics_comparison_figure
from llca.analytics.reporting.tables import build_publication_tables
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
        return_realization_lag=2,
        signal_buckets=2,
        target_threshold=0.0,
        minimum_acceptable_return=0.0,
        var_levels=(0.95,),
        autocorrelation_lags=(1,),
        worst_rolling_windows=(2,),
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
        normalization="bounded",
        return_type="simple",
        common_score_aversion=0.0,
        net_exposure_aversion=0.0,
        net_exposure_tolerance=1.0,
    )


class ComparisonEvaluationTest(unittest.TestCase):
    def test_models_share_items_tables_and_overlay_plots(self) -> None:
        dates = pd.date_range("2024-01-01", periods=5)
        index = pd.MultiIndex.from_product(
            [dates, ["A", "B", "C", "D"]], names=["date", "instrument"]
        )
        base = np.array([-0.02, -0.01, 0.01, 0.02])
        drift = np.repeat(np.arange(len(dates)) * 0.001, 4)
        target = pd.Series(np.tile(base, len(dates)) + drift, index=index)
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
                pd.Series(0.0, index=dates),
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
        )

        self.assertEqual(list(comparison.signal_metrics.index), ["first", "second"])
        self.assertEqual(list(comparison.portfolio_metrics.index), ["first", "second"])
        self.assertGreater(
            cast(float, comparison.signal_metrics.loc["first", "mean_daily_rank_ic"]),
            0.0,
        )
        self.assertLess(
            cast(float, comparison.signal_metrics.loc["second", "mean_daily_rank_ic"]),
            0.0,
        )
        single = build_comparison(
            (results[0],),
            start=dates[0],
            end=dates[-1],
        )
        single_figures = build_report_figures(single)
        multiple_figures = build_report_figures(comparison)
        try:
            self.assertEqual(
                [name for name, _ in single_figures],
                [name for name, _ in multiple_figures],
            )
            self.assertEqual(
                [name for name, _ in single_figures],
                ["portfolio_comparison", "signal_comparison", "confusion_roc"],
            )
        finally:
            for _, figure in (*single_figures, *multiple_figures):
                plt.close(figure)

        single_significance = build_model_significance_frame(single, _config())
        comparison_significance = build_model_significance_frame(comparison, _config())
        single_tables = {
            table.name: table
            for table in build_publication_tables(single, _config(), single_significance)
        }
        multiple_tables = {
            table.name: table
            for table in build_publication_tables(comparison, _config(), comparison_significance)
        }
        self.assertEqual(single_tables.keys(), multiple_tables.keys())
        for name in ("yearly_returns", "side_attribution", "signal_bucket_analysis"):
            with self.subTest(table=name):
                self.assertNotIsInstance(single_tables[name].frame.columns, pd.MultiIndex)
                self.assertEqual(single_tables[name].frame.columns.name, "Statistic")
                self.assertIsInstance(multiple_tables[name].frame.columns, pd.MultiIndex)
                self.assertEqual(
                    list(multiple_tables[name].frame.columns.names),
                    ["Statistic", "Model"],
                )
        single_inference = evaluate_comparison_inference(single, _config(), index)
        self.assertEqual(
            build_statistics_comparison_figure(
                single_inference.comparison_matrices, single_inference.model_confidence
            ),
            [],
        )
        with patch(
            "llca.analytics.comparison.inference.inference.model_confidence_set",
            return_value=pd.DataFrame(),
        ):
            confidence = build_model_confidence_summary(comparison, _config())
        assert confidence is not None
        self.assertIn("95% confidence set (5% significance)", confidence.caption)
        significance = build_model_significance_frame(comparison, _config())
        self.assertNotIn("pt_statistic", significance.columns)
        for label in ("first", "second"):
            self.assertAlmostEqual(
                cast(float, significance.loc[label, "mean_ic"]),
                cast(float, comparison.signal_metrics.loc[label, "mean_daily_rank_ic"]),
            )
            self.assertAlmostEqual(
                cast(float, significance.loc[label, "mean_hit_rate"]),
                cast(float, comparison.signal_metrics.loc[label, "directional_accuracy"]),
            )

        with TemporaryDirectory() as temporary_directory:
            report_config = replace(
                _config(),
                output_dir=Path(temporary_directory),
                table_formats=("csv", "tex", "pdf", "png"),
                table_dpi=72,
            )
            inference = evaluate_comparison_inference(comparison, report_config, index)
            with patch("builtins.print") as console:
                report = export_publication_report(comparison, report_config, inference)
            console.assert_not_called()
            self.assertRegex(report.directory.name, r"_[0-9a-f]{12}$")
            self.assertIn("signal_metrics", report.artifacts)
            self.assertIn("portfolio_performance", report.artifacts)
            self.assertIn("signal_bucket_analysis", report.artifacts)
            self.assertIn("yearly_returns", report.artifacts)
            self.assertIn("side_attribution", report.artifacts)
            self.assertIn("statistical_significance", report.artifacts)
            # The pairwise matrices share one grid figure; the confidence set stands alone.
            self.assertIn("model_comparison", report.artifacts)
            self.assertIn("model_confidence_set", report.artifacts)
            self.assertNotIn("diebold_mariano_pvalues", report.artifacts)
            self.assertNotIn("sharpe_difference_pvalues", report.artifacts)
            self.assertNotIn("portfolio_return_correlation", report.artifacts)
            self.assertNotIn("signal_correlation", report.artifacts)
            self.assertNotIn("position_overlap", report.artifacts)
            self.assertNotIn("signal_buckets", report.artifacts)
            self.assertNotIn("confusion_matrix", report.artifacts)
            self.assertNotIn("asset_attribution", report.artifacts)
            self.assertNotIn("maximum_drawdown_attribution", report.artifacts)
            self.assertNotIn("signal_decay", report.artifacts)
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
                common_observations=len(index),
                factor_data_manifest={"plan": "shared-factor-inputs"},
                ipca_diagnostics={"usable_returns": 10},
            )
            files = manifest["report"]["files"]
            self.assertEqual(manifest["schema_version"], 5)
            self.assertEqual(manifest["evaluation"]["common_observations"], len(index))
            self.assertEqual(manifest["evaluation"]["common_dates"], len(dates))
            self.assertEqual(
                manifest["evaluation"]["funding_convention"],
                "residual_cash_at_risk_free",
            )
            self.assertEqual(manifest["models"][0]["prediction_kind"], "portfolio")
            self.assertEqual(manifest["models"][0]["ic_basis"], "cross_sectional")
            self.assertIsNone(manifest["models"][0]["portfolio_accounting"])
            self.assertEqual(manifest["factor_inputs"]["pipeline"], {})
            self.assertEqual(
                manifest["factor_inputs"]["data_manifest"],
                {"plan": "shared-factor-inputs"},
            )
            self.assertEqual(manifest["ipca"]["diagnostics"], {"usable_returns": 10})
            self.assertEqual(len(files), sum(map(len, report.artifacts.values())))
            self.assertTrue(all(len(item["sha256"]) == 64 for item in files))


if __name__ == "__main__":
    unittest.main()
