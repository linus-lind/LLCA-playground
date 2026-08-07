"""Single-asset (entity-less) analytics path regressions.

A portfolio path is defined by chronological date order, so the entity-less path must be
invariant to incoming row order and must agree with the equivalent one-entity cross-sectional
panel. These guard the sort-symmetry fixes in ``_dense`` (portfolio) and ``_objective_matrices``
(predictions), and the shared-column-label fixes that let the entity-less path run at all.
"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from llca.analytics.evaluation.portfolio import build_portfolio_evaluation
from llca.analytics.evaluation.predictions import _objective_matrices
from llca.loss.portfolio import PortfolioLoss


def _objective(normalization: str = "gross") -> PortfolioLoss:
    return PortfolioLoss(
        leverage=1.0,
        risk_aversion=0.0,
        concentration_aversion=0.0,
        execution_fee=0.001,
        bid_ask_spread=0.0,
        slippage=0.0,
        borrow_cost=0.0,
        common_score_aversion=0.0,
        net_exposure_aversion=0.0,
        net_exposure_tolerance=0.0,
        normalization=normalization,  # type: ignore[arg-type]
        return_type="simple",
    )


def _kwargs(objective: PortfolioLoss, risk_free: pd.Series) -> dict:
    return dict(
        normalize=objective.normalize_weights,
        return_type="simple",
        annualization_periods=252,
        risk_free=risk_free,
        minimum_acceptable_return=0.0,
        var_levels=(0.95,),
        rolling_window=3,
        signal_buckets=2,
        active_weight_threshold=0.0,
        include_initial_trade=True,
        execution_fee=0.001,
        bid_ask_spread=0.0,
        slippage=0.0,
        borrow_cost=0.0,
    )


class SingleAssetPortfolioTest(unittest.TestCase):
    def setUp(self) -> None:
        self.dates = pd.date_range("2024-01-01", periods=12, name="date")
        rng = np.random.RandomState(0)
        self.scores = pd.Series(rng.normal(size=12), index=self.dates)
        self.returns = pd.Series(rng.normal(0.0, 0.02, size=12), index=self.dates)
        self.risk_free = pd.Series(0.0001, index=self.dates)
        self.objective = _objective()

    def test_entity_less_portfolio_is_invariant_to_input_row_order(self) -> None:
        kwargs = _kwargs(self.objective, self.risk_free)
        base = build_portfolio_evaluation(self.scores, self.returns, **kwargs)
        perm = np.random.RandomState(1).permutation(len(self.dates))
        shuffled = build_portfolio_evaluation(
            self.scores.iloc[perm], self.returns.iloc[perm], **kwargs
        )
        self.assertTrue(shuffled.daily.index.is_monotonic_increasing)
        self.assertTrue(base.daily.equals(shuffled.daily))

    def test_entity_less_matches_equivalent_one_entity_panel(self) -> None:
        kwargs = _kwargs(self.objective, self.risk_free)
        flat = build_portfolio_evaluation(self.scores, self.returns, **kwargs)
        multi = pd.MultiIndex.from_product([self.dates, ["only"]], names=["date", "entity"])
        panel = build_portfolio_evaluation(
            pd.Series(self.scores.to_numpy(), index=multi),
            pd.Series(self.returns.to_numpy(), index=multi),
            **kwargs,
        )
        np.testing.assert_allclose(
            flat.daily.to_numpy(dtype=float),
            panel.daily.to_numpy(dtype=float),
            rtol=1e-9,
            atol=1e-12,
        )

    def test_objective_matrices_entity_less_is_sorted_by_date(self) -> None:
        perm = np.random.RandomState(2).permutation(len(self.dates))
        sorted_scores, sorted_targets, sorted_valid, sorted_dates = _objective_matrices(
            self.scores, self.returns
        )
        shuffled_scores, shuffled_targets, shuffled_valid, shuffled_dates = _objective_matrices(
            self.scores.iloc[perm], self.returns.iloc[perm]
        )
        np.testing.assert_allclose(sorted_scores.numpy(), shuffled_scores.numpy())
        np.testing.assert_allclose(sorted_targets.numpy(), shuffled_targets.numpy())
        np.testing.assert_array_equal(sorted_valid.numpy(), shuffled_valid.numpy())
        # The recovered date axis is chronological regardless of input row order.
        self.assertTrue(sorted_dates.equals(shuffled_dates))


class SingleAssetComparisonTest(unittest.TestCase):
    """Synthetic single-asset MULTI-model path: evaluate two models, compare, infer, and export."""

    def test_two_single_asset_models_compare_infer_and_report(self) -> None:
        from dataclasses import replace
        from pathlib import Path
        from tempfile import TemporaryDirectory

        import matplotlib.pyplot as plt
        from omegaconf import OmegaConf

        from llca.analytics.comparison import (
            ModelEvaluationResult,
            build_comparison,
            evaluate_comparison_inference,
        )
        from llca.analytics.evaluation import evaluate_predictions
        from llca.analytics.modules.analytics_config import (
            ModelEvaluationConfig,
            RegisteredModelConfig,
        )
        from llca.analytics.modules.registered_model import RegisteredModelMetadata
        from llca.analytics.reporting import export_publication_report
        from llca.data.modules.masked_panel import MaskedPanel
        from llca.models.estimators.prediction import PredictionOutput

        dates = pd.date_range("2024-01-01", periods=48, name="date")
        rng = np.random.RandomState(7)
        returns = pd.Series(rng.normal(0.0, 0.02, size=len(dates)), index=dates)
        supervision = MaskedPanel(
            values=returns.rename("fwd_return").to_frame(),
            observed=pd.DataFrame(True, index=dates, columns=["fwd_return"]),
            age=pd.DataFrame(0, index=dates, columns=["fwd_return"]),
            segment=pd.Series(range(len(dates)), index=dates),
        )
        models = (
            RegisteredModelConfig("single-a", 1, "SA-A"),
            RegisteredModelConfig("single-b", 1, "SA-B"),
        )
        config = ModelEvaluationConfig(
            models=models,
            device="cpu",
            annualization_periods=252,
            return_type="simple",
            return_realization_lag=1,
            signal_buckets=3,
            target_threshold=0.0,
            minimum_acceptable_return=0.0,
            var_levels=(0.95,),
            autocorrelation_lags=(1,),
            worst_rolling_windows=(4,),
            rolling_window=6,
            signal_decay_periods=(0, 1),
            active_weight_threshold=0.0,
            include_initial_trade=True,
            show_plots=False,
            evaluation_end=None,
        )
        objective = _objective("gross")
        score_a = pd.Series(rng.normal(size=len(dates)), index=dates)
        raw_scores = {"SA-A": score_a, "SA-B": -score_a}  # opposite signals
        results = []
        for model in models:
            evaluation = evaluate_predictions(
                PredictionOutput(kind="portfolio", values=raw_scores[model.label].rename("score")),
                supervision,
                "fwd_return",
                objective,
                config,
                pd.Series(0.0001, index=dates),
            )
            metadata = RegisteredModelMetadata(
                config=model,
                run_id=f"run-{model.name}",
                model_uri=f"models:/{model.name}/1",
                test_start=dates[0],
                test_end=dates[-1],
                pipeline_config=OmegaConf.create({}),
                data_manifest={"schema_version": 1, "plan": {}, "sources": {}, "datasets": {}},
            )
            results.append(ModelEvaluationResult(metadata, evaluation))

        comparison = build_comparison(tuple(results), start=dates[0], end=dates[-1])
        self.assertEqual(list(comparison.signal_metrics.index), ["SA-A", "SA-B"])
        # A single-asset panel cannot identify a cross-section, so IC uses the rolling fallback.
        for result in results:
            self.assertEqual(result.evaluation.signal.ic_basis, "rolling_time_series")
        # Opposite scores over the same outcomes must give opposite-signed rank IC.
        self.assertAlmostEqual(
            float(comparison.signal_metrics.loc["SA-A", "mean_daily_rank_ic"]),
            -float(comparison.signal_metrics.loc["SA-B", "mean_daily_rank_ic"]),
            places=10,
        )

        with TemporaryDirectory() as tmp:
            report_config = replace(
                config, output_dir=Path(tmp), table_formats=("csv",), table_dpi=72
            )
            inference = evaluate_comparison_inference(comparison, report_config, dates)
            report = export_publication_report(comparison, report_config, inference)
            # Two models -> pairwise comparison + model-confidence-set artifacts are produced.
            self.assertIn("model_comparison", report.artifacts)
            self.assertIn("model_confidence_set", report.artifacts)
            self.assertIn("signal_metrics", report.artifacts)
            self.assertTrue(
                all(
                    path.exists() and path.stat().st_size > 0
                    for paths in report.artifacts.values()
                    for path in paths
                )
            )
        plt.close("all")


if __name__ == "__main__":
    unittest.main()
