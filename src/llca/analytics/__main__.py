from __future__ import annotations

from pathlib import Path
from typing import Any

import hydra
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from omegaconf import DictConfig

from llca.analytics.audit import build_analytics_manifest, log_analytics_report
from llca.analytics.candidates import (
    EvaluationCandidate,
    assert_common_targets,
    assert_portfolio_accounting_contract,
    assert_realization_lag_contract,
    build_evaluation_candidates,
    common_target_index,
    comparison_window,
)
from llca.analytics.comparison import (
    ModelEvaluationResult,
    build_comparison,
    evaluate_comparison_inference,
)
from llca.analytics.evaluation import evaluate_predictions
from llca.analytics.factors.report import estimate_ipca_for_report
from llca.analytics.inputs.preparation import prepare_factor_inputs
from llca.analytics.inputs.registry import get_registered_model_metadata
from llca.analytics.modules.factor_settings import FactorSources
from llca.analytics.reporting import (
    build_report_figures,
    export_publication_report,
)
from llca.analytics.reporting.factor_tables import FactorAnalysis, build_factor_analysis
from llca.core.paths import PROJECT_ROOT, chdir_to_project_root
from llca.core.resolvers import register_resolvers
from llca.data.versioning import verify_raw_sources
from llca.mappers import build_analytics
from llca.mappers.analytics.config_validator import validate_analytics_config

register_resolvers()
load_dotenv(PROJECT_ROOT / ".env")

_CONFIG_PATH = (
    "../configs/analytics"
    if (Path(__file__).resolve().parents[1] / "configs").is_dir()
    else "../../../hydra/configs/analytics"
)


@hydra.main(config_path=_CONFIG_PATH, config_name="analytics", version_base=None)
def main(cfg: DictConfig) -> None:
    """Run the full analytics report for the configured registry models.

    Validates the config, resolves each model's registry metadata and the shared evaluation
    window, checks the accounting and realization-lag contracts, and verifies raw-data sources.
    It then predicts every model on the common universe, builds the comparison, optionally runs
    the IPCA factor analysis, computes the inferential statistics, and exports the tables and
    figures, logging the report and its manifest to MLflow. Entry point for
    ``python -m llca.analytics``.
    """
    validate_analytics_config(cfg)
    analytics = build_analytics(cfg.analytics)
    tracking_uri = str(cfg.mlflow_tracking_uri)
    metadata = tuple(
        get_registered_model_metadata(model, tracking_uri) for model in analytics.models
    )
    assert_portfolio_accounting_contract(metadata, analytics.return_type)
    assert_realization_lag_contract(metadata, analytics.return_realization_lag)
    verified_hashes: dict[Path, str] = {}
    for model in metadata:
        verify_raw_sources(model.data_manifest, verified_hashes=verified_hashes)
    comparison_start, comparison_end = comparison_window(metadata, analytics.evaluation_end)

    candidates: list[EvaluationCandidate] = build_evaluation_candidates(
        metadata,
        device=analytics.device,
        comparison_start=comparison_start,
        comparison_end=comparison_end,
    )
    common_index = common_target_index(candidates)
    assert_common_targets(candidates, common_index)
    factor_inputs = prepare_factor_inputs(cfg)
    risk_free = factor_inputs.risk_free
    sources: FactorSources | None = factor_inputs.sources
    results: list[ModelEvaluationResult] = []
    for candidate in candidates:
        evaluation = evaluate_predictions(
            candidate.predictions,
            candidate.supervision,
            candidate.evaluation_spec.supervision_column,
            candidate.objective,
            analytics,
            risk_free,
            objective_layout=candidate.evaluation_spec.objective_layout,
            objective_adapter=candidate.evaluation_spec.objective_adapter,
        )
        results.append(ModelEvaluationResult(metadata=candidate.metadata, evaluation=evaluation))

    comparison = build_comparison(
        tuple(results),
        start=comparison_start,
        end=comparison_end,
    )
    factor_analysis: FactorAnalysis | None = None
    ipca_diagnostics: dict[str, Any] | None = None
    if sources is not None:
        ipca_factors, ipca_diagnostics = estimate_ipca_for_report(
            sources,
            risk_free,
            factor_inputs.prepared,
            start=comparison_start,
            end=comparison_end,
        )
        factor_analysis = build_factor_analysis(comparison, analytics, sources, ipca_factors)

    inference = evaluate_comparison_inference(comparison, analytics, common_index)
    figures = build_report_figures(comparison)
    report = export_publication_report(comparison, analytics, inference, figures, factor_analysis)
    manifest = build_analytics_manifest(
        cfg,
        metadata,
        comparison,
        report,
        common_observations=len(common_index),
        factor_data_manifest=factor_inputs.prepared.data_manifest,
        ipca_diagnostics=ipca_diagnostics,
    )
    analytics_run_id = log_analytics_report(
        manifest,
        report,
        tracking_uri=tracking_uri,
        experiment_name=str(cfg.analytics_experiment_name),
    )
    print(f"Analytics report written to: {report.directory} (MLflow run {analytics_run_id})")
    if analytics.show_plots:
        plt.show()
    for _, figure in figures:
        plt.close(figure)


if __name__ == "__main__":
    chdir_to_project_root()
    main()
