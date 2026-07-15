from __future__ import annotations

import gc
from dataclasses import dataclass
from functools import reduce
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from omegaconf import DictConfig
from pandas.api.types import is_numeric_dtype
from torch import nn

from llca.analytics.audit import build_analytics_manifest, log_analytics_report
from llca.analytics.comparison import (
    ModelEvaluationResult,
    build_comparison,
)
from llca.analytics.comparison_plots import plot_comparison
from llca.analytics.evaluation import evaluate_predictions
from llca.analytics.plots import plot_evaluation
from llca.analytics.reporting import export_publication_report
from llca.analytics.utils.data import (
    build_evaluation_panels,
    restrict_to_test_period,
    test_window_with_history,
)
from llca.analytics.utils.model_loader import (
    get_registered_model_metadata,
    load_registered_estimator,
)
from llca.analytics.utils.registered_model_metadata import RegisteredModelMetadata
from llca.core.paths import PROJECT_ROOT, chdir_to_project_root
from llca.core.resolvers import register_resolvers
from llca.data.modules.masked_panel import MaskedPanel
from llca.data.versioning import verify_raw_sources
from llca.mappers import build_analytics, build_loss
from llca.mappers.analytics.config_validator import validate_analytics_config
from llca.models.estimators.evaluation_spec import EvaluationSpec
from llca.models.estimators.prediction import PredictionOutput

register_resolvers()
load_dotenv(PROJECT_ROOT / ".env")

_CONFIG_PATH = (
    "../configs"
    if (Path(__file__).resolve().parents[1] / "configs").is_dir()
    else "../../../hydra/configs"
)


@dataclass(frozen=True, slots=True)
class _EvaluationCandidate:
    """Retain only model-specific state required after sequential prediction."""

    metadata: RegisteredModelMetadata
    predictions: PredictionOutput
    supervision: MaskedPanel
    evaluation_spec: EvaluationSpec
    objective: nn.Module | None


def _comparison_window(
    metadata: tuple[RegisteredModelMetadata, ...],
    configured_end: pd.Timestamp | None,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Intersect registered test periods and apply an optional common end boundary."""
    start = max(model.test_start for model in metadata)
    end = min(model.test_end for model in metadata)
    if configured_end is not None:
        end = min(end, configured_end)
    if end < start:
        windows = ", ".join(
            f"{model.config.label}={model.test_start.date()}..{model.test_end.date()}"
            for model in metadata
        )
        raise ValueError(f"configured models have no common test window: {windows}")
    return start, end


def _common_evaluation_index(
    candidates: list[_EvaluationCandidate],
) -> pd.Index:
    """Intersect prediction coverage and target validity for every configured model."""
    common = reduce(
        lambda left, right: left.intersection(right),
        (candidate.predictions.index for candidate in candidates),
    )
    common = common.sort_values()
    if common.empty:
        raise ValueError("configured models have no common prediction items")

    valid = pd.Series(True, index=common, dtype=bool)
    for candidate in candidates:
        column = candidate.evaluation_spec.supervision_column
        target = candidate.supervision.values[column].reindex(common)
        observed = (
            candidate.supervision.observed[column]
            .reindex(common)
            .astype("boolean")
            .fillna(False)
            .astype(bool)
        )
        candidate_valid = observed & target.notna()
        if is_numeric_dtype(target.dtype):
            candidate_valid &= pd.Series(
                np.isfinite(target.to_numpy(dtype=float)),
                index=target.index,
                dtype=bool,
            )
        valid &= candidate_valid
    evaluation_index = common[valid.to_numpy(dtype=bool)]
    if evaluation_index.empty:
        raise ValueError("configured models have no common items with valid supervision")
    return evaluation_index


def _assert_common_targets(candidates: list[_EvaluationCandidate], common_index: pd.Index) -> None:
    """Reject cross-model tables whose aligned outcomes are not actually comparable."""
    reference_candidate = candidates[0]
    reference_column = reference_candidate.evaluation_spec.supervision_column
    reference = reference_candidate.supervision.values[reference_column].reindex(common_index)
    for candidate in candidates[1:]:
        target = candidate.supervision.values[candidate.evaluation_spec.supervision_column].reindex(
            common_index
        )
        if is_numeric_dtype(reference.dtype) and is_numeric_dtype(target.dtype):
            equal = np.allclose(
                reference.to_numpy(dtype=float),
                target.to_numpy(dtype=float),
                rtol=1e-10,
                atol=1e-12,
            )
        else:
            equal = reference.equals(target)
        if not equal:
            raise ValueError(
                "configured models use different supervision values on the common "
                f"universe ({reference_candidate.metadata.config.label} versus "
                f"{candidate.metadata.config.label})"
            )


def _configured_objective(pipeline_config: DictConfig) -> nn.Module | None:
    """Build the objective stored with one model version, if training used one."""
    loss = pipeline_config.get("loss")
    if not isinstance(loss, DictConfig) or loss.get("name") is None:
        return None
    return build_loss(loss)


def _assert_return_convention(
    metadata: tuple[RegisteredModelMetadata, ...], configured_return_type: str
) -> None:
    """Prevent portfolio accounting under a convention different from training targets."""
    mismatches: list[str] = []
    for model in metadata:
        loss = model.pipeline_config.get("loss")
        if not isinstance(loss, DictConfig) or loss.get("name") != "portfolio":
            continue
        trained_return_type = str(loss.get("return_type"))
        if trained_return_type != configured_return_type:
            mismatches.append(
                f"{model.config.label}: training={trained_return_type}, "
                f"analytics={configured_return_type}"
            )
    if mismatches:
        raise ValueError(
            "analytics.return_type must match every portfolio model's training target "
            "convention: " + "; ".join(mismatches)
        )


def _target_panel(panel: MaskedPanel, column: str) -> MaskedPanel:
    """Retain one supervision column instead of the complete model-specific panel set."""
    return MaskedPanel(
        values=panel.values[[column]],
        observed=panel.observed[[column]],
        age=panel.age[[column]],
        segment=panel.segment,
    )


def _release_accelerator_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@hydra.main(config_path=_CONFIG_PATH, config_name="analytics", version_base=None)
def main(cfg: DictConfig) -> None:
    """Evaluate registry models sequentially on one shared date/instrument universe."""
    validate_analytics_config(cfg)
    analytics = build_analytics(cfg.analytics)
    tracking_uri = str(cfg.mlflow_tracking_uri)
    metadata = tuple(
        get_registered_model_metadata(model, tracking_uri) for model in analytics.models
    )
    verified_hashes: dict[Path, str] = {}
    for model in metadata:
        verify_raw_sources(model.data_manifest, verified_hashes=verified_hashes)
    _assert_return_convention(metadata, analytics.return_type)
    comparison_start, comparison_end = _comparison_window(metadata, analytics.evaluation_end)

    candidates: list[_EvaluationCandidate] = []
    for model in metadata:
        estimator = load_registered_estimator(model, analytics.device)
        panels = build_evaluation_panels(model.pipeline_config, model.data_manifest)
        evaluation_spec = estimator.evaluation_spec
        test = test_window_with_history(
            panels,
            evaluation_spec.primary_dataset,
            comparison_start,
            comparison_end,
            estimator.required_history,
        )
        prediction = restrict_to_test_period(
            estimator.predict(test),
            comparison_start,
            comparison_end,
        )
        supervision = _target_panel(
            panels[evaluation_spec.supervision_dataset],
            evaluation_spec.supervision_column,
        )
        candidates.append(
            _EvaluationCandidate(
                metadata=model,
                predictions=prediction,
                supervision=supervision,
                evaluation_spec=evaluation_spec,
                objective=_configured_objective(model.pipeline_config),
            )
        )
        del estimator, test, panels
        _release_accelerator_memory()

    common_index = _common_evaluation_index(candidates)
    _assert_common_targets(candidates, common_index)
    results: list[ModelEvaluationResult] = []
    for candidate in candidates:
        evaluation = evaluate_predictions(
            candidate.predictions.reindex(common_index),
            candidate.supervision,
            candidate.evaluation_spec.supervision_column,
            candidate.objective,
            analytics,
            objective_layout=candidate.evaluation_spec.objective_layout,
            objective_adapter=candidate.evaluation_spec.objective_adapter,
        )
        results.append(ModelEvaluationResult(metadata=candidate.metadata, evaluation=evaluation))

    comparison = build_comparison(
        tuple(results),
        start=comparison_start,
        end=comparison_end,
        common_index=common_index,
    )
    report = export_publication_report(comparison, analytics)
    manifest = build_analytics_manifest(cfg, metadata, comparison, report)
    analytics_run_id = log_analytics_report(
        manifest,
        report,
        tracking_uri=tracking_uri,
        experiment_name=str(cfg.analytics_experiment_name),
    )
    print(f"Analytics report written to: {report.directory} (MLflow run {analytics_run_id})")
    if analytics.show_plots:
        if len(results) == 1:
            plot_evaluation(results[0].evaluation)
        else:
            plot_comparison(comparison)


if __name__ == "__main__":
    chdir_to_project_root()
    main()
