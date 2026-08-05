"""Predict each registered model once into an aligned evaluation candidate.

Models are scored sequentially and their accelerator memory released between runs so several
registry versions never accumulate their neural-network weights on the GPU at once. Each
candidate retains only the model-specific state the later evaluation and alignment stages need.
"""

from __future__ import annotations

import gc
from dataclasses import dataclass

import pandas as pd
import torch
from omegaconf import DictConfig
from torch import nn

from llca.analytics.evaluation import require_supported_prediction_kind
from llca.analytics.inputs.preparation import (
    build_evaluation_panels,
    restrict_to_test_period,
    test_window_with_history,
)
from llca.analytics.inputs.registry import load_registered_estimator
from llca.analytics.modules.registered_model import RegisteredModelMetadata
from llca.data.modules.masked_panel import MaskedPanel
from llca.mappers import build_loss
from llca.models.estimators.evaluation_spec import EvaluationSpec
from llca.models.estimators.prediction import PredictionOutput


@dataclass(frozen=True, slots=True)
class EvaluationCandidate:
    """Retain only model-specific state required after sequential prediction."""

    metadata: RegisteredModelMetadata
    predictions: PredictionOutput
    supervision: MaskedPanel
    evaluation_spec: EvaluationSpec
    objective: nn.Module | None


def _configured_objective(pipeline_config: DictConfig) -> nn.Module | None:
    """Reconstruct the training objective from a model's archived config, or ``None``.

    Returns ``None`` when the pipeline recorded no named loss.
    """
    loss = pipeline_config.get("loss")
    if not isinstance(loss, DictConfig) or loss.get("name") is None:
        return None
    return build_loss(loss)


def _target_panel(panel: MaskedPanel, column: str) -> MaskedPanel:
    """Narrow a masked panel to the single supervision ``column``, dropping the rest."""
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


def build_evaluation_candidates(
    metadata: tuple[RegisteredModelMetadata, ...],
    *,
    device: str,
    comparison_start: pd.Timestamp,
    comparison_end: pd.Timestamp,
) -> list[EvaluationCandidate]:
    """Load, run, and package every registered model into an evaluation candidate.

    For each model in turn: load its estimator onto ``device``, build its input panels, predict
    over the shared window extended by the causal history its inputs need, and clip the
    predictions back to ``comparison_start``-``comparison_end``. The prediction kind is checked,
    the supervision column retained, and the estimator and panels freed before the next model so
    only one set of weights sits on the accelerator at a time.
    """
    candidates: list[EvaluationCandidate] = []
    for model in metadata:
        estimator = load_registered_estimator(model, device)
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
        require_supported_prediction_kind(prediction.kind)
        supervision = _target_panel(
            panels[evaluation_spec.supervision_dataset],
            evaluation_spec.supervision_column,
        )
        candidates.append(
            EvaluationCandidate(
                metadata=model,
                predictions=prediction,
                supervision=supervision,
                evaluation_spec=evaluation_spec,
                objective=_configured_objective(model.pipeline_config),
            )
        )
        del estimator, test, panels
        _release_accelerator_memory()
    return candidates
