"""Binary cross-entropy objective plugin with task-specific diagnostics."""

from __future__ import annotations

from omegaconf import DictConfig
from torch import nn

from llca.loss.target_loss import TargetLoss, binary_classification_output
from llca.mappers.config_validation import ConfigField, check_fields
from llca.mappers.loss.mapper import loss_registry, register_objective_kind
from llca.pipeline.contracts import ObjectiveKind


@loss_registry.register("binary-cross-entropy")
def build(cfg: DictConfig, **_: object) -> nn.Module:
    return TargetLoss(
        nn.BCEWithLogitsLoss(pos_weight=None, reduction=str(cfg.get("reduction", "mean"))),
        binary_classification_output,
    )


@loss_registry.register_validator("binary-cross-entropy")
def validate(cfg: DictConfig) -> list[str]:
    errors = check_fields(
        cfg,
        "loss",
        [ConfigField("reduction", "str", required=False)],
    )
    reduction = cfg.get("reduction")
    if reduction is not None and reduction not in ("mean", "sum"):
        errors.append("loss.reduction must be 'mean' or 'sum'")
    return errors


register_objective_kind("binary-cross-entropy", ObjectiveKind.BINARY_CLASSIFICATION)
