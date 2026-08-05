from __future__ import annotations

from omegaconf import DictConfig
from torch import nn

from llca.loss.target_loss import TargetLoss, regression_output
from llca.mappers.loss.mapper import loss_registry, register_objective_kind
from llca.pipeline.contracts import ObjectiveKind

_REDUCTIONS = ("mean", "sum")


@loss_registry.register("mse")
def _build_mse(cfg: DictConfig, **_: object) -> nn.Module:
    return TargetLoss(
        nn.MSELoss(reduction=str(cfg.get("reduction", "mean"))),
        regression_output,
    )


@loss_registry.register_validator("mse")
def _validate_mse(cfg: DictConfig) -> list[str]:
    reduction = cfg.get("reduction")
    if reduction is not None and reduction not in _REDUCTIONS:
        return [f"loss.reduction '{reduction}' must be one of {list(_REDUCTIONS)}"]
    return []


register_objective_kind("mse", ObjectiveKind.REGRESSION)
