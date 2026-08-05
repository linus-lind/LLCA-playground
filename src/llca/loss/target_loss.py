from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from llca.loss.modules.loss_output import LossOutput


@dataclass(frozen=True, slots=True)
class RegressionLossOutput(LossOutput):
    """Regression-specific diagnostics emitted without portfolio-only fields."""

    mean_squared_error: Tensor
    mean_absolute_error: Tensor


@dataclass(frozen=True, slots=True)
class BinaryClassificationLossOutput(LossOutput):
    """Binary classification diagnostics derived from logits and labels."""

    accuracy: Tensor
    positive_probability: Tensor
    positive_rate: Tensor


type OutputFactory = Callable[[Tensor, Tensor, Tensor], LossOutput]


def regression_output(loss: Tensor, scores: Tensor, target: Tensor) -> LossOutput:
    errors = scores.float() - target.float()
    return RegressionLossOutput(
        loss=loss,
        mean_squared_error=errors.square().mean(),
        mean_absolute_error=errors.abs().mean(),
    )


def binary_classification_output(loss: Tensor, scores: Tensor, target: Tensor) -> LossOutput:
    probabilities = torch.sigmoid(scores.float())
    labels = target.float()
    return BinaryClassificationLossOutput(
        loss=loss,
        accuracy=((probabilities >= 0.5) == (labels >= 0.5)).float().mean(),
        positive_probability=probabilities.mean(),
        positive_rate=labels.mean(),
    )


class TargetLoss(nn.Module):
    """Adapts a standard supervised loss (e.g. ``nn.MSELoss``) to the estimator's uniform
    ``forward(scores, supervision, mask)`` contract by applying the base loss over the valid
    entries selected by ``mask``."""

    def __init__(self, base: nn.Module, output_factory: OutputFactory | None = None) -> None:
        super().__init__()
        self._base = base
        self._output_factory = output_factory

    def forward(self, scores: Tensor, target: Tensor, mask: Tensor | None = None) -> LossOutput:
        """Evaluate the wrapped objective on valid entries of shape-compatible tensors.

        ``scores``, ``target``, and an optional boolean ``mask`` share their leading shape.
        Masking flattens selected entries before delegating to the wrapped loss.
        """
        if mask is not None:
            scores = scores[mask]
            target = target[mask]
        loss = self._base(scores, target)
        return (
            self._output_factory(loss, scores, target)
            if self._output_factory is not None
            else LossOutput(loss=loss)
        )
