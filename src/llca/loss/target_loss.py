from torch import Tensor, nn

from llca.loss.modules.loss_output import LossOutput


class TargetLoss(nn.Module):
    """Adapts a standard supervised loss (e.g. ``nn.MSELoss``) to the estimator's uniform
    ``forward(scores, supervision, mask)`` contract by applying the base loss over the valid
    entries selected by ``mask``."""

    def __init__(self, base: nn.Module) -> None:
        super().__init__()
        self._base = base

    def forward(self, scores: Tensor, target: Tensor, mask: Tensor | None = None) -> LossOutput:
        """Evaluate the wrapped objective on valid entries of shape-compatible tensors.

        ``scores``, ``target``, and an optional boolean ``mask`` share their leading shape.
        Masking flattens selected entries before delegating to the wrapped loss.
        """
        if mask is not None:
            scores = scores[mask]
            target = target[mask]
        return LossOutput(loss=self._base(scores, target))
