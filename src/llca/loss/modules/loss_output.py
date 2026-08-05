from __future__ import annotations

from dataclasses import dataclass, fields

from torch import Tensor


@dataclass(frozen=True, slots=True)
class LossOutput:
    """Provide a scalar objective and reusable diagnostics for any loss function.

    Subclasses may add scalar tensor fields. ``diagnostic_metrics`` exposes those fields
    without requiring the trainer or an estimator to know the concrete loss type.
    """

    loss: Tensor

    def diagnostic_metrics(self) -> dict[str, Tensor]:
        """Return scalar dataclass fields other than the differentiable objective itself."""
        metrics: dict[str, Tensor] = {}
        for field in fields(self):
            if field.name == "loss":
                continue
            value = getattr(self, field.name)
            if not isinstance(value, Tensor) or value.numel() != 1:
                raise TypeError(f"objective diagnostic '{field.name}' must be a scalar tensor")
            metrics[field.name] = value
        return metrics
