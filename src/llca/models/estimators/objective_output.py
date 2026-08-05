from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from torch import Tensor

type DiagnosticValue = float | Tensor
type DiagnosticMetrics = Mapping[str, DiagnosticValue]
type DiagnosticFactory = Callable[[], DiagnosticMetrics]


class StructuredObjectiveOutput(Protocol):
    """Minimal structural contract for an objective result with a named loss."""

    @property
    def loss(self) -> Tensor:
        """Return the scalar tensor used for differentiation."""


@runtime_checkable
class DiagnosticObjectiveOutput(StructuredObjectiveOutput, Protocol):
    """Optional extension through which objectives expose scalar components."""

    def diagnostic_metrics(self) -> Mapping[str, DiagnosticValue]:
        """Return scalar values explaining the objective without changing its gradient."""


type ObjectiveResult = Tensor | StructuredObjectiveOutput


def objective_loss(output: object) -> Tensor:
    """Extract the differentiable scalar from a tensor or structured objective output."""
    if isinstance(output, Tensor):
        return output
    loss = getattr(output, "loss", None)
    if not isinstance(loss, Tensor):
        raise TypeError("objective must return a Tensor or an object with a Tensor 'loss'")
    return loss


def objective_diagnostics(
    output: object, *, namespace: str = "objective"
) -> dict[str, DiagnosticValue]:
    """Extract namespaced diagnostics through the optional objective output protocol."""
    if not isinstance(output, DiagnosticObjectiveOutput):
        return {}
    return {
        f"{namespace}/{name}": value.detach() if isinstance(value, Tensor) else value
        for name, value in output.diagnostic_metrics().items()
    }


def tensor_distribution_diagnostics(
    values: Tensor,
    *,
    namespace: str,
    mask: Tensor | None = None,
    saturation_threshold: float | None = None,
) -> dict[str, Tensor]:
    """Summarize a tensor distribution over optional valid positions.

    Masking applies before flattening. The helper is independent of output semantics;
    callers choose the namespace and whether an absolute saturation threshold is meaningful.
    """
    selected = values[mask] if mask is not None else values.reshape(-1)
    if selected.numel() == 0:
        raise ValueError(f"cannot diagnose empty tensor '{namespace}'")
    selected = selected.float().detach()
    metrics = {
        f"{namespace}/mean": selected.mean(),
        f"{namespace}/std": selected.std(unbiased=False),
        f"{namespace}/minimum": selected.min(),
        f"{namespace}/maximum": selected.max(),
        f"{namespace}/mean_absolute": selected.abs().mean(),
    }
    if saturation_threshold is not None:
        metrics[f"{namespace}/saturation_fraction"] = (
            (selected.abs() >= saturation_threshold).float().mean()
        )
    return metrics


@dataclass(frozen=True, slots=True)
class TrainingBatchOutput:
    """Differentiable batch loss plus optional eager or lazy scalar diagnostics.

    Expensive model-specific metrics belong in ``metrics_factory`` and are evaluated only
    on steps selected by the trainer's diagnostic interval.
    """

    loss: Tensor
    metrics: DiagnosticMetrics = field(default_factory=dict)
    metrics_factory: DiagnosticFactory | None = None

    def diagnostic_metrics(self) -> dict[str, DiagnosticValue]:
        """Merge eager metrics with lazily computed diagnostics for the current step."""
        result = dict(self.metrics)
        if self.metrics_factory is not None:
            generated = dict(self.metrics_factory())
            overlap = result.keys() & generated.keys()
            if overlap:
                raise ValueError(f"duplicate training diagnostic names: {sorted(overlap)}")
            result.update(generated)
        return result


@dataclass(frozen=True, slots=True)
class BatchMetadata:
    """Stable identity and size of one recurring optimizer batch."""

    index: int
    observations: int

    def metrics(self) -> dict[str, float]:
        """Expose generic numeric batch dimensions in the step-level namespace."""
        return {
            "batch/index": float(self.index),
            "batch/observation_count": float(self.observations),
        }

    def record(self) -> dict[str, str | int]:
        """Return the generic portion of a batch manifest row."""
        return {
            "batch_index": self.index,
            "observation_count": self.observations,
        }


@dataclass(frozen=True, slots=True)
class PanelBatchMetadata(BatchMetadata):
    """Add temporal and entity dimensions for date-by-entity panel batches."""

    start_date: str
    end_date: str
    dates: int
    entities: int

    def metrics(self) -> dict[str, float]:
        """Extend generic batch metrics with panel dimensions."""
        return super().metrics() | {
            "batch/date_count": float(self.dates),
            "batch/entity_union_count": float(self.entities),
        }

    def record(self) -> dict[str, str | int]:
        """Extend the manifest row with date boundaries and panel dimensions."""
        return super().record() | {
            "start_date": self.start_date,
            "end_date": self.end_date,
            "date_count": self.dates,
            "entity_union_count": self.entities,
        }
