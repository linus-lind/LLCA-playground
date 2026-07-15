"""Model-independent configuration objects for training execution."""

from __future__ import annotations

from collections.abc import Iterable
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from typing import ClassVar

import torch
from torch import Tensor

from llca.pipeline.contracts import TrainingEngine
from llca.training.reproducibility import configure_determinism, seed_everything


class OptimizerConfig:
    """Build an optimizer for parameters placed on a resolved device."""

    optimizer_name: ClassVar[str]
    learning_rate: float

    def build(self, parameters: Iterable[Tensor], device: torch.device) -> torch.optim.Optimizer:
        """Create the configured optimizer for trainable parameters on ``device``."""
        raise NotImplementedError

    def tracking_parameters(self) -> dict[str, str | float | bool]:
        """Return stable scalar fields suitable for experiment comparison."""
        return {
            "name": self.optimizer_name,
            "learning_rate": self.learning_rate,
        }


@dataclass(frozen=True, slots=True)
class AdamConfig(OptimizerConfig):
    """Configuration for Adam, including its CUDA-only fused implementation."""

    optimizer_name: ClassVar[str] = "adam"
    learning_rate: float
    weight_decay: float
    fused: bool

    def build(self, parameters: Iterable[Tensor], device: torch.device) -> torch.optim.Optimizer:
        """Build Adam, enabling the fused implementation only on CUDA."""
        return torch.optim.Adam(
            parameters,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            fused=self.fused and device.type == "cuda",
        )

    def tracking_parameters(self) -> dict[str, str | float | bool]:
        return OptimizerConfig.tracking_parameters(self) | {
            "weight_decay": self.weight_decay,
            "fused": self.fused,
        }


@dataclass(frozen=True, slots=True)
class AdamWConfig(OptimizerConfig):
    """Adam with weight decay decoupled from gradient moments."""

    optimizer_name: ClassVar[str] = "adamw"
    learning_rate: float
    weight_decay: float
    fused: bool

    def build(self, parameters: Iterable[Tensor], device: torch.device) -> torch.optim.Optimizer:
        """Build AdamW, enabling the fused implementation only on CUDA."""
        return torch.optim.AdamW(
            parameters,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            fused=self.fused and device.type == "cuda",
        )

    def tracking_parameters(self) -> dict[str, str | float | bool]:
        return OptimizerConfig.tracking_parameters(self) | {
            "weight_decay": self.weight_decay,
            "fused": self.fused,
        }


@dataclass(frozen=True, slots=True)
class EarlyStoppingConfig:
    """Validation-loss stopping rule shared by trainable estimators."""

    patience: int
    min_delta: float


@dataclass(frozen=True, slots=True)
class TrainingDiagnosticsConfig:
    """Frequency and cost controls for reusable optimizer diagnostics."""

    interval: int
    component_gradient_norms: bool
    parameter_update_norms: bool


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Execution settings independent of model architecture and data splitting."""

    seed: int
    deterministic: bool
    epochs: int
    batch_size: int
    grad_clip: float
    device: str
    precision: str
    gradient_checkpointing: bool
    optimizer: OptimizerConfig
    early_stopping: EarlyStoppingConfig
    diagnostics: TrainingDiagnosticsConfig

    @property
    def engine(self) -> TrainingEngine:
        return TrainingEngine.TORCH

    @property
    def tracking_interval(self) -> int:
        return self.diagnostics.interval

    def tracking_parameters(self) -> dict[str, str | int | float | bool]:
        """Flatten the effective training configuration into comparable MLflow params."""
        parameters: dict[str, str | int | float | bool] = {
            "training.seed": self.seed,
            "training.deterministic": self.deterministic,
            "training.epochs": self.epochs,
            "training.batch_size": self.batch_size,
            "training.grad_clip": self.grad_clip,
            "training.device": self.device,
            "training.precision": self.precision,
            "training.gradient_checkpointing": self.gradient_checkpointing,
            "training.early_stopping.patience": self.early_stopping.patience,
            "training.early_stopping.min_delta": self.early_stopping.min_delta,
            "training.diagnostics.interval": self.diagnostics.interval,
            "training.diagnostics.component_gradient_norms": (
                self.diagnostics.component_gradient_norms
            ),
            "training.diagnostics.parameter_update_norms": (
                self.diagnostics.parameter_update_norms
            ),
        }
        parameters.update(
            {
                f"training.optimizer.{name}": value
                for name, value in self.optimizer.tracking_parameters().items()
            }
        )
        return parameters

    def prepare(self) -> torch.device:
        """Configure reproducibility, reset RNGs, and resolve the training device."""
        configure_determinism(self.deterministic)
        seed_everything(self.seed)
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)

    def use_bf16(self, device: torch.device) -> bool:
        """Return whether requested BF16 execution is supported on ``device``."""
        return self.precision == "bf16" and device.type == "cuda" and torch.cuda.is_bf16_supported()

    def precision_context(self, device: torch.device) -> AbstractContextManager[None]:
        """Return the configured batch-scoped numerical precision context."""
        if self.use_bf16(device):
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return nullcontext()
