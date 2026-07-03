"""Map validated Hydra training configuration to reusable runtime objects."""

from omegaconf import DictConfig

from llca.mappers.modules.registry import Registry
from llca.training.modules.training_config import (
    AdamConfig,
    AdamWConfig,
    EarlyStoppingConfig,
    OptimizerConfig,
    TrainingConfig,
    TrainingDiagnosticsConfig,
)

optimizer_registry: Registry[OptimizerConfig] = Registry("optimizer")


@optimizer_registry.register("adam")
def _build_adam(cfg: DictConfig) -> AdamConfig:
    return AdamConfig(
        learning_rate=float(cfg.learning_rate),
        weight_decay=float(cfg.weight_decay),
        fused=bool(cfg.fused),
    )


@optimizer_registry.register("adamw")
def _build_adamw(cfg: DictConfig) -> AdamWConfig:
    return AdamWConfig(
        learning_rate=float(cfg.learning_rate),
        weight_decay=float(cfg.weight_decay),
        fused=bool(cfg.fused),
    )


def build_training(cfg: DictConfig) -> TrainingConfig:
    """Build model-independent training runtime configuration."""
    optimizer = optimizer_registry.build(str(cfg.optimizer.name), cfg.optimizer)
    return TrainingConfig(
        seed=int(cfg.seed),
        deterministic=bool(cfg.deterministic),
        epochs=int(cfg.epochs),
        batch_size=int(cfg.batch_size),
        grad_clip=float(cfg.grad_clip),
        device=str(cfg.device),
        precision=str(cfg.precision),
        gradient_checkpointing=bool(cfg.gradient_checkpointing),
        optimizer=optimizer,
        early_stopping=EarlyStoppingConfig(
            patience=int(cfg.early_stopping.patience),
            min_delta=float(cfg.early_stopping.min_delta),
        ),
        diagnostics=TrainingDiagnosticsConfig(
            interval=int(cfg.diagnostics.interval),
            component_gradient_norms=bool(cfg.diagnostics.component_gradient_norms),
            parameter_update_norms=bool(cfg.diagnostics.parameter_update_norms),
        ),
    )
