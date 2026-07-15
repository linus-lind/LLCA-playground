"""Generic optimization loop shared by trainable estimators."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor, nn

from llca.training.checkpointer import Checkpointer
from llca.training.modules.tracking import TrainingTracker
from llca.training.modules.training_config import TrainingConfig
from llca.training.modules.training_diagnostics import TrainingBatchOutput, objective_loss
from llca.training.modules.training_task import TrainingTask
from llca.training.reproducibility import restore_rng_state

CheckpointStateFactory = Callable[
    [torch.optim.Optimizer, int, float, dict[str, Tensor] | None, int], dict[str, Any]
]


def _l2_norm(tensors: Sequence[Tensor]) -> float:
    squared = [tensor.detach().float().square().sum() for tensor in tensors]
    if not squared:
        return 0.0
    return float(torch.stack(squared).sum().sqrt().item())


def _component_gradient_norms(
    model: nn.Module, group_for_parameter: Callable[[str], str] | None = None
) -> dict[str, Tensor]:
    """Aggregate pre-clipping L2 gradient norms by top-level model component."""
    squared: dict[str, list[Tensor]] = {}
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        component = (
            group_for_parameter(name)
            if group_for_parameter is not None
            else name.split(".", maxsplit=1)[0]
        )
        squared.setdefault(component, []).append(parameter.grad.detach().float().square().sum())
    return {
        f"gradient/components_pre_clip/{component}": torch.stack(values).sum().sqrt()
        for component, values in squared.items()
    }


def _scalar_metrics(metrics: dict[str, float | Tensor]) -> dict[str, float]:
    """Detach scalar diagnostics in one device transfer and reject invalid values.

    Tensor diagnostics must contain exactly one element. Batched conversion avoids a
    device synchronization for each metric, which matters when diagnostics are collected
    on every optimizer step.
    """
    result = {
        name: float(value) for name, value in metrics.items() if not isinstance(value, Tensor)
    }
    tensor_items = [(name, value) for name, value in metrics.items() if isinstance(value, Tensor)]
    for name, value in tensor_items:
        if value.numel() != 1:
            raise ValueError(f"training diagnostic '{name}' must be scalar")
    if tensor_items:
        values = (
            torch.stack([value.detach().float().reshape(()) for _, value in tensor_items])
            .cpu()
            .tolist()
        )
        result.update(
            {name: float(value) for (name, _), value in zip(tensor_items, values, strict=True)}
        )
    for name in metrics:
        if not math.isfinite(result[name]):
            raise FloatingPointError(f"training diagnostic '{name}' is not finite")
    return result


class Trainer[BatchT]:
    """Optimize an ``nn.Module`` through model-specific batch callbacks.

    Estimators retain ownership of data preparation and the differentiable loss
    for one batch. This class owns the reusable execution policy: optimizer
    construction, gradient clipping, tracking, validation-based early stopping,
    checkpoint persistence, exact RNG resume, and restoration of best weights.
    """

    def __init__(
        self,
        config: TrainingConfig,
        task: TrainingTask[BatchT],
        *,
        tracker: TrainingTracker | None = None,
        checkpoint_dir: str | Path | None = None,
        checkpoint_state: CheckpointStateFactory | None = None,
    ) -> None:
        if checkpoint_dir is not None and checkpoint_state is None:
            raise ValueError("checkpoint_state is required when checkpoint_dir is configured")
        self._config = config
        self._task = task
        self._model = task.model
        self._batches = task.batches
        self._loss_for_batch = task.train_step
        self._validation_loss = task.validation_step
        self._batch_metadata = task.batch_metadata
        self._tracker = tracker
        self._checkpointer = Checkpointer(checkpoint_dir) if checkpoint_dir is not None else None
        self._checkpoint_state = checkpoint_state

    def fit(self, *, resume: bool = False) -> None:
        """Run optimization and restore the best validation state before returning.

        Estimator callbacks produce one scalar differentiable loss and optional diagnostics
        per batch. Gradients are checked, measured, and globally clipped before each update.
        Expensive parameter snapshots and component norms run only at the diagnostic
        interval. When resuming, model, optimizer, progress, early-stopping state, and RNG
        streams are restored together; changing optimizer families is rejected.
        """
        device = next(self._model.parameters()).device
        parameters = [
            parameter for parameter in self._model.parameters() if parameter.requires_grad
        ]
        optimizer = self._config.optimizer.build(parameters, device)
        best_val = math.inf
        best_state: dict[str, Tensor] | None = None
        epochs_without_improvement = 0
        start_epoch = 0

        if resume and self._checkpointer is None:
            raise ValueError("resume requires a checkpoint directory")
        if self._checkpointer is not None and resume:
            resumed = self._checkpointer.load_latest(map_location=device)
            checkpoint_optimizer = cast(str, resumed["optimizer_name"])
            configured_optimizer = self._config.optimizer.optimizer_name
            if checkpoint_optimizer != configured_optimizer:
                raise ValueError(
                    "cannot resume checkpoint created with optimizer "
                    f"'{checkpoint_optimizer}' using configured optimizer "
                    f"'{configured_optimizer}'"
                )
            self._model.load_state_dict(resumed["model_state_dict"])
            optimizer.load_state_dict(resumed["optimizer_state_dict"])
            best_val = float(resumed["best_val"])
            best_state = cast(dict[str, Tensor] | None, resumed["best_state"])
            epochs_without_improvement = int(resumed["epochs_without_improvement"])
            start_epoch = int(resumed["epoch"]) + 1
            rng_state = resumed.get("rng_state")
            if isinstance(rng_state, dict):
                restore_rng_state(rng_state)

            terminal = start_epoch >= self._config.epochs or (
                self._validation_loss is not None
                and epochs_without_improvement > 0
                and epochs_without_improvement >= self._config.early_stopping.patience
            )
            if terminal:
                if best_state is not None:
                    self._model.load_state_dict(best_state)
                return

        metadata = (
            [self._batch_metadata(batch, index) for index, batch in enumerate(self._batches)]
            if self._batch_metadata is not None
            else []
        )
        if self._tracker is not None:
            self._tracker.begin(
                self._config.epochs,
                steps_per_epoch=len(self._batches),
                start_step=start_epoch * len(self._batches),
                batch_manifest=metadata,
            )

        for epoch in range(start_epoch, self._config.epochs):
            self._model.train()
            epoch_loss = 0.0
            for batch_index, batch in enumerate(self._batches):
                optimizer.zero_grad(set_to_none=True)
                with self._config.precision_context(device):
                    output = self._loss_for_batch(batch)
                loss = objective_loss(output)
                if loss.numel() != 1 or not bool(torch.isfinite(loss).item()):
                    raise FloatingPointError("training loss must be one finite scalar")
                loss.backward()  # type: ignore[no-untyped-call]
                parameters_with_grad = [
                    parameter for parameter in parameters if parameter.grad is not None
                ]
                diagnostic_step = (
                    epoch * len(self._batches) + batch_index + 1
                ) % self._config.diagnostics.interval == 0
                diagnostics: dict[str, float | Tensor] = {}
                if diagnostic_step and isinstance(output, TrainingBatchOutput):
                    diagnostics.update(output.diagnostic_metrics())
                snapshot: list[Tensor] | None = None
                parameter_norm_before = 0.0
                if diagnostic_step and self._config.diagnostics.parameter_update_norms:
                    snapshot = [parameter.detach().clone() for parameter in parameters]
                    parameter_norm_before = _l2_norm(parameters)
                if diagnostic_step and self._config.diagnostics.component_gradient_norms:
                    diagnostics.update(
                        _component_gradient_norms(self._model, self._task.gradient_group)
                    )
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    parameters_with_grad,
                    self._config.grad_clip,
                    error_if_nonfinite=True,
                )
                grad_norm_value = float(grad_norm.item())
                clip_factor = min(
                    self._config.grad_clip / (grad_norm_value + 1e-6),
                    1.0,
                )
                clipped = grad_norm_value > self._config.grad_clip
                if diagnostic_step:
                    diagnostics |= {
                        "gradient/norm_before_clip": grad_norm_value,
                        "gradient/norm_after_clip": _l2_norm(
                            [cast(Tensor, parameter.grad) for parameter in parameters_with_grad]
                        ),
                        "gradient/clip_factor": clip_factor,
                        "gradient/clipped": float(clipped),
                    }
                optimizer.step()
                if snapshot is not None:
                    updates = [
                        parameter.detach() - previous
                        for parameter, previous in zip(parameters, snapshot, strict=True)
                    ]
                    update_norm = _l2_norm(updates)
                    parameter_norm = _l2_norm(parameters)
                    diagnostics |= {
                        "parameters/norm": parameter_norm,
                        "parameters/norm_before_update": parameter_norm_before,
                        "parameters/update_norm": update_norm,
                        "parameters/update_to_parameter_ratio": (
                            update_norm / parameter_norm_before
                            if parameter_norm_before > 0.0
                            else math.nan
                        ),
                    }
                learning_rates = [float(group["lr"]) for group in optimizer.param_groups]
                diagnostics["optimizer/learning_rate"] = learning_rates[0]
                diagnostics["batch/learning_rate"] = learning_rates[0]
                if len(learning_rates) > 1:
                    diagnostics.update(
                        {
                            f"optimizer/groups/{index}/learning_rate": learning_rate
                            for index, learning_rate in enumerate(learning_rates)
                        }
                    )
                batch_info = metadata[batch_index] if metadata else None
                if batch_info is not None:
                    diagnostics.update(batch_info.metrics())
                epoch_loss += float(loss.item())
                if self._tracker is not None:
                    self._tracker.on_batch_end(
                        loss=float(loss.item()),
                        grad_norm=grad_norm_value,
                        clipped=clipped,
                        observations=(batch_info.observations if batch_info is not None else None),
                        diagnostics=(_scalar_metrics(diagnostics) if diagnostic_step else None),
                    )

            metrics = {"loss": epoch_loss / max(len(self._batches), 1)}
            stop = False
            improved = False
            if self._validation_loss is not None:
                with self._config.precision_context(device):
                    val_loss = self._validation_loss()
                metrics["val_loss"] = val_loss
                if val_loss < best_val - self._config.early_stopping.min_delta:
                    best_val = val_loss
                    best_state = {
                        name: value.detach().clone()
                        for name, value in self._model.state_dict().items()
                    }
                    epochs_without_improvement = 0
                    improved = True
                else:
                    epochs_without_improvement += 1
                    stop = epochs_without_improvement >= self._config.early_stopping.patience

            if self._tracker is not None:
                self._tracker.on_epoch_end(
                    epoch,
                    metrics=metrics,
                    learning_rate=float(optimizer.param_groups[0]["lr"]),
                )

            if self._checkpointer is not None and self._checkpoint_state is not None:
                checkpoint = self._checkpoint_state(
                    optimizer,
                    epoch,
                    best_val,
                    best_state,
                    epochs_without_improvement,
                )
                self._checkpointer.save_latest(checkpoint)
                if improved:
                    self._checkpointer.save_best(checkpoint)

            if stop:
                break

        if best_state is not None:
            self._model.load_state_dict(best_state)
