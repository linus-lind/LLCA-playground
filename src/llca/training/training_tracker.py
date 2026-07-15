import time
from collections.abc import Mapping

import pandas as pd
from mlflow.entities import Metric
from mlflow.tracking import MlflowClient

from llca.training.modules.training_diagnostics import BatchMetadata


class MlflowTrainingTracker:
    """Stream training progress and optimizer diagnostics to an existing MLflow run.

    Batch metrics share a global optimizer-step axis, including resumed sessions. Session-
    local counters drive throughput and ETA, while cumulative clipping fraction reflects
    the current process session. Static batch boundaries and dimensions are stored once as
    a manifest table instead of repeated string metrics.
    """

    def __init__(self, run_id: str, *, batch_log_interval: int = 1) -> None:
        if batch_log_interval < 1:
            raise ValueError(f"batch_log_interval must be >= 1, got {batch_log_interval}")
        self._client = MlflowClient()
        self._run_id = run_id
        self._batch_log_interval = batch_log_interval
        self._total_epochs = 0
        self._steps_per_epoch: int | None = None
        self._global_step = 0
        self._session_steps = 0
        self._samples_seen = 0
        self._clipped_steps = 0
        self._start_time = 0.0

    def begin(
        self,
        total_epochs: int,
        steps_per_epoch: int | None = None,
        *,
        start_step: int = 0,
        batch_manifest: list[BatchMetadata] | None = None,
    ) -> None:
        """Initialize progress counters and optionally log the recurring batch manifest."""
        self._total_epochs = total_epochs
        self._steps_per_epoch = steps_per_epoch
        self._global_step = start_step
        self._session_steps = 0
        self._samples_seen = 0
        self._clipped_steps = 0
        self._start_time = time.perf_counter()
        if batch_manifest:
            self._client.log_table(
                self._run_id,
                pd.DataFrame([metadata.record() for metadata in batch_manifest]),
                "training/batch_manifest.json",
            )

    def on_batch_end(
        self,
        loss: float,
        *,
        grad_norm: float | None = None,
        clipped: bool = False,
        observations: int | None = None,
        diagnostics: Mapping[str, float] | None = None,
    ) -> None:
        """Advance one optimizer step and log diagnostics at the configured interval.

        Progress counters update on every call even when metric emission is skipped. This
        keeps throughput, sample counts, clipping frequency, and resume-aware global steps
        consistent with the actual optimization history.
        """
        self._global_step += 1
        self._session_steps += 1
        self._clipped_steps += int(clipped)
        if observations is not None:
            self._samples_seen += observations
        if self._global_step % self._batch_log_interval != 0:
            return

        elapsed = self._elapsed()
        payload: dict[str, float] = {
            "batch/loss": float(loss),
            "throughput/steps_per_sec": self._session_steps / elapsed,
        }
        if grad_norm is not None:
            payload["batch/grad_norm"] = float(grad_norm)
        payload["gradient/clipped_step_fraction"] = self._clipped_steps / self._session_steps
        if diagnostics is not None:
            payload.update(diagnostics)
        if self._samples_seen:
            payload["throughput/samples_per_sec"] = self._samples_seen / elapsed
        payload.update(self._progress_metrics(elapsed))
        self._log(payload)

    def on_epoch_end(
        self, epoch: int, *, metrics: Mapping[str, float], learning_rate: float | None = None
    ) -> None:
        """Log epoch aggregates on the same global step used by batch diagnostics."""
        payload: dict[str, float] = {f"epoch/{key}": float(value) for key, value in metrics.items()}
        payload["progress/epoch"] = float(epoch + 1)
        if learning_rate is not None:
            payload["epoch/learning_rate"] = float(learning_rate)
        payload.update(self._progress_metrics(self._elapsed()))
        self._log(payload)

    def _elapsed(self) -> float:
        return max(time.perf_counter() - self._start_time, 1e-9)

    def _progress_metrics(self, elapsed: float) -> dict[str, float]:
        """Estimate completion and remaining wall time from session-average step duration."""
        if not self._steps_per_epoch or self._total_epochs <= 0 or self._global_step <= 0:
            return {"time/elapsed_sec": elapsed}
        total_steps = self._steps_per_epoch * self._total_epochs
        fraction = min(self._global_step / total_steps, 1.0)
        avg_step_time = elapsed / max(self._session_steps, 1)
        remaining = max(total_steps - self._global_step, 0)
        return {
            "progress/fraction": fraction,
            "progress/percent": fraction * 100.0,
            "time/elapsed_sec": elapsed,
            "time/remaining_sec": remaining * avg_step_time,
        }

    def _log(self, metrics: dict[str, float]) -> None:
        timestamp = int(time.time() * 1000)
        entries = [
            Metric(key, value, timestamp, self._global_step) for key, value in metrics.items()
        ]
        self._client.log_batch(self._run_id, metrics=entries)
