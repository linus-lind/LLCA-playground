from __future__ import annotations

from pathlib import Path
from typing import Any

import mlflow.pyfunc
import torch

from llca.data.modules.masked_panel import MaskedPanels
from llca.models.estimators.estimator import Estimator


class Pyfunc[EstimatorT: Estimator](mlflow.pyfunc.PythonModel):  # type: ignore[misc]
    """Expose any pipeline ``Estimator`` through MLflow's Python model protocol.

    MLflow serializes this adapter separately from the estimator bundle. The estimator is
    therefore restored during ``load_context`` on the best available execution device,
    after which ``predict`` delegates to the typed pipeline output contract.
    """

    def __init__(self, estimator_cls: type[EstimatorT], bundle_artifact: str) -> None:
        super().__init__()
        self._estimator_cls = estimator_cls
        self._estimator: EstimatorT
        self._bundle_artifact = bundle_artifact

    def load_context(
        self,
        context: mlflow.pyfunc.PythonModelContext,
    ) -> None:
        """Load the bundled estimator artifact on CUDA when available, otherwise CPU."""
        bundle = Path(context.artifacts[self._bundle_artifact])
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._estimator = self._estimator_cls.load(bundle, device)

    @property
    def estimator(self) -> EstimatorT:
        return self._estimator

    def predict(
        self,
        context: mlflow.pyfunc.PythonModelContext,
        model_input: Any,
        params: dict[str, Any] | None = None,
    ) -> Any:
        """Return native prediction values for the supplied ``MaskedPanels`` input."""
        panels: MaskedPanels = model_input
        return self._estimator.predict(panels).values
