from __future__ import annotations

from pathlib import Path
from typing import Any

import mlflow.pyfunc

from llca.models.estimators.estimator import Estimator


class Pyfunc[EstimatorT: Estimator[Any]](mlflow.pyfunc.PythonModel):  # type: ignore[misc]
    """Expose any pipeline ``Estimator`` through MLflow's Python model protocol.

    MLflow serializes this adapter separately from the estimator bundle. The estimator is
    therefore restored during ``load_context`` on the best available execution device,
    after which ``predict`` delegates to the typed pipeline output contract.
    """

    # Estimators consume LLCA's compound data contracts (for example MaskedPanels), not
    # MLflow's row-oriented ``list[...]`` schema.  Asking MLflow to infer and enforce a
    # schema from ``Any`` both emits a warning and would misrepresent that contract.
    _skip_type_hint_validation = True

    def __init__(self, estimator_cls: type[EstimatorT], bundle_artifact: str) -> None:
        super().__init__()
        self._estimator_cls = estimator_cls
        self._estimator: EstimatorT
        self._bundle_artifact = bundle_artifact

    def load_context(
        self,
        context: mlflow.pyfunc.PythonModelContext,
    ) -> None:
        """Load the bundle through its backend-neutral estimator contract."""
        bundle = Path(context.artifacts[self._bundle_artifact])
        self._estimator = self._estimator_cls.load(bundle, "auto")

    @property
    def estimator(self) -> EstimatorT:
        return self._estimator

    def predict(
        self,
        context: mlflow.pyfunc.PythonModelContext,
        model_input: Any,
        params: dict[str, Any] | None = None,
    ) -> Any:
        """Return native prediction values for the estimator's registered data contract."""
        return self._estimator.predict(model_input).values
