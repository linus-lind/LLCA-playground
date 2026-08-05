from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import pandas as pd
from torch import Tensor

from llca.models.estimators.prediction import PredictionOutput

type ObjectiveLayout = Literal["panel", "rows"]
type ObjectiveTensorAdapter = Callable[[PredictionOutput, pd.Series], tuple[Tensor, Tensor, Tensor]]


@dataclass(frozen=True, slots=True)
class EvaluationSpec:
    """Declare the panel roles required to evaluate an estimator consistently.

    ``primary_dataset`` supplies the calendar and causal history. The supervision binding
    identifies the aligned target independently of a concrete model architecture.
    ``objective_layout`` selects a built-in conversion to either dense date-by-entity
    tensors or independent observation rows. Architectures with a different objective
    contract can provide a pure ``objective_adapter`` without changing analytics code.
    """

    primary_dataset: str
    supervision_dataset: str
    supervision_column: str
    objective_layout: ObjectiveLayout = "panel"
    objective_adapter: ObjectiveTensorAdapter | None = None
