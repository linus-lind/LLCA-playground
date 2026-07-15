from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

type PredictionKind = Literal["portfolio", "regression", "binary", "multiclass"]
type PredictionValues = pd.Series | pd.DataFrame
PREDICTION_KINDS: tuple[PredictionKind, ...] = (
    "portfolio",
    "regression",
    "binary",
    "multiclass",
)


def validate_prediction_kind(value: object) -> PredictionKind:
    """Return one supported prediction kind or reject stale/unknown persisted values."""
    if not isinstance(value, str) or value not in PREDICTION_KINDS:
        raise ValueError(
            f"unsupported prediction kind {value!r}; expected one of {list(PREDICTION_KINDS)}"
        )
    return value


def prediction_kind_from_bundle(value: object) -> PredictionKind:
    """Canonicalize the two retired portfolio labels stored by registered FMG bundles."""
    if value in ("ranking", "allocation"):
        return "portfolio"
    return validate_prediction_kind(value)


@dataclass(frozen=True, slots=True)
class PredictionOutput:
    """Typed model output kept separate from downstream portfolio construction.

    `values` contains the model's native decision output: portfolio scores or weights, a
    numerical forecast, a binary decision score, or one score column per multiclass label.
    Classification models may additionally expose calibrated `probabilities`; binary
    probabilities use a Series for the positive class and multiclass probabilities use one
    column per class.
    """

    kind: PredictionKind
    values: PredictionValues
    probabilities: PredictionValues | None = None
    quantiles: pd.DataFrame | None = None

    def __post_init__(self) -> None:
        """Validate index alignment, finite values, and kind-specific output constraints."""
        validate_prediction_kind(self.kind)
        if self.values.empty:
            raise ValueError("prediction output must not be empty")
        if self.values.index.has_duplicates:
            raise ValueError("prediction output contains duplicate index rows")
        if not np.isfinite(self.values.to_numpy(dtype=float)).all():
            raise ValueError("prediction output contains non-finite values")
        if self.kind == "multiclass":
            if not isinstance(self.values, pd.DataFrame):
                raise ValueError("multiclass predictions require one score column per class")
            if len(self.values.columns) < 3:
                raise ValueError("multiclass predictions require at least three classes")
            if self.values.columns.has_duplicates:
                raise ValueError("multiclass prediction columns must be unique")
        elif not isinstance(self.values, pd.Series):
            raise ValueError(f"{self.kind} predictions require one scalar value per item")
        if self.probabilities is not None:
            if self.kind not in ("binary", "multiclass"):
                raise ValueError("only binary or multiclass outputs may expose probabilities")
            if not self.probabilities.index.equals(self.values.index):
                raise ValueError("classification probabilities must share the prediction index")
            if self.kind == "binary" and not isinstance(self.probabilities, pd.Series):
                raise ValueError("binary probabilities require one positive-class Series")
            if self.kind == "multiclass" and not isinstance(self.probabilities, pd.DataFrame):
                raise ValueError("multiclass probabilities require one column per class")
            if self.kind == "multiclass" and not self.values.columns.equals(
                self.probabilities.columns
            ):
                raise ValueError(
                    "multiclass scores and probabilities must have identical class columns"
                )
            probability_values = self.probabilities.to_numpy(dtype=float)
            if not np.isfinite(probability_values).all():
                raise ValueError("classification probabilities contain non-finite values")
            if ((probability_values < 0.0) | (probability_values > 1.0)).any():
                raise ValueError("classification probabilities must lie in [0, 1]")
            if self.kind == "multiclass":
                row_sums = probability_values.sum(axis=1)
                if not np.allclose(row_sums, 1.0, rtol=1e-6, atol=1e-8):
                    raise ValueError("multiclass probability rows must sum to one")
        if self.quantiles is not None:
            if self.kind != "regression":
                raise ValueError("only regression outputs may expose predictive quantiles")
            if not self.quantiles.index.equals(self.values.index):
                raise ValueError("predictive quantiles must share the prediction index")
            try:
                levels = np.asarray(self.quantiles.columns, dtype=float)
            except (TypeError, ValueError) as exc:
                raise ValueError("predictive quantile columns must be numeric levels") from exc
            if len(levels) < 2 or ((levels <= 0.0) | (levels >= 1.0)).any():
                raise ValueError("predictive quantiles require at least two levels in (0, 1)")
            if not np.all(np.diff(levels) > 0.0):
                raise ValueError("predictive quantile levels must be strictly increasing")
            quantile_values = self.quantiles.to_numpy(dtype=float)
            if not np.isfinite(quantile_values).all():
                raise ValueError("predictive quantiles contain non-finite values")
            if (np.diff(quantile_values, axis=1) < 0.0).any():
                raise ValueError("predictive quantile values must not cross")

    @property
    def index(self) -> pd.Index:
        return self.values.index

    def select(self, keep: np.ndarray) -> PredictionOutput:
        """Return the same prediction contract restricted by one positional mask."""
        probabilities = self.probabilities.iloc[keep] if self.probabilities is not None else None
        quantiles = self.quantiles.iloc[keep] if self.quantiles is not None else None
        return PredictionOutput(
            kind=self.kind,
            values=self.values.iloc[keep],
            probabilities=probabilities,
            quantiles=quantiles,
        )

    def reindex(self, index: pd.Index) -> PredictionOutput:
        """Align every component to an already validated common comparison index."""
        return PredictionOutput(
            kind=self.kind,
            values=self.values.reindex(index),
            probabilities=(
                self.probabilities.reindex(index) if self.probabilities is not None else None
            ),
            quantiles=self.quantiles.reindex(index) if self.quantiles is not None else None,
        )
