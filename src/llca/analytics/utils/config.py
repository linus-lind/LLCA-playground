from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd

from llca.core.returns import ReturnType

type TableFormat = Literal["csv", "tex", "pdf", "png"]


@dataclass(frozen=True, slots=True)
class RegisteredModelConfig:
    """One exact registry model version and its human-readable comparison label."""

    name: str
    version: int
    label: str


@dataclass(frozen=True, slots=True)
class ModelEvaluationConfig:
    """Define models and shared conventions for a comparable analytical report.

    All configured models use one item universe, annualization basis, return convention,
    risk thresholds, rolling window, signal horizons, and portfolio construction settings.
    This prevents plots and tables from comparing metrics computed under different rules.
    """

    models: tuple[RegisteredModelConfig, ...]
    device: str
    annualization_periods: int
    return_type: ReturnType
    signal_buckets: int
    probability_bins: int
    classification_threshold: float
    target_threshold: float
    risk_free_rate: float
    minimum_acceptable_return: float
    var_levels: tuple[float, ...]
    rolling_window: int
    signal_decay_periods: tuple[int, ...]
    active_weight_threshold: float
    include_initial_trade: bool
    show_plots: bool
    evaluation_end: pd.Timestamp | None
    output_dir: Path = Path("reports/analytics")
    table_formats: tuple[TableFormat, ...] = ("csv", "tex", "pdf", "png")
    table_dpi: int = 300
