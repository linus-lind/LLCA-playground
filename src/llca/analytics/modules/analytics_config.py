from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd

from llca.core.returns import ReturnType

type TableFormat = Literal["csv", "tex", "pdf", "png"]
type PlotFormat = Literal["png", "pdf", "svg"]


@dataclass(frozen=True, slots=True)
class RegisteredModelConfig:
    """One exact registry model version and its human-readable comparison label."""

    name: str
    version: int
    label: str


@dataclass(frozen=True, slots=True)
class RiskFreeReference:
    """Dataset and feature-output column of the prepared daily risk-free return."""

    dataset: str
    column: str


@dataclass(frozen=True, slots=True)
class ModelEvaluationConfig:
    """Define models and shared conventions for a comparable analytical report.

    Every configured model is scored on its own native item universe (e.g. a single-asset
    model keeps its one entity while a cross-sectional model keeps its full universe), but
    all of them share one annualization basis, return convention, risk thresholds, rolling
    window, signal horizons, and portfolio construction settings. This prevents plots and
    tables from comparing metrics computed under different rules.
    """

    models: tuple[RegisteredModelConfig, ...]
    device: str
    annualization_periods: int
    return_type: ReturnType
    return_realization_lag: int
    signal_buckets: int
    target_threshold: float
    minimum_acceptable_return: float
    var_levels: tuple[float, ...]
    autocorrelation_lags: tuple[int, ...]
    worst_rolling_windows: tuple[int, ...]
    rolling_window: int
    signal_decay_periods: tuple[int, ...]
    active_weight_threshold: float
    include_initial_trade: bool
    show_plots: bool
    evaluation_end: pd.Timestamp | None
    hac_lag: int | None = None
    bootstrap_resamples: int = 2000
    bootstrap_block_length: float = 10.0
    bootstrap_seed: int = 0
    test_significance_level: float = 0.05
    multiple_testing_correction: str = "holm"
    output_dir: Path = Path("reports/analytics")
    table_formats: tuple[TableFormat, ...] = ("csv", "tex", "pdf", "png")
    table_dpi: int = 300
    plot_formats: tuple[PlotFormat, ...] = ("png",)
    plot_dpi: int = 200
