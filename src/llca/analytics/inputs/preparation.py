"""Reconstruct model panels and shared factor inputs from validated Hydra settings."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from llca.analytics.inputs.risk_free import align_risk_free
from llca.analytics.modules.analytics_config import RiskFreeReference
from llca.analytics.modules.factor_settings import FactorSettings, FactorSources
from llca.data.index_spec import time_level
from llca.data.modules.masked_panel import MaskedPanels
from llca.mappers.analytics import (
    analytics_data_requirements,
    build_factor_settings,
    build_risk_free_reference,
)
from llca.mappers.model.mapper import model_capabilities
from llca.models.estimators.prediction import PredictionOutput
from llca.pipeline.preparation import (
    PreparedAnalysisData,
    prepare_analysis_data,
    prepare_model_data,
)
from llca.splitting.slice_by_date import slice_by_date


@dataclass(frozen=True, slots=True)
class PreparedFactorInputs:
    """One shared pipeline result and every factor input extracted from it."""

    prepared: PreparedAnalysisData
    risk_free: pd.Series
    sources: FactorSources | None


def build_evaluation_panels(
    cfg: DictConfig,
    data_manifest: Mapping[str, Any],
) -> MaskedPanels:
    """Reproduce one model's training input panels from its config and data manifest.

    Resolves the model's data requirements and re-runs the shared preparation pipeline so the
    feature-engineered, membership-masked panels match those seen in training. Raises
    ``TypeError`` if the estimator does not use a mapping-based data view.
    """
    capabilities = model_capabilities(str(cfg.model.name))
    requirements = capabilities.resolve_data(cfg.model)
    prepared = prepare_model_data(
        cfg,
        requirements,
        data_view=capabilities.data_view,
        data_manifest=data_manifest,
    )
    if not isinstance(prepared.data, dict):
        raise TypeError("analytics currently requires a mapping-based estimator data view")
    return prepared.data


def test_window_with_history(
    panels: MaskedPanels,
    primary_dataset: str,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
    lookback: int,
) -> MaskedPanels:
    """Slice the panels to the test window extended backward by ``lookback`` dates.

    The extra leading dates give the first test observations the past context their input
    sequences need; they are dropped again from the reported predictions downstream. Raises
    ``ValueError`` if ``test_start`` or ``test_end`` is not on the primary dataset's calendar.
    """
    calendar = panels[primary_dataset].values
    dates = pd.DatetimeIndex(calendar.index.get_level_values(time_level(calendar)))
    ordered_dates = dates.unique().sort_values()
    start_position = int(ordered_dates.searchsorted(test_start))
    if start_position >= len(ordered_dates) or ordered_dates[start_position] != test_start:
        raise ValueError(f"test start {test_start.date()} is not present in the primary calendar")
    if test_end not in ordered_dates:
        raise ValueError(f"test end {test_end.date()} is not present in the primary calendar")

    history_start = ordered_dates[max(0, start_position - lookback)]
    return slice_by_date(panels, history_start, test_end)


def restrict_to_test_period(
    predictions: PredictionOutput,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
) -> PredictionOutput:
    """Drop history-only predictions and return the test window in chronological order.

    Keeps predictions dated within ``test_start``-``test_end`` and sorts them by date. Raises
    ``ValueError`` if nothing falls inside the window.
    """
    dates = pd.DatetimeIndex(predictions.index.get_level_values(time_level(predictions.values)))
    keep = (dates >= test_start) & (dates <= test_end)
    selected = predictions.select(keep)
    if selected.values.empty:
        raise ValueError(
            f"model produced no predictions in test period {test_start.date()} to {test_end.date()}"
        )
    order = np.argsort(selected.values.index)
    return selected.select(order)


def _native_frame(prepared: PreparedAnalysisData, dataset: str) -> pd.DataFrame:
    """Fetch one prepared date-indexed feature panel, sorted by date.

    Raises ``ValueError`` if the dataset is absent or is not a date-only (context) panel.
    """
    if dataset not in prepared.feature_panels:
        raise ValueError(f"factor source '{dataset}' is absent from prepared feature panels")
    frame = prepared.feature_panels[dataset]
    if frame.index.nlevels != 1:
        raise ValueError(f"factor source '{dataset}' must be a date-only (context) dataset")
    return frame.sort_index()


def extract_risk_free(
    reference: RiskFreeReference,
    prepared: PreparedAnalysisData,
) -> pd.Series:
    """Pull the risk-free rate series named by ``reference`` from prepared features.

    Returns the column, dropping missing dates and naming it ``risk_free``. Raises
    ``ValueError`` if the column is absent from its dataset.
    """
    frame = _native_frame(prepared, reference.dataset)
    if reference.column not in frame.columns:
        raise ValueError(f"risk-free column '{reference.column}' is absent from its dataset")
    series = frame[reference.column].dropna().copy()
    series.name = "risk_free"
    return series


def _extract_spanning_benchmark(
    settings: FactorSettings,
    risk_free: pd.Series,
    prepared: PreparedAnalysisData,
) -> pd.DataFrame:
    """Load the spanning benchmark portfolios as decimal excess returns.

    The stored portfolio returns are rescaled by ``settings.spanning_scale`` (percent to
    decimal) and, when ``settings.spanning_excess`` is set, reduced by the causally aligned
    daily risk-free rate so they share the excess-return convention of the spanning test's
    left-hand side. Raises ``ValueError`` if a configured portfolio column is missing.
    """
    frame = _native_frame(prepared, settings.spanning_dataset)
    missing = [column for column in settings.spanning_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"spanning benchmark portfolios absent from their dataset: {missing}")
    benchmark = frame[list(settings.spanning_columns)].astype(float) * settings.spanning_scale
    if settings.spanning_excess:
        aligned_risk_free = align_risk_free(risk_free, benchmark.index)
        benchmark = benchmark.sub(aligned_risk_free, axis=0)
    return benchmark.dropna()


def extract_factor_sources(
    settings: FactorSettings,
    risk_free: pd.Series,
    prepared: PreparedAnalysisData,
) -> FactorSources:
    """Assemble the factor sources — FF6, timing instruments, IPCA config — from features.

    Reads the FF6 factor columns and the timing-instrument columns named in ``settings`` from
    their prepared datasets and packages them with the risk-free series and the timing/IPCA
    options into a ``FactorSources``. Raises ``ValueError`` if any configured column is missing.
    """
    ff_frame = _native_frame(prepared, settings.factors_dataset)
    missing = [column for column in settings.ff6_columns if column not in ff_frame.columns]
    if missing:
        raise ValueError(f"factor columns absent from their dataset: {missing}")
    ff6 = ff_frame[list(settings.ff6_columns)].dropna()

    spanning_benchmark = _extract_spanning_benchmark(settings, risk_free, prepared)

    instrument_frame = _native_frame(prepared, settings.instruments_dataset)
    missing_instruments = [
        column for column in settings.instrument_columns if column not in instrument_frame.columns
    ]
    if missing_instruments:
        raise ValueError(f"timing instruments absent from their dataset: {missing_instruments}")
    timing_instruments = instrument_frame[list(settings.instrument_columns)].copy()

    return FactorSources(
        risk_free=risk_free,
        ff6=ff6,
        market_column=settings.market_column,
        spanning_benchmark=spanning_benchmark,
        timing_instruments=timing_instruments,
        ipca=settings.ipca,
        timing_instrument_lag=settings.timing_instrument_lag,
        market_squared=settings.market_squared,
        conditional_alpha=settings.conditional_alpha,
        rolling_beta_window=settings.rolling_beta_window,
    )


def prepare_factor_inputs(cfg: DictConfig) -> PreparedFactorInputs:
    """Run the shared preparation once and extract every factor input from it.

    Resolves the analytics data requirements, runs the model-independent pipeline, and pulls out
    the risk-free series and — when factor analysis is configured — the factor sources, bundling
    them with the prepared data. The ``analytics`` config is trusted as already validated by the
    compose-time gate.
    """
    analytics_cfg = cast(DictConfig, cfg.analytics)
    requirements, data_view = analytics_data_requirements(analytics_cfg)
    reference = build_risk_free_reference(analytics_cfg)
    settings = build_factor_settings(analytics_cfg)
    prepared = prepare_analysis_data(cfg, requirements, data_view=data_view)
    risk_free = extract_risk_free(reference, prepared)
    sources = (
        extract_factor_sources(settings, risk_free, prepared) if settings is not None else None
    )
    return PreparedFactorInputs(prepared=prepared, risk_free=risk_free, sources=sources)
