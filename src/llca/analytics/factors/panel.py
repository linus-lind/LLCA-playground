"""Select the IPCA estimation sample from the shared, membership-masked panels.

The firm-characteristic instruments and the asset returns are prepared once by the shared
Hydra pipeline: preprocessing, feature creation, and the membership-masked backward-as-of
alignment in :mod:`llca.data.masking`. This module therefore only restricts the common
evaluation window, converts the return convention, and reads the already carried-forward
characteristic values together with their observation age (in trading sessions) from the
aligned panel. It never re-implements alignment or staleness logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pandas as pd

from llca.analytics.inputs.risk_free import align_risk_free
from llca.analytics.modules.factor_settings import IpcaSettings
from llca.data.index_spec import entity_level, time_level
from llca.data.modules.masked_panel import MaskedPanels
from llca.pipeline.preparation import PreparedAnalysisData

_COVERAGE_QUANTILES = (0.0, 0.05, 0.5, 0.95, 1.0)


@dataclass(frozen=True, slots=True)
class IpcaPanelData:
    """Aligned IPCA response, instruments, and instrument age from the shared panel."""

    returns: pd.Series
    characteristics: pd.DataFrame
    characteristic_ages: pd.DataFrame
    diagnostics: dict[str, Any]


def _window_index(index: pd.Index, start: pd.Timestamp, end: pd.Timestamp) -> pd.Index:
    dates = pd.DatetimeIndex(index.get_level_values(str(index.names[0])))
    return index[(dates >= start) & (dates <= end)]


def _return_series(
    panels: MaskedPanels,
    settings: IpcaSettings,
    risk_free: pd.Series,
    grid: pd.Index,
) -> tuple[pd.Series, dict[str, int]]:
    panel = panels[settings.returns_dataset]
    column = settings.return_column
    if column not in panel.values:
        raise ValueError(
            f"configured IPCA return column '{column}' is absent from feature dataset "
            f"'{settings.returns_dataset}'"
        )
    values = panel.values[column].reindex(grid).astype(float)
    observed = panel.observed[column].reindex(grid).fillna(False).astype(bool)
    finite = pd.Series(np.isfinite(values.to_numpy(dtype=float)), index=grid)
    valid = observed & finite
    response = values.where(valid)
    if settings.return_type == "log":
        response = pd.Series(
            np.expm1(response.to_numpy(dtype=float)),
            index=response.index,
            name=response.name,
        )

    if settings.excess_returns:
        time = time_level(response)
        dates = pd.DatetimeIndex(response.index.get_level_values(time))
        aligned_rf = align_risk_free(risk_free, dates)
        response = response - aligned_rf.to_numpy(dtype=float)
    response = response.where(np.isfinite(response))
    response.name = "ipca_excess_return" if settings.excess_returns else "ipca_return"
    return response, {
        "grid_rows": int(len(grid)),
        "observed_finite_return_rows": int(response.notna().sum()),
        "rejected_carried_or_invalid_return_rows": int(len(grid) - response.notna().sum()),
    }


def _characteristic_frame(
    panels: MaskedPanels,
    settings: IpcaSettings,
    grid: pd.Index,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Pull the characteristic values and their observation ages over the evaluation grid.

    Reads the configured characteristics dataset, restricts it to ``grid``, and returns the
    value frame, the matching age frame, and coverage diagnostics (per-column non-missing
    fractions and quantiles of mean daily row coverage). Raises if the dataset has no columns.
    """
    panel = panels[settings.characteristics_dataset]
    values = panel.values.reindex(grid).astype(float)
    ages = panel.age.reindex(grid)
    if values.shape[1] == 0:
        raise ValueError(
            f"IPCA characteristics dataset '{settings.characteristics_dataset}' produced "
            "no feature columns"
        )
    time = time_level(values)
    row_coverage = values.notna().mean(axis=1)
    time_coverage = row_coverage.groupby(level=time).mean()
    diagnostics: dict[str, Any] = {
        "configured_characteristics": [str(column) for column in values.columns],
        "characteristic_non_missing_fraction": {
            str(column): float(values[column].notna().mean()) for column in values.columns
        },
        "mean_daily_row_coverage_quantiles": {
            str(quantile): float(time_coverage.quantile(quantile))
            for quantile in _COVERAGE_QUANTILES
        },
    }
    return values, ages, diagnostics


def prepare_ipca_panel(
    settings: IpcaSettings,
    risk_free: pd.Series,
    prepared: PreparedAnalysisData,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> IpcaPanelData:
    """Build the IPCA estimation sample over the common evaluation window.

    Validates that the aligning, returns, and characteristics datasets exist in the prepared
    aligned panel, derives the date-entity grid from the aligning dataset within
    ``start``-``end``, and reads the return series and characteristic values/ages onto it.
    Returns them bundled with selection diagnostics. Raises if a dataset is missing, the panel
    is not the aligned view, the grid is empty, or the aligning dataset is not entity-indexed.
    """
    if not isinstance(prepared.data, dict):
        raise TypeError("IPCA requires the aligned_panel data view")
    panels = cast(MaskedPanels, prepared.data)
    for name in (
        settings.aligning_dataset,
        settings.returns_dataset,
        settings.characteristics_dataset,
    ):
        if name not in panels:
            raise ValueError(f"IPCA dataset '{name}' is unavailable in the aligned panel")

    aligning = panels[settings.aligning_dataset]
    grid = _window_index(aligning.values.index, start, end)
    if grid.empty:
        raise ValueError("IPCA aligning grid is empty in the common evaluation window")
    aligning_entity = entity_level(aligning.values)
    if aligning_entity is None:
        raise ValueError("IPCA aligning dataset must be entity-indexed")

    returns, return_diagnostics = _return_series(panels, settings, risk_free, grid)
    characteristics, ages, characteristic_diagnostics = _characteristic_frame(
        panels, settings, grid
    )
    diagnostics = {
        "evaluation_start": start.isoformat(),
        "evaluation_end": end.isoformat(),
        "reference_entities": int(grid.get_level_values(aligning_entity).nunique()),
        **return_diagnostics,
        **characteristic_diagnostics,
    }
    return IpcaPanelData(
        returns=returns,
        characteristics=characteristics,
        characteristic_ages=ages,
        diagnostics=diagnostics,
    )
