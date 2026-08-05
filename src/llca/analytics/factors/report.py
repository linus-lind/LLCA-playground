"""Estimate the report's configured IPCA factors from the shared point-in-time panel.

Enabling IPCA is a required report contract, so panel preparation and estimation errors
deliberately propagate: a configuration that intentionally does not require IPCA must set
``factor_analysis.ipca.enabled: false``.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from llca.analytics.factors import estimate_ipca_factors
from llca.analytics.factors.panel import prepare_ipca_panel
from llca.analytics.modules.factor_settings import FactorSources
from llca.pipeline.preparation import PreparedAnalysisData


def estimate_ipca_for_report(
    sources: FactorSources,
    risk_free: pd.Series,
    prepared: PreparedAnalysisData,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> tuple[pd.DataFrame | None, dict[str, Any] | None]:
    """Prepare the panel and estimate the report's IPCA factors, or skip when disabled.

    Returns ``(None, None)`` when IPCA is turned off. Otherwise it builds the estimation sample
    over ``start``-``end``, runs the estimator with the configured factor count, coverage
    threshold, and age caps, and returns the factor frame together with the merged panel and
    estimation diagnostics. Errors propagate, since enabling IPCA is a hard report requirement.
    """
    if not sources.ipca.enabled:
        return None, None
    panel = prepare_ipca_panel(sources.ipca, risk_free, prepared, start=start, end=end)
    diagnostics = dict(panel.diagnostics)
    maximum_age = {
        "default": sources.ipca.default_max_age,
        "columns": sources.ipca.column_max_age,
    }
    factors, estimation_diagnostics = estimate_ipca_factors(
        panel.returns,
        panel.characteristics,
        n_factors=sources.ipca.n_factors,
        min_characteristic_coverage=sources.ipca.min_characteristic_coverage,
        characteristic_ages=panel.characteristic_ages,
        feature_max_age=maximum_age,
        return_diagnostics=True,
    )
    diagnostics["estimation"] = estimation_diagnostics.to_dict()
    return factors, diagnostics
