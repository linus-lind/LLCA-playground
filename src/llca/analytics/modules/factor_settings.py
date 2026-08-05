from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True, slots=True)
class IpcaSettings:
    """Resolved model-independent IPCA data, return, and instrument-coverage policy.

    The instrument panel is the complete feature output of ``characteristics_dataset``;
    per-date cross-sectional ranking with neutral imputation of residual missing values is
    intrinsic to IPCA and applied unconditionally by the estimator. Staleness of the raw
    fundamentals is handled upstream by the shared preprocessing and masking pipeline;
    ``default_max_age``/``column_max_age`` only cap how old a carried instrument may be, read
    from the aligned panel's observation age. ``aligning_dataset`` names the entity-indexed
    dataset whose masked ``(date, entity)`` grid the returns and instruments are read from;
    it is the shared factor-analysis alignment target.
    """

    enabled: bool
    n_factors: int
    aligning_dataset: str
    returns_dataset: str
    return_column: str
    return_type: str
    realization_lag: int
    excess_returns: bool
    characteristics_dataset: str
    min_characteristic_coverage: float
    default_max_age: int | None
    column_max_age: dict[str, int]


@dataclass(frozen=True, slots=True)
class FactorSettings:
    """Resolved, data-independent factor-model configuration.

    ``ff6`` name the Hydra feature outputs for daily Fama-French factor returns and the
    market alias used by the timing model. Timing instruments are arbitrary configured
    macro feature outputs (levels, changes, or other registered transforms). Realization
    alignment belongs to feature configuration; the timing information lag is an explicit
    regression setting. Data extraction into :class:`FactorSources` happens separately so
    this object stays a pure translation of validated Hydra settings.

    ``spanning_*`` name the benchmark-portfolio feature outputs regressed on in the
    mean-variance spanning test. ``spanning_scale`` rescales their stored units (percent to
    decimal) and ``spanning_excess`` requests risk-free subtraction, matching the excess-return
    left-hand side.
    """

    factors_dataset: str
    ff6_columns: tuple[str, ...]
    market_column: str
    spanning_dataset: str
    spanning_columns: tuple[str, ...]
    spanning_scale: float
    spanning_excess: bool
    instruments_dataset: str
    instrument_columns: tuple[str, ...]
    timing_instrument_lag: int
    market_squared: bool
    conditional_alpha: bool
    rolling_beta_window: int
    ipca: IpcaSettings


@dataclass(frozen=True, slots=True)
class FactorSources:
    """Loaded factor-model inputs paired with the resolved analysis configuration.

    The frames are extracted from one shared, model-independent Hydra preparation; the
    scalar fields are copied from the resolved :class:`FactorSettings` so downstream
    reporting consumes a single typed object without re-reading configuration.
    """

    risk_free: pd.Series
    ff6: pd.DataFrame
    market_column: str
    spanning_benchmark: pd.DataFrame
    timing_instruments: pd.DataFrame
    ipca: IpcaSettings
    timing_instrument_lag: int
    market_squared: bool
    conditional_alpha: bool
    rolling_beta_window: int
