"""Map validated Hydra factor-analysis settings to immutable runtime objects."""

from __future__ import annotations

from typing import cast

from omegaconf import DictConfig

from llca.analytics.modules.factor_settings import FactorSettings, IpcaSettings


def build_ipca_settings(cfg: DictConfig, aligning_dataset: str) -> IpcaSettings:
    """Translate one validated ``analytics.factor_analysis.ipca`` block to runtime settings.

    ``aligning_dataset`` is the shared ``analytics.factor_analysis.aligning_dataset`` grid the
    IPCA cross-section and returns are read from; it is not an IPCA-specific setting.
    """
    inputs = cast(DictConfig, cfg["inputs"])
    returns = cast(DictConfig, inputs["returns"])
    characteristics = cast(DictConfig, inputs["characteristics"])
    max_age = cfg.get("max_age")
    default_age = max_age.get("default") if isinstance(max_age, DictConfig) else None
    overrides = max_age.get("columns") if isinstance(max_age, DictConfig) else None
    return IpcaSettings(
        enabled=bool(cfg.get("enabled", True)),
        n_factors=int(cfg["n_factors"]),
        aligning_dataset=aligning_dataset,
        returns_dataset=str(returns["dataset"]),
        return_column=str(returns["column"]),
        return_type=str(returns.get("return_type", "simple")),
        realization_lag=int(returns.get("realization_lag", 0)),
        excess_returns=bool(returns.get("excess", True)),
        characteristics_dataset=str(characteristics["dataset"]),
        min_characteristic_coverage=float(cfg.get("min_characteristic_coverage", 0.5)),
        default_max_age=int(default_age) if default_age is not None else None,
        column_max_age=(
            {str(column): int(age) for column, age in overrides.items()}
            if isinstance(overrides, DictConfig)
            else {}
        ),
    )


def build_factor_settings(cfg: DictConfig) -> FactorSettings | None:
    """Translate ``analytics.factor_analysis`` to runtime settings, or ``None`` when disabled."""
    settings = cfg.get("factor_analysis")
    if not isinstance(settings, DictConfig) or not bool(settings.get("enabled", True)):
        return None

    factors = cast(DictConfig, settings["factors"])
    ff6_columns = tuple(str(column) for column in factors["ff6"])
    market_column = str(factors["market"])
    if market_column not in ff6_columns:
        raise ValueError(f"factor_analysis.factors.market '{market_column}' must be one of ff6")

    spanning = cast(DictConfig, settings["spanning"])
    timing = cast(DictConfig, settings["timing"])
    instruments = cast(DictConfig, timing["instruments"])
    return FactorSettings(
        factors_dataset=str(factors["dataset"]),
        ff6_columns=ff6_columns,
        market_column=market_column,
        spanning_dataset=str(spanning["dataset"]),
        spanning_columns=tuple(str(column) for column in spanning["portfolios"]),
        spanning_scale=float(spanning.get("scale", 1.0)),
        spanning_excess=bool(spanning.get("excess", True)),
        instruments_dataset=str(instruments["dataset"]),
        instrument_columns=tuple(str(column) for column in instruments["columns"]),
        timing_instrument_lag=int(timing.get("instrument_lag", 1)),
        market_squared=bool(timing.get("market_squared", True)),
        conditional_alpha=bool(timing.get("conditional_alpha", True)),
        rolling_beta_window=int(settings["rolling_beta_window"]),
        ipca=build_ipca_settings(
            cast(DictConfig, settings["ipca"]), str(settings["aligning_dataset"])
        ),
    )
