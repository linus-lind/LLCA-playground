from __future__ import annotations

from pathlib import Path
from typing import cast

import pandas as pd
from omegaconf import DictConfig

from llca.analytics.modules.analytics_config import (
    ModelEvaluationConfig,
    PlotFormat,
    RegisteredModelConfig,
    RiskFreeReference,
    TableFormat,
)
from llca.core.paths import PROJECT_ROOT
from llca.core.returns import ReturnType
from llca.mappers.analytics.factors import build_ipca_settings
from llca.pipeline.contracts import DataRequirements, DatasetRequirement, EntityScope


def build_analytics(cfg: DictConfig) -> ModelEvaluationConfig:
    """Map validated Hydra analytics settings to an immutable runtime configuration."""
    evaluation_end = cfg.get("evaluation_end")
    configured_output = Path(str(cfg.output_dir))
    output_dir = (
        configured_output if configured_output.is_absolute() else PROJECT_ROOT / configured_output
    )
    return ModelEvaluationConfig(
        models=tuple(
            RegisteredModelConfig(
                name=str(model.name),
                version=int(model.version),
                label=str(model.label),
            )
            for model in cfg.models
        ),
        device=str(cfg.device),
        annualization_periods=int(cfg.annualization_periods),
        return_type=cast(ReturnType, str(cfg.return_type)),
        return_realization_lag=int(cfg.return_realization_lag),
        signal_buckets=int(cfg.signal_buckets),
        target_threshold=float(cfg.target_threshold),
        minimum_acceptable_return=float(cfg.minimum_acceptable_return),
        var_levels=tuple(float(level) for level in cfg.var_levels),
        autocorrelation_lags=tuple(int(lag) for lag in cfg.autocorrelation_lags),
        worst_rolling_windows=tuple(int(window) for window in cfg.worst_rolling_windows),
        rolling_window=int(cfg.rolling_window),
        signal_decay_periods=tuple(int(period) for period in cfg.signal_decay_periods),
        active_weight_threshold=float(cfg.active_weight_threshold),
        include_initial_trade=bool(cfg.include_initial_trade),
        show_plots=bool(cfg.show_plots),
        evaluation_end=(pd.Timestamp(str(evaluation_end)) if evaluation_end is not None else None),
        hac_lag=(int(cfg.hac_lag) if cfg.get("hac_lag") is not None else None),
        bootstrap_resamples=int(cfg.get("bootstrap_resamples", 2000)),
        bootstrap_block_length=float(cfg.get("bootstrap_block_length", 10.0)),
        bootstrap_seed=int(cfg.get("bootstrap_seed", 0)),
        test_significance_level=float(cfg.get("test_significance_level", 0.05)),
        multiple_testing_correction=str(cfg.get("multiple_testing_correction", "holm")),
        output_dir=output_dir,
        table_formats=tuple(cast(TableFormat, str(value)) for value in cfg.table_formats),
        table_dpi=int(cfg.table_dpi),
        plot_formats=tuple(cast(PlotFormat, str(value)) for value in cfg.plot_formats),
        plot_dpi=int(cfg.plot_dpi),
    )


def build_risk_free_reference(cfg: DictConfig) -> RiskFreeReference:
    """Resolve the prepared daily risk-free feature output selected by analytics."""
    reference = cfg.get("risk_free")
    if not isinstance(reference, DictConfig):
        raise ValueError("analytics.risk_free must declare a dataset and column")
    return RiskFreeReference(dataset=str(reference["dataset"]), column=str(reference["column"]))


def analytics_data_requirements(cfg: DictConfig) -> tuple[DataRequirements, str]:
    """Resolve the union of analytics factor datasets and the required data view.

    The risk-free dataset is always required. When factor analysis is enabled the FF,
    spanning-benchmark, and timing datasets join the union, and an enabled IPCA additionally
    pulls in its own reference universe and promotes the assembly to the aligned,
    membership-masked panel built on the configured ``aligning_dataset`` grid.
    """
    risk_free = cfg.get("risk_free")
    if not isinstance(risk_free, DictConfig):
        raise ValueError("analytics.risk_free must declare a dataset and column")
    risk_free_dataset = str(risk_free["dataset"])
    names: dict[str, None] = {risk_free_dataset: None}
    primary_dataset = risk_free_dataset
    data_view = "independent"

    settings = cfg.get("factor_analysis")
    if isinstance(settings, DictConfig) and bool(settings.get("enabled", True)):
        factors = cast(DictConfig, settings["factors"])
        spanning = cast(DictConfig, settings["spanning"])
        timing = cast(DictConfig, settings["timing"])
        instruments = cast(DictConfig, timing["instruments"])
        names[str(factors["dataset"])] = None
        names[str(spanning["dataset"])] = None
        names[str(instruments["dataset"])] = None
        aligning_dataset = str(settings["aligning_dataset"])
        ipca = build_ipca_settings(cast(DictConfig, settings["ipca"]), aligning_dataset)
        if ipca.enabled:
            primary_dataset = aligning_dataset
            names[aligning_dataset] = None
            names[ipca.returns_dataset] = None
            names[ipca.characteristics_dataset] = None
            data_view = "aligned_panel"

    requirements = DataRequirements(
        primary_dataset=primary_dataset,
        datasets=tuple(DatasetRequirement(name, EntityScope.UNIVERSE) for name in names),
    )
    return requirements, data_view
