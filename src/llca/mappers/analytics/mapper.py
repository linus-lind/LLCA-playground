from pathlib import Path
from typing import cast

import pandas as pd
from omegaconf import DictConfig

from llca.analytics.utils.config import (
    ModelEvaluationConfig,
    RegisteredModelConfig,
    TableFormat,
)
from llca.core.paths import PROJECT_ROOT
from llca.core.returns import ReturnType


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
        signal_buckets=int(cfg.signal_buckets),
        probability_bins=int(cfg.probability_bins),
        classification_threshold=float(cfg.classification_threshold),
        target_threshold=float(cfg.target_threshold),
        risk_free_rate=float(cfg.risk_free_rate),
        minimum_acceptable_return=float(cfg.minimum_acceptable_return),
        var_levels=tuple(float(level) for level in cfg.var_levels),
        rolling_window=int(cfg.rolling_window),
        signal_decay_periods=tuple(int(period) for period in cfg.signal_decay_periods),
        active_weight_threshold=float(cfg.active_weight_threshold),
        include_initial_trade=bool(cfg.include_initial_trade),
        show_plots=bool(cfg.show_plots),
        evaluation_end=(pd.Timestamp(str(evaluation_end)) if evaluation_end is not None else None),
        output_dir=output_dir,
        table_formats=tuple(cast(TableFormat, str(value)) for value in cfg.table_formats),
        table_dpi=int(cfg.table_dpi),
    )
