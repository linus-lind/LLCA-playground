import pandas as pd
import torch
from omegaconf import DictConfig, ListConfig

from llca.core.returns import RETURN_TYPES
from llca.mappers.config_validation import (
    ConfigField,
    check_fields,
    is_number,
    register_validator,
)
from llca.mappers.modules.config_validation_error import ConfigValidationError

_TABLE_FORMATS = ("csv", "tex", "pdf", "png")


@register_validator
def _validate_analytics(cfg: DictConfig) -> list[str]:
    """Validate comparable model identities and shared evaluation conventions.

    Labels and registry identities must be unique because all models share plot and table
    axes. Risk levels, decay horizons, device strings, return conventions, and optional
    evaluation boundaries are checked before any model or dataset is loaded.
    """
    analytics = cfg.get("analytics")
    if analytics is None:
        return []
    if not isinstance(analytics, DictConfig):
        return ["analytics must be a mapping"]
    errors = check_fields(
        analytics,
        "analytics",
        [
            ConfigField("models", "list", non_empty=True),
            ConfigField("device", "str"),
            ConfigField("annualization_periods", "int", positive=True),
            ConfigField("return_type", "str"),
            ConfigField("signal_buckets", "int", minimum=2),
            ConfigField("probability_bins", "int", minimum=2),
            ConfigField("classification_threshold", "number", minimum=0.0, maximum=1.0),
            ConfigField("target_threshold", "number"),
            ConfigField("risk_free_rate", "number"),
            ConfigField("minimum_acceptable_return", "number"),
            ConfigField("var_levels", "list", non_empty=True),
            ConfigField("rolling_window", "int", positive=True),
            ConfigField("signal_decay_periods", "list", non_empty=True),
            ConfigField("active_weight_threshold", "number", minimum=0.0),
            ConfigField("include_initial_trade", "bool"),
            ConfigField("show_plots", "bool"),
            ConfigField("evaluation_end", "str", required=False),
            ConfigField("output_dir", "str"),
            ConfigField("table_formats", "list", non_empty=True),
            ConfigField("table_dpi", "int", positive=True),
        ],
    )
    models = analytics.get("models")
    if isinstance(models, list | ListConfig):
        labels: list[str] = []
        identities: list[tuple[str, int]] = []
        for index, model in enumerate(models):
            prefix = f"analytics.models[{index}]"
            if not isinstance(model, DictConfig):
                errors.append(f"{prefix} must be a mapping")
                continue
            errors.extend(
                check_fields(
                    model,
                    prefix,
                    [
                        ConfigField("name", "str"),
                        ConfigField("version", "int", positive=True),
                        ConfigField("label", "str"),
                    ],
                )
            )
            name = model.get("name")
            version = model.get("version")
            label = model.get("label")
            if isinstance(label, str):
                if not label.strip():
                    errors.append(f"{prefix}.label must not be blank")
                labels.append(label)
            if isinstance(name, str) and isinstance(version, int):
                identities.append((name, version))
        if len(labels) != len(set(labels)):
            errors.append("analytics.models labels must be unique")
        if len(identities) != len(set(identities)):
            errors.append("analytics.models must not repeat the same name/version")
    device = analytics.get("device")
    if isinstance(device, str) and device != "auto":
        try:
            torch.device(device)
        except (RuntimeError, ValueError):
            errors.append(f"analytics.device '{device}' must be 'auto' or a valid PyTorch device")
    return_type = analytics.get("return_type")
    if isinstance(return_type, str) and return_type not in RETURN_TYPES:
        errors.append(f"analytics.return_type '{return_type}' must be one of {list(RETURN_TYPES)}")
    levels = analytics.get("var_levels")
    if isinstance(levels, list | ListConfig):
        for index, level in enumerate(levels):
            if not is_number(level) or not 0.0 < float(level) < 1.0:
                errors.append(f"analytics.var_levels[{index}] must be a number in (0, 1)")
        if len(set(levels)) != len(levels):
            errors.append("analytics.var_levels must not contain duplicates")
    decay_periods = analytics.get("signal_decay_periods")
    if isinstance(decay_periods, list | ListConfig):
        if any(
            not isinstance(period, int) or isinstance(period, bool) or period < 0
            for period in decay_periods
        ):
            errors.append("analytics.signal_decay_periods must contain non-negative integers")
        elif list(decay_periods) != sorted(set(decay_periods)):
            errors.append("analytics.signal_decay_periods must be sorted and unique")
        elif 0 not in decay_periods:
            errors.append("analytics.signal_decay_periods must include 0")
    evaluation_end = analytics.get("evaluation_end")
    if isinstance(evaluation_end, str):
        try:
            parsed = pd.Timestamp(evaluation_end)
            if pd.isna(parsed):
                raise ValueError
        except ValueError:
            errors.append("analytics.evaluation_end must be an ISO-compatible date")
    table_formats = analytics.get("table_formats")
    if isinstance(table_formats, list | ListConfig):
        unknown = sorted({str(value) for value in table_formats} - set(_TABLE_FORMATS))
        if unknown:
            errors.append(
                f"analytics.table_formats contains unsupported values {unknown}; "
                f"available: {list(_TABLE_FORMATS)}"
            )
        if len(table_formats) != len(set(table_formats)):
            errors.append("analytics.table_formats must not contain duplicates")
    return errors


def validate_analytics_config(cfg: DictConfig) -> None:
    """Validate the standalone analytics entrypoint without training-only groups."""
    errors = _validate_analytics(cfg)
    if errors:
        raise ConfigValidationError(errors)
