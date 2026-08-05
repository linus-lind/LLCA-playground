from __future__ import annotations

from typing import cast

import pandas as pd
import torch
from omegaconf import DictConfig, ListConfig

from llca.core.resolvers import register_resolvers
from llca.core.returns import RETURN_TYPES
from llca.data.modules.column_selection import is_all_columns
from llca.mappers.config_validation import (
    ConfigField,
    check_fields,
    check_required_columns,
    is_int,
    is_number,
    register_validator,
)
from llca.mappers.data.config_validator import _validate_data
from llca.mappers.features.config_validator import _validate_features
from llca.mappers.features.mapper import feature_registry
from llca.mappers.modules.column_ref import referenced_columns
from llca.mappers.modules.config_validation_error import ConfigValidationError
from llca.mappers.preprocessing.config_validator import _validate_preprocessing_group

_TABLE_FORMATS = ("csv", "tex", "pdf", "png")
_PLOT_FORMATS = ("png", "pdf", "svg")
_MULTIPLE_TESTING_CORRECTIONS = ("none", "holm", "bh")


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
    fields = [
        ConfigField("models", "list", non_empty=True),
        ConfigField("device", "str"),
        ConfigField("annualization_periods", "int", positive=True),
        ConfigField("return_type", "str"),
        ConfigField("return_realization_lag", "int", minimum=0),
        ConfigField("signal_buckets", "int", minimum=2),
        ConfigField("target_threshold", "number"),
        ConfigField("minimum_acceptable_return", "number"),
        ConfigField("risk_free", "mapping"),
        ConfigField("factor_analysis", "mapping", required=False),
        ConfigField("var_levels", "list", non_empty=True),
        ConfigField("autocorrelation_lags", "list", non_empty=True),
        ConfigField("worst_rolling_windows", "list", non_empty=True),
        ConfigField("rolling_window", "int", positive=True),
        ConfigField("signal_decay_periods", "list", non_empty=True),
        ConfigField("active_weight_threshold", "number", minimum=0.0),
        ConfigField("include_initial_trade", "bool"),
        ConfigField("hac_lag", "int", minimum=0, required=False),
        ConfigField("bootstrap_resamples", "int", positive=True),
        ConfigField("bootstrap_block_length", "number", positive=True),
        ConfigField("bootstrap_seed", "int", minimum=0),
        ConfigField("test_significance_level", "number"),
        ConfigField("multiple_testing_correction", "str"),
        ConfigField("show_plots", "bool"),
        ConfigField("evaluation_end", "str", required=False),
        ConfigField("output_dir", "str"),
        ConfigField("table_formats", "list", non_empty=True),
        ConfigField("table_dpi", "int", positive=True),
        ConfigField("plot_formats", "list", non_empty=True),
        ConfigField("plot_dpi", "int", positive=True),
    ]
    errors = _unsupported_fields(
        analytics,
        "analytics",
        {field.name for field in fields},
    )
    errors.extend(
        check_fields(
            analytics,
            "analytics",
            fields,
        )
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
            errors.extend(_unsupported_fields(model, prefix, {"name", "version", "label"}))
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
    level = analytics.get("test_significance_level")
    if is_number(level) and not 0.0 < float(level) < 1.0:
        errors.append("analytics.test_significance_level must be a number in (0, 1)")
    correction = analytics.get("multiple_testing_correction")
    if isinstance(correction, str) and correction not in _MULTIPLE_TESTING_CORRECTIONS:
        errors.append(
            f"analytics.multiple_testing_correction '{correction}' must be one of "
            f"{list(_MULTIPLE_TESTING_CORRECTIONS)}"
        )
    levels = analytics.get("var_levels")
    if isinstance(levels, list | ListConfig):
        for index, level in enumerate(levels):
            if not is_number(level) or not 0.0 < float(level) < 1.0:
                errors.append(f"analytics.var_levels[{index}] must be a number in (0, 1)")
        if len(set(levels)) != len(levels):
            errors.append("analytics.var_levels must not contain duplicates")
    errors.extend(_validate_positive_int_list(analytics, "autocorrelation_lags"))
    errors.extend(_validate_positive_int_list(analytics, "worst_rolling_windows"))
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
    plot_formats = analytics.get("plot_formats")
    if isinstance(plot_formats, list | ListConfig):
        unknown = sorted({str(value) for value in plot_formats} - set(_PLOT_FORMATS))
        if unknown:
            errors.append(
                f"analytics.plot_formats contains unsupported values {unknown}; "
                f"available: {list(_PLOT_FORMATS)}"
            )
        if len(plot_formats) != len(set(plot_formats)):
            errors.append("analytics.plot_formats must not contain duplicates")
    errors.extend(_validate_factor_sources(cfg, analytics))
    errors.extend(_validate_analysis_pipeline(cfg))
    return errors


def _validate_positive_int_list(analytics: DictConfig, name: str) -> list[str]:
    """Require a unique, positive-integer horizon list for period-window analytics settings."""
    values = analytics.get(name)
    if not isinstance(values, list | ListConfig):
        return []
    errors = []
    if any(not is_int(value) or int(value) <= 0 for value in values):
        errors.append(f"analytics.{name} must contain positive integers")
    elif len(set(values)) != len(values):
        errors.append(f"analytics.{name} must not contain duplicates")
    return errors


def _unsupported_fields(config: DictConfig, prefix: str, allowed: set[str]) -> list[str]:
    unknown = sorted(str(key) for key in config if str(key) not in allowed)
    return [f"{prefix} has unsupported field(s) {unknown}"] if unknown else []


def _configured_feature_outputs(
    cfg: DictConfig, dataset: str, prefix: str
) -> tuple[dict[str, DictConfig], list[str]]:
    """Resolve stable analytics feature aliases and verify their raw-column bindings."""
    features = cfg.get("features")
    if not isinstance(features, DictConfig):
        return {}, ["features must be a mapping for analytics"]
    specs = features.get(dataset)
    if not isinstance(specs, list | ListConfig) or not specs:
        return {}, [f"features.{dataset} must be a non-empty list for {prefix}"]

    data = cfg.get("data")
    datasets = data.get("datasets") if isinstance(data, DictConfig) else None
    dataset_spec = datasets.get(dataset) if isinstance(datasets, DictConfig) else None
    available: set[str] = set()
    wildcard = False
    if isinstance(dataset_spec, DictConfig):
        for group in ("index", "columns", "auxiliary"):
            mapping = dataset_spec.get(group)
            if group == "columns" and is_all_columns(mapping):
                wildcard = True
            elif isinstance(mapping, DictConfig):
                available.update(str(column) for column in mapping)

    outputs: dict[str, DictConfig] = {}
    errors: list[str] = []
    for index, spec in enumerate(specs):
        spec_prefix = f"features.{dataset}[{index}]"
        if not isinstance(spec, DictConfig):
            errors.append(f"{spec_prefix} must be a feature mapping")
            continue
        name = spec.get("name")
        if not isinstance(name, str) or not feature_registry.is_registered(name):
            # The shared feature validator supplies the detailed registry error.
            continue
        alias = spec.get("as")
        if not isinstance(alias, str) or not alias.strip():
            errors.append(f"{spec_prefix}.as must explicitly name the analytics output")
            continue
        if alias in outputs:
            errors.append(f"features.{dataset} defines duplicate analytics output '{alias}'")
            continue
        outputs[alias] = spec
        errors.extend(
            check_required_columns(
                spec,
                f"{spec_prefix} feature '{name}'",
                feature_registry.column_refs(name),
            )
        )
        if not wildcard and available:
            missing = sorted(
                set(referenced_columns(spec, feature_registry.column_refs(name))) - available
            )
            if missing:
                errors.append(
                    f"{spec_prefix} references columns absent from data.datasets.{dataset}: "
                    f"{missing}"
                )
    return outputs, errors


def _validate_analysis_dataset(
    cfg: DictConfig,
    dataset: object,
    prefix: str,
    *,
    require_entity: bool,
) -> tuple[str | None, list[str]]:
    """Require an analytics role to resolve to an explicitly prepared dataset."""
    if not isinstance(dataset, str) or not dataset.strip():
        return None, [f"{prefix} must be a non-empty dataset name"]
    data = cfg.get("data")
    datasets = data.get("datasets") if isinstance(data, DictConfig) else None
    if not isinstance(datasets, DictConfig) or dataset not in datasets:
        return dataset, [f"{prefix} '{dataset}' is not declared in data.datasets"]

    errors: list[str] = []
    data_index = data.get("index")
    time_role = data_index.get("time") if isinstance(data_index, DictConfig) else None
    entity_role = data_index.get("entity") if isinstance(data_index, DictConfig) else None
    dataset_spec = datasets.get(dataset)
    dataset_index = dataset_spec.get("index") if isinstance(dataset_spec, DictConfig) else None
    required_roles = [("time", time_role)]
    if require_entity:
        required_roles.append(("entity", entity_role))
    for role, name in required_roles:
        if (
            not isinstance(name, str)
            or not isinstance(dataset_index, DictConfig)
            or name not in dataset_index
        ):
            errors.append(
                f"data.datasets.{dataset}.index must bind the global {role} role for {prefix}"
            )

    preprocessing = cfg.get("preprocessing")
    if not isinstance(preprocessing, DictConfig) or dataset not in preprocessing:
        errors.append(f"preprocessing.{dataset} must be explicitly configured for {prefix}")
    return dataset, errors


def _validate_dataset_frequency(
    cfg: DictConfig,
    dataset: object,
    prefix: str,
    *,
    expected: str,
) -> list[str]:
    """Require return-like analytics sources to declare their economic frequency."""
    if not isinstance(dataset, str):
        return []
    data = cfg.get("data")
    datasets = data.get("datasets") if isinstance(data, DictConfig) else None
    specification = datasets.get(dataset) if isinstance(datasets, DictConfig) else None
    if not isinstance(specification, DictConfig):
        return []
    frequency = specification.get("frequency")
    if frequency != expected:
        return [
            f"{prefix} dataset '{dataset}' must declare frequency '{expected}', got {frequency!r}"
        ]
    return []


def _validate_ipca(cfg: DictConfig, ipca: object) -> list[str]:
    """Validate IPCA's independent data pipeline and instrument-coverage policy."""
    prefix = "analytics.factor_analysis.ipca"
    if not isinstance(ipca, DictConfig):
        return [f"{prefix} must be a mapping"]

    errors = _unsupported_fields(
        ipca,
        prefix,
        {"enabled", "n_factors", "inputs", "min_characteristic_coverage", "max_age"},
    )
    errors.extend(
        check_fields(
            ipca,
            prefix,
            [
                ConfigField("enabled", "bool", required=False),
                ConfigField("n_factors", "int", positive=True),
                ConfigField("inputs", "mapping"),
                ConfigField("min_characteristic_coverage", "number"),
                ConfigField("max_age", "mapping", required=False),
            ],
        )
    )
    if ipca.get("enabled") is False:
        return errors
    n_factors = ipca.get("n_factors")

    inputs = ipca.get("inputs")
    if not isinstance(inputs, DictConfig):
        return errors
    inputs_prefix = f"{prefix}.inputs"
    errors.extend(_unsupported_fields(inputs, inputs_prefix, {"returns", "characteristics"}))
    errors.extend(
        check_fields(
            inputs,
            inputs_prefix,
            [
                ConfigField("returns", "mapping"),
                ConfigField("characteristics", "mapping"),
            ],
        )
    )

    returns = inputs.get("returns")
    return_outputs: dict[str, DictConfig] = {}
    if isinstance(returns, DictConfig):
        returns_prefix = f"{inputs_prefix}.returns"
        errors.extend(
            _unsupported_fields(
                returns,
                returns_prefix,
                {"dataset", "column", "return_type", "realization_lag", "excess"},
            )
        )
        errors.extend(
            check_fields(
                returns,
                returns_prefix,
                [
                    ConfigField("dataset", "str"),
                    ConfigField("column", "str"),
                    ConfigField("return_type", "str"),
                    ConfigField("realization_lag", "int", minimum=0),
                    ConfigField("excess", "bool"),
                ],
            )
        )
        return_dataset, dataset_errors = _validate_analysis_dataset(
            cfg,
            returns.get("dataset"),
            f"{returns_prefix}.dataset",
            require_entity=True,
        )
        errors.extend(dataset_errors)
        errors.extend(
            _validate_dataset_frequency(
                cfg,
                return_dataset,
                returns_prefix,
                expected="daily",
            )
        )
        if return_dataset is not None:
            return_outputs, output_errors = _configured_feature_outputs(
                cfg, return_dataset, returns_prefix
            )
            errors.extend(output_errors)
        return_type = returns.get("return_type")
        if isinstance(return_type, str) and return_type not in RETURN_TYPES:
            errors.append(
                f"{returns_prefix}.return_type '{return_type}' must be one of {list(RETURN_TYPES)}"
            )
        column = returns.get("column")
        if isinstance(column, str) and return_outputs and column not in return_outputs:
            errors.append(
                f"{returns_prefix}.column '{column}' is not produced by features.{return_dataset}"
            )
        if isinstance(column, str) and column in return_outputs:
            return_spec = return_outputs[column]
            expected_transform = f"{return_type}_change"
            if return_type in RETURN_TYPES and return_spec.get("name") != expected_transform:
                errors.append(
                    f"features.{return_dataset} output '{column}' must use "
                    f"'{expected_transform}' for return_type '{return_type}'"
                )
            lag = returns.get("realization_lag")
            configured_shift = return_spec.get("shift")
            effective_shift = 0 if configured_shift is None else configured_shift
            if is_int(lag) and effective_shift != -int(lag):
                errors.append(
                    f"features.{return_dataset} output '{column}' must set shift to "
                    f"{-int(lag)} (or omit shift when lag is zero) to match realization_lag"
                )
            analytics = cfg.get("analytics")
            global_lag = (
                analytics.get("return_realization_lag")
                if isinstance(analytics, DictConfig)
                else None
            )
            if is_int(lag) and is_int(global_lag) and int(lag) != cast(int, global_lag):
                errors.append(
                    f"{returns_prefix}.realization_lag must equal the global "
                    "analytics.return_realization_lag"
                )
    elif "returns" in inputs:
        errors.append(f"{inputs_prefix}.returns must be a mapping")

    characteristics = inputs.get("characteristics")
    characteristic_outputs: dict[str, DictConfig] = {}
    selected_characteristics: set[str] = set()
    if isinstance(characteristics, DictConfig):
        characteristics_prefix = f"{inputs_prefix}.characteristics"
        errors.extend(_unsupported_fields(characteristics, characteristics_prefix, {"dataset"}))
        errors.extend(
            check_fields(
                characteristics,
                characteristics_prefix,
                [ConfigField("dataset", "str")],
            )
        )
        characteristics_dataset, dataset_errors = _validate_analysis_dataset(
            cfg,
            characteristics.get("dataset"),
            f"{characteristics_prefix}.dataset",
            require_entity=True,
        )
        errors.extend(dataset_errors)
        if characteristics_dataset is not None:
            characteristic_outputs, output_errors = _configured_feature_outputs(
                cfg, characteristics_dataset, characteristics_prefix
            )
            errors.extend(output_errors)
        # Every feature output of the characteristics dataset is used as an IPCA instrument.
        selected_characteristics = set(characteristic_outputs)
        if (
            is_int(n_factors)
            and selected_characteristics
            and int(n_factors) > len(selected_characteristics) + 1
        ):
            errors.append(
                f"{prefix}.n_factors must not exceed the {len(selected_characteristics)} "
                "characteristic feature outputs plus the explicit constant instrument"
            )
    elif "characteristics" in inputs:
        errors.append(f"{inputs_prefix}.characteristics must be a mapping")

    coverage = ipca.get("min_characteristic_coverage")
    if is_number(coverage) and not 0.0 <= float(coverage) <= 1.0:
        errors.append(f"{prefix}.min_characteristic_coverage must be in [0, 1]")

    max_age = ipca.get("max_age")
    if isinstance(max_age, DictConfig):
        max_age_prefix = f"{prefix}.max_age"
        errors.extend(_unsupported_fields(max_age, max_age_prefix, {"default", "columns"}))
        errors.extend(
            check_fields(
                max_age,
                max_age_prefix,
                [
                    ConfigField("default", "int", minimum=0, required=False),
                    ConfigField("columns", "mapping", required=False),
                ],
            )
        )
        overrides = max_age.get("columns")
        if isinstance(overrides, DictConfig):
            for column, age in overrides.items():
                if not is_int(age) or int(age) < 0:
                    column_name = str(column)
                    errors.append(
                        f"{max_age_prefix}.columns.{column_name} must be a non-negative integer"
                    )
            unknown = sorted(
                str(column) for column in overrides if str(column) not in selected_characteristics
            )
            if unknown and selected_characteristics:
                errors.append(
                    f"{max_age_prefix}.columns contains unselected or unknown "
                    f"characteristics: {unknown}"
                )

    return errors


def _validate_analysis_pipeline(cfg: DictConfig) -> list[str]:
    """Apply the shared data/preprocessing/feature validators to analytics-owned inputs."""
    if not isinstance(cfg.get("data"), DictConfig):
        return ["data must be a mapping for analytics"]
    errors = _validate_data(cfg)
    if isinstance(cfg.get("preprocessing"), DictConfig):
        errors.extend(_validate_preprocessing_group(cfg))
    else:
        errors.append("preprocessing must be a mapping for analytics")
    if isinstance(cfg.get("features"), DictConfig):
        errors.extend(_validate_features(cfg))
    else:
        errors.append("features must be a mapping for analytics")
    if not isinstance(cfg.get("masking"), DictConfig):
        errors.append("masking must be a mapping for analytics")
    return errors


def _selected_columns(value: object, prefix: str) -> tuple[list[str], list[str]]:
    """Resolve a non-empty, unique list of stable feature-output names."""
    if not isinstance(value, list | ListConfig) or not value:
        return [], [f"{prefix} must be a non-empty list"]
    columns = [str(column) for column in value if isinstance(column, str) and column]
    errors = [] if len(columns) == len(value) else [f"{prefix} must contain non-empty strings"]
    if len(columns) != len(set(columns)):
        errors.append(f"{prefix} must be unique")
    return columns, errors


def _referenced_feature_specs(
    cfg: DictConfig,
    dataset: object,
    columns: list[str],
    prefix: str,
) -> tuple[dict[str, DictConfig], list[str]]:
    """Validate one date-indexed source and return its selected feature declarations."""
    dataset_name, errors = _validate_analysis_dataset(
        cfg, dataset, f"{prefix}.dataset", require_entity=False
    )
    if dataset_name is None:
        return {}, errors
    outputs, output_errors = _configured_feature_outputs(cfg, dataset_name, prefix)
    errors.extend(output_errors)
    missing = sorted(set(columns) - set(outputs))
    if missing and outputs:
        errors.append(
            f"{prefix} references columns not produced by features.{dataset_name}: {missing}"
        )
    return {column: outputs[column] for column in columns if column in outputs}, errors


def _validate_realized_return_shifts(
    specs: dict[str, DictConfig], lag: object, prefix: str
) -> list[str]:
    """Ensure realized factor/risk-free features share the global decision-date label."""
    if not is_int(lag):
        return []
    expected = -cast(int, lag)
    errors: list[str] = []
    for column, spec in specs.items():
        configured = spec.get("shift")
        effective = 0 if configured is None else configured
        if effective != expected:
            errors.append(
                f"features output '{column}' selected by {prefix} must set shift to "
                f"{expected} (or omit shift when lag is zero) to match "
                "analytics.return_realization_lag"
            )
    return errors


def _validate_timing_shifts(specs: dict[str, DictConfig], prefix: str) -> list[str]:
    """Keep the timing regression's instrument_lag as the only lag declaration."""
    errors: list[str] = []
    for column, spec in specs.items():
        shift = spec.get("shift")
        if shift is not None and (not is_int(shift) or int(shift) != 0):
            errors.append(
                f"features output '{column}' selected by {prefix} must not set a non-zero "
                "shift; analytics.factor_analysis.timing.instrument_lag is the single "
                "timing-lag convention"
            )
    return errors


def _validate_spanning(cfg: DictConfig, settings: DictConfig, realization_lag: object) -> list[str]:
    """Validate the mean-variance spanning benchmark-portfolio references.

    The dataset must be an explicitly prepared daily, date-indexed source; every listed
    portfolio must be produced by its feature outputs and carry the shared decision-date shift.
    ``scale`` and ``excess`` are the optional unit-conversion and risk-free-subtraction knobs.
    """
    prefix = "analytics.factor_analysis.spanning"
    spanning = settings.get("spanning")
    if not isinstance(spanning, DictConfig):
        return [f"{prefix} must be a mapping"]
    errors = _unsupported_fields(spanning, prefix, {"dataset", "portfolios", "scale", "excess"})
    errors.extend(
        check_fields(
            spanning,
            prefix,
            [
                ConfigField("dataset", "str"),
                ConfigField("portfolios", "list", non_empty=True),
                ConfigField("scale", "number", positive=True, required=False),
                ConfigField("excess", "bool", required=False),
            ],
        )
    )
    portfolios, column_errors = _selected_columns(
        spanning.get("portfolios"), f"{prefix}.portfolios"
    )
    errors.extend(column_errors)
    specs, reference_errors = _referenced_feature_specs(
        cfg, spanning.get("dataset"), portfolios, prefix
    )
    errors.extend(reference_errors)
    errors.extend(
        _validate_dataset_frequency(cfg, spanning.get("dataset"), prefix, expected="daily")
    )
    errors.extend(_validate_realized_return_shifts(specs, realization_lag, prefix))
    return errors


def _validate_factor_sources(cfg: DictConfig, analytics: DictConfig) -> list[str]:
    """Validate analytics-owned risk-free, FF6, and timing feature references."""
    errors: list[str] = []
    realization_lag = analytics.get("return_realization_lag")
    risk_free = analytics.get("risk_free")
    if isinstance(risk_free, DictConfig):
        risk_free_prefix = "analytics.risk_free"
        errors.extend(_unsupported_fields(risk_free, risk_free_prefix, {"dataset", "column"}))
        errors.extend(
            check_fields(
                risk_free,
                risk_free_prefix,
                [ConfigField("dataset", "str"), ConfigField("column", "str")],
            )
        )
        risk_free_column = risk_free.get("column")
        risk_free_specs, reference_errors = _referenced_feature_specs(
            cfg,
            risk_free.get("dataset"),
            [risk_free_column] if isinstance(risk_free_column, str) else [],
            risk_free_prefix,
        )
        errors.extend(reference_errors)
        errors.extend(
            _validate_dataset_frequency(
                cfg,
                risk_free.get("dataset"),
                risk_free_prefix,
                expected="daily",
            )
        )
        errors.extend(
            _validate_realized_return_shifts(risk_free_specs, realization_lag, risk_free_prefix)
        )
    elif risk_free is not None:
        errors.append("analytics.risk_free must be a mapping")

    settings = analytics.get("factor_analysis")
    if not isinstance(settings, DictConfig):
        return errors
    settings_prefix = "analytics.factor_analysis"
    errors.extend(
        _unsupported_fields(
            settings,
            settings_prefix,
            {
                "enabled",
                "aligning_dataset",
                "factors",
                "spanning",
                "ipca",
                "timing",
                "rolling_beta_window",
            },
        )
    )
    errors.extend(
        check_fields(
            settings,
            settings_prefix,
            [
                ConfigField("enabled", "bool", required=False),
                ConfigField("aligning_dataset", "str"),
                ConfigField("factors", "mapping"),
                ConfigField("spanning", "mapping"),
                ConfigField("ipca", "mapping"),
                ConfigField("timing", "mapping"),
                ConfigField("rolling_beta_window", "int", positive=True),
            ],
        )
    )
    if settings.get("enabled") is False:
        return errors

    # The entity-indexed grid every factor-analysis panel is forward-filled and masked onto.
    # It is consumed only when IPCA is enabled, so its source is validated under that gate.
    ipca_block = settings.get("ipca")
    ipca_enabled = not (isinstance(ipca_block, DictConfig) and ipca_block.get("enabled") is False)
    if ipca_enabled:
        aligning_dataset, aligning_errors = _validate_analysis_dataset(
            cfg,
            settings.get("aligning_dataset"),
            f"{settings_prefix}.aligning_dataset",
            require_entity=True,
        )
        errors.extend(aligning_errors)
        if aligning_dataset is not None:
            _, aligning_output_errors = _configured_feature_outputs(
                cfg, aligning_dataset, f"{settings_prefix}.aligning_dataset"
            )
            errors.extend(aligning_output_errors)
    factors = settings.get("factors")
    ff6: list[str] = []
    if isinstance(factors, DictConfig):
        factors_prefix = "analytics.factor_analysis.factors"
        errors.extend(_unsupported_fields(factors, factors_prefix, {"dataset", "ff6", "market"}))
        errors.extend(
            check_fields(
                factors,
                factors_prefix,
                [
                    ConfigField("dataset", "str"),
                    ConfigField("ff6", "list", non_empty=True),
                    ConfigField("market", "str"),
                ],
            )
        )
        ff6, column_errors = _selected_columns(factors.get("ff6"), f"{factors_prefix}.ff6")
        errors.extend(column_errors)
        factor_specs, reference_errors = _referenced_feature_specs(
            cfg, factors.get("dataset"), ff6, factors_prefix
        )
        errors.extend(reference_errors)
        errors.extend(
            _validate_dataset_frequency(
                cfg,
                factors.get("dataset"),
                factors_prefix,
                expected="daily",
            )
        )
        errors.extend(
            _validate_realized_return_shifts(factor_specs, realization_lag, factors_prefix)
        )
        market = factors.get("market")
        if isinstance(market, str) and market not in ff6:
            errors.append("analytics.factor_analysis.factors.market must be one of the ff6 columns")
    else:
        errors.append("analytics.factor_analysis.factors must be a mapping")
    errors.extend(_validate_spanning(cfg, settings, realization_lag))
    errors.extend(_validate_ipca(cfg, settings.get("ipca")))
    timing = settings.get("timing")
    if isinstance(timing, DictConfig):
        timing_prefix = "analytics.factor_analysis.timing"
        errors.extend(
            _unsupported_fields(
                timing,
                timing_prefix,
                {"instruments", "instrument_lag", "market_squared", "conditional_alpha"},
            )
        )
        errors.extend(
            check_fields(
                timing,
                timing_prefix,
                [
                    ConfigField("instruments", "mapping"),
                    ConfigField("instrument_lag", "int", minimum=0),
                    ConfigField("market_squared", "bool"),
                    ConfigField("conditional_alpha", "bool"),
                ],
            )
        )
        instruments = timing.get("instruments")
        if isinstance(instruments, DictConfig):
            instruments_prefix = f"{timing_prefix}.instruments"
            errors.extend(
                _unsupported_fields(instruments, instruments_prefix, {"dataset", "columns"})
            )
            errors.extend(
                check_fields(
                    instruments,
                    instruments_prefix,
                    [ConfigField("dataset", "str"), ConfigField("columns", "list", non_empty=True)],
                )
            )
            instrument_columns, column_errors = _selected_columns(
                instruments.get("columns"), f"{instruments_prefix}.columns"
            )
            errors.extend(column_errors)
            instrument_specs, reference_errors = _referenced_feature_specs(
                cfg,
                instruments.get("dataset"),
                instrument_columns,
                instruments_prefix,
            )
            errors.extend(reference_errors)
            errors.extend(_validate_timing_shifts(instrument_specs, instruments_prefix))
        elif "instruments" in timing:
            errors.append(f"{timing_prefix}.instruments must be a mapping")
    else:
        errors.append("analytics.factor_analysis.timing must be a mapping")
    window = settings.get("rolling_beta_window")
    if not isinstance(window, int) or isinstance(window, bool) or window <= 0:
        errors.append("analytics.factor_analysis.rolling_beta_window must be a positive integer")
    elif ff6 and window <= len(ff6) + 1:
        errors.append(
            "analytics.factor_analysis.rolling_beta_window must exceed the number of "
            f"rolling-regression coefficients (intercept plus {len(ff6)} factors)"
        )
    return errors


def validate_analytics_config(cfg: DictConfig) -> None:
    """Validate the standalone analytics entrypoint without training-only groups."""
    register_resolvers()
    errors = _validate_analytics(cfg)
    if errors:
        raise ConfigValidationError(errors)
