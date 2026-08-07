"""Map Hydra hyperparameter-selection configuration to its validated runtime form.

The shared ``hyperparameter_selection`` group carries the inner-CV geometry, search method, and
adoption margin; each tunable model contributes its own ``search_space`` and the baseline values
(read from the model's ordinary top-level hyperparameters). Validation runs before any data I/O
and rejects unknown methods, non-tunable parameter names, malformed dimensions, impossible fold
geometry, and a purge shorter than the supervision label horizon.
"""

from __future__ import annotations

from typing import Any, cast

from omegaconf import DictConfig, ListConfig

from llca.mappers.config_validation import (
    ConfigField,
    check_fields,
    is_number,
    register_validator,
)
from llca.mappers.model.mapper import model_capabilities, model_registry
from llca.mappers.supervision import supervision_forward_horizon
from llca.training.tuning import (
    ChoiceDimension,
    HyperparameterSelection,
    InnerCvSettings,
    IntRangeDimension,
    LogRangeDimension,
    ParameterValue,
    SearchDimension,
    SearchSettings,
    SearchSpace,
)
from llca.training.tuning.search import SEARCH_METHODS

_DEFAULT_MARGIN = 1.0
_DIMENSION_TYPES = ("choice", "log_range", "int_range")


def _plain(value: Any) -> ParameterValue:
    if isinstance(value, ListConfig | DictConfig):
        raise TypeError("hyperparameter values must be scalars")
    return cast(ParameterValue, value)


def _dimension(name: str, spec: DictConfig) -> SearchDimension:
    kind = str(spec.get("type"))
    if kind == "choice":
        return ChoiceDimension(name, tuple(_plain(value) for value in spec["values"]))
    if kind == "log_range":
        return LogRangeDimension(name, float(spec["low"]), float(spec["high"]), int(spec["num"]))
    if kind == "int_range":
        return IntRangeDimension(name, int(spec["low"]), int(spec["high"]))
    raise ValueError(f"unknown search dimension type '{kind}' for parameter '{name}'")


def _search_space(cfg: object) -> SearchSpace:
    if not isinstance(cfg, DictConfig):
        return SearchSpace(())
    return SearchSpace(tuple(_dimension(str(name), cfg[name]) for name in cfg))


def build_hyperparameter_selection(
    hp_cfg: DictConfig | None, model_cfg: DictConfig
) -> HyperparameterSelection | None:
    """Build the runtime selection object, or ``None`` when no selection group is composed."""
    if not isinstance(hp_cfg, DictConfig):
        return None
    space = _search_space(model_cfg.get("search_space"))
    baseline = {name: _plain(model_cfg.get(name)) for name in space.names()}
    cv = hp_cfg.get("cv") or DictConfig({})
    search = hp_cfg.get("search") or DictConfig({})
    selection = hp_cfg.get("selection") or DictConfig({})
    return HyperparameterSelection(
        enabled=bool(hp_cfg.get("enabled", False)),
        inner_cv=InnerCvSettings(
            train_size=int(cv.get("train_size", 0)),
            val_size=int(cv.get("val_size", 0)),
            step_size=int(cv.get("step_size", 0)),
            purge=int(cv.get("purge", 0)),
            lookback=int(cv.get("lookback", 0)),
            min_folds=int(cv.get("min_folds", 2)),
        ),
        search=SearchSettings(
            method=str(search.get("method", "grid")),
            n_trials=int(search.get("n_trials", 0)),
            seed=int(search.get("seed", 0)),
        ),
        search_space=space,
        baseline=baseline,
        standard_error_margin=float(selection.get("standard_error_margin", _DEFAULT_MARGIN)),
    )


def _validate_dimension(name: str, spec: object, prefix: str) -> list[str]:
    if not isinstance(spec, DictConfig):
        return [f"{prefix}.{name} must be a mapping with a 'type'"]
    kind = spec.get("type")
    if kind == "choice":
        values = spec.get("values")
        if not isinstance(values, ListConfig | list) or len(values) == 0:
            return [f"{prefix}.{name}.values must be a non-empty list"]
        return []
    if kind == "log_range":
        errors = check_fields(
            spec,
            f"{prefix}.{name}",
            [
                ConfigField("low", "number", positive=True),
                ConfigField("high", "number", positive=True),
                ConfigField("num", "int", positive=True),
            ],
        )
        return errors
    if kind == "int_range":
        return check_fields(
            spec, f"{prefix}.{name}", [ConfigField("low", "int"), ConfigField("high", "int")]
        )
    return [f"{prefix}.{name}.type must be one of {list(_DIMENSION_TYPES)}"]


@register_validator
def _validate_hyperparameter_selection(cfg: DictConfig) -> list[str]:
    """Validate the hyperparameter-selection group and its model-specific search space."""
    hp = cfg.get("hyperparameter_selection")
    if not isinstance(hp, DictConfig):
        return []
    errors: list[str] = []
    enabled = hp.get("enabled")
    if enabled is not None and not isinstance(enabled, bool):
        errors.append("hyperparameter_selection.enabled must be a boolean")
    search = hp.get("search")
    method = search.get("method") if isinstance(search, DictConfig) else None
    if method is not None and method not in SEARCH_METHODS:
        errors.append(
            f"hyperparameter_selection.search.method must be one of {list(SEARCH_METHODS)}"
        )
    selection = hp.get("selection")
    margin = selection.get("standard_error_margin") if isinstance(selection, DictConfig) else None
    if margin is not None and (not is_number(margin) or float(margin) < 0.0):
        errors.append("hyperparameter_selection.selection.standard_error_margin must be >= 0")
    if not bool(enabled):
        return errors

    model = cfg.get("model")
    model_name = str(model.name) if isinstance(model, DictConfig) and model.get("name") else ""
    tunable = (
        model_capabilities(model_name).tunable_parameters
        if model_name and model_registry.is_registered(model_name)
        else frozenset()
    )
    if not tunable:
        errors.append(
            f"model '{model_name}' does not support hyperparameter selection; "
            "set hyperparameter_selection.enabled=false"
        )
        return errors

    loss = cfg.get("loss")
    if not isinstance(loss, DictConfig) or loss.get("name") is None:
        errors.append(
            "hyperparameter_selection.enabled requires a loss to score candidates against; "
            "set a non-null loss"
        )

    cv = hp.get("cv")
    if not isinstance(cv, DictConfig):
        errors.append("hyperparameter_selection.cv must be a mapping when selection is enabled")
    else:
        errors.extend(
            check_fields(
                cv,
                "hyperparameter_selection.cv",
                [
                    ConfigField("train_size", "int", positive=True),
                    ConfigField("val_size", "int", positive=True),
                    ConfigField("step_size", "int", positive=True),
                    ConfigField("purge", "int", minimum=0, required=False),
                    ConfigField("lookback", "int", minimum=0, required=False),
                    ConfigField("min_folds", "int", minimum=2, required=False),
                ],
            )
        )
        horizon = supervision_forward_horizon(cfg)
        purge = cv.get("purge", 0)
        if (
            horizon is not None
            and isinstance(purge, int)
            and not isinstance(purge, bool)
            and purge < horizon
        ):
            errors.append(
                f"hyperparameter_selection.cv.purge ({purge}) must be >= the supervision "
                f"label horizon ({horizon}) to prevent inner train/validation leakage"
            )

    if method == "random":
        n_trials = search.get("n_trials") if isinstance(search, DictConfig) else None
        if not isinstance(n_trials, int) or isinstance(n_trials, bool) or n_trials < 1:
            errors.append("hyperparameter_selection.search.n_trials must be >= 1 for random search")

    space_cfg = model.get("search_space") if isinstance(model, DictConfig) else None
    if not isinstance(space_cfg, DictConfig) or len(space_cfg) == 0:
        errors.append(
            "model.search_space must be a non-empty mapping when hyperparameter selection is enabled"
        )
        return errors
    for raw_name in space_cfg:
        name = str(raw_name)
        if name not in tunable:
            errors.append(
                f"model.search_space.{name} is not a tunable parameter of '{model_name}'; "
                f"allowed: {sorted(tunable)}"
            )
        errors.extend(_validate_dimension(name, space_cfg[name], "model.search_space"))
        if model.get(name) is None:
            errors.append(f"model.{name} baseline value is required to search '{name}'")
    return errors
