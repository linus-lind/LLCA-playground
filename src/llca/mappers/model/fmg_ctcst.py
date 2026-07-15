from functools import partial
from typing import cast

from omegaconf import DictConfig, ListConfig
from torch import nn

from llca.mappers.config_validation import ConfigField, as_list, check_fields, is_int
from llca.mappers.model.mapper import EstimatorFactory, model_registry
from llca.models.estimators.fmg_ctcst_estimator import FmgCtcstEstimator
from llca.models.estimators.fmg_ctct_1_estimator import FmgCtct1Estimator
from llca.models.estimators.fmg_ctt_estimator import FmgCttEstimator

_SCORE_ACTIVATIONS = ("identity", "softsign", "tanh")

_TOP_FIELDS = [
    ConfigField("d_model", "int", positive=True),
    ConfigField("feature_embedding_dim", "int", positive=True),
    ConfigField("sequence_length", "int", positive=True),
    ConfigField("dropout", "number", minimum=0.0, maximum=1.0),
    ConfigField("score_activation", "str"),
]
_SUPERVISION_FIELDS = [
    ConfigField("dataset", "str"),
    ConfigField("column", "str"),
]
_DIAGNOSTIC_FIELDS = [
    ConfigField("score_saturation_threshold", "number", positive=True),
]
_TARGET_FIELDS = [ConfigField("entity_id", "int", positive=True)]


def _validate_score_activation(model: DictConfig) -> list[str]:
    activation = model.get("score_activation")
    if activation is not None and activation not in _SCORE_ACTIVATIONS:
        return [f"model.score_activation '{activation}' must be one of {list(_SCORE_ACTIVATIONS)}"]
    return []


def _validate_inputs(cfg: DictConfig) -> list[str]:
    """Validate named feature/context roles against configured reusable datasets."""
    model = cfg.model
    errors = check_fields(model, "model", [ConfigField("inputs", "mapping")])
    if errors:
        return errors
    inputs = model.inputs
    errors = check_fields(
        inputs,
        "model.inputs",
        [
            ConfigField("features", "str"),
            ConfigField("context", "list", non_empty=True, allow_scalar=True),
        ],
    )
    datasets = cfg.data.get("datasets") if cfg.get("data") is not None else None
    available = set(datasets.keys()) if isinstance(datasets, DictConfig) else set()
    for role in ("features", "context"):
        for name in (str(entry) for entry in as_list(inputs.get(role))):
            if name not in available:
                errors.append(
                    f"model.inputs.{role} references '{name}' which is not a configured dataset in data.datasets"
                )
    return errors


def _validate_kernel(
    kernel: object, prefix: str, sequence_length: object, d_model: object
) -> list[str]:
    """Validate temporal and embedding-axis kernel dimensions against model widths."""
    if not isinstance(kernel, list | ListConfig) or len(kernel) != 2:
        return [f"{prefix}.kernel_size must be a [height, width] pair of positive integers"]
    height, width = kernel[0], kernel[1]
    errors = [
        f"{prefix}.kernel_size {axis} must be a positive integer"
        for axis, value in (("height", height), ("width", width))
        if not (is_int(value) and value > 0)
    ]
    if errors:
        return errors

    if is_int(sequence_length) and height > cast(int, sequence_length):
        errors.append(
            f"{prefix}.kernel_size height ({height}) must not exceed model.sequence_length ({sequence_length})"
        )
    if width % 2 == 0:
        errors.append(
            f"{prefix}.kernel_size width ({width}) must be odd for symmetric same-padding"
        )
    if is_int(d_model) and width > cast(int, d_model):
        errors.append(
            f"{prefix}.kernel_size width ({width}) must not exceed model.d_model ({d_model})"
        )
    return errors


def _validate_padding(padding: object, kernel: object, prefix: str) -> list[str]:
    """Require width preservation and non-expanding temporal convolution padding."""
    if not isinstance(padding, list | ListConfig) or len(padding) != 2:
        return [f"{prefix}.padding must be a [height, width] pair of non-negative integers"]
    pad_height, pad_width = padding[0], padding[1]
    errors = [
        f"{prefix}.padding {axis} must be a non-negative integer"
        for axis, value in (("height", pad_height), ("width", pad_width))
        if not (is_int(value) and value >= 0)
    ]
    if errors:
        return errors

    if (
        isinstance(kernel, list | ListConfig)
        and len(kernel) == 2
        and is_int(kernel[0])
        and is_int(kernel[1])
    ):
        kernel_height, kernel_width = kernel[0], kernel[1]
        if 2 * pad_width + 1 != kernel_width:
            errors.append(
                f"{prefix}.padding width ({pad_width}) must equal (kernel_width - 1) / 2 to preserve model.d_model"
            )
        if 2 * pad_height > kernel_height - 1:
            errors.append(
                f"{prefix}.padding height ({pad_height}) must not exceed (kernel_height - 1) / 2 "
                "to avoid expanding the time axis"
            )
    return errors


def _validate_cnn(model: DictConfig) -> list[str]:
    """Validate every convolution layer and its cumulative shape-compatible primitives."""
    errors = check_fields(model, "model", [ConfigField("cnn", "mapping")])
    if errors:
        return errors
    cnn = model.cnn
    errors = check_fields(cnn, "model.cnn", [ConfigField("layers", "list", non_empty=True)])
    if errors:
        return errors

    for index, layer in enumerate(cnn.layers):
        prefix = f"model.cnn.layers[{index}]"
        if not isinstance(layer, DictConfig):
            errors.append(f"{prefix} must be a mapping")
            continue
        errors.extend(
            check_fields(layer, prefix, [ConfigField("out_channels", "int", positive=True)])
        )
        kernel = layer.get("kernel_size")
        errors.extend(
            _validate_kernel(kernel, prefix, model.get("sequence_length"), model.get("d_model"))
        )
        errors.extend(_validate_padding(layer.get("padding"), kernel, prefix))
    return errors


def _validate_transformer(model: DictConfig) -> list[str]:
    """Require a positive head count that evenly partitions the model dimension."""
    errors = check_fields(model, "model", [ConfigField("transformer", "mapping")])
    if errors:
        return errors
    transformer = model.transformer
    errors = check_fields(
        transformer,
        "model.transformer",
        [
            ConfigField("n_heads", "int", positive=True),
        ],
    )
    d_model = model.get("d_model")
    n_heads = transformer.get("n_heads")
    if is_int(d_model) and is_int(n_heads) and n_heads > 0 and d_model % n_heads != 0:
        errors.append(
            f"model.d_model ({d_model}) must be divisible by model.transformer.n_heads ({n_heads})"
        )
    return errors


def _validate_diagnostics(model: DictConfig) -> list[str]:
    """Validate diagnostics whose interpretation is specific to FMG score outputs."""
    errors = check_fields(model, "model", [ConfigField("diagnostics", "mapping")])
    if errors:
        return errors
    return check_fields(model.diagnostics, "model.diagnostics", _DIAGNOSTIC_FIELDS)


def _validate_supervision(cfg: DictConfig) -> list[str]:
    """Validate the target dataset binding independently of feature inputs."""
    model = cfg.model
    errors = check_fields(model, "model", [ConfigField("supervision", "mapping")])
    if errors:
        return errors
    supervision = model.supervision
    errors = check_fields(supervision, "model.supervision", _SUPERVISION_FIELDS)
    datasets = cfg.data.get("datasets") if cfg.get("data") is not None else None
    available = set(datasets.keys()) if isinstance(datasets, DictConfig) else set()
    dataset = supervision.get("dataset")
    if dataset is not None and str(dataset) not in available:
        errors.append(
            f"model.supervision.dataset '{dataset}' is not a configured dataset in data.datasets"
        )
    return errors


def _validate_fmg_common(cfg: DictConfig) -> list[str]:
    """Compose architecture, input, supervision, convolution, and attention validation."""
    model = cfg.model
    errors = check_fields(model, "model", _TOP_FIELDS)
    errors.extend(_validate_score_activation(model))
    errors.extend(_validate_inputs(cfg))
    errors.extend(_validate_supervision(cfg))
    errors.extend(_validate_cnn(model))
    errors.extend(_validate_transformer(model))
    errors.extend(_validate_diagnostics(model))
    return errors


@model_registry.register_validator("fmg-ctct-2")
@model_registry.register_validator("fmg-ctcst")
def _validate_fmg_ctcst(cfg: DictConfig) -> list[str]:
    """Validate the full cross-sectional FMG variant and its legacy alias."""
    return _validate_fmg_common(cfg)


def _validate_single_asset_allocation(cfg: DictConfig, model_name: str) -> list[str]:
    """Validate target identity and the shared direct-allocation contract."""
    errors = _validate_fmg_common(cfg)
    model = cfg.model
    target_errors = check_fields(model, "model", [ConfigField("target", "mapping")])
    errors.extend(target_errors)
    if not target_errors:
        errors.extend(check_fields(model.target, "model.target", _TARGET_FIELDS))

    if model.get("score_activation") != "tanh":
        errors.append(f"model.score_activation must be 'tanh' for {model_name} allocations")

    loss = cfg.get("loss")
    if not isinstance(loss, DictConfig) or loss.get("name") != "portfolio":
        errors.append(f"{model_name} requires loss.name 'portfolio'")
        return errors
    if loss.get("normalization") != "bounded":
        errors.append(f"{model_name} requires loss.normalization 'bounded'")
    leverage = loss.get("leverage")
    if not isinstance(leverage, int | float) or isinstance(leverage, bool) or leverage != 1.0:
        errors.append(f"{model_name} requires loss.leverage 1.0")
    return errors


@model_registry.register_validator("fmg-ctct-1")
def _validate_fmg_ctct_1(cfg: DictConfig) -> list[str]:
    """Validate the target-query single-asset allocation model."""
    return _validate_single_asset_allocation(cfg, "fmg-ctct-1")


@model_registry.register_validator("fmg-ctt")
def _validate_fmg_ctt(cfg: DictConfig) -> list[str]:
    """Validate the temporal-only single-asset allocation model."""
    return _validate_single_asset_allocation(cfg, "fmg-ctt")


def _validate_single_asset_objective(model_name: str, loss: nn.Module | None) -> None:
    """Fail safely when builders are called without package-level validation."""
    if loss is None:
        raise ValueError(f"{model_name} requires a loss function")
    if getattr(loss, "normalization", None) != "bounded":
        raise ValueError(f"{model_name} requires a bounded portfolio objective")
    if float(getattr(loss, "leverage", float("nan"))) != 1.0:
        raise ValueError(f"{model_name} requires portfolio leverage 1.0")


@model_registry.register("fmg-ctct-1")
def _build_fmg_ctct_1(
    cfg: DictConfig,
    *,
    loss: nn.Module | None = None,
    **_: object,
) -> EstimatorFactory:
    """Bind the target-query model configuration to its allocation estimator."""
    _validate_single_asset_objective("fmg-ctct-1", loss)
    return partial(FmgCtct1Estimator, config=cfg, loss=loss)


@model_registry.register("fmg-ctt")
def _build_fmg_ctt(
    cfg: DictConfig,
    *,
    loss: nn.Module | None = None,
    **_: object,
) -> EstimatorFactory:
    """Bind the temporal-only model configuration to its target estimator."""
    _validate_single_asset_objective("fmg-ctt", loss)
    return partial(FmgCttEstimator, config=cfg, loss=loss)


@model_registry.register("fmg-ctct-2")
@model_registry.register("fmg-ctcst")
def _build_fmg_ctcst(
    cfg: DictConfig,
    *,
    loss: nn.Module | None = None,
    **_: object,
) -> EstimatorFactory:
    """Bind validated model configuration and objective into a fresh estimator factory."""
    if loss is None:
        raise ValueError(f"{cfg.name} requires a loss function")

    return partial(FmgCtcstEstimator, config=cfg, loss=loss)
