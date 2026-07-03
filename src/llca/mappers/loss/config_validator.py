from omegaconf import DictConfig, ListConfig

from llca.mappers.config_validation import register_validator
from llca.mappers.loss.mapper import loss_registry

_FEATURE_RETURN_TYPES = {
    "log_change": "log",
    "log_difference": "log",
    "log_ratio": "log",
    "simple_change": "simple",
}


def _validate_supervision_return_type(cfg: DictConfig) -> list[str]:
    """Cross-check known target transforms against the portfolio return convention.

    Return semantics belong to the objective rather than every model feature. The check
    follows the model's supervision alias and only constrains transforms whose convention
    is unambiguous; passthrough or custom targets remain extensible.
    """
    loss = cfg.get("loss")
    model = cfg.get("model")
    features = cfg.get("features")
    if (
        not isinstance(loss, DictConfig)
        or loss.get("name") != "portfolio"
        or not isinstance(model, DictConfig)
        or not isinstance(features, DictConfig)
    ):
        return []
    supervision = model.get("supervision")
    if not isinstance(supervision, DictConfig):
        return []
    specs = features.get(supervision.get("dataset"))
    if not isinstance(specs, ListConfig | list):
        return []
    column = supervision.get("column")
    for spec in specs:
        if spec.get("as") != column:
            continue
        expected = _FEATURE_RETURN_TYPES.get(str(spec.get("name")))
        configured = loss.get("return_type")
        if expected is not None and configured != expected:
            return [
                f"loss.return_type '{configured}' conflicts with supervision "
                f"'{supervision.get('dataset')}.{column}' produced by "
                f"'{spec.get('name')}' ({expected})"
            ]
        return []
    return []


@register_validator
def _validate_loss(cfg: DictConfig) -> list[str]:
    """Validate the selected objective name and delegate its component-specific fields."""
    loss = cfg.get("loss")
    if loss is None or loss.get("name") is None:
        return []

    name = loss.name
    if not loss_registry.is_registered(name):
        return [f"loss.name '{name}' is not registered"]

    return loss_registry.validate(name, loss) + _validate_supervision_return_type(cfg)
