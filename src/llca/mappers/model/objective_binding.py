"""Shared validation for objective-driven model data bindings.

The portfolio objective funds residual cash at the risk-free rate, so any model trained with
it must name the dataset and column supplying that rate. This validation is architecture
independent (FMG and tabular models share it), so it lives apart from any one model family.
"""

from __future__ import annotations

from omegaconf import DictConfig

from llca.mappers.config_validation import ConfigField, check_fields

_RISK_FREE_FIELDS = [ConfigField("dataset", "str"), ConfigField("column", "str")]


def loss_is_portfolio(cfg: DictConfig) -> bool:
    """Return whether the configured objective is the residual-cash portfolio loss."""
    loss = cfg.get("loss")
    return isinstance(loss, DictConfig) and loss.get("name") == "portfolio"


def validate_risk_free_binding(cfg: DictConfig, model_name: str) -> list[str]:
    """Require a resolvable ``model.risk_free`` binding for a portfolio objective.

    Validated like ``supervision``: the binding must name a dataset configured in
    ``data.datasets`` and a column. Callers invoke this only when the objective is the
    portfolio loss, so a pointwise loss such as MSE is never forced to declare a rate.
    """
    model = cfg.get("model")
    risk_free = model.get("risk_free") if isinstance(model, DictConfig) else None
    if not isinstance(risk_free, DictConfig):
        return [
            f"{model_name} portfolio objective requires a model.risk_free binding "
            "(dataset and column) so residual cash earns the risk-free rate"
        ]
    errors = check_fields(risk_free, "model.risk_free", _RISK_FREE_FIELDS)
    datasets = cfg.data.get("datasets") if cfg.get("data") is not None else None
    available = set(datasets.keys()) if isinstance(datasets, DictConfig) else set()
    dataset = risk_free.get("dataset")
    if dataset is not None and str(dataset) not in available:
        errors.append(
            f"model.risk_free.dataset '{dataset}' is not a configured dataset in data.datasets"
        )
    return errors
