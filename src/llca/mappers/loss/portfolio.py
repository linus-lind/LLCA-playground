from omegaconf import DictConfig
from torch import nn

from llca.core.returns import RETURN_TYPES
from llca.loss.portfolio import PORTFOLIO_NORMALIZATIONS, PortfolioLoss
from llca.mappers.config_validation import ConfigField, check_fields, is_number
from llca.mappers.loss.mapper import loss_registry, register_objective_kind
from llca.pipeline.contracts import ObjectiveKind

_PORTFOLIO_FIELDS = [
    ConfigField("leverage", "number", positive=True),
    ConfigField("normalization", "str"),
    ConfigField("return_type", "str"),
    ConfigField("risk_aversion", "number", minimum=0.0),
    ConfigField("concentration_aversion", "number", minimum=0.0),
    ConfigField("common_score_aversion", "number", minimum=0.0),
    ConfigField("net_exposure_aversion", "number", minimum=0.0),
    ConfigField("net_exposure_tolerance", "number", minimum=0.0),
    ConfigField("execution_fee", "number", minimum=0.0),
    ConfigField("bid_ask_spread", "number", minimum=0.0),
    ConfigField("slippage", "number", minimum=0.0),
    ConfigField("borrow_cost", "number", minimum=0.0),
]


@loss_registry.register("portfolio")
def _build_portfolio(cfg: DictConfig, **_: object) -> nn.Module:
    """Map portfolio objective coefficients to a model-independent loss module."""
    return PortfolioLoss(
        leverage=cfg.leverage,
        normalization=cfg.normalization,
        return_type=cfg.return_type,
        risk_aversion=cfg.risk_aversion,
        concentration_aversion=cfg.concentration_aversion,
        execution_fee=cfg.execution_fee,
        bid_ask_spread=cfg.bid_ask_spread,
        slippage=cfg.slippage,
        borrow_cost=cfg.borrow_cost,
        common_score_aversion=float(cfg.common_score_aversion),
        net_exposure_aversion=float(cfg.net_exposure_aversion),
        net_exposure_tolerance=float(cfg.net_exposure_tolerance),
    )


@loss_registry.register_validator("portfolio")
def _validate_portfolio(cfg: DictConfig) -> list[str]:
    errors = check_fields(cfg, "loss", _PORTFOLIO_FIELDS)
    normalization = cfg.get("normalization")
    if isinstance(normalization, str) and normalization not in PORTFOLIO_NORMALIZATIONS:
        errors.append(
            f"loss.normalization '{normalization}' must be one of {list(PORTFOLIO_NORMALIZATIONS)}"
        )
    return_type = cfg.get("return_type")
    if isinstance(return_type, str) and return_type not in RETURN_TYPES:
        errors.append(f"loss.return_type '{return_type}' must be one of {list(RETURN_TYPES)}")
    tolerance = cfg.get("net_exposure_tolerance")
    leverage = cfg.get("leverage")
    if is_number(tolerance) and is_number(leverage) and tolerance > leverage:
        errors.append("loss.net_exposure_tolerance must be <= loss.leverage")
    return errors


register_objective_kind("portfolio", ObjectiveKind.PORTFOLIO)
