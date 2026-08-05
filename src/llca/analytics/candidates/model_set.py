"""Validate that the configured registry models form a comparable set.

These guards run on registry metadata alone, before any model is loaded: they derive the shared
evaluation window and reject models whose realized-accounting contract cannot be compared.
"""

from __future__ import annotations

import pandas as pd
from omegaconf import DictConfig, ListConfig

from llca.analytics.modules.registered_model import RegisteredModelMetadata


def comparison_window(
    metadata: tuple[RegisteredModelMetadata, ...],
    configured_end: pd.Timestamp | None,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Derive the shared out-of-sample window across every registered model.

    Takes the latest test start and the earliest test end across the models, optionally clipping
    the end to ``configured_end``. Raises ``ValueError``, listing each model's window, when the
    models leave no overlapping period.
    """
    start = max(model.test_start for model in metadata)
    end = min(model.test_end for model in metadata)
    if configured_end is not None:
        end = min(end, configured_end)
    if end < start:
        windows = ", ".join(
            f"{model.config.label}={model.test_start.date()}..{model.test_end.date()}"
            for model in metadata
        )
        raise ValueError(f"configured models have no common test window: {windows}")
    return start, end


def assert_portfolio_accounting_contract(
    metadata: tuple[RegisteredModelMetadata, ...], configured_return_type: str
) -> None:
    """Require all portfolio models to agree on return convention and trading-cost inputs.

    Checks each portfolio objective's training return type against ``configured_return_type`` and
    that its execution-fee, spread, slippage, and borrow-cost fields are present and identical
    across models. These describe the market rather than a model's mandate, so a mismatch would
    make the realized-accounting comparison invalid. Collects every violation and raises one
    ``ValueError`` listing them. Non-portfolio models are ignored.
    """
    accounting_fields = (
        "execution_fee",
        "bid_ask_spread",
        "slippage",
        "borrow_cost",
    )
    contracts: list[tuple[str, dict[str, object]]] = []
    errors: list[str] = []
    for model in metadata:
        loss = model.pipeline_config.get("loss")
        if not isinstance(loss, DictConfig) or loss.get("name") != "portfolio":
            continue
        trained_return_type = str(loss.get("return_type"))
        if trained_return_type != configured_return_type:
            errors.append(
                f"{model.config.label}: training={trained_return_type}, "
                f"analytics={configured_return_type}"
            )
        missing = [field for field in accounting_fields if loss.get(field) is None]
        if missing:
            errors.append(f"{model.config.label}: missing accounting fields {missing}")
            continue
        contracts.append(
            (
                model.config.label,
                {field: loss.get(field) for field in accounting_fields},
            )
        )

    if contracts:
        reference_label, reference = contracts[0]
        for label, contract in contracts[1:]:
            differences = [
                field for field in accounting_fields if contract[field] != reference[field]
            ]
            if differences:
                detail = ", ".join(
                    f"{field}: {reference_label}={reference[field]!r}, {label}={contract[field]!r}"
                    for field in differences
                )
                errors.append(detail)

    if errors:
        raise ValueError(
            "portfolio models must share the analytics return convention and identical "
            "realized-accounting settings for a comparable report: " + "; ".join(errors)
        )


def _supervision_realization_lag(model: RegisteredModelMetadata) -> int | None:
    """Recover the return-realization lag a model was trained on, or ``None`` if unavailable.

    Locates the supervision column's feature spec in the archived training config and returns
    the negation of its ``shift`` — the number of periods the supervision return was pulled back
    onto the decision date. Returns ``None`` whenever the configuration cannot be resolved to a
    plain integer shift.
    """
    pipeline = model.pipeline_config
    if not isinstance(pipeline, DictConfig):
        return None
    supervision = pipeline.get("model", {}).get("supervision")
    features = pipeline.get("features")
    if not isinstance(supervision, DictConfig) or not isinstance(features, DictConfig):
        return None
    column = supervision.get("column")
    specs = features.get(supervision.get("dataset"))
    if not isinstance(specs, ListConfig) or not isinstance(column, str):
        return None
    for spec in specs:
        if isinstance(spec, DictConfig) and spec.get("as") == column:
            shift = spec.get("shift", 0)
            return -int(shift) if isinstance(shift, int) and not isinstance(shift, bool) else None
    return None


def assert_realization_lag_contract(
    metadata: tuple[RegisteredModelMetadata, ...], configured_lag: int
) -> None:
    """Require every model's trained realization lag to equal the analytics ``configured_lag``.

    Analytics rebuilds returns on the decision-date convention set by ``configured_lag``; a model
    trained against a different supervision shift optimized a different realized target and cannot
    be fairly scored here. Raises ``ValueError`` for any model whose lag differs or cannot be
    recovered from its archived training configuration.
    """
    errors: list[str] = []
    for model in metadata:
        trained_lag = _supervision_realization_lag(model)
        if trained_lag is None:
            errors.append(
                f"{model.config.label}: trained supervision-return lag could not be determined "
                "from its archived training configuration"
            )
        elif trained_lag != configured_lag:
            errors.append(
                f"{model.config.label}: training={trained_lag}, analytics={configured_lag}"
            )
    if errors:
        raise ValueError(
            "portfolio models must share the analytics return-realization lag "
            f"({configured_lag}): " + "; ".join(errors)
        )
