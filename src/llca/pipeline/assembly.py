"""Registered adapters from independently processed datasets to estimator-native data."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from omegaconf import DictConfig

from llca.data.masking import align_and_mask
from llca.data.modules.panels import Panels
from llca.mappers.masking import build_masking

type DataAssembler = Callable[[Panels, Panels, str, DictConfig], Any]

_assemblers: dict[str, DataAssembler] = {}


def register_data_assembler(name: str) -> Callable[[DataAssembler], DataAssembler]:
    """Register one reusable data-view adapter."""

    def decorator(assembler: DataAssembler) -> DataAssembler:
        if name in _assemblers:
            raise ValueError(f"data assembler '{name}' is already registered")
        _assemblers[name] = assembler
        return assembler

    return decorator


def assemble_data(
    name: str,
    processed: Panels,
    features: Panels,
    primary_dataset: str,
    cfg: DictConfig,
) -> Any:
    """Construct the estimator-native data representation selected by its capability."""
    try:
        assembler = _assemblers[name]
    except KeyError as exc:
        raise KeyError(
            f"unknown data assembler '{name}', available: {sorted(_assemblers)}"
        ) from exc
    return assembler(processed, features, primary_dataset, cfg)


@register_data_assembler("aligned_panel")
def _aligned_panel(
    processed: Panels,
    features: Panels,
    primary_dataset: str,
    cfg: DictConfig,
) -> Any:
    """Create a leakage-safe as-of panel for cross-sectional and temporal estimators."""
    unsupported = {
        name: list(panel.index.names) for name, panel in features.items() if panel.index.nlevels > 2
    }
    if unsupported:
        raise ValueError(
            "aligned_panel accepts only time or (time, entity) indices; use the "
            f"'independent' data view for event-keyed datasets: {unsupported}"
        )
    membership = build_masking(cfg.get("masking"))
    return align_and_mask(processed, features, primary_dataset, membership)


@register_data_assembler("independent")
def _independent(
    processed: Panels,
    features: Panels,
    primary_dataset: str,
    cfg: DictConfig,
) -> Any:
    """Preserve native dataset frequencies and indices for custom/event estimators."""
    del processed, primary_dataset, cfg
    return features
