"""Canonical model-training manifest, excluding invocation and analytics concerns."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any, cast

from omegaconf import DictConfig, OmegaConf

TRAINING_MANIFEST_SCHEMA_VERSION = 1

_TRAINING_SECTIONS = (
    "experiment_name",
    "data",
    "preprocessing",
    "features",
    "masking",
    "loss",
    "model",
    "training",
    "split",
)


def build_training_manifest(config: Mapping[str, Any] | DictConfig) -> dict[str, Any]:
    """Select only state that defines or reproduces a trained model."""
    if isinstance(config, DictConfig):
        resolved = cast(dict[str, Any], OmegaConf.to_container(config, resolve=True))
    else:
        resolved = dict(config)
    missing = [section for section in _TRAINING_SECTIONS if section not in resolved]
    if missing:
        raise ValueError(f"training configuration is missing manifest sections: {missing}")
    sections = {section: deepcopy(resolved[section]) for section in _TRAINING_SECTIONS}
    data = sections["data"]
    if isinstance(data, dict):
        data.pop("cache", None)
        selection = data.get("selection")
        if isinstance(selection, dict):
            selection.pop("csv_chunk_size", None)
    return {
        "schema_version": TRAINING_MANIFEST_SCHEMA_VERSION,
        **sections,
    }


def validate_training_manifest(value: object) -> dict[str, Any]:
    """Require the canonical manifest schema emitted by current training runs."""
    if not isinstance(value, dict):
        raise ValueError("training manifest must be a JSON object")
    version = value.get("schema_version")
    if version != TRAINING_MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            "training manifest schema_version must be "
            f"{TRAINING_MANIFEST_SCHEMA_VERSION}, got {version!r}"
        )
    expected = {"schema_version", *_TRAINING_SECTIONS}
    missing = sorted(expected - value.keys())
    unexpected = sorted(value.keys() - expected)
    if missing:
        raise ValueError(f"training manifest is missing sections: {missing}")
    if unexpected:
        raise ValueError(f"training manifest contains unexpected sections: {unexpected}")
    return deepcopy(value)
