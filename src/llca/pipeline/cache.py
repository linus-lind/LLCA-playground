"""Disposable, content-addressed cache for expensive deterministic data preparation."""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

from omegaconf import DictConfig, ListConfig, OmegaConf

from llca.core.paths import PROJECT_ROOT
from llca.pipeline.contracts import DataPlan

_CACHE_SCHEMA_VERSION = 1
_CODE_ROOTS = (
    "data",
    "preprocessing",
    "transforms",
    "mappers/data",
    "mappers/features",
    "mappers/preprocessing",
    "mappers/masking",
    "mappers/modules",
    "pipeline/assembly.py",
    "pipeline/preparation.py",
)


def _plain(value: object) -> object:
    """Resolve an OmegaConf node to native containers with interpolations applied.

    Both mapping and list config forms are resolved: preprocessing and feature chains may
    be supplied as a top-level ``ListConfig``, whose interpolated values would otherwise be
    serialized verbatim (``${...}``) and collide across genuinely different resolved runs.
    """
    if isinstance(value, DictConfig | ListConfig):
        return OmegaConf.to_container(value, resolve=True)
    return value


def _code_fingerprint() -> str:
    root = PROJECT_ROOT / "src" / "llca"
    digest = hashlib.sha256()
    files: list[Path] = []
    for relative in _CODE_ROOTS:
        path = root / relative
        files.extend(path.rglob("*.py") if path.is_dir() else [path])
    for path in sorted(set(files)):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def preparation_cache_key(
    cfg: DictConfig,
    plan: DataPlan,
    logical_sources: Mapping[str, Path],
    *,
    data_view: str,
    source_versions: Mapping[str, str] | None = None,
) -> str:
    """Hash source state, selected config, model query, and transformation implementation."""
    if source_versions is not None and set(source_versions) != set(logical_sources):
        raise ValueError("source versions must match the planned logical source names")
    payload = {
        "schema_version": _CACHE_SCHEMA_VERSION,
        "code": _code_fingerprint(),
        "sources": {
            name: {
                "path": path.resolve().as_posix(),
                **(
                    {"sha256": source_versions[name]}
                    if source_versions is not None
                    else {
                        "size": path.stat().st_size,
                        "mtime_ns": path.stat().st_mtime_ns,
                    }
                ),
            }
            for name, path in sorted(logical_sources.items())
        },
        "data": {
            "index": _plain(cfg.data.get("index")),
            "datasets": {name: _plain(cfg.data.datasets[name]) for name in sorted(plan.datasets)},
        },
        "selection": {
            name: list(query.entity_ids) if query.entity_ids is not None else None
            for name, query in sorted(plan.datasets.items())
        },
        "preprocessing": _plain(cfg.get("preprocessing")),
        "features": _plain(cfg.get("features")),
        "masking": _plain(cfg.get("masking")),
        "data_view": data_view,
        "primary_dataset": plan.primary_dataset,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def cache_directory(cfg: DictConfig) -> Path | None:
    """Resolve the configured cache root, or disable caching explicitly."""
    cache = cfg.data.get("cache")
    if cache is None:
        return PROJECT_ROOT / ".cache/llca/data"
    if not isinstance(cache, DictConfig) or not bool(cache.get("enabled", False)):
        return None
    configured = Path(str(cache.get("directory", ".cache/llca/data")))
    return configured if configured.is_absolute() else PROJECT_ROOT / configured


def load_cached_preparation(directory: Path, key: str) -> dict[str, Any] | None:
    """Load one trusted project-local cache entry and validate its lightweight schema."""
    path = directory / f"{key}.pkl"
    if not path.is_file():
        return None
    try:
        with path.open("rb") as stream:
            payload = pickle.load(stream)  # noqa: S301 - trusted project-local cache only
    except (OSError, EOFError, ImportError, AttributeError, ValueError, pickle.UnpicklingError):
        return None
    if not isinstance(payload, dict) or payload.get("schema_version") != _CACHE_SCHEMA_VERSION:
        return None
    return cast(dict[str, Any], payload)


def save_cached_preparation(directory: Path, key: str, payload: Mapping[str, Any]) -> None:
    """Atomically publish a complete cache entry so interruptions cannot corrupt it."""
    directory.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{key}.", suffix=".tmp", dir=directory)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            pickle.dump(
                {"schema_version": _CACHE_SCHEMA_VERSION, **dict(payload)},
                stream,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, directory / f"{key}.pkl")
    finally:
        try:
            Path(temporary).unlink(missing_ok=True)
        except OSError:
            pass
