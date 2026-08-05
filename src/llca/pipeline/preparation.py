"""Model-aware execution of ingestion, preprocessing, features, assembly, and audit."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from omegaconf import DictConfig

from llca.core.paths import PROJECT_ROOT
from llca.data.modules.panels import Panels
from llca.data.versioning import (
    archive_raw_sources,
    build_data_manifest,
    sha256_file,
    validate_data_manifest,
)
from llca.mappers.data import build_datasets, data_source_path
from llca.mappers.features import build_feature_panels
from llca.mappers.preprocessing import build_preprocessing
from llca.pipeline.assembly import assemble_data
from llca.pipeline.cache import (
    cache_directory,
    load_cached_preparation,
    preparation_cache_key,
    save_cached_preparation,
)
from llca.pipeline.contracts import DataPlan, DataRequirements
from llca.pipeline.data_planning import build_data_plan


@dataclass(frozen=True, slots=True)
class PreparedModelData:
    """Model-ready data plus the exact logical inputs selected for its construction."""

    data: Any
    processed_datasets: Panels
    feature_panels: Panels
    plan: DataPlan
    logical_sources: dict[str, Path]


@dataclass(frozen=True, slots=True)
class PreparedTrainingData(PreparedModelData):
    """Prepared model data extended with immutable training evidence."""

    data_manifest: dict[str, Any]


@dataclass(frozen=True, slots=True)
class PreparedAnalysisData(PreparedModelData):
    """Model-independent analysis data plus read-only local input evidence."""

    data_manifest: dict[str, Any]


def _prepare(
    cfg: DictConfig,
    plan: DataPlan,
    logical_sources: dict[str, Path],
    data_view: str,
    *,
    source_versions: Mapping[str, str] | None = None,
) -> PreparedModelData:
    directory = cache_directory(cfg)
    cache_key = (
        preparation_cache_key(
            cfg,
            plan,
            logical_sources,
            data_view=data_view,
            source_versions=source_versions,
        )
        if directory is not None
        else None
    )
    cached = (
        load_cached_preparation(directory, cache_key)
        if directory is not None and cache_key is not None
        else None
    )
    if cached is not None:
        return PreparedModelData(
            data=cached["data"],
            processed_datasets=cached["processed_datasets"],
            feature_panels=cached["feature_panels"],
            plan=plan,
            logical_sources=logical_sources,
        )

    loaded = build_datasets(cfg.data, plan)
    processed = build_preprocessing(cfg.get("preprocessing"), loaded)
    feature_panels = build_feature_panels(cfg.get("features"), processed)
    data = assemble_data(
        data_view,
        processed,
        feature_panels,
        plan.primary_dataset,
        cfg,
    )
    prepared = PreparedModelData(
        data=data,
        processed_datasets=processed,
        feature_panels=feature_panels,
        plan=plan,
        logical_sources=logical_sources,
    )
    if directory is not None and cache_key is not None:
        save_cached_preparation(
            directory,
            cache_key,
            {
                "data": prepared.data,
                "processed_datasets": prepared.processed_datasets,
                "feature_panels": prepared.feature_panels,
            },
        )
    return prepared


def prepare_model_data(
    cfg: DictConfig,
    requirements: DataRequirements,
    *,
    data_manifest: Mapping[str, Any],
    data_view: str = "aligned_panel",
) -> PreparedModelData:
    """Prepare model data without mutating version-control or tracking state."""
    data_manifest = validate_data_manifest(data_manifest)
    plan = build_data_plan(cfg.data, requirements)
    logical_sources = {name: data_source_path(cfg.data.datasets[name]) for name in plan.datasets}
    _assert_manifest_plan(plan, data_manifest)
    source_versions = _manifest_source_versions(logical_sources, data_manifest)
    return _prepare(
        cfg,
        plan,
        logical_sources,
        data_view=data_view,
        source_versions=source_versions,
    )


def prepare_analysis_data(
    cfg: DictConfig,
    requirements: DataRequirements,
    *,
    data_view: str = "aligned_panel",
) -> PreparedAnalysisData:
    """Prepare an analysis universe without relying on a model run or mutating DVC.

    The current bytes of every planned raw source are hashed before preparation. Those
    hashes both version the disposable preparation cache and form the source evidence
    returned to the analytics manifest. Feature-panel fingerprints bind the evidence to
    the exact configured preprocessing and feature creation result.
    """
    plan = build_data_plan(cfg.data, requirements)
    logical_sources = {name: data_source_path(cfg.data.datasets[name]) for name in plan.datasets}
    source_records = _current_source_records(logical_sources)
    source_versions = {
        name: str(source_records[_project_source_key(path)]["sha256"])
        for name, path in logical_sources.items()
    }
    prepared = _prepare(
        cfg,
        plan,
        logical_sources,
        data_view=data_view,
        source_versions=source_versions,
    )
    data_manifest = build_data_manifest(
        logical_sources,
        prepared.feature_panels,
        archived_sources=source_records,
        data_plan=_resolved_plan(plan),
    )
    return PreparedAnalysisData(
        data=prepared.data,
        processed_datasets=prepared.processed_datasets,
        feature_panels=prepared.feature_panels,
        plan=prepared.plan,
        logical_sources=prepared.logical_sources,
        data_manifest=data_manifest,
    )


def prepare_training_data(
    cfg: DictConfig, requirements: DataRequirements, data_view: str
) -> PreparedTrainingData:
    """Execute only the logical datasets and rows declared by the selected model."""
    plan = build_data_plan(cfg.data, requirements)
    logical_sources = {name: data_source_path(cfg.data.datasets[name]) for name in plan.datasets}
    archived_sources = archive_raw_sources(logical_sources)
    source_versions = {
        name: str(
            archived_sources[path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()][
                "sha256"
            ]
        )
        for name, path in logical_sources.items()
    }
    prepared = _prepare(
        cfg,
        plan,
        logical_sources,
        data_view=data_view,
        source_versions=source_versions,
    )
    data_manifest = build_data_manifest(
        logical_sources,
        prepared.feature_panels,
        archived_sources=archived_sources,
        data_plan={
            "primary_dataset": plan.primary_dataset,
            "datasets": {
                name: {
                    "entity_ids": list(query.entity_ids) if query.entity_ids is not None else None
                }
                for name, query in plan.datasets.items()
            },
        },
    )
    return PreparedTrainingData(
        data=prepared.data,
        processed_datasets=prepared.processed_datasets,
        feature_panels=prepared.feature_panels,
        plan=prepared.plan,
        logical_sources=prepared.logical_sources,
        data_manifest=data_manifest,
    )


def _project_source_key(path: Path) -> str:
    """Return the canonical project-relative source key used by data manifests."""
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(
            f"analysis data source '{resolved}' must be inside project root "
            f"'{PROJECT_ROOT.resolve()}'"
        ) from exc


def _current_source_records(
    logical_sources: Mapping[str, Path],
) -> dict[str, dict[str, Any]]:
    """Fingerprint unique local inputs without DVC operations or filesystem writes."""
    records: dict[str, dict[str, Any]] = {}
    for path in logical_sources.values():
        source_key = _project_source_key(path)
        if source_key in records:
            continue
        if not path.is_file():
            raise FileNotFoundError(f"analysis data source does not exist: {path}")
        records[source_key] = {
            "path": source_key,
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    return records


def _resolved_plan(plan: DataPlan) -> dict[str, Any]:
    """Serialize the logical selection plan consistently across run types."""
    return {
        "primary_dataset": plan.primary_dataset,
        "datasets": {
            name: {"entity_ids": list(query.entity_ids) if query.entity_ids is not None else None}
            for name, query in plan.datasets.items()
        },
    }


def _manifest_source_versions(
    logical_sources: Mapping[str, Path], manifest: Mapping[str, Any]
) -> dict[str, str]:
    """Bind planned logical datasets to the exact raw hashes stored by their run."""
    datasets = manifest.get("datasets")
    sources = manifest.get("sources")
    if not isinstance(datasets, Mapping) or not isinstance(sources, Mapping):
        raise ValueError("data manifest must contain 'datasets' and 'sources' mappings")

    versions: dict[str, str] = {}
    for name, path in logical_sources.items():
        dataset = datasets.get(name)
        if not isinstance(dataset, Mapping):
            raise ValueError(f"data manifest has no dataset record for '{name}'")
        source_key = str(dataset.get("raw_source"))
        source = sources.get(source_key)
        if not isinstance(source, Mapping):
            raise ValueError(f"data manifest has no raw source record for '{source_key}'")
        actual_key = path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
        recorded_path = str(source.get("path", source_key))
        if actual_key != source_key or recorded_path != source_key:
            raise ValueError(
                f"configured path for dataset '{name}' differs from its archived source: "
                f"configured={actual_key!r}, archived={source_key!r}"
            )
        sha256 = source.get("sha256")
        if not isinstance(sha256, str) or not sha256:
            raise ValueError(f"raw source '{source_key}' has no SHA-256 fingerprint")
        versions[name] = sha256
    return versions


def _assert_manifest_plan(plan: DataPlan, manifest: Mapping[str, Any]) -> None:
    """Reject analytical reconstruction under a different logical selection plan."""
    recorded = manifest.get("plan")
    if not isinstance(recorded, Mapping) or not recorded:
        raise ValueError("data manifest has no resolved data selection plan")
    expected = {
        "primary_dataset": plan.primary_dataset,
        "datasets": {
            name: {"entity_ids": list(query.entity_ids) if query.entity_ids is not None else None}
            for name, query in plan.datasets.items()
        },
    }
    if dict(recorded) != expected:
        raise ValueError("resolved data plan differs from the model run's archived data selection")
