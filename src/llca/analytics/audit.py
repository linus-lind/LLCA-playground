"""Persist analytics settings and report artifacts as independent MLflow runs."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import mlflow
from omegaconf import DictConfig, OmegaConf

from llca.analytics.comparison import ComparisonEvaluation
from llca.analytics.reporting import PublicationReport
from llca.analytics.utils.registered_model_metadata import RegisteredModelMetadata
from llca.training.manifests import build_environment_manifest
from llca.training.manifests.source import SOURCE_FINGERPRINT_TAG, source_fingerprint
from llca.utils.utils import git_commit, git_dirty

ANALYTICS_MANIFEST_SCHEMA_VERSION = 1
ANALYTICS_MANIFEST_ARTIFACT = "analytics/manifest.json"


def build_analytics_manifest(
    config: DictConfig,
    metadata: tuple[RegisteredModelMetadata, ...],
    comparison: ComparisonEvaluation,
    report: PublicationReport,
) -> dict[str, Any]:
    """Describe one evaluation independently from every model's training manifest."""
    analytics = cast(dict[str, Any], OmegaConf.to_container(config.analytics, resolve=True))
    commit = git_commit()
    dirty = git_dirty()
    return {
        "schema_version": ANALYTICS_MANIFEST_SCHEMA_VERSION,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "analytics": analytics,
        "evaluation": {
            "start": comparison.start.isoformat(),
            "end": comparison.end.isoformat(),
            "common_observations": len(comparison.common_index),
        },
        "models": [
            {
                "name": model.config.name,
                "version": model.config.version,
                "label": model.config.label,
                "run_id": model.run_id,
                "model_uri": model.model_uri,
                "test_start": model.test_start.isoformat(),
                "test_end": model.test_end.isoformat(),
            }
            for model in metadata
        ],
        "source": {
            "git_commit": commit,
            "git_dirty": dirty,
            "sha256": source_fingerprint(),
        },
        "environment": build_environment_manifest(),
        "report": {
            "files": [
                {
                    "path": path.relative_to(report.directory).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
                for path in sorted(
                    (path for paths in report.artifacts.values() for path in paths),
                    key=lambda path: path.as_posix(),
                )
            ]
        },
    }


def _sha256(path: Path) -> str:
    """Hash one report artifact without loading publication figures into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def log_analytics_report(
    manifest: dict[str, Any],
    report: PublicationReport,
    *,
    tracking_uri: str,
    experiment_name: str,
) -> str:
    """Write a local manifest and archive the complete report in a dedicated MLflow run."""
    manifest_path = report.directory / "analytics_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    mlflow.set_tracking_uri(tracking_uri)
    experiment = mlflow.set_experiment(experiment_name)
    model_refs = ",".join(f"{model['name']}:{model['version']}" for model in manifest["models"])
    tags = {
        "llca.run_kind": "analytics",
        "llca.model_versions": model_refs,
        SOURCE_FINGERPRINT_TAG: str(manifest["source"]["sha256"]),
    }
    commit = manifest["source"].get("git_commit")
    if commit is not None:
        tags["git_commit"] = str(commit)
    with mlflow.start_run(
        experiment_id=experiment.experiment_id,
        run_name="analytics",
        tags=tags,
    ) as active:
        mlflow.log_dict(manifest, ANALYTICS_MANIFEST_ARTIFACT)
        mlflow.log_artifacts(str(Path(report.directory)), artifact_path="analytics/report")
        return str(active.info.run_id)
