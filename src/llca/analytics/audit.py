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
from llca.analytics.modules.registered_model import RegisteredModelMetadata
from llca.analytics.reporting import PublicationReport
from llca.core.provenance.environment import build_environment_manifest
from llca.core.provenance.source import SOURCE_FINGERPRINT_TAG, source_fingerprint
from llca.utils.git import git_commit, git_dirty

ANALYTICS_MANIFEST_SCHEMA_VERSION = 5
ANALYTICS_MANIFEST_ARTIFACT = "analytics/manifest.json"

_ACCOUNTING_FIELDS = (
    "return_type",
    "normalization",
    "leverage",
    "execution_fee",
    "bid_ask_spread",
    "slippage",
    "borrow_cost",
)


def _portfolio_accounting(config: DictConfig) -> dict[str, object] | None:
    """Capture a model's trading-cost and allocation settings, or ``None`` if not a portfolio.

    Records the accounting fields from the training loss so the report is self-describing.
    """
    loss = config.get("loss")
    if not isinstance(loss, DictConfig) or loss.get("name") != "portfolio":
        return None
    return {field: loss.get(field) for field in _ACCOUNTING_FIELDS}


def build_analytics_manifest(
    config: DictConfig,
    metadata: tuple[RegisteredModelMetadata, ...],
    comparison: ComparisonEvaluation,
    report: PublicationReport,
    *,
    common_observations: int,
    factor_data_manifest: dict[str, Any] | None = None,
    ipca_diagnostics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the self-contained provenance manifest for one analytics run.

    Captures the analytics config, factor-input pipeline and data manifest, IPCA diagnostics,
    the evaluation window and universe size, per-model registry and accounting details, the
    source fingerprint and environment, and a hashed inventory of every report artifact.
    """
    analytics = cast(dict[str, Any], OmegaConf.to_container(config.analytics, resolve=True))
    factor_pipeline = {
        name: OmegaConf.to_container(config.get(name), resolve=True)
        for name in ("data", "preprocessing", "features", "masking")
        if config.get(name) is not None
    }
    commit = git_commit()
    dirty = git_dirty()
    evaluations = {result.label: result.evaluation for result in comparison.results}
    return {
        "schema_version": ANALYTICS_MANIFEST_SCHEMA_VERSION,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "analytics": analytics,
        "factor_inputs": {
            "pipeline": factor_pipeline,
            "data_manifest": factor_data_manifest,
        },
        "ipca": {
            "diagnostics": ipca_diagnostics,
        },
        "evaluation": {
            "start": comparison.start.isoformat(),
            "end": comparison.end.isoformat(),
            "common_observations": common_observations,
            "common_dates": int(
                comparison.results[0].evaluation.dates if comparison.results else 0
            ),
            "funding_convention": "residual_cash_at_risk_free",
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
                "prediction_kind": evaluations[model.config.label].predictions.kind,
                "ic_basis": evaluations[model.config.label].signal.ic_basis,
                "portfolio_accounting": _portfolio_accounting(model.pipeline_config),
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
    """Return the SHA-256 hex digest of a file, read in chunks to bound memory use."""
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
    """Save the manifest locally and log it with the full report to a new MLflow run.

    Writes the manifest into the report directory, opens an ``analytics``-tagged run under
    ``experiment_name`` at ``tracking_uri``, uploads the manifest and every report artifact, and
    returns the run id.
    """
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
