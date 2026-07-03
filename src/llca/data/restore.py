"""Restore the exact DVC raw inputs recorded by an MLflow data manifest."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

import mlflow
import yaml
from mlflow import MlflowClient

from llca.core.artifacts import DATA_MANIFEST_ARTIFACT
from llca.core.paths import PROJECT_ROOT
from llca.data.versioning import DataVersioningError, sha256_file

_DVC_EXECUTABLE = str(Path(sys.executable).parent / "dvc")


def _safe_path(project_root: Path, relative: str) -> Path:
    target = (project_root / relative).resolve()
    try:
        target.relative_to(project_root.resolve())
    except ValueError as exc:
        raise DataVersioningError(f"manifest path escapes project root: {relative}") from exc
    return target


def _write_pointer(path: Path, pointer: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(
            yaml.safe_dump(dict(pointer), sort_keys=False),
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def restore_data_manifest(
    manifest: Mapping[str, Any],
    *,
    project_root: Path = PROJECT_ROOT,
    force: bool = False,
) -> tuple[Path, ...]:
    """Recreate DVC pointers, pull missing bytes, and verify every raw SHA-256."""
    sources = manifest.get("sources")
    if not isinstance(sources, Mapping):
        raise DataVersioningError("data manifest has no valid 'sources' mapping")
    restored: list[Path] = []
    for record_value in sources.values():
        if not isinstance(record_value, Mapping):
            raise DataVersioningError("data manifest contains an invalid source record")
        record = cast(Mapping[str, Any], record_value)
        relative_data = str(record["path"])
        expected_sha256 = str(record["sha256"])
        dvc = cast(Mapping[str, Any], record["dvc"])
        relative_pointer = str(dvc["pointer_path"])
        data_path = _safe_path(project_root, relative_data)
        pointer_path = _safe_path(project_root, relative_pointer)

        if data_path.is_file() and sha256_file(data_path) == expected_sha256:
            _write_pointer(pointer_path, cast(Mapping[str, Any], dvc["pointer"]))
            restored.append(data_path)
            continue
        if data_path.exists() and not force:
            raise DataVersioningError(
                f"refusing to replace data with a different hash: {data_path}; use --force"
            )
        if data_path.exists():
            data_path.unlink()

        _write_pointer(pointer_path, cast(Mapping[str, Any], dvc["pointer"]))
        try:
            subprocess.run(
                [
                    _DVC_EXECUTABLE,
                    "pull",
                    "--remote",
                    str(dvc["remote"]),
                    relative_pointer,
                ],
                cwd=project_root,
                check=True,
            )
        except (FileNotFoundError, subprocess.CalledProcessError) as exc:
            raise DataVersioningError(f"could not restore {relative_data} from DVC") from exc
        if not data_path.is_file() or sha256_file(data_path) != expected_sha256:
            raise DataVersioningError(f"restored data failed SHA-256 verification: {data_path}")
        restored.append(data_path)
    return tuple(restored)


def restore_run_data(
    run_id: str,
    tracking_uri: str,
    *,
    project_root: Path = PROJECT_ROOT,
    force: bool = False,
) -> tuple[Path, ...]:
    """Download one run's manifest from MLflow and restore its raw sources."""
    mlflow.set_tracking_uri(tracking_uri)
    local = MlflowClient().download_artifacts(run_id, DATA_MANIFEST_ARTIFACT)
    manifest = json.loads(Path(local).read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise DataVersioningError("MLflow data manifest must be a JSON object")
    return restore_data_manifest(manifest, project_root=project_root, force=force)


def main(arguments: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True, help="MLflow parent or fold run ID")
    parser.add_argument("--tracking-uri", default="sqlite:///mlflow.db")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--force", action="store_true")
    options = parser.parse_args(arguments)
    restored = restore_run_data(
        options.run_id,
        options.tracking_uri,
        project_root=options.project_root,
        force=options.force,
    )
    for path in restored:
        print(path)


if __name__ == "__main__":
    main()
