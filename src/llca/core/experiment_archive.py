"""Create and verify immutable snapshots of the local MLflow store."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sqlite3
from collections.abc import Sequence
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from llca.core.paths import PROJECT_ROOT

ARCHIVE_SCHEMA_VERSION = 1
_CHUNK_SIZE = 8 * 1024 * 1024


class ExperimentArchiveError(RuntimeError):
    """Raised when an MLflow snapshot cannot be created or verified safely."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def _assert_no_active_runs(database: Path) -> None:
    try:
        with closing(sqlite3.connect(database)) as connection:
            rows = connection.execute(
                "SELECT run_uuid FROM runs WHERE status IN ('RUNNING', 'SCHEDULED')"
            ).fetchall()
    except sqlite3.Error as exc:
        raise ExperimentArchiveError(f"cannot inspect MLflow database {database}") from exc
    if rows:
        ids = ", ".join(str(row[0]) for row in rows[:10])
        raise ExperimentArchiveError(
            f"refusing to archive while MLflow runs are active: {ids}"
        )


def _backup_database(source: Path, destination: Path) -> None:
    with closing(sqlite3.connect(source)) as original, closing(
        sqlite3.connect(destination)
    ) as backup:
        original.backup(backup)


def _file_manifest(directory: Path) -> list[dict[str, object]]:
    return [
        {
            "path": path.relative_to(directory).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path.name != "archive_manifest.json"
    ]


def archive_experiment_store(
    *,
    database: Path = PROJECT_ROOT / "mlflow.db",
    artifacts: Path = PROJECT_ROOT / "mlruns",
    archive_root: Path,
) -> Path:
    """Snapshot a quiescent local MLflow backend and content-address all copied files."""
    if not database.is_file():
        raise ExperimentArchiveError(f"MLflow database does not exist: {database}")
    if not artifacts.is_dir():
        raise ExperimentArchiveError(f"MLflow artifact directory does not exist: {artifacts}")
    _assert_no_active_runs(database)
    archive_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    final = archive_root / f"mlflow-{timestamp}-{uuid4().hex[:8]}"
    temporary = archive_root / f".{final.name}.tmp"
    try:
        temporary.mkdir()
        _backup_database(database, temporary / "mlflow.db")
        shutil.copytree(artifacts, temporary / "mlruns")
        manifest = {
            "schema_version": ARCHIVE_SCHEMA_VERSION,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "source": {
                "database": str(database.resolve()),
                "artifacts": str(artifacts.resolve()),
            },
            "files": _file_manifest(temporary),
        }
        (temporary / "archive_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        os.replace(temporary, final)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return final


def verify_experiment_archive(directory: Path) -> None:
    """Reject missing, modified, or unexpected files in one immutable snapshot."""
    manifest_path = directory / "archive_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected = {
            str(item["path"]): (int(item["size_bytes"]), str(item["sha256"]))
            for item in manifest["files"]
        }
    except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ExperimentArchiveError(f"invalid archive manifest: {manifest_path}") from exc
    actual_paths = {
        path.relative_to(directory).as_posix()
        for path in directory.rglob("*")
        if path.is_file() and path != manifest_path
    }
    if actual_paths != set(expected):
        raise ExperimentArchiveError("archive file set differs from its manifest")
    for relative, (size, digest) in expected.items():
        path = directory / relative
        if path.stat().st_size != size or _sha256(path) != digest:
            raise ExperimentArchiveError(f"archive verification failed: {relative}")


def restore_experiment_archive(directory: Path, *, force: bool = False) -> tuple[Path, Path]:
    """Restore a verified snapshot to its original MLflow paths with rollback safety."""
    verify_experiment_archive(directory)
    manifest = json.loads((directory / "archive_manifest.json").read_text(encoding="utf-8"))
    database = Path(str(manifest["source"]["database"]))
    artifacts = Path(str(manifest["source"]["artifacts"]))
    existing = [path for path in (database, artifacts) if path.exists()]
    if existing and not force:
        raise ExperimentArchiveError(
            f"refusing to replace existing MLflow store: {existing}; use --force"
        )
    if database.is_file():
        _assert_no_active_runs(database)
    if database.parent != artifacts.parent:
        raise ExperimentArchiveError(
            "archive source paths must share a parent for atomic restoration"
        )

    root = database.parent
    root.mkdir(parents=True, exist_ok=True)
    token = uuid4().hex
    staged_database = root / f".{database.name}.{token}.restore"
    staged_artifacts = root / f".{artifacts.name}.{token}.restore"
    backup_database = root / f".{database.name}.{token}.backup"
    backup_artifacts = root / f".{artifacts.name}.{token}.backup"
    try:
        shutil.copy2(directory / "mlflow.db", staged_database)
        shutil.copytree(directory / "mlruns", staged_artifacts)
        if database.exists():
            os.replace(database, backup_database)
        if artifacts.exists():
            os.replace(artifacts, backup_artifacts)
        os.replace(staged_database, database)
        os.replace(staged_artifacts, artifacts)
    except Exception:
        if backup_database.exists():
            database.unlink(missing_ok=True)
            os.replace(backup_database, database)
        if backup_artifacts.exists():
            shutil.rmtree(artifacts, ignore_errors=True)
            os.replace(backup_artifacts, artifacts)
        raise
    finally:
        staged_database.unlink(missing_ok=True)
        shutil.rmtree(staged_artifacts, ignore_errors=True)
    backup_database.unlink(missing_ok=True)
    shutil.rmtree(backup_artifacts, ignore_errors=True)
    return database, artifacts


def main(arguments: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive-root",
        type=Path,
        default=Path(
            os.environ.get(
                "LLCA_AUDIT_ARCHIVE_DIR",
                str(PROJECT_ROOT.parent / "LLCA-audit-archive"),
            )
        ),
    )
    parser.add_argument("--verify", type=Path)
    parser.add_argument("--restore", type=Path)
    parser.add_argument("--force", action="store_true")
    options = parser.parse_args(arguments)
    if options.verify is not None and options.restore is not None:
        parser.error("--verify and --restore are mutually exclusive")
    if options.restore is not None:
        database, artifacts = restore_experiment_archive(
            options.restore,
            force=options.force,
        )
        print(f"Experiment archive restored: {database}, {artifacts}")
        return
    if options.verify is not None:
        verify_experiment_archive(options.verify)
        print(f"Archive verified: {options.verify}")
        return
    archive = archive_experiment_store(archive_root=options.archive_root)
    verify_experiment_archive(archive)
    print(f"Experiment archive created and verified: {archive}")


if __name__ == "__main__":
    main()
