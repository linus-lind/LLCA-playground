"""Capture restorable raw-data versions and deterministic processed-data evidence."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import pandas as pd
import yaml

from llca.core.paths import PROJECT_ROOT

DATA_MANIFEST_SCHEMA_VERSION = 1
DATA_MANIFEST_FINGERPRINT_TAG = "llca.data_manifest_sha256"
_DVC_EXECUTABLE = str(Path(sys.executable).parent / "dvc")
_HASH_CHUNK_SIZE = 8 * 1024 * 1024
_FRAME_HASH_CHUNK_ROWS = 250_000
_ANSI_ESCAPE = re.compile(r"\x1b(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")


class DataVersioningError(RuntimeError):
    """Raised when data cannot be archived or described reproducibly."""


def _run_dvc(*arguments: str, project_root: Path = PROJECT_ROOT) -> str:
    try:
        completed = subprocess.run(
            [_DVC_EXECUTABLE, *arguments],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise DataVersioningError(f"DVC executable not found: {_DVC_EXECUTABLE}") from exc
    except subprocess.CalledProcessError as exc:
        details = (exc.stderr or exc.stdout or str(exc)).strip()
        raise DataVersioningError(
            f"DVC command failed ({' '.join(arguments)}): {details}"
        ) from exc
    return completed.stdout.strip()


def _project_relative(path: Path, project_root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(project_root.resolve()).as_posix()
    except ValueError as exc:
        raise DataVersioningError(
            f"data source '{resolved}' must be inside project root '{project_root.resolve()}'"
        ) from exc


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(_HASH_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def _dvc_remote(project_root: Path) -> str:
    remote = _run_dvc("config", "core.remote", project_root=project_root)
    if not remote:
        raise DataVersioningError(
            "DVC has no default remote; configure one before starting an auditable run"
        )
    return remote


def _parse_dvc_status(output: str) -> dict[str, Any]:
    """Parse one or more JSON documents emitted by DVC into one status mapping.

    Some Windows DVC invocations append a second JSON document or terminal control
    sequence to stdout. ``json.loads`` rejects that otherwise valid JSON stream as
    ``Extra data``. Every document is still inspected so a real pending remote change
    cannot be hidden by the compatibility handling.
    """
    remaining = _ANSI_ESCAPE.sub("", output).lstrip("\ufeff \t\r\n")
    decoder = json.JSONDecoder()
    combined: dict[str, Any] = {}
    while remaining:
        try:
            value, end = decoder.raw_decode(remaining)
        except json.JSONDecodeError as exc:
            raise DataVersioningError(f"DVC returned invalid remote status: {output}") from exc
        if not isinstance(value, dict):
            raise DataVersioningError("DVC remote status must contain JSON objects")
        combined.update(value)
        remaining = remaining[end:].lstrip(" \t\r\n")
    return combined


def archive_raw_file(path: Path, *, project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    """Track one raw file, push its content immediately, and return a restorable record."""
    if not path.is_file():
        raise DataVersioningError(f"raw data source does not exist: {path}")
    relative_path = _project_relative(path, project_root)
    remote = _dvc_remote(project_root)
    _run_dvc("add", relative_path, project_root=project_root)

    pointer_path = path.with_name(f"{path.name}.dvc")
    relative_pointer = _project_relative(pointer_path, project_root)
    try:
        pointer = cast(dict[str, Any], yaml.safe_load(pointer_path.read_text(encoding="utf-8")))
        output = pointer["outs"][0]
        algorithm = str(output.get("hash", "md5"))
        content_hash = str(output[algorithm])
    except (KeyError, IndexError, TypeError, yaml.YAMLError) as exc:
        raise DataVersioningError(f"invalid DVC pointer generated at {pointer_path}") from exc

    _run_dvc("push", relative_pointer, project_root=project_root)
    remote_status = _run_dvc(
        "status", "--cloud", "--json", relative_pointer, project_root=project_root
    )
    pending = _parse_dvc_status(remote_status or "{}")
    if pending:
        raise DataVersioningError(
            f"DVC remote '{remote}' did not confirm archived data for {relative_path}: "
            f"{remote_status}"
        )

    return {
        "path": relative_path,
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "dvc": {
            "remote": remote,
            "pointer_path": relative_pointer,
            "hash_algorithm": algorithm,
            "content_hash": content_hash,
            "pointer": pointer,
        },
    }


def fingerprint_frame(frame: pd.DataFrame) -> dict[str, Any]:
    """Hash values, order, labels, and dtypes without materializing an audit Parquet."""
    schema = {
        "columns": [repr(column) for column in frame.columns],
        "dtypes": [str(dtype) for dtype in frame.dtypes],
        "index_names": [repr(name) for name in frame.index.names],
        "index_dtypes": [
            str(frame.index.get_level_values(level).dtype)
            for level in range(frame.index.nlevels)
        ],
    }
    digest = hashlib.sha256(
        json.dumps(schema, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    try:
        for start in range(0, len(frame), _FRAME_HASH_CHUNK_ROWS):
            rows = frame.iloc[start : start + _FRAME_HASH_CHUNK_ROWS]
            row_hashes = pd.util.hash_pandas_object(rows, index=True, categorize=True)
            digest.update(row_hashes.to_numpy(dtype="<u8", copy=False).tobytes())
    except (TypeError, ValueError) as exc:
        raise DataVersioningError("processed panel contains unhashable values") from exc
    return {
        "fingerprint_algorithm": "sha256:pandas-hash-v1",
        "sha256": digest.hexdigest(),
        "rows": int(frame.shape[0]),
        "columns": int(frame.shape[1]),
        "schema": schema,
    }


def build_data_manifest(
    logical_sources: Mapping[str, Path],
    processed_panels: Mapping[str, pd.DataFrame],
    *,
    project_root: Path = PROJECT_ROOT,
    archived_sources: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Archive unique raw files once and bind every logical dataset to its evidence."""
    source_records = (
        {key: dict(value) for key, value in archived_sources.items()}
        if archived_sources is not None
        else archive_raw_sources(logical_sources, project_root=project_root)
    )
    datasets: dict[str, dict[str, Any]] = {}
    for name, path in logical_sources.items():
        source_key = _project_relative(path, project_root)
        if source_key not in source_records:
            raise DataVersioningError(f"raw source was not archived: {source_key}")
        if name not in processed_panels:
            raise DataVersioningError(f"dataset '{name}' has no processed feature panel")
        datasets[name] = {
            "raw_source": source_key,
            "processed": fingerprint_frame(processed_panels[name]),
        }
    unexpected = sorted(set(processed_panels) - set(logical_sources))
    if unexpected:
        raise DataVersioningError(
            f"processed panels have no configured raw source: {unexpected}"
        )
    return {
        "schema_version": DATA_MANIFEST_SCHEMA_VERSION,
        "sources": source_records,
        "datasets": datasets,
    }


def archive_raw_sources(
    logical_sources: Mapping[str, Path],
    *,
    project_root: Path = PROJECT_ROOT,
) -> dict[str, dict[str, Any]]:
    """Archive every unique configured source before downstream data processing starts."""
    records: dict[str, dict[str, Any]] = {}
    for path in logical_sources.values():
        source_key = _project_relative(path, project_root)
        if source_key not in records:
            records[source_key] = archive_raw_file(path, project_root=project_root)
    return records


def provenance_tags(manifest: Mapping[str, Any]) -> dict[str, str]:
    """Flatten stable hashes into searchable MLflow tags; the manifest stays authoritative."""
    sources = cast(Mapping[str, Mapping[str, Any]], manifest["sources"])
    datasets = cast(Mapping[str, Mapping[str, Any]], manifest["datasets"])
    tags: dict[str, str] = {
        DATA_MANIFEST_FINGERPRINT_TAG: data_manifest_fingerprint(manifest)
    }
    for name, record in datasets.items():
        source = sources[str(record["raw_source"])]
        dvc = cast(Mapping[str, Any], source["dvc"])
        tags[f"raw_data_sha256_{name}"] = str(source["sha256"])
        tags[f"raw_data_dvc_{name}"] = str(dvc["content_hash"])
        if dvc["hash_algorithm"] == "md5":
            tags[f"raw_data_md5_{name}"] = str(dvc["content_hash"])
        processed = cast(Mapping[str, Any], record["processed"])
        tags[f"processed_data_sha256_{name}"] = str(processed["sha256"])
    return tags


def data_manifest_fingerprint(manifest: Mapping[str, Any]) -> str:
    """Return the canonical digest used to bind an artifact to MLflow metadata."""
    encoded = json.dumps(
        dict(manifest), sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def version_file(path: Path) -> str:
    """Compatibility wrapper returning the DVC content hash after durable archival."""
    record = archive_raw_file(path)
    return str(cast(Mapping[str, Any], record["dvc"])["content_hash"])
