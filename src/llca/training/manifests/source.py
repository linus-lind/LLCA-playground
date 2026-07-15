"""Exact source snapshot for reconstruction from a dirty worktree."""

import base64
import hashlib
from pathlib import Path
from typing import Any

from llca.core.paths import PROJECT_ROOT

SOURCE_SNAPSHOT_SCHEMA_VERSION = 1
SOURCE_FINGERPRINT_TAG = "llca.source_sha256"


def source_fingerprint(source_root: Path | None = None) -> str:
    """Hash executable package sources in stable path order."""
    root = source_root or PROJECT_ROOT / "src" / "llca"
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def build_source_snapshot(source_root: Path | None = None) -> dict[str, Any]:
    """Archive every Python package source with a deterministic fingerprint."""
    root = source_root or PROJECT_ROOT / "src" / "llca"
    files = {
        path.relative_to(root).as_posix(): base64.b64encode(path.read_bytes()).decode("ascii")
        for path in sorted(root.rglob("*.py"))
    }
    return {
        "schema_version": SOURCE_SNAPSHOT_SCHEMA_VERSION,
        "source_sha256": source_fingerprint(root),
        "content_encoding": "base64",
        "files": files,
    }
