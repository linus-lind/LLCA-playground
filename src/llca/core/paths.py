from __future__ import annotations

import os
from pathlib import Path


def _find_project_root() -> Path:
    configured = os.environ.get("LLCA_PROJECT_ROOT")
    if configured:
        root = Path(configured).expanduser().resolve()
        if not root.is_dir():
            raise RuntimeError(f"LLCA_PROJECT_ROOT is not a directory: {root}")
        return root

    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").exists():
            return parent
    working_directory = Path.cwd().resolve()
    for candidate in (working_directory, *working_directory.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate
    return working_directory


PROJECT_ROOT = _find_project_root()
DATA_DIR = PROJECT_ROOT / "data"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
REPORTS_DIR = PROJECT_ROOT / "reports"


def chdir_to_project_root() -> None:
    os.chdir(PROJECT_ROOT)
