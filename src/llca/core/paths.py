import os
from pathlib import Path


def _find_project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("could not locate project root (no pyproject.toml found)")


PROJECT_ROOT = _find_project_root()
DATA_DIR = PROJECT_ROOT / "data"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
REPORTS_DIR = PROJECT_ROOT / "reports"


def chdir_to_project_root() -> None:
    os.chdir(PROJECT_ROOT)
