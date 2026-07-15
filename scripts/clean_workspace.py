"""Remove disposable build state without touching research evidence."""

from __future__ import annotations

import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _remove(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def clean_workspace() -> tuple[Path, ...]:
    """Delete caches, build products, and ephemeral Hydra output directories."""
    targets = [
        PROJECT_ROOT / "dist",
        PROJECT_ROOT / "build",
        PROJECT_ROOT / ".coverage",
        PROJECT_ROOT / "htmlcov",
        PROJECT_ROOT / ".pytest_cache",
        PROJECT_ROOT / ".mypy_cache",
        PROJECT_ROOT / ".ruff_cache",
        PROJECT_ROOT / ".cache",
        PROJECT_ROOT / "hydra" / "outputs",
        PROJECT_ROOT / "hydra" / "multirun",
    ]
    for source_root in (PROJECT_ROOT / "src", PROJECT_ROOT / "tests", PROJECT_ROOT / "scripts"):
        if source_root.exists():
            targets.extend(source_root.rglob("__pycache__"))
            targets.extend(source_root.rglob("*.egg-info"))
    log_dir = PROJECT_ROOT / "logs"
    if log_dir.exists():
        targets.extend(log_dir.glob("*.log"))

    unique = tuple(sorted(set(targets), key=lambda path: len(path.parts), reverse=True))
    removed: list[Path] = []
    for target in unique:
        if target.exists():
            _remove(target)
            removed.append(target.relative_to(PROJECT_ROOT))
    return tuple(removed)


def main() -> None:
    removed = clean_workspace()
    if removed:
        print("Removed disposable workspace state:")
        for path in removed:
            print(f"  - {path}")
    else:
        print("Workspace is already clean.")


if __name__ == "__main__":
    main()
