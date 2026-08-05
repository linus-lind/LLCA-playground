from __future__ import annotations


class ConfigValidationError(Exception):
    """Aggregate independent Hydra validation failures into one actionable exception."""

    def __init__(self, errors: list[str]) -> None:
        self.errors = errors
        message = "Invalid Hydra configuration:\n" + "\n".join(f"  - {error}" for error in errors)
        super().__init__(message)
