from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ConfigField:
    """Declare reusable type, presence, range, and cardinality checks for one Hydra field."""

    name: str
    kind: str
    required: bool = True
    positive: bool = False
    minimum: float | None = None
    maximum: float | None = None
    non_empty: bool = False
    allow_scalar: bool = False
