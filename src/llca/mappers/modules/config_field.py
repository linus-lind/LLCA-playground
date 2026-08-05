from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

type FieldKind = Literal["int", "number", "str", "bool", "list", "mapping"]


@dataclass(frozen=True, slots=True)
class ConfigField:
    """Declare reusable type, presence, range, and cardinality checks for one Hydra field."""

    name: str
    kind: FieldKind
    required: bool = True
    positive: bool = False
    minimum: float | None = None
    maximum: float | None = None
    non_empty: bool = False
    allow_scalar: bool = False
