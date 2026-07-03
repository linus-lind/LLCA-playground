from dataclasses import dataclass

import pandas as pd
from omegaconf import DictConfig


@dataclass(frozen=True, slots=True)
class IndexSpec:
    """Describe the canonical time axis and optional entity axis of every panel.

    Date-only datasets use a one-level index; cross-sectional datasets use the ordered
    ``(time, entity)`` contract. Downstream code relies on time being the first level.
    """

    time: str
    entity: str | None = None

    @property
    def levels(self) -> list[str]:
        return [self.time] if self.entity is None else [self.time, self.entity]

    @property
    def has_entity(self) -> bool:
        return self.entity is not None


def index_spec(cfg: DictConfig) -> IndexSpec:
    """Read the global panel index contract from the validated data configuration."""
    index = cfg.get("index")
    if index is None or index.get("time") is None:
        raise ValueError("data.index must define at least a 'time' axis")
    entity = index.get("entity")
    return IndexSpec(time=str(index.time), entity=str(entity) if entity is not None else None)


def time_level(panel: pd.DataFrame | pd.Series) -> str:
    return str(panel.index.names[0])


def entity_level(panel: pd.DataFrame | pd.Series) -> str | None:
    return str(panel.index.names[1]) if panel.index.nlevels > 1 else None


def require_entity_level(panel: pd.DataFrame) -> str:
    """Return the entity level name or reject a date-only panel."""
    entity = entity_level(panel)
    if entity is None:
        raise ValueError(
            f"expected an entity-indexed panel, got index levels {list(panel.index.names)}"
        )
    return entity
