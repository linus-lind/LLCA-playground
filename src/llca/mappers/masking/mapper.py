from omegaconf import DictConfig


def build_masking(cfg: DictConfig | None) -> list[tuple[str, str]]:
    """Map configured activity-spell fields to canonical start/end column pairs."""
    if cfg is None:
        return []
    subgroups = cfg.get("subgroups")
    if subgroups is None:
        return []
    return [(str(pair.start), str(pair.end)) for pair in subgroups]
