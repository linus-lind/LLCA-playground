"""Resolve explicit user selection and model requirements into an ingestion plan."""

from __future__ import annotations

from omegaconf import DictConfig, ListConfig

from llca.pipeline.contracts import DataPlan, DataRequirements, DatasetQuery, EntityId, EntityScope


def _configured_entities(data_cfg: DictConfig) -> tuple[EntityId, ...] | None:
    selection = data_cfg.get("selection")
    if not isinstance(selection, DictConfig):
        return None
    raw = selection.get("entity_ids")
    if raw is None:
        return None
    if not isinstance(raw, list | ListConfig):
        raise ValueError("data.selection.entity_ids must be a list or null")
    entities = tuple(raw)
    if len(set(entities)) != len(entities):
        raise ValueError("data.selection.entity_ids must not contain duplicates")
    return entities or None


def build_data_plan(data_cfg: DictConfig, requirements: DataRequirements) -> DataPlan:
    """Select only required datasets and push safe entity filters as early as possible."""
    available = set(data_cfg.datasets.keys())
    missing = sorted(
        requirement.name
        for requirement in requirements.datasets
        if requirement.name not in available
    )
    if missing:
        raise ValueError(f"model requires unconfigured datasets: {missing}")

    configured_entities = _configured_entities(data_cfg)
    if (
        requirements.target_entity is not None
        and configured_entities is not None
        and requirements.target_entity not in configured_entities
    ):
        raise ValueError(
            f"target entity {requirements.target_entity!r} is excluded by data.selection.entity_ids"
        )

    queries: dict[str, DatasetQuery] = {}
    entity_role = (
        data_cfg.index.get("entity") if isinstance(data_cfg.get("index"), DictConfig) else None
    )
    for requirement in requirements.datasets:
        entity_ids: tuple[EntityId, ...] | None
        dataset_index = data_cfg.datasets[requirement.name].get("index")
        has_entity = (
            entity_role is not None
            and isinstance(dataset_index, DictConfig)
            and entity_role in dataset_index
        )
        if not has_entity:
            entity_ids = None
        elif requirement.entity_scope is EntityScope.TARGET:
            entity_ids = (
                (requirements.target_entity,) if requirements.target_entity is not None else None
            )
        else:
            entity_ids = configured_entities
        queries[requirement.name] = DatasetQuery(entity_ids=entity_ids)

    selection = data_cfg.get("selection")
    chunk_size = (
        int(selection.get("csv_chunk_size", 250_000))
        if isinstance(selection, DictConfig)
        else 250_000
    )
    if chunk_size <= 0:
        raise ValueError("data.selection.csv_chunk_size must be positive")
    return DataPlan(
        primary_dataset=requirements.primary_dataset,
        datasets=queries,
        csv_chunk_size=chunk_size,
    )
