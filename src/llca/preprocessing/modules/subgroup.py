import pandas as pd

from llca.data.index_spec import entity_level


def subgroup_key(panel: pd.DataFrame, subgroup_keys: list[str]) -> list[pd.Index | pd.Series]:
    """Build the composite grouping key from entity index and configured columns."""
    entity = entity_level(panel)
    keys: list[pd.Index | pd.Series] = []
    if entity is not None:
        keys.append(panel.index.get_level_values(entity))
    keys.extend(panel[key] for key in subgroup_keys)
    return keys
