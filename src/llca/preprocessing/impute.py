import pandas as pd

from llca.preprocessing.modules.subgroup import subgroup_key


def _ffill(frame: pd.DataFrame, keys: list[pd.Index | pd.Series]) -> pd.DataFrame:
    if not keys:
        return frame.ffill()
    return frame.groupby(keys, group_keys=False).ffill()


def impute(
    panel: pd.DataFrame,
    ffill: list[str],
    fill_zero: list[str],
    subgroup_keys: list[str],
) -> pd.DataFrame:
    """Apply configured imputations within entity and optional subgroup boundaries.

    Forward filling never crosses a composite subgroup. Rows still missing any column that
    was declared for forward- or zero-filling are removed, making the postcondition for
    configured required columns explicit.
    """
    panel = panel.copy()
    keys = subgroup_key(panel, subgroup_keys)

    if ffill:
        panel[ffill] = _ffill(panel[ffill], keys)
    for column in fill_zero:
        panel[column] = panel[column].fillna(0)

    required = [*ffill, *fill_zero]
    return panel[panel[required].notna().all(axis=1)] if required else panel
