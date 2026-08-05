from __future__ import annotations

import pandas as pd


def deduplicate(panel: pd.DataFrame) -> pd.DataFrame:
    """Collapse duplicate index rows by column-wise mean and restore sorted index order."""
    if not panel.index.duplicated().any():
        return panel
    return panel.groupby(level=list(panel.index.names)).mean().sort_index()
