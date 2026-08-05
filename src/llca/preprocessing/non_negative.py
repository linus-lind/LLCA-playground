from __future__ import annotations

import numpy as np
import pandas as pd


def non_negative_check(panel: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    panel = panel.copy()
    for column in columns:
        panel.loc[panel[column] < 0, column] = np.nan
    return panel
