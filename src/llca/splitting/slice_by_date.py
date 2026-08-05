from __future__ import annotations

import pandas as pd

from llca.data.index_spec import time_level
from llca.data.modules.masked_panel import MaskedPanels


def slice_by_date(panels: MaskedPanels, start: pd.Timestamp, end: pd.Timestamp) -> MaskedPanels:
    """Slice every named panel to the same inclusive time interval."""
    sliced: MaskedPanels = {}
    for name, panel in panels.items():
        dates = panel.values.index.get_level_values(time_level(panel.values))
        keep = (dates >= start) & (dates <= end)
        sliced[name] = panel.slice_rows(keep)
    return sliced
