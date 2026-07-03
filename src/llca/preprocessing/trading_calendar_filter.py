import exchange_calendars as xcals
import pandas as pd

from llca.data.index_spec import time_level


def trading_calendar_filter(panel: pd.DataFrame, calendar: str) -> pd.DataFrame:
    """Retain rows whose normalized time index belongs to the named exchange calendar."""
    dates = pd.DatetimeIndex(panel.index.get_level_values(time_level(panel)))
    sessions = xcals.get_calendar(calendar, start=dates.min(), end=dates.max()).sessions
    return panel.loc[dates.normalize().isin(sessions)]
