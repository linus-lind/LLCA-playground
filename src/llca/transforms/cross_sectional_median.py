import pandas as pd

from llca.data.index_spec import require_entity_level, time_level
from llca.transforms.primitives import log_change


def cross_sectional_median(panel: pd.DataFrame, horizon: int, column: str) -> pd.Series:
    """Label whether each entity's forward log return exceeds its date cross-section.

    The entity-local ``horizon`` return ending at ``t + horizon`` is shifted back to ``t``
    and compared with the median across entities at ``t``. The output is binary with NaN
    where the forward return is unavailable, making this transform suitable for target
    construction rather than causal inference features.
    """
    entity = require_entity_level(panel)
    time = time_level(panel)
    oo_returns = (
        panel[column]
        .groupby(level=entity)
        .transform(lambda prices: log_change(prices.to_numpy(dtype=float), horizon=horizon))
    )
    forward_returns = oo_returns.groupby(level=entity).shift(-horizon)
    median = forward_returns.groupby(level=time).transform("median")
    label = (forward_returns > median).astype(int)
    return label.where(forward_returns.notna())
