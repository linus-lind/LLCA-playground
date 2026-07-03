import pandas as pd


def corporate_adjustment(
    panel: pd.DataFrame,
    *,
    price_columns: list[str] | None = None,
    price_factor: str | None = None,
    volume: str | None = None,
    shares_outstanding: str | None = None,
    share_factor: str | None = None,
) -> pd.DataFrame:
    """Apply split-style factors to prices, volume, and shares outstanding.

    Prices and volume are divided by their configured factors, while shares outstanding
    are multiplied by the share factor. The input is copied so preprocessing steps remain
    composable and do not mutate the ingested dataset.
    """
    panel = panel.copy()

    if price_columns:
        if price_factor is None:
            raise ValueError(
                "corporate_adjustment: price_factor is required when price_columns is set"
            )
        panel[price_columns] = panel[price_columns].div(panel[price_factor], axis=0)

    share_columns = [column for column in (volume, shares_outstanding) if column is not None]
    if share_columns:
        if share_factor is None:
            raise ValueError(
                "corporate_adjustment: share_factor is required when volume/shares_outstanding is set"
            )
        if volume is not None:
            panel[volume] = panel[volume].div(panel[share_factor], axis=0)
        if shares_outstanding is not None:
            panel[shares_outstanding] = panel[shares_outstanding].mul(panel[share_factor], axis=0)

    return panel
