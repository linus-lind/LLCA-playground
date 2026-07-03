from __future__ import annotations

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from llca.data.index_spec import time_level
from llca.data.masking import align_and_mask
from llca.data.modules.masked_panel import MaskedPanels
from llca.mappers import build_datasets, build_feature_panels, build_masking
from llca.mappers.preprocessing import build_preprocessing
from llca.models.estimators.prediction import PredictionOutput
from llca.splitting.slice_by_date import slice_by_date


def build_evaluation_panels(cfg: DictConfig) -> MaskedPanels:
    """Rebuild the same canonical, feature-engineered and masked panels used for training."""
    datasets = build_datasets(cfg.get("data"))
    datasets = build_preprocessing(cfg.get("preprocessing"), datasets)
    feature_panels = build_feature_panels(cfg.get("features"), datasets)
    subgroups = build_masking(cfg.get("masking"))
    return align_and_mask(
        datasets,
        feature_panels,
        str(cfg.model.inputs.features),
        subgroups,
    )


def test_window_with_history(
    panels: MaskedPanels,
    primary_dataset: str,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
    lookback: int,
) -> MaskedPanels:
    """Slice the tagged test period plus preceding dates needed for causal input windows.

    The history is provided to sequence construction but is excluded again from reported
    predictions. This gives the first test observations their past context without leaking
    validation or test targets into model fitting.
    """
    calendar = panels[primary_dataset].values
    dates = pd.DatetimeIndex(calendar.index.get_level_values(time_level(calendar)))
    ordered_dates = dates.unique().sort_values()
    start_position = int(ordered_dates.searchsorted(test_start))
    if start_position >= len(ordered_dates) or ordered_dates[start_position] != test_start:
        raise ValueError(f"test start {test_start.date()} is not present in the primary calendar")
    if test_end not in ordered_dates:
        raise ValueError(f"test end {test_end.date()} is not present in the primary calendar")

    history_start = ordered_dates[max(0, start_position - lookback)]
    return slice_by_date(panels, history_start, test_end)


def restrict_to_test_period(
    predictions: PredictionOutput,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
) -> PredictionOutput:
    """Remove lookback-only predictions and return a chronologically ordered test series."""
    dates = pd.DatetimeIndex(predictions.index.get_level_values(time_level(predictions.values)))
    keep = (dates >= test_start) & (dates <= test_end)
    selected = predictions.select(keep)
    if selected.values.empty:
        raise ValueError(
            f"model produced no predictions in test period {test_start.date()} to {test_end.date()}"
        )
    order = np.argsort(selected.values.index)
    return selected.select(order)
