"""Causal, entity- and feature-specific EWMA standardization for stock-level panels."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Hashable, Mapping
from typing import Any, Self

import numpy as np
import pandas as pd

from llca.data.index_spec import require_entity_level, time_level
from llca.data.modules.masked_panel import MaskedPanel

_EPS_DEFAULT = 1e-8


def half_life_to_decay(half_life: float) -> float:
    """Convert an EWMA half-life in observations to the per-step retention ``lambda``.

    Uses ``lambda = 2 ** (-1 / half_life)``, so a weight observed ``half_life`` steps in
    the past carries exactly half the influence of the most recent observation.
    """
    if not half_life > 0:
        raise ValueError(f"half_life must be positive, got {half_life}")
    return float(2.0 ** (-1.0 / half_life))


class EwmaStandardizer:
    """Standardize a stock/entity-specific panel with a causal, resumable EWMA z-score.

    Every ``(entity, feature)`` pair owns an independent recursive mean/variance estimate;
    values are never pooled across entities or across features. For observation ``x_t`` of
    one entity-feature pair, the transform uses only state accumulated strictly before
    ``t``::

        diff_t   = x_t - mean_{t-1}
        z_t      = diff_t / max(sqrt(var_{t-1}), eps)
        mean_t   = mean_{t-1} + (1 - lambda) * diff_t
        var_t    = lambda * (var_{t-1} + (1 - lambda) * diff_t ** 2)

    ``lambda = 2 ** (-1 / half_life)`` (see :func:`half_life_to_decay`). ``mean_t``/``var_t``
    become the state used for the *next* observation, so a row's own value never enters its
    own normalization. This is the standard exponentially-weighted analogue of Welford's
    online variance update, evaluated at the pre-update mean; it is implemented directly
    (no ``pandas.ewm``) so the causal one-step-lag semantics below are exact and unambiguous.

    ``fit`` computes, once, the initial ``(mean_0, var_0)`` prior for every entity-feature
    pair from *only* the training panel passed to it, plus a pooled (cross-entity) fallback
    prior for entities that never appear in that training panel. Every subsequent
    ``transform`` call — whether over the remaining training rows, validation rows, or
    (much later, from restored state) test rows — advances the recursion further from
    wherever it last stopped; splits are never refit independently. Because ``fit``'s prior
    is estimated from the whole training panel, the very first training observation for an
    entity-feature pair is technically normalized against a statistic that "knows about"
    later training rows for that same pair — this is the one intentional exception to
    causality, confined to initialization. Every subsequent step depends only on strictly
    earlier observations.

    Validity is per entity-feature cell, not per row: a cell with ``age < 0`` (never yet
    observed) or a non-finite value is skipped for state updates entirely (the previous
    valid state for that one feature is retained) and standardizes to ``0.0``, matching the
    missing-value convention used elsewhere in this pipeline, even when a sibling feature
    on the same row is valid and does advance.

    ``transform`` may be called with panels whose date range overlaps rows already advanced
    in an earlier call (this pipeline's causal history buffers can request slightly more
    lookback than the walk-forward purge window guarantees is "new"). Each entity retains a
    bounded replay buffer of its last ``history_buffer`` transformed rows; a row whose date
    was already advanced is served verbatim from that buffer instead of being re-applied to
    the recursion, so state is never double-updated and replayed outputs are bit-exact. A
    replay request older than the buffer window raises, rather than guessing.
    """

    def __init__(self, half_life: float, *, history_buffer: int, eps: float = _EPS_DEFAULT) -> None:
        if history_buffer < 0:
            raise ValueError(f"history_buffer must be non-negative, got {history_buffer}")
        if not eps > 0:
            raise ValueError(f"eps must be positive, got {eps}")
        self._half_life = float(half_life)
        self._decay = half_life_to_decay(half_life)
        self._history_buffer = int(history_buffer)
        self._eps = float(eps)

        self._columns: list[str] | None = None
        self._mean: dict[Hashable, np.ndarray] = {}
        self._var: dict[Hashable, np.ndarray] = {}
        self._watermark: dict[Hashable, pd.Timestamp] = {}
        self._history: dict[Hashable, OrderedDict[pd.Timestamp, np.ndarray]] = {}
        self._fallback_mean: np.ndarray | None = None
        self._fallback_var: np.ndarray | None = None

    @property
    def half_life(self) -> float:
        return self._half_life

    def _require_fitted(self) -> tuple[list[str], np.ndarray, np.ndarray]:
        if self._columns is None or self._fallback_mean is None or self._fallback_var is None:
            raise RuntimeError("EwmaStandardizer must be fit before it can transform")
        return self._columns, self._fallback_mean, self._fallback_var

    def fit(self, panel: MaskedPanel) -> None:
        """Seed every entity-feature prior from training data only.

        The prior is the population mean/variance of that entity's valid (``age >= 0``,
        finite) observations in ``panel``. An entity-feature pair with zero valid training
        observations, and any entity encountered later that never appeared in ``panel`` at
        all, falls back to the pooled statistic over every valid training observation
        across all entities.
        """
        columns = list(panel.columns)
        entity_level = require_entity_level(panel.values)
        valid = (panel.age.to_numpy() >= 0) & np.isfinite(panel.values.to_numpy(dtype=np.float64))
        masked = panel.values.where(pd.DataFrame(valid, index=panel.values.index, columns=columns))
        entities = masked.index.get_level_values(entity_level)

        pooled = masked.to_numpy(dtype=np.float64)
        with np.errstate(invalid="ignore"):
            fallback_mean = np.nanmean(pooled, axis=0)
            fallback_var = np.nanvar(pooled, axis=0, ddof=0)
        fallback_mean = np.nan_to_num(fallback_mean, nan=0.0)
        fallback_var = np.nan_to_num(fallback_var, nan=0.0)

        grouped = masked.groupby(entities)
        with np.errstate(invalid="ignore"):
            entity_mean = grouped.mean()
            entity_var = grouped.var(ddof=0)

        self._columns = columns
        self._fallback_mean = fallback_mean
        self._fallback_var = fallback_var
        self._mean = {}
        self._var = {}
        self._watermark = {}
        self._history = {}
        mean_matrix = entity_mean.to_numpy(dtype=np.float64)
        var_matrix = entity_var.reindex(entity_mean.index).to_numpy(dtype=np.float64)
        for row_index, entity_id in enumerate(entity_mean.index):
            mean_row = np.where(
                np.isfinite(mean_matrix[row_index]), mean_matrix[row_index], fallback_mean
            )
            var_row = np.where(
                np.isfinite(var_matrix[row_index]), var_matrix[row_index], fallback_var
            )
            self._mean[entity_id] = mean_row
            self._var[entity_id] = var_row

    def _seed_new_entity(self, entity_id: Hashable) -> None:
        assert self._fallback_mean is not None
        assert self._fallback_var is not None
        self._mean[entity_id] = self._fallback_mean.copy()
        self._var[entity_id] = self._fallback_var.copy()

    def _remember(self, entity_id: Hashable, date: pd.Timestamp, row: np.ndarray) -> None:
        if self._history_buffer == 0:
            return
        buffer = self._history.setdefault(entity_id, OrderedDict())
        buffer[date] = row
        while len(buffer) > self._history_buffer:
            buffer.popitem(last=False)

    def transform(self, panel: MaskedPanel) -> pd.DataFrame:
        """Causally standardize ``panel`` in chronological order, advancing state in place.

        Rows at-or-before an entity's current watermark are treated as replays of
        already-advanced observations and served from the per-entity history buffer rather
        than re-applied to the recursion (see the class docstring). Remaining rows are
        processed strictly in ascending date order; state advances only for genuinely new,
        valid observations.
        """
        columns, fallback_mean, fallback_var = self._require_fitted()
        if list(panel.columns) != columns:
            raise ValueError(
                f"panel columns {list(panel.columns)} do not match fitted columns {columns}"
            )
        values = panel.values
        if len(values) == 0:
            return values.copy()

        entity_level = require_entity_level(values)
        date_level = time_level(values)
        entities = values.index.get_level_values(entity_level)
        dates_index = pd.DatetimeIndex(values.index.get_level_values(date_level))
        dates = dates_index.to_numpy()
        raw = values.to_numpy(dtype=np.float64)
        valid = (panel.age.to_numpy() >= 0) & np.isfinite(raw)

        n_rows, n_features = raw.shape
        output = np.empty((n_rows, n_features), dtype=np.float64)

        # Vectorized replay test: NaT watermarks (entity never seen) compare False, so an
        # unseen entity is correctly classified as "new" without a per-row branch. Every
        # unique entity present gets an explicit (possibly NaT) entry so the mapped result
        # is unambiguously datetime64, regardless of pandas' dict-lookup dtype inference.
        watermark_lookup = {
            entity_id: self._watermark.get(entity_id, pd.NaT) for entity_id in pd.unique(entities)
        }
        row_watermark = pd.to_datetime(entities.map(watermark_lookup).to_numpy()).to_numpy()
        is_replay = dates <= row_watermark

        for position in np.flatnonzero(is_replay):
            entity_id = entities[int(position)]
            date = dates_index[int(position)]
            buffer = self._history.get(entity_id)
            if buffer is None or date not in buffer:
                raise ValueError(
                    f"cannot replay entity {entity_id!r} at {date.date()}: outside the "
                    f"retained history_buffer window ({self._history_buffer} observations); "
                    "increase history_buffer or avoid re-presenting already-advanced dates "
                    "this far in the past"
                )
            output[position] = buffer[date]

        # New rows are walked one calendar date at a time (not one row at a time) so the
        # Python-level loop scales with unique dates rather than date x entity rows; the
        # mean/variance recursion itself is vectorized across every entity active that date.
        new_positions = np.flatnonzero(~is_replay)
        order = new_positions[np.argsort(dates[new_positions], kind="stable")]
        ordered_dates = dates[order]
        boundaries = np.flatnonzero(np.diff(ordered_dates).astype(np.int64) != 0) + 1
        for day_positions in np.split(order, boundaries):
            if len(day_positions) == 0:
                continue
            date = dates_index[day_positions[0]]
            day_entities = [entities[position] for position in day_positions]
            for entity_id in day_entities:
                if entity_id not in self._mean:
                    self._seed_new_entity(entity_id)

            mean_before = np.stack([self._mean[entity_id] for entity_id in day_entities])
            var_before = np.stack([self._var[entity_id] for entity_id in day_entities])
            day_valid = valid[day_positions]  # [n_today, n_features]; validity is per-cell
            day_raw = raw[day_positions]

            diff = day_raw - mean_before
            std_before = np.maximum(np.sqrt(var_before), self._eps)
            day_output = np.where(day_valid, diff / std_before, 0.0)
            # Per-feature selection: a feature invalid this row keeps its prior state
            # exactly, even when a sibling feature on the same row is valid and updates.
            mean_after = np.where(day_valid, mean_before + (1.0 - self._decay) * diff, mean_before)
            var_after = np.where(
                day_valid,
                self._decay * (var_before + (1.0 - self._decay) * diff * diff),
                var_before,
            )

            for row, entity_id in enumerate(day_entities):
                self._mean[entity_id] = mean_after[row]
                self._var[entity_id] = var_after[row]
                self._watermark[entity_id] = date
                self._remember(entity_id, date, day_output[row])
            output[day_positions] = day_output

        return pd.DataFrame(output, index=values.index, columns=values.columns)

    def state_dict(self) -> dict[str, Any]:
        """Return everything required to resume normalization from exactly this point."""
        return {
            "half_life": self._half_life,
            "eps": self._eps,
            "history_buffer": self._history_buffer,
            "columns": list(self._columns) if self._columns is not None else None,
            "fallback_mean": self._fallback_mean,
            "fallback_var": self._fallback_var,
            "mean": dict(self._mean),
            "var": dict(self._var),
            "watermark": dict(self._watermark),
            "history": {entity: dict(buffer) for entity, buffer in self._history.items()},
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> Self:
        """Restore a normalizer that continues advancing exactly where it left off."""
        standardizer = cls(
            half_life=float(state["half_life"]),
            history_buffer=int(state["history_buffer"]),
            eps=float(state["eps"]),
        )
        columns = state["columns"]
        standardizer._columns = list(columns) if columns is not None else None
        standardizer._fallback_mean = state["fallback_mean"]
        standardizer._fallback_var = state["fallback_var"]
        standardizer._mean = dict(state["mean"])
        standardizer._var = dict(state["var"])
        standardizer._watermark = dict(state["watermark"])
        standardizer._history = {
            entity: OrderedDict(buffer) for entity, buffer in state["history"].items()
        }
        return standardizer
