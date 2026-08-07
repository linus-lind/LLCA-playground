"""Typed, safe search-space representation shared by grid and random search.

Each dimension exposes both a deterministic enumeration (``grid_values``) and a seeded draw
(``sample``), so one representation serves every search method without evaluating arbitrary
strings from configuration. Values are plain scalars (numbers, strings, or ``None``) that a
scikit-learn constructor accepts directly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

type ParameterValue = float | int | str | bool | None


@runtime_checkable
class SearchDimension(Protocol):
    """One named hyperparameter axis that can be enumerated or sampled."""

    @property
    def name(self) -> str:
        """The hyperparameter this dimension varies."""
        ...

    def grid_values(self) -> tuple[ParameterValue, ...]:
        """Return the deterministic set of values enumerated by grid search."""
        ...

    def sample(self, rng: np.random.Generator) -> ParameterValue:
        """Draw one value for random search from the configured distribution."""
        ...


@dataclass(frozen=True, slots=True)
class ChoiceDimension:
    """A discrete set of candidate values, enumerated for grid and sampled uniformly."""

    name: str
    values: tuple[ParameterValue, ...]

    def grid_values(self) -> tuple[ParameterValue, ...]:
        return self.values

    def sample(self, rng: np.random.Generator) -> ParameterValue:
        return self.values[int(rng.integers(len(self.values)))]


@dataclass(frozen=True, slots=True)
class LogRangeDimension:
    """A positive range explored on a base-10 logarithmic scale.

    Grid search enumerates ``num`` log-spaced points inclusive of both endpoints; random search
    draws log-uniformly over ``[low, high]``.
    """

    name: str
    low: float
    high: float
    num: int

    def grid_values(self) -> tuple[ParameterValue, ...]:
        points = np.logspace(math.log10(self.low), math.log10(self.high), self.num)
        return tuple(float(value) for value in points)

    def sample(self, rng: np.random.Generator) -> ParameterValue:
        return float(10.0 ** rng.uniform(math.log10(self.low), math.log10(self.high)))


@dataclass(frozen=True, slots=True)
class IntRangeDimension:
    """An inclusive integer range, enumerated for grid and sampled uniformly for random."""

    name: str
    low: int
    high: int

    def grid_values(self) -> tuple[ParameterValue, ...]:
        return tuple(range(self.low, self.high + 1))

    def sample(self, rng: np.random.Generator) -> ParameterValue:
        return int(rng.integers(self.low, self.high + 1))


@dataclass(frozen=True, slots=True)
class SearchSpace:
    """An ordered collection of independent search dimensions."""

    dimensions: tuple[SearchDimension, ...]

    def names(self) -> tuple[str, ...]:
        return tuple(dimension.name for dimension in self.dimensions)

    def grid_size(self) -> int:
        size = 1
        for dimension in self.dimensions:
            size *= len(dimension.grid_values())
        return size

    def is_empty(self) -> bool:
        return len(self.dimensions) == 0
