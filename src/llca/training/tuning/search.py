"""Deterministic and reproducible candidate generation for hyperparameter search.

Candidate generation is deliberately separated from fold evaluation: a search method turns a
:class:`SearchSpace` into a list of complete parameter mappings, each formed by overlaying the
searched values on the baseline so every candidate is self-contained over the tunable keys.
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping

import numpy as np

from llca.training.tuning.search_space import ParameterValue, SearchSpace
from llca.training.tuning.settings import SearchSettings

SEARCH_METHODS: tuple[str, ...] = ("grid", "random")


def _complete(
    baseline: Mapping[str, ParameterValue], overrides: Mapping[str, ParameterValue]
) -> dict[str, ParameterValue]:
    """Overlay searched values on the baseline to form a complete tunable mapping."""
    return {**baseline, **overrides}


def _candidate_key(candidate: Mapping[str, ParameterValue]) -> tuple[tuple[str, object], ...]:
    return tuple(sorted(candidate.items()))


def generate_candidates(
    search: SearchSettings,
    space: SearchSpace,
    baseline: Mapping[str, ParameterValue],
) -> list[dict[str, ParameterValue]]:
    """Return the candidate hyperparameter mappings to evaluate, excluding the baseline itself.

    ``grid`` enumerates the Cartesian product of every dimension's values in a stable order.
    ``random`` draws ``n_trials`` unique candidates from the seeded generator, silently skipping
    repeated draws; if the space cannot yield that many distinct points, the unique candidates
    found are returned. Every candidate is completed against ``baseline`` over the tunable keys.
    """
    if space.is_empty():
        raise ValueError("cannot generate candidates from an empty search space")
    if search.method == "grid":
        names = space.names()
        combinations = itertools.product(
            *(dimension.grid_values() for dimension in space.dimensions)
        )
        return [_complete(baseline, dict(zip(names, combo, strict=True))) for combo in combinations]
    if search.method == "random":
        if search.n_trials <= 0:
            raise ValueError(f"random search requires n_trials >= 1, got {search.n_trials}")
        rng = np.random.default_rng(search.seed)
        seen: set[tuple[tuple[str, object], ...]] = set()
        candidates: list[dict[str, ParameterValue]] = []
        max_attempts = search.n_trials * 50 + 100
        for _ in range(max_attempts):
            if len(candidates) >= search.n_trials:
                break
            drawn = {dimension.name: dimension.sample(rng) for dimension in space.dimensions}
            candidate = _complete(baseline, drawn)
            key = _candidate_key(candidate)
            if key in seen:
                continue
            seen.add(key)
            candidates.append(candidate)
        return candidates
    raise ValueError(
        f"unknown search method '{search.method}', expected one of {list(SEARCH_METHODS)}"
    )
