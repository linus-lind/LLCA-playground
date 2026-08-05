from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew

from llca.data.index_spec import time_level

EPS = 1e-12


def significance_stars(p_value: float) -> str:
    """Map a p-value to the star notation used throughout the report.

    Returns ``***``, ``**``, or ``*`` when the p-value falls below the 1%, 5%, or 10%
    threshold respectively, and an empty string when it clears none of them or is not finite.
    """
    if not np.isfinite(p_value):
        return ""
    if p_value < 0.01:
        return "***"
    if p_value < 0.05:
        return "**"
    if p_value < 0.10:
        return "*"
    return ""


def significance_marker(p_value: float) -> str:
    """Format a p-value's stars as a suffix to append after a printed value.

    The stars are wrapped in parentheses and prefixed with a space (`` (***)``) so they read
    cleanly next to the number; a p-value that clears no level yields an empty suffix. The
    output is plain text and renders identically in every export format.
    """
    stars = significance_stars(p_value)
    return f" ({stars})" if stars else ""


def significance_marker_bold(p_value: float) -> str:
    """Variant of :func:`significance_marker` whose stars render bold in matplotlib.

    Because matplotlib can only embolden a substring via mathtext, the stars are emitted as a
    ``$\\mathbf{...}$`` fragment with the parentheses left outside the math group so they stay
    upright. Negative thin spaces close the gaps between adjacent stars. An empty suffix is
    returned when no level is cleared.
    """
    stars = significance_stars(p_value)
    if not stars:
        return ""
    inner = r"\!\!".join([r"\ast"] * len(stars))
    return rf" ($\mathbf{{{inner}}}$)"


def shape_statistics(values: pd.Series) -> tuple[float, float]:
    """Return the sample skewness and excess kurtosis of ``values`` as a ``(skew, kurtosis)`` pair.

    Both moments use the bias-corrected estimators. They are only meaningful with enough spread
    and enough observations, so the pair is ``(nan, nan)`` when there are fewer than four values
    or the standard deviation is effectively zero.
    """
    if len(values) < 4 or float(values.std(ddof=1)) <= EPS:
        return float("nan"), float("nan")
    array = values.to_numpy(dtype=float)
    return (
        float(skew(array, bias=False)),
        float(kurtosis(array, fisher=True, bias=False)),
    )


def rank_buckets(scores: pd.Series, bucket_count: int, *, pooled: bool = False) -> pd.Series:
    """Sort ``scores`` into ordered rank buckets labelled ``1`` (lowest) upward.

    Two regimes are handled explicitly, where ``N`` is a decision set's observation count and
    ``K`` is ``bucket_count``:

    * ``N >= K`` forms **equal-count** groups via ``bucket(r) = floor((r - 1) * K / N) + 1`` for
      ordinal rank ``r`` in ``1..N``, so bucket sizes differ by at most one (e.g. ``N=100, K=5``
      gives five buckets of twenty).
    * ``N < K`` spreads the few observations across the full ``[1, K]`` label range via
      ``floor((r - 1) * (K - 1) / (N - 1)) + 1`` (e.g. ``N=2, K=10`` labels ``1`` and ``10``) so
      the requested range is still represented rather than collapsed into the low buckets.

    A panel with an entity level is ranked within each date so a bucket never spans dates, unless
    ``pooled`` is set, in which case every observation is ranked together; a date-only series is
    always ranked as a whole. The series is ordered by index before ranking and ties are broken
    ordinally, so the labelling is deterministic and invariant to incoming row order. Missing
    scores are never ranked or counted, and an observation alone on its date has no relative rank
    and is left unlabelled (``<NA>``) rather than assigned a bucket.
    """
    order = scores.index.argsort(kind="stable")
    ordered = scores.iloc[order]
    if not pooled and ordered.index.nlevels > 1:
        time = time_level(ordered)
        ranks = ordered.groupby(level=time).rank(method="first")
        counts = ordered.groupby(level=time).transform("count").astype(float)
    else:
        ranks = ordered.rank(method="first")
        counts = pd.Series(float(ordered.notna().sum()), index=ordered.index)
    equal_count = np.floor((ranks - 1.0) * bucket_count / counts + EPS) + 1.0
    span_denominator = (counts - 1.0).where(counts > 1, other=1.0)
    spanned = np.floor((ranks - 1.0) * (bucket_count - 1) / span_denominator + EPS) + 1.0
    numbers = equal_count.where(counts >= bucket_count, spanned)
    # A singleton decision set has no relative rank and must not be forced into a bucket.
    numbers = numbers.where(counts > 1).clip(1, bucket_count)
    # Restore the caller's original row order positionally, so a duplicate index never raises.
    restored = np.full(len(scores), np.nan)
    restored[order] = numbers.to_numpy(dtype=float)
    return pd.Series(restored, index=scores.index, name="bucket").astype("Int64")
