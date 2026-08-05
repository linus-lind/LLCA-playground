"""Instrumented PCA (Kelly-Pruitt-Su 2019) factor estimation.

IPCA maps observable firm characteristics to latent-factor loadings,
``beta(z_{i,t}) = z_{i,t}' Gamma``, and estimates the factor realizations by
alternating least squares.  The factors are estimated from an independent asset
cross-section and can therefore be used to evaluate portfolios whose own models do
not consume firm characteristics.

The restricted model (``intercept=False``, i.e. ``Gamma_alpha = 0``) is used because
the object of interest is the portfolio's time-series alpha against the factors, not
the cross-sectional mispricing test of the characteristics themselves.  Estimation
runs on the available cross-section over the evaluation window; like conventional
Fama-French factors, these factor realizations are known ex post.

Observed characteristics are ranked cross-sectionally by date, and residual missing
ranks receive the neutral exposure zero, so an instrument with no fresh fundamentals
contributes no cross-sectional tilt rather than dropping the whole cross-section.
"""

from __future__ import annotations

import contextlib
import io
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Literal, overload

import numpy as np
import pandas as pd
from ipca import InstrumentedPCA

from llca.data.index_spec import entity_level, time_level

EPS = 1e-12
INDEPENDENCE_TOLERANCE = 1e-7
FeatureMaxAge = int | Mapping[str, object] | None


@dataclass(frozen=True, slots=True)
class IpcaEstimationDiagnostics:
    """Auditable sample-selection and instrument diagnostics for one IPCA fit.

    Counts are sequential where their names start with ``dropped``: a row with a
    missing return and missing characteristics is counted as a return exclusion only.
    Characteristic-value counts such as ``stale_values`` are independent of the row
    selection and therefore describe the full aligned input panel.
    """

    min_characteristic_coverage: float
    requested_factors: int
    estimated_factors: int
    input_observations: int
    input_dates: int
    finite_return_observations: int
    dropped_missing_return: int
    dropped_all_missing_characteristics: int
    dropped_low_characteristic_coverage: int
    dropped_thin_date_observations: int
    dropped_thin_dates: int
    dropped_rank_deficient_date_observations: int
    dropped_rank_deficient_dates: int
    estimation_observations: int
    estimation_dates: int
    configured_characteristics: tuple[str, ...]
    used_characteristics: tuple[str, ...]
    dropped_characteristics: dict[str, str]
    characteristic_observations: dict[str, int]
    neutral_imputations: dict[str, int]
    stale_values: dict[str, int]
    feature_max_age: dict[str, int]

    def to_dict(self) -> dict[str, object]:
        """Flatten these diagnostics into a plain dict for serialization into a manifest."""
        return asdict(self)


def _cross_sectional_ranks(characteristics: pd.DataFrame) -> pd.DataFrame:
    """Map each characteristic to a cross-sectional rank in ``[-0.5, 0.5]`` per date.

    Ranking is done separately within every date and column, using only that column's observed
    values, so instruments are put on a common scale. Missing values stay missing (to be imputed
    to the neutral zero later), and a characteristic with a single observation on a date maps to
    zero because no relative ordering exists.
    """
    time = time_level(characteristics)

    def rank_frame(group: pd.DataFrame) -> pd.DataFrame:
        ranks = group.rank(method="average", na_option="keep")
        counts = group.notna().sum(axis=0)
        scaled = pd.DataFrame(np.nan, index=group.index, columns=group.columns, dtype=float)
        multiple = counts > 1
        if bool(multiple.any()):
            columns = counts.index[multiple]
            scaled.loc[:, columns] = (
                ranks.loc[:, columns]
                .sub(1.0)
                .div(counts.loc[columns].sub(1.0), axis="columns")
                .sub(0.5)
            )
        single = counts == 1
        if bool(single.any()):
            columns = counts.index[single]
            scaled.loc[:, columns] = ranks.loc[:, columns].where(ranks.loc[:, columns].isna(), 0.0)
        return scaled

    return characteristics.groupby(level=time, group_keys=False).apply(rank_frame)


def _validate_age_limit(value: object, location: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{location} must be a non-negative integer")
    limit = int(value)
    if limit < 0:
        raise ValueError(f"{location} must be a non-negative integer")
    return limit


def _resolve_feature_max_age(
    columns: pd.Index, feature_max_age: FeatureMaxAge
) -> dict[object, int]:
    """Expand a ``feature_max_age`` declaration into a per-column age limit map.

    Accepts ``None`` (no limits), a single integer applied to every column, a flat
    ``{column: limit}`` mapping, or a nested ``{default, columns}`` form where ``default`` seeds
    every column and ``columns`` overrides individuals. Validates that limits are non-negative
    integers and that no unknown column or section is referenced, raising ``ValueError``
    otherwise.
    """
    if feature_max_age is None:
        return {}
    if isinstance(feature_max_age, (int, np.integer)) and not isinstance(feature_max_age, bool):
        limit = _validate_age_limit(feature_max_age, "feature_max_age")
        return {column: limit for column in columns}
    if not isinstance(feature_max_age, Mapping):
        raise ValueError(
            "feature_max_age must be a non-negative integer or a mapping of columns to limits"
        )

    declaration = dict(feature_max_age)
    nested = "default" in declaration or "columns" in declaration
    if nested:
        unknown_sections = set(declaration).difference({"default", "columns"})
        if unknown_sections:
            raise ValueError(
                "feature_max_age has unknown sections: "
                + ", ".join(sorted(str(key) for key in unknown_sections))
            )
        default = declaration.get("default")
        resolved = (
            {column: _validate_age_limit(default, "feature_max_age.default") for column in columns}
            if default is not None
            else {}
        )
        overrides = declaration.get("columns", {})
        if not isinstance(overrides, Mapping):
            raise ValueError("feature_max_age.columns must be a mapping")
        raw_overrides = dict(overrides)
    else:
        resolved = {}
        raw_overrides = declaration

    unknown_columns = set(raw_overrides).difference(columns)
    if unknown_columns:
        raise ValueError(
            "feature_max_age references unknown characteristics: "
            + ", ".join(sorted(str(column) for column in unknown_columns))
        )
    for column, value in raw_overrides.items():
        resolved[column] = _validate_age_limit(
            value, f"feature_max_age.columns.{column}" if nested else f"feature_max_age.{column}"
        )
    return resolved


def _numeric_characteristics(characteristics: pd.DataFrame) -> pd.DataFrame:
    try:
        numeric = characteristics.astype(float)
    except (TypeError, ValueError) as error:
        raise ValueError("IPCA characteristics must be numeric") from error
    return numeric.replace([np.inf, -np.inf], np.nan)


def _apply_age_limits(
    characteristics: pd.DataFrame,
    characteristic_ages: pd.DataFrame | None,
    age_limits: Mapping[object, int],
) -> tuple[pd.DataFrame, dict[str, int]]:
    stale_counts = {str(column): 0 for column in characteristics.columns}
    if not age_limits:
        return characteristics, stale_counts
    if characteristic_ages is None:
        raise ValueError("characteristic_ages is required when feature_max_age is configured")
    if not characteristic_ages.index.is_unique:
        raise ValueError("characteristic_ages index must be unique")
    if not characteristic_ages.columns.is_unique:
        raise ValueError("characteristic_ages columns must be unique")

    missing_columns = set(age_limits).difference(characteristic_ages.columns)
    if missing_columns:
        raise ValueError(
            "characteristic_ages is missing capped characteristics: "
            + ", ".join(sorted(str(column) for column in missing_columns))
        )

    result = characteristics.copy()
    ages = characteristic_ages.reindex(result.index)
    for column, limit in age_limits.items():
        try:
            age = pd.to_numeric(ages[column], errors="raise")
        except (TypeError, ValueError) as error:
            raise ValueError(f"characteristic age for '{column}' must be numeric") from error
        invalid_age = age.isna() | (age < 0) | (age > limit)
        stale = result[column].notna() & invalid_age
        stale_counts[str(column)] = int(stale.sum())
        result.loc[invalid_age, column] = np.nan
    return result, stale_counts


def _matrix_rank(matrix: np.ndarray, tolerance: float = INDEPENDENCE_TOLERANCE) -> int:
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    if singular_values.size == 0 or singular_values[0] <= EPS:
        return 0
    return int(np.sum(singular_values > tolerance * singular_values[0]))


def _independent_instruments(
    instruments: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Keep a full-rank subset of the ranked instruments, honouring the configured order.

    Walking the columns in order (against a constant), each is retained only if it adds a new
    linearly independent direction; constant or redundant columns are discarded. Returns the
    kept instruments and a mapping of every dropped column to the reason it was removed, so no
    specification change happens silently.
    """
    constant = np.ones((len(instruments), 1), dtype=float)
    design = constant
    rank = 1
    keep: list[object] = []
    dropped: dict[str, str] = {}

    for column in instruments.columns:
        values = instruments[column].to_numpy(dtype=float)
        if float(np.ptp(values)) <= EPS:
            dropped[str(column)] = "zero_variance_after_ranking"
            continue
        candidate = np.column_stack([design, values])
        candidate_rank = _matrix_rank(candidate)
        if candidate_rank <= rank:
            dropped[str(column)] = "linearly_dependent_after_ranking"
            continue
        keep.append(column)
        design = candidate
        rank = candidate_rank

    return instruments.loc[:, keep], dropped


@overload
def estimate_ipca_factors(
    returns: pd.Series,
    characteristics: pd.DataFrame,
    *,
    n_factors: int,
    min_characteristic_coverage: float = 0.5,
    characteristic_ages: pd.DataFrame | None = None,
    feature_max_age: FeatureMaxAge = None,
    return_diagnostics: Literal[False] = False,
) -> pd.DataFrame: ...


@overload
def estimate_ipca_factors(
    returns: pd.Series,
    characteristics: pd.DataFrame,
    *,
    n_factors: int,
    min_characteristic_coverage: float = 0.5,
    characteristic_ages: pd.DataFrame | None = None,
    feature_max_age: FeatureMaxAge = None,
    return_diagnostics: Literal[True],
) -> tuple[pd.DataFrame, IpcaEstimationDiagnostics]: ...


def estimate_ipca_factors(
    returns: pd.Series,
    characteristics: pd.DataFrame,
    *,
    n_factors: int,
    min_characteristic_coverage: float = 0.5,
    characteristic_ages: pd.DataFrame | None = None,
    feature_max_age: FeatureMaxAge = None,
    return_diagnostics: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, IpcaEstimationDiagnostics]:
    """Estimate ``n_factors`` latent IPCA return factors from a characteristics panel.

    Rows with non-finite returns, no observed characteristics, or coverage below
    ``min_characteristic_coverage`` are dropped — returns are never imputed. The survivors are
    cross-sectionally ranked, their remaining gaps set to the neutral zero exposure, and dates
    too thin or rank-deficient to support the requested factors are removed. Optionally,
    ``characteristic_ages`` with ``feature_max_age`` blanks stale carried values before coverage
    is measured. A restricted (zero-intercept) InstrumentedPCA is then fit, yielding a
    date-by-factor frame. The sample-selection diagnostics are always stored on
    ``DataFrame.attrs['ipca_diagnostics']`` and, when ``return_diagnostics`` is true, also
    returned as a typed object. Raises ``ValueError`` if the panel is unusable or over-requested.
    """
    n_factors = int(n_factors)
    if characteristics.empty or characteristics.shape[1] == 0:
        raise ValueError("IPCA characteristics panel must contain at least one column")
    if not characteristics.index.is_unique:
        raise ValueError("IPCA characteristics index must be unique")
    if not characteristics.columns.is_unique:
        raise ValueError("IPCA characteristics columns must be unique")
    if not returns.index.is_unique:
        raise ValueError("IPCA returns index must be unique")

    time = time_level(characteristics)
    entity = entity_level(characteristics)
    if entity is None:
        raise ValueError("IPCA estimation requires an entity-indexed characteristics panel")

    age_limits = _resolve_feature_max_age(characteristics.columns, feature_max_age)
    characteristic_values = _numeric_characteristics(characteristics)
    characteristic_values, stale_values = _apply_age_limits(
        characteristic_values, characteristic_ages, age_limits
    )

    aligned_returns = pd.to_numeric(returns.reindex(characteristic_values.index), errors="coerce")
    finite_return = pd.Series(
        np.isfinite(aligned_returns.to_numpy(dtype=float)), index=characteristic_values.index
    )
    observed_count = characteristic_values.notna().sum(axis=1)
    coverage = observed_count.div(characteristic_values.shape[1])
    all_missing = observed_count == 0
    low_coverage = coverage < float(min_characteristic_coverage)

    eligible = finite_return & ~all_missing & ~low_coverage
    dropped_missing_return = int((~finite_return).sum())
    dropped_all_missing = int((finite_return & all_missing).sum())
    dropped_low_coverage = int((finite_return & ~all_missing & low_coverage).sum())

    selected_characteristics = characteristic_values.loc[eligible]
    response = aligned_returns.loc[eligible].astype(float)
    if selected_characteristics.empty:
        raise ValueError(
            "IPCA panel is empty after filtering returns, characteristic coverage, and ages"
        )

    ranked = _cross_sectional_ranks(selected_characteristics)
    neutral_imputations = {
        str(column): int(ranked[column].isna().sum()) for column in ranked.columns
    }
    ranked = ranked.fillna(0.0)

    # Each date needs more names than requested factors.  Thin dates are removed, but
    # the requested factor count is never silently reduced.
    date_sizes = ranked.groupby(level=time).size()
    thin_dates = date_sizes.index[date_sizes <= n_factors]
    thin = ranked.index.get_level_values(time).isin(thin_dates)
    dropped_thin_observations = int(thin.sum())
    if dropped_thin_observations:
        ranked = ranked.loc[~thin]
        response = response.loc[ranked.index]
    if ranked.empty:
        raise ValueError(
            f"IPCA panel has no date with more than {n_factors} cross-sectional observations"
        )

    working = ranked
    dropped_characteristics: dict[str, str] = {}
    dropped_rank_deficient_observations = 0
    dropped_rank_deficient_dates: list[object] = []
    while True:
        selected, newly_dropped = _independent_instruments(working)
        dropped_characteristics.update(newly_dropped)
        if selected.shape[1] == 0:
            details = ", ".join(
                f"{name} ({reason})" for name, reason in dropped_characteristics.items()
            )
            raise ValueError(f"IPCA has no usable characteristic instrument: {details}")
        supported_factors = selected.shape[1] + 1  # explicit constant instrument
        if n_factors > supported_factors:
            details = ", ".join(
                f"{name} ({reason})" for name, reason in dropped_characteristics.items()
            )
            suffix = f"; dropped: {details}" if details else ""
            raise ValueError(
                f"IPCA requests {n_factors} factors but the fixed instrument set supports at "
                f"most {supported_factors}{suffix}"
            )

        instruments = selected.copy()
        constant_column = "__ipca_constant__"
        while constant_column in instruments.columns:
            constant_column += "_"
        instruments[constant_column] = 1.0
        date_ranks = instruments.groupby(level=time, sort=False).apply(
            lambda group: _matrix_rank(group.to_numpy(dtype=float))
        )
        deficient_dates = date_ranks.index[date_ranks < n_factors]
        if deficient_dates.empty:
            break
        deficient = instruments.index.get_level_values(time).isin(deficient_dates)
        dropped_rank_deficient_observations += int(deficient.sum())
        dropped_rank_deficient_dates.extend(deficient_dates.tolist())
        working = working.loc[~deficient]
        response = response.loc[working.index]
        if working.empty:
            raise ValueError(
                "IPCA panel has no date whose final characteristic-instrument rank supports "
                f"the requested {n_factors} factors"
            )

    used_characteristics = tuple(str(column) for column in selected.columns)

    # The estimator expects an (entity, time) MultiIndex and returns a (K, T) factor
    # matrix over sorted unique time values.
    ordered = instruments.swaplevel(time, entity).sort_index()
    y = response.swaplevel(time, entity).reindex(ordered.index)

    model = InstrumentedPCA(n_factors=n_factors, intercept=False)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        model = model.fit(X=ordered, y=y)

    times = np.sort(ordered.index.get_level_values(time).unique())
    factor_values = np.asarray(model.Factors, dtype=float)
    if factor_values.ndim == 1:
        factor_values = factor_values.reshape(1, -1)
    if factor_values.shape != (n_factors, len(times)):
        raise ValueError(
            "IPCA estimator returned an unexpected factor matrix shape: "
            f"{factor_values.shape}, expected {(n_factors, len(times))}"
        )
    factors = pd.DataFrame(
        factor_values.T,
        index=pd.Index(times, name=time),
        columns=[f"ipca_{i + 1}" for i in range(n_factors)],
    ).sort_index()

    diagnostics = IpcaEstimationDiagnostics(
        min_characteristic_coverage=float(min_characteristic_coverage),
        requested_factors=n_factors,
        estimated_factors=n_factors,
        input_observations=len(characteristic_values),
        input_dates=characteristic_values.index.get_level_values(time).nunique(),
        finite_return_observations=int(finite_return.sum()),
        dropped_missing_return=dropped_missing_return,
        dropped_all_missing_characteristics=dropped_all_missing,
        dropped_low_characteristic_coverage=dropped_low_coverage,
        dropped_thin_date_observations=dropped_thin_observations,
        dropped_thin_dates=len(thin_dates),
        dropped_rank_deficient_date_observations=dropped_rank_deficient_observations,
        dropped_rank_deficient_dates=len(dropped_rank_deficient_dates),
        estimation_observations=len(ordered),
        estimation_dates=len(times),
        configured_characteristics=tuple(str(column) for column in characteristics.columns),
        used_characteristics=used_characteristics,
        dropped_characteristics=dropped_characteristics,
        characteristic_observations={
            str(column): int(characteristic_values[column].notna().sum())
            for column in characteristic_values.columns
        },
        neutral_imputations=neutral_imputations,
        stale_values=stale_values,
        feature_max_age={str(column): limit for column, limit in age_limits.items()},
    )
    factors.attrs["ipca_diagnostics"] = diagnostics.to_dict()
    if return_diagnostics:
        return factors, diagnostics
    return factors
