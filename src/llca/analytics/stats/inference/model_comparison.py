"""Cross-model comparison tests: equal predictive accuracy, Sharpe equality, confidence set.

The Diebold-Mariano and Sharpe-difference tests operate on paired daily series; the model
confidence set eliminates dominated models from a date-by-model loss frame. All three use the
module's shared HAC and stationary-bootstrap machinery so serial dependence is respected.

References
----------
Diebold & Mariano (1995) with the Harvey, Leybourne & Newbold (1997) small-sample correction;
Jobson & Korkie (1981) / Memmel (2003); Hansen, Lunde & Nason (2011).
"""

from __future__ import annotations

import contextlib
import io

import numpy as np
import pandas as pd
from arch.bootstrap import MCS

from llca.analytics.stats.inference.foundation import (
    _p_value,
    long_run_variance,
    newey_west_bandwidth,
    stationary_bootstrap_indices,
)
from llca.analytics.stats.statistics import EPS


def _finite_pairs(left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    a = np.asarray(left, dtype=float)
    b = np.asarray(right, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3:
        return None
    return a[mask], b[mask]


def diebold_mariano(
    loss_a: np.ndarray,
    loss_b: np.ndarray,
    *,
    horizon: int = 1,
    lag: int | None = None,
) -> dict[str, float]:
    """Compare two forecasts' accuracy from their daily losses (Diebold-Mariano, 1995).

    Works on the per-date loss differential ``loss_a - loss_b``; a positive mean indicates
    model A loses more. The mean is standardised by a Bartlett-kernel long-run standard error
    (truncation lag ``lag``, or an automatic bandwidth respecting the forecast ``horizon``),
    scaled by the Harvey-Leybourne-Newbold (1997) small-sample factor, and graded two-sided
    against a Student-t with ``n - 1`` degrees of freedom. Fewer than three overlapping finite
    observations return ``nan``.
    """
    pair = _finite_pairs(loss_a, loss_b)
    if pair is None:
        return {
            "dm_statistic": float("nan"),
            "dm_p_value": float("nan"),
            "mean_difference": float("nan"),
        }
    diff = pair[0] - pair[1]
    n = diff.shape[0]
    resolved_lag = max(horizon - 1, newey_west_bandwidth(n)) if lag is None else lag
    variance = long_run_variance(diff, resolved_lag)
    mean = float(diff.mean())
    std_error = float(np.sqrt(variance / n)) if variance > EPS else float("nan")
    statistic = mean / std_error if std_error and std_error > EPS else float("nan")
    correction = np.sqrt((n + 1 - 2 * horizon + horizon * (horizon - 1) / n) / n)
    adjusted = statistic * correction
    return {
        "dm_statistic": float(adjusted),
        "dm_p_value": _p_value(float(adjusted), "two-sided", df=n - 1),
        "mean_difference": mean,
    }


def sharpe_difference(
    returns_a: np.ndarray,
    returns_b: np.ndarray,
    *,
    annualization_periods: int,
    n_boot: int = 2000,
    block_length: float = 10.0,
    seed: int = 0,
) -> dict[str, float]:
    """Test whether two return series share the same Sharpe ratio.

    Reports the annualized gap between the two Sharpe ratios alongside two p-values: the
    Jobson-Korkie / Memmel (2003) closed-form z-test, which assumes joint normality, and a
    paired stationary-bootstrap p-value that resamples both series on common indices
    (``n_boot`` resamples, mean block length ``block_length``, seeded by ``seed``) to preserve
    their autocorrelation and cross-correlation. Fewer than three overlapping finite
    observations return ``nan``.
    """
    pair = _finite_pairs(returns_a, returns_b)
    if pair is None:
        return {
            "delta_sharpe": float("nan"),
            "memmel_p_value": float("nan"),
            "bootstrap_p_value": float("nan"),
        }
    a, b = pair
    n = a.shape[0]
    sr_a = _period_sharpe(a)
    sr_b = _period_sharpe(b)
    correlation = float(np.corrcoef(a, b)[0, 1]) if n > 1 else float("nan")
    variance = (
        2.0 - 2.0 * correlation + 0.5 * (sr_a**2 + sr_b**2 - 2.0 * sr_a * sr_b * correlation**2)
    ) / n
    z_statistic = (sr_a - sr_b) / np.sqrt(variance) if variance > EPS else float("nan")
    bootstrap_p = float("nan")
    if n_boot > 0 and n >= 2:
        rng = np.random.default_rng(seed)
        indices = stationary_bootstrap_indices(n, block_length, n_boot, rng)
        deltas = _period_sharpe_batch(a[indices]) - _period_sharpe_batch(b[indices])
        finite = deltas[np.isfinite(deltas)]
        if finite.size:
            observed = sr_a - sr_b
            centred_distance = np.abs(finite - observed)
            exceedances = int(np.count_nonzero(centred_distance + EPS >= abs(observed)))
            # The add-one correction avoids a zero Monte-Carlo p-value and returns one for
            # identical series, where both the observed and every resampled difference are zero.
            bootstrap_p = float((exceedances + 1) / (finite.size + 1))
    return {
        "delta_sharpe": (sr_a - sr_b) * float(np.sqrt(annualization_periods)),
        "memmel_z_statistic": float(z_statistic),
        "memmel_p_value": _p_value(float(z_statistic), "two-sided", df=None),
        "bootstrap_p_value": bootstrap_p,
    }


def _period_sharpe(returns: np.ndarray) -> float:
    """Return the raw, per-period Sharpe ratio (mean over standard deviation) of one series.

    Left unannualized for use inside the bootstrap; ``nan`` when the series has fewer than two
    points or no dispersion.
    """
    array = np.asarray(returns, dtype=float)
    if array.shape[0] < 2:
        return float("nan")
    std = float(array.std(ddof=1))
    return float(array.mean() / std) if std > EPS else float("nan")


def _period_sharpe_batch(samples: np.ndarray) -> np.ndarray:
    """Vectorised :func:`_period_sharpe` over each row of a ``[n_boot, n]`` resample matrix.

    Returns one per-period Sharpe ratio per row, with ``nan`` for any row lacking dispersion.
    """
    mean = samples.mean(axis=1)
    std = samples.std(axis=1, ddof=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(std > EPS, mean / std, np.nan)


def model_confidence_set(
    losses: pd.DataFrame,
    alpha: float,
    n_boot: int,
    block_length: float,
    seed: int,
) -> pd.DataFrame:
    """Identify the models statistically indistinguishable from the best (Hansen-Lunde-Nason, 2011).

    ``losses`` is a date-by-model frame of per-date losses, lower being better; rows with any
    missing value are dropped. Elimination runs on :class:`arch.bootstrap.MCS` with the range
    statistic and a stationary bootstrap (``n_boot`` reps, block length ``block_length``, seed
    ``seed``), at test size ``alpha``. Models with identical loss paths are collapsed to a
    single representative first so duplicates share a verdict. Returns a frame, indexed by the
    original columns, of each model's MCS p-value, its membership flag, and its mean loss;
    trivial cases (one model, or too few usable dates) short-circuit to a degenerate result.
    """
    clean = losses.dropna(axis=0, how="any")
    models = list(losses.columns)
    n_dates, n_models = clean.shape
    source = clean if n_dates else losses
    mean_loss = {model: float(source[model].mean()) for model in models}
    if n_models < 2 or n_dates < 2:
        # One model, or too few usable dates to run the bootstrap: no model can be eliminated,
        # so the confidence set retains every model. The Hansen-Lunde-Nason MCS is never empty
        # (it always contains at least the best performer); p-values are undefined here.
        return pd.DataFrame(
            {
                "mcs_p_value": {model: float("nan") for model in models},
                "in_confidence_set": {model: True for model in models},
                "mean_loss": mean_loss,
            }
        )
    representatives: list[object] = []
    representative_for: dict[object, object] = {}
    for model in models:
        representative = next(
            (
                candidate
                for candidate in representatives
                if np.allclose(
                    clean[model].to_numpy(dtype=float),
                    clean[candidate].to_numpy(dtype=float),
                    rtol=1e-12,
                    atol=1e-14,
                )
            ),
            None,
        )
        if representative is None:
            representatives.append(model)
            representative = model
        representative_for[model] = representative

    if len(representatives) == 1:
        return pd.DataFrame(
            {
                "mcs_p_value": {model: 1.0 for model in models},
                "in_confidence_set": {model: True for model in models},
                "mean_loss": mean_loss,
            }
        )

    unique_losses = clean[representatives]
    mcs = MCS(
        unique_losses,
        size=alpha,
        reps=n_boot,
        block_size=max(int(round(block_length)), 1),
        method="R",
        seed=seed,
    )
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        mcs.compute()
    p_by_model = {name: float(value) for name, value in mcs.pvalues["Pvalue"].items()}
    included = set(mcs.included)
    return pd.DataFrame(
        {
            "mcs_p_value": {
                model: p_by_model.get(representative_for[model], float("nan")) for model in models
            },
            "in_confidence_set": {model: representative_for[model] in included for model in models},
            "mean_loss": mean_loss,
        }
    )
