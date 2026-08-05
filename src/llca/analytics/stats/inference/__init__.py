"""Inferential statistics for signal quality and model comparison.

This package isolates the hypothesis tests used by the analytics report from any pandas
plumbing: every function takes plain arrays (or already-aligned daily series) and returns
finite floats or small frames. Autocorrelation in daily financial series is handled with
Newey-West heteroskedasticity-and-autocorrelation-consistent (HAC) variances, and the
tests that lack a trustworthy closed form (Sharpe ratios, the model confidence set) use a
stationary bootstrap so the reported p-values stay valid under serial dependence.

The shared HAC and bootstrap machinery lives in :mod:`.foundation`; the test families are
split into :mod:`.predictive`, :mod:`.performance`, :mod:`.model_comparison`, and
:mod:`.multiple_testing`. Every public name is re-exported here so callers continue to use
``from llca.analytics.stats import inference`` and ``inference.<test>`` unchanged.

References
----------
Pesaran & Timmermann (1992); Anatolyev & Gerko (2005); Newey & West (1987, 1994);
Lo (2002); Diebold & Mariano (1995) with the Harvey, Leybourne & Newbold (1997)
small-sample correction; Jobson & Korkie (1981) / Memmel (2003);
Politis & Romano (1994); Hansen, Lunde & Nason (2011).
"""

from llca.analytics.stats.inference.foundation import (
    Alternative,
    MeanTest,
    hac_mean_test,
    long_run_variance,
    newey_west_bandwidth,
    stationary_bootstrap_indices,
)
from llca.analytics.stats.inference.model_comparison import (
    diebold_mariano,
    model_confidence_set,
    sharpe_difference,
)
from llca.analytics.stats.inference.multiple_testing import (
    adjust_pairwise,
    benjamini_hochberg,
    holm_adjust,
)
from llca.analytics.stats.inference.performance import sharpe_ratio, sharpe_significance
from llca.analytics.stats.inference.predictive import (
    directional_accuracy_test,
    excess_profitability_test,
    information_coefficient_test,
    pesaran_timmermann,
)

__all__ = [
    "Alternative",
    "MeanTest",
    "adjust_pairwise",
    "benjamini_hochberg",
    "diebold_mariano",
    "directional_accuracy_test",
    "excess_profitability_test",
    "hac_mean_test",
    "holm_adjust",
    "information_coefficient_test",
    "long_run_variance",
    "model_confidence_set",
    "newey_west_bandwidth",
    "pesaran_timmermann",
    "sharpe_difference",
    "sharpe_ratio",
    "sharpe_significance",
    "stationary_bootstrap_indices",
]
