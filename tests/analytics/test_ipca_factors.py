from __future__ import annotations

import json
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from llca.analytics.factors import estimate_ipca_factors


class _CapturingInstrumentedPCA:
    """Small deterministic stand-in that exposes the estimator's prepared sample."""

    last_x: pd.DataFrame | None = None
    last_y: pd.Series | None = None

    def __init__(self, n_factors: int, intercept: bool) -> None:
        self.n_factors = n_factors
        self.intercept = intercept
        self.Factors = np.empty((n_factors, 0))

    def fit(self, X: pd.DataFrame, y: pd.Series) -> _CapturingInstrumentedPCA:
        type(self).last_x = X.copy()
        type(self).last_y = y.copy()
        periods = X.index.get_level_values("date").nunique()
        self.Factors = np.arange(self.n_factors * periods, dtype=float).reshape(
            self.n_factors, periods
        )
        return self


class IpcaMissingPolicyTest(unittest.TestCase):
    def setUp(self) -> None:
        _CapturingInstrumentedPCA.last_x = None
        _CapturingInstrumentedPCA.last_y = None

    @staticmethod
    def _missing_panel() -> tuple[pd.Series, pd.DataFrame]:
        rng = np.random.default_rng(42)
        dates = pd.bdate_range("2021-01-04", periods=3)
        entities = np.arange(1, 8)
        index = pd.MultiIndex.from_product([dates, entities], names=["date", "instrument_id"])
        characteristics = pd.DataFrame(
            rng.normal(size=(len(index), 4)), index=index, columns=["c0", "c1", "c2", "c3"]
        )
        returns = pd.Series(rng.normal(size=len(index)), index=index, name="return")

        # Entity 2 remains eligible with 75% coverage and must receive neutral c0.
        characteristics.loc[(slice(None), 2), "c0"] = np.nan
        # No-fundamental rows and rows below 50% coverage leave the factor cross-section.
        characteristics.loc[(slice(None), 6), :] = np.nan
        characteristics.loc[(slice(None), 7), ["c1", "c2", "c3"]] = np.nan
        # Missing or infinite returns leave independently, irrespective of characteristics.
        returns.loc[(slice(None), 5)] = np.inf
        return returns, characteristics

    def test_rank_neutral_policy_preserves_partial_rows_and_reports_exclusions(self) -> None:
        returns, characteristics = self._missing_panel()

        with patch("llca.analytics.factors.ipca.InstrumentedPCA", _CapturingInstrumentedPCA):
            factors, diagnostics = estimate_ipca_factors(
                returns,
                characteristics,
                n_factors=1,
                min_characteristic_coverage=0.5,
                return_diagnostics=True,
            )

        prepared = _CapturingInstrumentedPCA.last_x
        assert prepared is not None
        self.assertEqual(set(prepared.index.get_level_values("instrument_id")), {1, 2, 3, 4})
        # Missing c0 was filled only after ranking and is therefore the neutral exposure.
        self.assertTrue((prepared.loc[(2, slice(None)), "c0"] == 0.0).all())
        for date in factors.index:
            observed = prepared.loc[([1, 3, 4], date), "c0"].sort_values().to_numpy()
            np.testing.assert_allclose(observed, [-0.5, 0.0, 0.5])

        self.assertEqual(diagnostics.input_observations, 21)
        self.assertEqual(diagnostics.finite_return_observations, 18)
        self.assertEqual(diagnostics.dropped_missing_return, 3)
        self.assertEqual(diagnostics.dropped_all_missing_characteristics, 3)
        self.assertEqual(diagnostics.dropped_low_characteristic_coverage, 3)
        self.assertEqual(diagnostics.estimation_observations, 12)
        self.assertEqual(diagnostics.neutral_imputations["c0"], 3)
        self.assertEqual(factors.attrs["ipca_diagnostics"], diagnostics.to_dict())
        json.dumps(factors.attrs["ipca_diagnostics"])

    def test_feature_age_caps_invalidate_values_before_coverage_and_ranking(self) -> None:
        rng = np.random.default_rng(7)
        dates = pd.bdate_range("2022-02-01", periods=3)
        index = pd.MultiIndex.from_product(
            [dates, np.arange(1, 7)], names=["date", "instrument_id"]
        )
        characteristics = pd.DataFrame(
            rng.normal(size=(len(index), 3)), index=index, columns=["quarterly", "annual", "other"]
        )
        ages = pd.DataFrame(0, index=index, columns=characteristics.columns)
        ages.loc[(slice(None), 1), ["quarterly", "annual"]] = 2
        returns = pd.Series(rng.normal(size=len(index)), index=index)

        with patch("llca.analytics.factors.ipca.InstrumentedPCA", _CapturingInstrumentedPCA):
            _, diagnostics = estimate_ipca_factors(
                returns,
                characteristics,
                n_factors=1,
                characteristic_ages=ages,
                feature_max_age={"default": 1, "columns": {"annual": 3}},
                return_diagnostics=True,
            )

        prepared = _CapturingInstrumentedPCA.last_x
        assert prepared is not None
        self.assertTrue((prepared.loc[(1, slice(None)), "quarterly"] == 0.0).all())
        self.assertEqual(diagnostics.stale_values["quarterly"], 3)
        self.assertEqual(diagnostics.stale_values["annual"], 0)
        self.assertEqual(diagnostics.feature_max_age, {"quarterly": 1, "annual": 3, "other": 1})

    def test_age_configuration_is_strict(self) -> None:
        returns, characteristics = self._missing_panel()
        ages = pd.DataFrame(0, index=characteristics.index, columns=characteristics.columns)

        with self.assertRaisesRegex(ValueError, "unknown characteristics"):
            estimate_ipca_factors(
                returns,
                characteristics,
                n_factors=1,
                characteristic_ages=ages,
                feature_max_age={"typo": 10},
            )
        with self.assertRaisesRegex(ValueError, "characteristic_ages is required"):
            estimate_ipca_factors(returns, characteristics, n_factors=1, feature_max_age=10)
        with self.assertRaisesRegex(ValueError, "non-negative integer"):
            estimate_ipca_factors(
                returns,
                characteristics,
                n_factors=1,
                characteristic_ages=ages,
                feature_max_age=-1,
            )

    def test_collinear_and_zero_variance_instruments_are_disclosed(self) -> None:
        dates = pd.bdate_range("2023-01-02", periods=3)
        entities = np.arange(1, 7)
        index = pd.MultiIndex.from_product([dates, entities], names=["date", "instrument_id"])
        base = np.tile(np.arange(len(entities), dtype=float), len(dates))
        characteristics = pd.DataFrame(
            {"first": base, "duplicate": base, "constant_raw": 1.0}, index=index
        )
        returns = pd.Series(np.linspace(-0.01, 0.01, len(index)), index=index)

        with patch("llca.analytics.factors.ipca.InstrumentedPCA", _CapturingInstrumentedPCA):
            factors, diagnostics = estimate_ipca_factors(
                returns, characteristics, n_factors=2, return_diagnostics=True
            )

        self.assertEqual(factors.shape, (3, 2))
        self.assertEqual(diagnostics.used_characteristics, ("first",))
        self.assertEqual(
            diagnostics.dropped_characteristics,
            {
                "duplicate": "linearly_dependent_after_ranking",
                "constant_raw": "zero_variance_after_ranking",
            },
        )

        with self.assertRaisesRegex(ValueError, "requests 3 factors.*supports at most 2"):
            estimate_ipca_factors(returns, characteristics, n_factors=3)

    def test_rank_deficient_dates_are_removed_without_reducing_factor_count(self) -> None:
        dates = pd.date_range("2024-01-01", periods=3, name="date")
        index = pd.MultiIndex.from_product(
            [dates, ["A", "B", "C", "D"]], names=["date", "instrument"]
        )
        characteristics = pd.DataFrame(
            {
                "varying": [
                    -2.0,
                    -1.0,
                    1.0,
                    2.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    -2.0,
                    -1.0,
                    1.0,
                    2.0,
                ]
            },
            index=index,
        )
        returns = pd.Series(np.linspace(-0.01, 0.01, len(index)), index=index)

        with patch("llca.analytics.factors.ipca.InstrumentedPCA", _CapturingInstrumentedPCA):
            factors, diagnostics = estimate_ipca_factors(
                returns,
                characteristics,
                n_factors=2,
                return_diagnostics=True,
            )

        self.assertEqual(list(factors.index), [dates[0], dates[2]])
        self.assertEqual(diagnostics.dropped_rank_deficient_dates, 1)
        self.assertEqual(diagnostics.dropped_rank_deficient_date_observations, 4)


if __name__ == "__main__":
    unittest.main()
