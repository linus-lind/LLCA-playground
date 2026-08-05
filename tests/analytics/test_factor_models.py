import unittest

import numpy as np
import pandas as pd

from llca.analytics.factors import estimate_ipca_factors
from llca.analytics.factors import factor_models as fm


def _factors(n: int, k: int = 3, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2015-01-01", periods=n)
    data = 0.01 * rng.standard_normal((n, k))
    return pd.DataFrame(data, index=dates, columns=[f"f{i}" for i in range(k)])


class FactorModelsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.factors = _factors(2000)
        self.rng = np.random.default_rng(11)

    def _portfolio(self, alpha: float, betas: list[float], noise: float = 0.002) -> pd.Series:
        exposure = self.factors.to_numpy() @ np.array(betas)
        values = alpha + exposure + noise * self.rng.standard_normal(len(self.factors))
        return pd.Series(values, index=self.factors.index, name="p")

    def test_factor_alpha_recovers_alpha_and_betas(self) -> None:
        result = fm.factor_alpha(
            self._portfolio(0.0005, [0.8, 0.3, -0.2]),
            self.factors,
            annualization_periods=252,
        )
        assert result is not None
        self.assertAlmostEqual(result.alpha, 0.0005, delta=1e-4)
        self.assertLess(result.alpha_p_value, 0.01)
        self.assertAlmostEqual(result.betas["f0"], 0.8, delta=0.05)
        self.assertAlmostEqual(result.betas["f1"], 0.3, delta=0.05)
        self.assertAlmostEqual(result.annualized_alpha, 0.0005 * 252, delta=0.03)

    def test_factor_alpha_does_not_flag_zero_alpha(self) -> None:
        result = fm.factor_alpha(
            self._portfolio(0.0, [1.0, 0.0, 0.5]),
            self.factors,
            annualization_periods=252,
        )
        assert result is not None
        self.assertGreater(result.alpha_p_value, 0.05)

    def test_factor_alpha_returns_none_when_underdetermined(self) -> None:
        short = self.factors.iloc[:2]
        portfolio = pd.Series([0.01, -0.01], index=short.index)
        self.assertIsNone(fm.factor_alpha(portfolio, short, annualization_periods=252))

    def test_alpha_difference_detects_superior_model(self) -> None:
        better = self._portfolio(0.0006, [0.7, 0.0, 0.0])
        worse = self._portfolio(0.0, [0.7, 0.0, 0.0])
        outcome = fm.alpha_difference(better, worse, self.factors)
        self.assertGreater(outcome["alpha_difference"], 0.0)
        self.assertLess(outcome["alpha_difference_p_value"], 0.05)

    def test_joint_alpha_test_rejects_alpha_and_is_milder_without(self) -> None:
        with_alpha = pd.concat(
            [
                self._portfolio(0.001, [0.8, 0.0, 0.0]).rename("A"),
                self._portfolio(0.0009, [0.2, 0.3, 0.0]).rename("B"),
            ],
            axis=1,
        )
        without = pd.concat(
            [
                self._portfolio(0.0, [0.8, 0.0, 0.0]).rename("A"),
                self._portfolio(0.0, [0.2, 0.3, 0.0]).rename("B"),
            ],
            axis=1,
        )
        rejecting = fm.joint_alpha_test(with_alpha, self.factors)
        mild = fm.joint_alpha_test(without, self.factors)
        self.assertLess(rejecting["joint_alpha_p_value"], 0.05)
        self.assertGreater(mild["joint_alpha_p_value"], rejecting["joint_alpha_p_value"])

    def test_joint_alpha_test_deduplicates_identical_portfolios(self) -> None:
        portfolio = self._portfolio(0.001, [0.8, 0.0, 0.0]).rename("A")
        single = fm.joint_alpha_test(portfolio.to_frame(), self.factors)
        duplicated = fm.joint_alpha_test(
            pd.concat([portfolio, portfolio.rename("A duplicate")], axis=1),
            self.factors,
        )

        self.assertAlmostEqual(duplicated["joint_alpha_statistic"], single["joint_alpha_statistic"])
        self.assertAlmostEqual(duplicated["joint_alpha_p_value"], single["joint_alpha_p_value"])

    def test_spanning_flags_alpha_and_passes_spanned_portfolio(self) -> None:
        benchmark = self.factors
        spanned = pd.Series(
            benchmark.to_numpy() @ np.array([0.5, 0.3, 0.2])  # betas sum to 1, no alpha
            + 0.0005 * self.rng.standard_normal(len(benchmark)),
            index=benchmark.index,
        )
        expanding = self._portfolio(0.001, [0.5, 0.3, 0.2])
        self.assertGreater(fm.spanning_test(spanned, benchmark)["spanning_p_value"], 0.05)
        self.assertLess(fm.spanning_test(expanding, benchmark)["spanning_p_value"], 0.05)

    def test_timing_model_reports_alpha_and_gamma(self) -> None:
        instruments = pd.DataFrame(
            self.rng.standard_normal((len(self.factors), 2)),
            index=self.factors.index,
            columns=["z0", "z1"],
        )
        result = fm.timing_model(
            self._portfolio(0.0004, [0.9, 0.1, 0.0]),
            self.factors,
            "f0",
            instruments,
            annualization_periods=252,
        )
        assert result is not None
        self.assertTrue(np.isfinite(result.alpha))
        self.assertTrue(np.isfinite(result.timing_gamma))
        self.assertEqual(result.observations, len(self.factors) - 1)
        self.assertEqual(
            set(result.coefficients),
            {
                "alpha_z0",
                "alpha_z1",
                "f0",
                "f0_x_z0",
                "f0_x_z1",
                "f1",
                "f2",
                "f0_squared",
            },
        )
        self.assertEqual(set(result.coefficient_p_values), set(result.coefficients))

    def test_timing_model_exposes_zero_instrument_lag(self) -> None:
        instruments = pd.DataFrame(
            self.rng.standard_normal((len(self.factors), 1)),
            index=self.factors.index,
            columns=["state"],
        )

        result = fm.timing_model(
            self._portfolio(0.0, [0.8, 0.1, 0.0]),
            self.factors,
            "f0",
            instruments,
            annualization_periods=252,
            instrument_lag=0,
        )

        assert result is not None
        self.assertEqual(result.observations, len(self.factors))

    def test_rolling_betas_trace_exposure(self) -> None:
        portfolio = self._portfolio(0.0, [0.8, 0.0, 0.0])
        rolling = fm.rolling_betas(portfolio, self.factors, window=250)
        self.assertEqual(list(rolling.columns), list(self.factors.columns))
        self.assertEqual(len(rolling), len(self.factors) - 250 + 1)
        self.assertEqual(rolling.index[0], self.factors.index[249])
        design = np.column_stack([np.ones(250), self.factors.iloc[:250].to_numpy()])
        expected, *_ = np.linalg.lstsq(design, portfolio.iloc[:250].to_numpy(), rcond=None)
        np.testing.assert_allclose(rolling.iloc[0].to_numpy(), expected[1:], rtol=1e-12, atol=1e-12)
        self.assertAlmostEqual(float(rolling["f0"].mean()), 0.8, delta=0.1)

    def test_rolling_betas_rejects_underdetermined_window(self) -> None:
        with self.assertRaises(ValueError):
            fm.rolling_betas(self._portfolio(0.0, [0.8, 0.0, 0.0]), self.factors, window=3)
        with self.assertRaises(ValueError):
            fm.rolling_betas(self._portfolio(0.0, [0.8, 0.0, 0.0]), self.factors, window=4)

    def test_cumulative_alpha_endpoint_matches_alpha(self) -> None:
        portfolio = self._portfolio(0.0005, [0.7, 0.2, 0.0])
        result = fm.factor_alpha(portfolio, self.factors, annualization_periods=252)
        cumulative = fm.cumulative_alpha(portfolio, self.factors)
        assert result is not None
        # residuals sum to zero under OLS with an intercept, so the endpoint is alpha * n.
        self.assertAlmostEqual(
            float(cumulative.iloc[-1]), result.alpha * result.observations, delta=1e-6
        )


class IpcaFactorTest(unittest.TestCase):
    def test_estimate_ipca_factors_shape_and_usability(self) -> None:
        rng = np.random.default_rng(1)
        periods, names, chars, k = 90, 40, 4, 2
        dates = pd.bdate_range("2018-01-01", periods=periods)
        entities = np.arange(1, names + 1)
        index = pd.MultiIndex.from_product([dates, entities], names=["date", "instrument_id"])
        characteristics = pd.DataFrame(
            rng.standard_normal((periods * names, chars)),
            index=index,
            columns=[f"c{i}" for i in range(chars)],
        )
        returns = pd.Series(0.01 * rng.standard_normal(periods * names), index=index)
        # Two entities have no characteristics at all and must simply drop out.
        characteristics.loc[(slice(None), entities[:2]), :] = np.nan

        factors = estimate_ipca_factors(returns, characteristics, n_factors=k)
        self.assertEqual(factors.shape, (periods, k))
        self.assertEqual(factors.index.name, "date")
        self.assertTrue(np.isfinite(factors.to_numpy()).all())


if __name__ == "__main__":
    unittest.main()
