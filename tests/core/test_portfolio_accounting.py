import unittest

import torch

from llca.core.portfolio_accounting import (
    FUNDING_CONVENTION,
    cash_return_contribution,
    drifted_weights,
    gross_return,
    net_exposure,
    portfolio_nav_growth,
    residual_cash_weight,
    risky_return,
)


class PortfolioAccountingTest(unittest.TestCase):
    def test_funding_convention_name_is_stable(self) -> None:
        self.assertEqual(FUNDING_CONVENTION, "residual_cash_at_risk_free")

    def test_net_exposure_and_residual_cash_are_complementary(self) -> None:
        weights = torch.tensor([[0.3, 0.2], [0.5, -0.5], [1.0, 0.5]])
        torch.testing.assert_close(net_exposure(weights), torch.tensor([0.5, 0.0, 1.5]))
        torch.testing.assert_close(residual_cash_weight(weights), torch.tensor([0.5, 1.0, -0.5]))

    def test_fully_invested_book_ignores_risk_free(self) -> None:
        weights = torch.tensor([[0.6, 0.4]])
        returns = torch.tensor([[0.10, -0.05]])
        without = gross_return(weights, returns)
        with_rf = gross_return(weights, returns, torch.tensor([0.02]))
        torch.testing.assert_close(without, with_rf)
        torch.testing.assert_close(without, torch.tensor([0.6 * 0.10 + 0.4 * -0.05]))
        torch.testing.assert_close(
            cash_return_contribution(weights, torch.tensor([0.02])), torch.tensor([0.0])
        )

    def test_underinvested_book_earns_risk_free_on_residual_cash(self) -> None:
        weights = torch.tensor([[0.3, 0.2]])
        returns = torch.tensor([[0.10, 0.20]])
        risk_free = torch.tensor([0.01])
        expected = 0.3 * 0.10 + 0.2 * 0.20 + 0.5 * 0.01
        torch.testing.assert_close(
            gross_return(weights, returns, risk_free), torch.tensor([expected])
        )

    def test_market_neutral_book_earns_full_risk_free_on_cash(self) -> None:
        weights = torch.tensor([[0.5, -0.5]])
        returns = torch.tensor([[0.08, 0.03]])
        risk_free = torch.tensor([0.02])
        # cash weight is 1, so gross = rf + (0.5*r1 - 0.5*r2)
        expected = 0.02 + (0.5 * 0.08 - 0.5 * 0.03)
        torch.testing.assert_close(
            gross_return(weights, returns, risk_free), torch.tensor([expected])
        )
        torch.testing.assert_close(residual_cash_weight(weights), torch.tensor([1.0]))

    def test_leveraged_net_long_book_pays_risk_free_on_negative_cash(self) -> None:
        weights = torch.tensor([[1.0, 0.5]])
        returns = torch.tensor([[0.04, 0.02]])
        risk_free = torch.tensor([0.03])
        self.assertAlmostEqual(float(residual_cash_weight(weights)), -0.5)
        expected = 1.0 * 0.04 + 0.5 * 0.02 + (-0.5) * 0.03
        torch.testing.assert_close(
            gross_return(weights, returns, risk_free), torch.tensor([expected])
        )

    def test_single_asset_short_borrows_two_units_of_cash(self) -> None:
        # w = -1 (fully short) leaves residual cash 1 - (-1) = 2 under this balance-sheet convention.
        weights = torch.tensor([[-1.0]])
        returns = torch.tensor([[0.05]])
        risk_free = torch.tensor([0.01])
        self.assertAlmostEqual(float(residual_cash_weight(weights)), 2.0)
        expected = -1.0 * 0.05 + 2.0 * 0.01
        torch.testing.assert_close(
            gross_return(weights, returns, risk_free), torch.tensor([expected])
        )

    def test_zero_risky_exposure_returns_risk_free(self) -> None:
        weights = torch.zeros(1, 3)
        returns = torch.tensor([[0.1, -0.2, 0.3]])
        risk_free = torch.tensor([0.004])
        torch.testing.assert_close(gross_return(weights, returns, risk_free), torch.tensor([0.004]))

    def test_absent_risk_free_is_zero_and_matches_risky_return(self) -> None:
        weights = torch.tensor([[0.3, 0.2]])
        returns = torch.tensor([[0.1, 0.2]])
        torch.testing.assert_close(gross_return(weights, returns), risky_return(weights, returns))

    def test_risk_free_units_pass_through_without_rescaling(self) -> None:
        # A 5 bps daily rate is 0.0005; zero risky exposure must return exactly that, not 0.05.
        weights = torch.zeros(1, 2)
        returns = torch.zeros(1, 2)
        gross = gross_return(weights, returns, torch.tensor([0.0005]))
        # places=7 tolerates float32 rounding yet still rejects a 100x percent/decimal error.
        self.assertAlmostEqual(float(gross), 0.0005, places=7)

    def test_risk_free_shape_must_match_date_axis(self) -> None:
        weights = torch.zeros(2, 3)
        returns = torch.zeros(2, 3)
        with self.assertRaisesRegex(ValueError, "one rate per date"):
            gross_return(weights, returns, torch.zeros(3))  # per-asset, not per-date

    def test_drift_advances_risky_and_cash_under_one_wealth_path(self) -> None:
        weights = torch.tensor([[0.5, 0.25]])  # net 0.75, residual cash 0.25
        returns = torch.tensor([[0.10, -0.20]])
        risk_free = torch.tensor([0.02])
        growth = 1.0 + (0.5 * 0.10 + 0.25 * -0.20 + 0.25 * 0.02)
        torch.testing.assert_close(
            portfolio_nav_growth(weights, returns, risk_free), torch.tensor([growth])
        )
        expected = torch.tensor([[0.5 * 1.10 / growth, 0.25 * 0.80 / growth]])
        torch.testing.assert_close(drifted_weights(weights, returns, risk_free), expected)

    def test_drift_ignoring_cash_would_differ_from_cash_inclusive_drift(self) -> None:
        # Guards against reintroducing a zero-return-cash NAV denominator.
        weights = torch.tensor([[0.4, 0.1]])  # residual cash 0.5
        returns = torch.tensor([[0.10, 0.10]])
        risk_free = torch.tensor([0.05])
        cash_inclusive = drifted_weights(weights, returns, risk_free)
        zero_cash = drifted_weights(weights, returns, torch.tensor([0.0]))
        self.assertFalse(torch.allclose(cash_inclusive, zero_cash))

    def test_gross_return_keeps_gradient_to_weights_through_residual_cash(self) -> None:
        weights = torch.tensor([[0.3, 0.2]], requires_grad=True)
        returns = torch.tensor([[0.10, 0.20]])
        risk_free = torch.tensor([0.05])
        gross_return(weights, returns, risk_free).sum().backward()  # type: ignore[no-untyped-call]
        # d gross / d w_i = r_i - rf (the excess-return form), non-zero and rf-aware.
        assert weights.grad is not None
        torch.testing.assert_close(weights.grad, torch.tensor([[0.10 - 0.05, 0.20 - 0.05]]))


if __name__ == "__main__":
    unittest.main()
