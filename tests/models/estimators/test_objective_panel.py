"""Objective-panel packing: alignment, all-invalid-date dropping, and within-fold turnover."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from llca.loss.portfolio import PortfolioLoss
from llca.models.estimators.objective_panel import pack_objective_panel


def _single_asset_series(values: list[float], name: str) -> pd.Series:
    dates = pd.bdate_range("2021-01-01", periods=len(values))
    index = pd.MultiIndex.from_product([dates, [1]], names=["date", "entity"])
    return pd.Series(values, index=index, name=name)


def _loss() -> PortfolioLoss:
    return PortfolioLoss(
        leverage=1.0,
        risk_aversion=1.0,
        concentration_aversion=0.0,
        execution_fee=0.0001,
        bid_ask_spread=0.0003,
        slippage=0.0002,
        borrow_cost=0.00002,
        normalization="gross",
        return_type="simple",
        common_score_aversion=0.0,
        net_exposure_aversion=0.0,
        net_exposure_tolerance=1.0,
    )


class PackObjectivePanelTest(unittest.TestCase):
    def test_drops_dates_without_a_jointly_observed_entity(self) -> None:
        scores = _single_asset_series([1.0, np.nan, 3.0, 4.0], "score")
        returns = _single_asset_series([0.1, 0.2, np.nan, 0.4], "fwd_return")

        packed_scores, packed_returns, mask, dates = pack_objective_panel(scores, returns)

        # Only the first and last dates carry both a score and a return.
        np.testing.assert_allclose(packed_scores.numpy(), [[1.0], [4.0]])
        np.testing.assert_allclose(packed_returns.numpy(), [[0.1], [0.4]])
        np.testing.assert_array_equal(mask.numpy(), [[True], [True]])
        self.assertEqual(list(dates), [scores.index[0][0], scores.index[-1][0]])

    def test_aligns_returns_to_the_score_index(self) -> None:
        scores = _single_asset_series([1.0, 2.0, 3.0], "score")
        returns = _single_asset_series([0.1, 0.2, 0.3], "fwd_return").iloc[::-1]  # reordered

        _, packed_returns, mask, _ = pack_objective_panel(scores, returns)

        np.testing.assert_allclose(packed_returns.numpy(), [[0.1], [0.2], [0.3]])
        self.assertTrue(mask.all())

    def test_no_valid_observation_raises(self) -> None:
        scores = _single_asset_series([np.nan, np.nan], "score")
        returns = _single_asset_series([0.1, 0.2], "fwd_return")
        with self.assertRaisesRegex(ValueError, "no jointly observed"):
            pack_objective_panel(scores, returns)

    def test_turnover_reflects_within_fold_sign_changes(self) -> None:
        returns = _single_asset_series([0.01, 0.01, 0.01], "fwd_return")
        objective = _loss()

        steady_scores, steady_returns, steady_mask, _ = pack_objective_panel(
            _single_asset_series([1.0, 1.0, 1.0], "s"), returns
        )
        steady = objective(steady_scores, steady_returns, steady_mask)
        flip_scores, flip_returns, flip_mask, _ = pack_objective_panel(
            _single_asset_series([1.0, -1.0, 1.0], "s"), returns
        )
        flipping = objective(flip_scores, flip_returns, flip_mask)

        # A steady fully-invested position turns over nothing; alternating sign turns over a lot.
        self.assertAlmostEqual(float(steady.turnover.item()), 0.0, places=6)
        self.assertGreater(float(flipping.turnover.item()), 1.0)
        # Higher turnover means higher cost, hence a higher (worse) loss on identical returns.
        self.assertGreater(float(flipping.loss.item()), float(steady.loss.item()))


if __name__ == "__main__":
    unittest.main()
