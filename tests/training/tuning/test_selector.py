"""Model-agnostic selector orchestration: known winner, per-fold scoring, coverage guard."""

from __future__ import annotations

import unittest
from collections.abc import Mapping

import numpy as np
import pandas as pd

from llca.data.modules.masked_panel import MaskedPanel, MaskedPanels
from llca.models.estimators.prediction import PredictionOutput
from llca.training.modules.sklearn_config import SklearnTrainingConfig
from llca.training.tuning.search_space import ChoiceDimension, ParameterValue, SearchSpace
from llca.training.tuning.selector import select_hyperparameters
from llca.training.tuning.settings import HyperparameterSelection, InnerCvSettings, SearchSettings

_TRAINING = SklearnTrainingConfig(seed=0, n_jobs=1)
_INNER = InnerCvSettings(train_size=5, val_size=2, step_size=3, purge=1, lookback=0, min_folds=1)


def _panels(periods: int = 20) -> MaskedPanels:
    dates = pd.bdate_range("2021-01-01", periods=periods)
    index = pd.MultiIndex.from_product([dates, [1]], names=["date", "entity"])
    values = pd.DataFrame({"f": np.arange(len(index), dtype=float)}, index=index)
    panel = MaskedPanel(
        values=values,
        observed=pd.DataFrame(True, index=index, columns=values.columns),
        age=pd.DataFrame(0.0, index=index, columns=values.columns),
        segment=pd.Series(index.get_level_values("entity"), index=index),
    )
    return {"daily_values": panel}


class _ConstantModel:
    """A fake estimator whose prediction is a constant equal to its ``k`` hyperparameter."""

    def __init__(self, k: float, *, drop_first: bool = False) -> None:
        self._k = k
        self._drop_first = drop_first

    def fit(self, train: MaskedPanels, *, training: object) -> None:
        del train, training

    def predict(self, test: MaskedPanels) -> PredictionOutput:
        index = test["daily_values"].values.index
        if self._drop_first:
            index = index[1:]
        return PredictionOutput(
            kind="portfolio", values=pd.Series(float(self._k), index=index, name="score")
        )


def _returns(split: MaskedPanels) -> pd.Series:
    index = split["daily_values"].values.index
    return pd.Series(0.0, index=index, name="fwd_return")


class SelectorTest(unittest.TestCase):
    def _selection(
        self, choices: tuple[ParameterValue, ...], margin: float = 1.0
    ) -> HyperparameterSelection:
        return HyperparameterSelection(
            enabled=True,
            inner_cv=_INNER,
            search=SearchSettings("grid", 0, 0),
            search_space=SearchSpace((ChoiceDimension("k", choices),)),
            baseline={"k": 1.0},
            standard_error_margin=margin,
        )

    def test_selects_clear_winner_and_reports_provenance(self) -> None:
        calls: list[int] = []

        def objective(scores: pd.Series, returns: pd.Series) -> float:
            del returns
            calls.append(len(scores))
            return abs(float(scores.mean()))  # loss == |k|

        result = select_hyperparameters(
            train=_panels(),
            primary="daily_values",
            selection=self._selection((0.0, 1.0, 2.0)),
            candidate_factory=lambda p: _ConstantModel(float(p["k"])),
            realized_returns=_returns,
            fold_objective=objective,
            training=_TRAINING,
        )

        self.assertEqual(result.selected_parameters, {"k": 0.0})
        self.assertFalse(result.selected_is_baseline)
        self.assertEqual(result.best_candidate_mean_loss, 0.0)
        self.assertEqual(result.baseline_mean_loss, 1.0)
        self.assertEqual(result.evaluated_candidates, 3)
        self.assertEqual(result.fold_count, 5)
        # Scored per fold, never pooled: (baseline + 3 candidates) x 5 folds.
        self.assertEqual(len(calls), 4 * 5)
        # Each call sees exactly one fold's validation window (val_size == 2), not a pooled series.
        self.assertTrue(all(count == 2 for count in calls))

    def test_retains_baseline_when_every_candidate_is_worse(self) -> None:
        def objective(scores: pd.Series, returns: pd.Series) -> float:
            del returns
            return abs(float(scores.mean()))

        result = select_hyperparameters(
            train=_panels(),
            primary="daily_values",
            selection=self._selection((2.0, 3.0)),  # both worse than baseline k=1
            candidate_factory=lambda p: _ConstantModel(float(p["k"])),
            realized_returns=_returns,
            fold_objective=objective,
            training=_TRAINING,
        )
        self.assertTrue(result.selected_is_baseline)
        self.assertEqual(result.selected_parameters, {"k": 1.0})

    def test_inconsistent_candidate_coverage_is_rejected(self) -> None:
        def objective(scores: pd.Series, returns: pd.Series) -> float:
            del returns
            return abs(float(scores.mean()))

        def factory(params: Mapping[str, ParameterValue]) -> _ConstantModel:
            # The k=2 candidate drops its first validation row, breaking coverage parity.
            return _ConstantModel(float(params["k"]), drop_first=float(params["k"]) == 2.0)

        with self.assertRaisesRegex(ValueError, "different validation observations"):
            select_hyperparameters(
                train=_panels(),
                primary="daily_values",
                selection=self._selection((0.0, 2.0)),
                candidate_factory=factory,
                realized_returns=_returns,
                fold_objective=objective,
                training=_TRAINING,
            )

    def test_non_finite_fold_loss_fails_loudly(self) -> None:
        # A non-finite objective value aborts selection rather than being silently dropped.
        with self.assertRaisesRegex(ValueError, "non-finite loss"):
            select_hyperparameters(
                train=_panels(),
                primary="daily_values",
                selection=self._selection((0.0, 2.0)),
                candidate_factory=lambda p: _ConstantModel(float(p["k"])),
                realized_returns=_returns,
                fold_objective=lambda s, r: float("nan"),
                training=_TRAINING,
            )

    def test_insufficient_folds_is_rejected(self) -> None:
        selection = HyperparameterSelection(
            enabled=True,
            inner_cv=InnerCvSettings(
                train_size=5, val_size=2, step_size=3, purge=1, lookback=0, min_folds=99
            ),
            search=SearchSettings("grid", 0, 0),
            search_space=SearchSpace((ChoiceDimension("k", (0.0,)),)),
            baseline={"k": 1.0},
            standard_error_margin=1.0,
        )
        with self.assertRaisesRegex(ValueError, "at least 99 are required"):
            select_hyperparameters(
                train=_panels(),
                primary="daily_values",
                selection=selection,
                candidate_factory=lambda p: _ConstantModel(float(p["k"])),
                realized_returns=_returns,
                fold_objective=lambda s, r: 0.0,
                training=_TRAINING,
            )


if __name__ == "__main__":
    unittest.main()
