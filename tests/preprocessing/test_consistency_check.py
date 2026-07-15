import unittest

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from llca.mappers.modules.column_ref import referenced_columns
from llca.mappers.preprocessing.config_validator import _validate_consistency_check
from llca.mappers.preprocessing.mapper import preprocessing_registry
from llca.preprocessing.consistency_check import (
    ConstraintExpression,
    ConstraintRule,
    consistency_check,
)
from llca.preprocessing.impute import impute


def _ohlc_rule() -> ConstraintRule:
    return ConstraintRule(
        name="ohlc_order",
        expressions=(
            ConstraintExpression(("high",), "ge", "low"),
            ConstraintExpression(("high",), "ge", "open"),
            ConstraintExpression(("high",), "ge", "close"),
            ConstraintExpression(("low",), "le", "open"),
            ConstraintExpression(("low",), "le", "close"),
        ),
        invalidate=("open", "high", "low", "close"),
    )


def _valid_spec() -> object:
    return OmegaConf.create(
        {
            "name": "consistency_check",
            "constraints": [
                {
                    "name": "ohlc_order",
                    "expressions": [
                        {"left": "high", "op": "ge", "right": "low"},
                        {"left": "low", "op": "le", "right": ["invalid"]},
                    ],
                    "invalidate": ["open", "high", "low", "close"],
                }
            ],
        }
    )


class ConsistencyCheckTest(unittest.TestCase):
    def test_scalar_and_relational_rules_apply_their_declared_invalidation_scope(self) -> None:
        panel = pd.DataFrame(
            {
                "open": [9.0, 10.0, 9.0],
                "high": [10.0, 9.0, np.nan],
                "low": [8.0, 8.0, 8.0],
                "close": [9.5, 8.5, 9.0],
                "volume": [10.0, -1.0, 2.0],
            }
        )
        rules = (
            ConstraintRule(
                name="non_negative",
                expressions=(ConstraintExpression(("volume",), "ge", 0),),
            ),
            _ohlc_rule(),
        )

        checked = consistency_check(panel, rules)

        self.assertTrue(checked.loc[1, ["open", "high", "low", "close"]].isna().all())
        self.assertTrue(np.isnan(checked.loc[1, "volume"]))
        self.assertEqual(checked.loc[0].to_dict(), panel.loc[0].to_dict())
        self.assertTrue(np.isnan(checked.loc[2, "high"]))
        self.assertEqual(checked.loc[2, "open"], 9.0)
        self.assertEqual(
            checked.attrs["consistency_reports"][0]["rules"][0]["name"], "non_negative"
        )

    def test_post_imputation_rule_detects_a_new_ohlc_contradiction(self) -> None:
        panel = pd.DataFrame(
            {
                "open": [9.0, 11.0],
                "high": [10.0, np.nan],
                "low": [8.0, 10.0],
                "close": [9.0, 12.0],
            }
        )

        filled = impute(
            panel,
            ffill=["open", "high", "low", "close"],
            fill_zero=[],
            subgroup_keys=[],
        )
        self.assertEqual(filled.loc[1, "high"], 10.0)

        checked = consistency_check(filled, (_ohlc_rule(),))

        self.assertTrue(checked.loc[1, ["open", "high", "low", "close"]].isna().all())
        self.assertTrue(checked.loc[0].notna().all())

    def test_missing_referenced_columns_fail_before_evaluation(self) -> None:
        with self.assertRaisesRegex(KeyError, "low"):
            consistency_check(
                pd.DataFrame({"high": [1.0]}),
                (
                    ConstraintRule(
                        name="order",
                        expressions=(ConstraintExpression(("high",), "ge", "low"),),
                    ),
                ),
            )


class ConsistencyConfigurationTest(unittest.TestCase):
    def test_generic_contract_validates_and_resolves_all_column_dependencies(self) -> None:
        spec = OmegaConf.create(
            {
                "name": "consistency_check",
                "constraints": [
                    {
                        "name": "ohlc_order",
                        "expressions": [
                            {"left": ["high"], "op": "ge", "right": "low"},
                        ],
                        "invalidate": ["open", "high", "low", "close"],
                    }
                ],
            }
        )

        self.assertEqual(_validate_consistency_check(spec), [])
        columns = referenced_columns(
            spec,
            preprocessing_registry.column_refs("consistency_check"),
        )
        self.assertEqual(columns, ["high", "low", "open", "close"])

    def test_contract_rejects_legacy_and_malformed_constraints(self) -> None:
        spec = _valid_spec()
        spec.positive = ["open"]  # type: ignore[attr-defined]
        errors = _validate_consistency_check(spec)  # type: ignore[arg-type]

        self.assertTrue(any("unsupported field" in error for error in errors))
        self.assertTrue(any("right must be a column name or a number" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
