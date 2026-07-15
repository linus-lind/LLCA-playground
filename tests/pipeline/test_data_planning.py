import unittest

from omegaconf import OmegaConf

from llca.pipeline.contracts import (
    DataPlan,
    DataRequirements,
    DatasetQuery,
    DatasetRequirement,
    EntityScope,
)
from llca.pipeline.data_planning import build_data_plan
from llca.pipeline.preparation import _assert_manifest_plan


class DataPlanningTest(unittest.TestCase):
    def test_archived_plan_must_match_analytical_reconstruction(self) -> None:
        plan = DataPlan(
            primary_dataset="values",
            datasets={"values": DatasetQuery(entity_ids=(1,))},
            csv_chunk_size=100,
        )
        manifest = {
            "plan": {
                "primary_dataset": "values",
                "datasets": {"values": {"entity_ids": [2]}},
            }
        }

        with self.assertRaisesRegex(ValueError, "archived data selection"):
            _assert_manifest_plan(plan, manifest)

    def test_model_scopes_select_only_required_datasets_and_entities(self) -> None:
        data = OmegaConf.create(
            {
                "index": {"time": "date", "entity": "asset"},
                "selection": {"entity_ids": None, "csv_chunk_size": 100},
                "datasets": {
                    "daily": {"index": {"asset": "asset"}},
                    "context": {"index": {"asset": "asset"}},
                    "target": {"index": {"asset": "asset"}},
                    "unused": {"index": {"asset": "asset"}},
                },
            }
        )
        requirements = DataRequirements(
            primary_dataset="daily",
            datasets=(
                DatasetRequirement("daily", EntityScope.TARGET),
                DatasetRequirement("context", EntityScope.TARGET),
                DatasetRequirement("target", EntityScope.TARGET),
            ),
            target_entity=14593,
        )

        plan = build_data_plan(data, requirements)  # type: ignore[arg-type]

        self.assertEqual(set(plan.datasets), {"daily", "context", "target"})
        self.assertNotIn("unused", plan.datasets)
        self.assertTrue(all(query.entity_ids == (14593,) for query in plan.datasets.values()))
        self.assertEqual(plan.csv_chunk_size, 100)

    def test_explicit_universe_subset_is_preserved_for_cross_sectional_inputs(self) -> None:
        data = OmegaConf.create(
            {
                "index": {"time": "date", "entity": "asset"},
                "selection": {"entity_ids": [1, 2], "csv_chunk_size": 100},
                "datasets": {
                    "daily": {"index": {"asset": "asset"}},
                    "target": {"index": {"asset": "asset"}},
                },
            }
        )
        requirements = DataRequirements(
            primary_dataset="daily",
            datasets=(
                DatasetRequirement("daily", EntityScope.UNIVERSE),
                DatasetRequirement("target", EntityScope.TARGET),
            ),
            target_entity=1,
        )

        plan = build_data_plan(data, requirements)  # type: ignore[arg-type]

        self.assertEqual(plan.datasets["daily"].entity_ids, (1, 2))
        self.assertEqual(plan.datasets["target"].entity_ids, (1,))

    def test_rejects_selection_that_excludes_required_target(self) -> None:
        data = OmegaConf.create(
            {
                "index": {"time": "date", "entity": "asset"},
                "selection": {"entity_ids": [2], "csv_chunk_size": 100},
                "datasets": {"daily": {"index": {"asset": "asset"}}},
            }
        )
        requirements = DataRequirements(
            primary_dataset="daily",
            datasets=(DatasetRequirement("daily", EntityScope.TARGET),),
            target_entity=1,
        )

        with self.assertRaisesRegex(ValueError, "excluded"):
            build_data_plan(data, requirements)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
