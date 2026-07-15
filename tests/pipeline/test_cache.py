import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from omegaconf import OmegaConf

from llca.pipeline.cache import (
    load_cached_preparation,
    preparation_cache_key,
    save_cached_preparation,
)
from llca.pipeline.contracts import DataPlan, DatasetQuery


class PreparationCacheTest(unittest.TestCase):
    def test_key_covers_selection_and_atomic_entry_roundtrip(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.csv"
            source.write_text("date,asset,value\n2024-01-01,1,2\n", encoding="utf-8")
            cfg = OmegaConf.create(
                {
                    "data": {
                        "index": {"time": "date", "entity": "asset"},
                        "datasets": {"values": {"path": "source.csv"}},
                    },
                    "preprocessing": {},
                    "features": {},
                    "masking": {},
                }
            )
            target_plan = DataPlan(
                primary_dataset="values",
                datasets={"values": DatasetQuery(entity_ids=(1,))},
                csv_chunk_size=10,
            )
            universe_plan = DataPlan(
                primary_dataset="values",
                datasets={"values": DatasetQuery()},
                csv_chunk_size=10,
            )

            target_key = preparation_cache_key(  # type: ignore[arg-type]
                cfg,
                target_plan,
                {"values": source},
                data_view="independent",
            )
            universe_key = preparation_cache_key(  # type: ignore[arg-type]
                cfg,
                universe_plan,
                {"values": source},
                data_view="independent",
            )
            archived_key = preparation_cache_key(  # type: ignore[arg-type]
                cfg,
                target_plan,
                {"values": source},
                data_view="independent",
                source_versions={"values": "archived-sha"},
            )
            save_cached_preparation(root, target_key, {"data": {"value": 1}})

            self.assertNotEqual(target_key, universe_key)
            self.assertNotEqual(target_key, archived_key)
            self.assertEqual(
                load_cached_preparation(root, target_key)["data"],  # type: ignore[index]
                {"value": 1},
            )
            self.assertIsNone(load_cached_preparation(root, universe_key))


if __name__ == "__main__":
    unittest.main()
