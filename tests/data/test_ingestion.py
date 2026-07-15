import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd
from omegaconf import OmegaConf

from llca.data.index_spec import IndexSpec
from llca.data.ingestion import load_datasets
from llca.pipeline.contracts import DatasetQuery


def _spec(column: str) -> object:
    return OmegaConf.create(
        {
            "path": "shared.csv",
            "date_format": "%Y-%m-%d",
            "index": {"date": {"raw": "date", "dtype": "date"}, "asset": "permno"},
            "columns": {column: column},
        }
    )


class DatasetIngestionTest(unittest.TestCase):
    def test_target_filter_does_not_truncate_shared_entity_free_view(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            pd.DataFrame(
                {
                    "date": ["2024-01-01", "2024-01-02"],
                    "permno": [1, 2],
                    "context": [10.0, 20.0],
                    "target": [0.1, 0.2],
                }
            ).to_csv(root / "shared.csv", index=False)
            context = OmegaConf.create(
                {
                    "path": "shared.csv",
                    "date_format": "%Y-%m-%d",
                    "index": {"date": {"raw": "date", "dtype": "date"}},
                    "columns": {"context": "context"},
                }
            )
            with patch("llca.data.ingestion.DATA_DIR", root):
                panels = load_datasets(  # type: ignore[arg-type]
                    {"context": context, "target": _spec("target")},
                    IndexSpec("date", "asset"),
                    {
                        "context": DatasetQuery(),
                        "target": DatasetQuery(entity_ids=(1,)),
                    },
                )

        self.assertEqual(len(panels["context"]), 2)
        self.assertEqual(len(panels["target"]), 1)

    def test_shared_source_is_scanned_once_and_target_filter_is_pushed_down(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            pd.DataFrame(
                {
                    "date": ["2024-01-01", "2024-01-01", "2024-01-02"],
                    "permno": [1, 2, 1],
                    "feature": [10.0, 20.0, 11.0],
                    "target": [0.1, 0.2, 0.3],
                }
            ).to_csv(root / "shared.csv", index=False)
            specs = {"features": _spec("feature"), "target": _spec("target")}
            queries = {
                "features": DatasetQuery(entity_ids=(1,)),
                "target": DatasetQuery(entity_ids=(1,)),
            }
            original = pd.read_csv
            with (
                patch("llca.data.ingestion.DATA_DIR", root),
                patch("llca.data.ingestion.pd.read_csv", wraps=original) as read_csv,
            ):
                panels = load_datasets(  # type: ignore[arg-type]
                    specs,
                    IndexSpec("date", "asset"),
                    queries,
                    csv_chunk_size=1,
                )

        self.assertEqual(read_csv.call_count, 2)  # one header read and one physical scan
        self.assertEqual(set(panels["features"].index.get_level_values("asset")), {1})
        self.assertEqual(len(panels["features"]), 2)
        self.assertEqual(len(panels["target"]), 2)

    def test_logical_target_can_filter_after_a_full_universe_shared_scan(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            pd.DataFrame(
                {
                    "date": ["2024-01-01", "2024-01-01"],
                    "permno": [1, 2],
                    "feature": [10.0, 20.0],
                    "target": [0.1, 0.2],
                }
            ).to_csv(root / "shared.csv", index=False)
            with patch("llca.data.ingestion.DATA_DIR", root):
                panels = load_datasets(  # type: ignore[arg-type]
                    {"features": _spec("feature"), "target": _spec("target")},
                    IndexSpec("date", "asset"),
                    {
                        "features": DatasetQuery(),
                        "target": DatasetQuery(entity_ids=(1,)),
                    },
                )

        self.assertEqual(len(panels["features"]), 2)
        self.assertEqual(len(panels["target"]), 1)


if __name__ == "__main__":
    unittest.main()
