import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd
from omegaconf import OmegaConf

from llca.core.paths import PROJECT_ROOT
from llca.data.versioning import fingerprint_frame, sha256_file
from llca.pipeline.contracts import (
    DataRequirements,
    DatasetRequirement,
    EntityScope,
)
from llca.pipeline.preparation import (
    PreparedAnalysisData,
    PreparedModelData,
    prepare_analysis_data,
)


class AnalysisPreparationTest(unittest.TestCase):
    def test_prepares_independent_universe_with_read_only_evidence(self) -> None:
        with TemporaryDirectory(dir=PROJECT_ROOT) as directory:
            root = Path(directory)
            source = root / "shared.csv"
            source.write_text(
                "date,asset,return,value\n2024-01-02,1,0.01,2.0\n",
                encoding="utf-8",
            )
            cfg = OmegaConf.create(
                {
                    "data": {
                        "index": {"time": "date", "entity": "asset"},
                        "selection": {"entity_ids": [1, 2], "csv_chunk_size": 100},
                        "datasets": {
                            "returns": {
                                "path": "shared.csv",
                                "index": {"asset": "asset"},
                            },
                            "characteristics": {
                                "path": "shared.csv",
                                "index": {"asset": "asset"},
                            },
                        },
                    },
                    "preprocessing": {"returns": []},
                    "features": {"characteristics": []},
                    "masking": {},
                }
            )
            requirements = DataRequirements(
                primary_dataset="returns",
                datasets=(
                    DatasetRequirement("returns", EntityScope.UNIVERSE),
                    DatasetRequirement("characteristics", EntityScope.UNIVERSE),
                ),
            )
            index = pd.MultiIndex.from_tuples(
                [(pd.Timestamp("2024-01-02"), 1)], names=["date", "asset"]
            )
            returns = pd.DataFrame({"return": [0.01]}, index=index)
            characteristics = pd.DataFrame({"value": [2.0]}, index=index)
            panels = {"returns": returns, "characteristics": characteristics}
            captured: dict[str, object] = {}

            def fake_prepare(
                config: object,
                plan: object,
                logical_sources: dict[str, Path],
                data_view: str,
                *,
                source_versions: dict[str, str] | None = None,
            ) -> PreparedModelData:
                captured["config"] = config
                captured["data_view"] = data_view
                captured["source_versions"] = source_versions
                captured["logical_sources"] = logical_sources
                return PreparedModelData(
                    data={"aligned": True},
                    processed_datasets=panels,
                    feature_panels=panels,
                    plan=plan,  # type: ignore[arg-type]
                    logical_sources=logical_sources,
                )

            with (
                patch(
                    "llca.pipeline.preparation.data_source_path",
                    side_effect=lambda _spec: source,
                ),
                patch("llca.pipeline.preparation._prepare", side_effect=fake_prepare),
                patch("llca.pipeline.preparation.archive_raw_sources") as archive,
            ):
                prepared = prepare_analysis_data(
                    cfg,
                    requirements,
                    data_view="aligned_panel",
                )

            digest = sha256_file(source)
            relative_source = source.relative_to(PROJECT_ROOT).as_posix()
            self.assertIsInstance(prepared, PreparedAnalysisData)
            self.assertEqual(prepared.data, {"aligned": True})
            self.assertIs(captured["config"], cfg)
            self.assertEqual(captured["data_view"], "aligned_panel")
            self.assertEqual(
                captured["source_versions"],
                {"returns": digest, "characteristics": digest},
            )
            self.assertEqual(
                prepared.data_manifest["plan"],
                {
                    "primary_dataset": "returns",
                    "datasets": {
                        "returns": {"entity_ids": [1, 2]},
                        "characteristics": {"entity_ids": [1, 2]},
                    },
                },
            )
            self.assertEqual(
                prepared.data_manifest["sources"],
                {
                    relative_source: {
                        "path": relative_source,
                        "size_bytes": source.stat().st_size,
                        "sha256": digest,
                    }
                },
            )
            self.assertNotIn("dvc", prepared.data_manifest["sources"][relative_source])
            self.assertEqual(
                prepared.data_manifest["datasets"]["characteristics"]["processed"],
                fingerprint_frame(characteristics),
            )
            archive.assert_not_called()

    def test_missing_local_source_fails_before_preparation(self) -> None:
        cfg = OmegaConf.create(
            {
                "data": {
                    "index": {"time": "date", "entity": "asset"},
                    "datasets": {
                        "returns": {
                            "path": "missing.csv",
                            "index": {"asset": "asset"},
                        }
                    },
                }
            }
        )
        requirements = DataRequirements(
            primary_dataset="returns",
            datasets=(DatasetRequirement("returns", EntityScope.UNIVERSE),),
        )
        missing = PROJECT_ROOT / "does-not-exist" / "missing.csv"

        with (
            patch("llca.pipeline.preparation.data_source_path", return_value=missing),
            patch("llca.pipeline.preparation._prepare") as prepare,
        ):
            with self.assertRaisesRegex(FileNotFoundError, "does not exist"):
                prepare_analysis_data(cfg, requirements)

        prepare.assert_not_called()


if __name__ == "__main__":
    unittest.main()
