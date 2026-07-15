import hashlib
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import call, patch

import pandas as pd

from llca.data.versioning import (
    DataVersioningError,
    _parse_dvc_status,
    archive_raw_file,
    build_data_manifest,
    fingerprint_frame,
    provenance_tags,
    validate_data_manifest,
    verify_raw_sources,
)


class DataVersioningTest(unittest.TestCase):
    def test_rejects_noncanonical_data_manifest_schema(self) -> None:
        with self.assertRaisesRegex(DataVersioningError, "schema_version must be 1"):
            validate_data_manifest({"schema_version": 2})

    def test_dvc_status_parser_accepts_repeated_windows_json_output(self) -> None:
        self.assertEqual(_parse_dvc_status("{}\r\n{}\r\n"), {})
        self.assertEqual(
            _parse_dvc_status('\x1b[0m{"data/prices.csv": "new"}\r\n{}'),
            {"data/prices.csv": "new"},
        )

    def test_dvc_status_parser_rejects_non_json_trailing_output(self) -> None:
        with self.assertRaisesRegex(DataVersioningError, "invalid remote status"):
            _parse_dvc_status("{}\r\nnot-json")

    def test_processed_fingerprint_covers_values_order_and_schema(self) -> None:
        frame = pd.DataFrame(
            {"value": [1.0, 2.0]},
            index=pd.Index(["a", "b"], name="entity"),
        )

        original = fingerprint_frame(frame)
        same = fingerprint_frame(frame.copy())
        reordered = fingerprint_frame(frame.iloc[::-1])
        changed = fingerprint_frame(frame.assign(value=[1.0, 3.0]))

        self.assertEqual(original["sha256"], same["sha256"])
        self.assertNotEqual(original["sha256"], reordered["sha256"])
        self.assertNotEqual(original["sha256"], changed["sha256"])
        self.assertEqual(original["rows"], 2)

    def test_manifest_archives_shared_raw_source_once(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "data" / "prices.csv"
            raw.parent.mkdir()
            raw.write_text("value\n1\n", encoding="utf-8")
            archived = {
                "path": "data/prices.csv",
                "size_bytes": raw.stat().st_size,
                "sha256": "raw-sha",
                "dvc": {
                    "remote": "archive",
                    "pointer_path": "data/prices.csv.dvc",
                    "hash_algorithm": "md5",
                    "content_hash": "raw-md5",
                    "pointer": {"outs": []},
                },
            }
            panels = {
                "prices": pd.DataFrame({"x": [1.0]}),
                "target": pd.DataFrame({"x": [1.0]}),
            }
            with patch("llca.data.versioning.archive_raw_file", return_value=archived) as archive:
                manifest = build_data_manifest(
                    {"prices": raw, "target": raw}, panels, project_root=root
                )

            archive.assert_called_once_with(raw, project_root=root)
            self.assertEqual(len(manifest["sources"]), 1)
            self.assertEqual(manifest["datasets"]["target"]["raw_source"], "data/prices.csv")

    def test_raw_archive_pushes_pointer_and_records_sha256(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "data" / "prices.csv"
            raw.parent.mkdir()
            raw.write_bytes(b"value\n1\n")
            pointer = raw.with_name("prices.csv.dvc")
            pointer.write_text(
                "outs:\n- md5: abc123\n  size: 8\n  hash: md5\n  path: prices.csv\n",
                encoding="utf-8",
            )

            def dvc(*arguments: str, project_root: Path) -> str:
                del project_root
                if arguments == ("config", "core.remote"):
                    return "archive"
                if arguments[:3] == ("status", "--cloud", "--json"):
                    return "{}"
                return ""

            with patch("llca.data.versioning._run_dvc", side_effect=dvc) as run:
                record = archive_raw_file(raw, project_root=root)

            self.assertEqual(record["dvc"]["content_hash"], "abc123")
            self.assertEqual(len(record["sha256"]), 64)
            self.assertIn(
                call("push", "data/prices.csv.dvc", project_root=root),
                run.call_args_list,
            )

    def test_provenance_tags_use_canonical_source_hash_names(self) -> None:
        manifest = {
            "sources": {
                "data/prices.csv": {
                    "sha256": "raw-sha",
                    "dvc": {"hash_algorithm": "md5", "content_hash": "raw-md5"},
                }
            },
            "datasets": {
                "prices": {
                    "raw_source": "data/prices.csv",
                    "processed": {"sha256": "processed-sha"},
                }
            },
        }

        tags = provenance_tags(manifest)

        self.assertEqual(tags["raw_data_dvc_prices"], "raw-md5")
        self.assertEqual(tags["raw_data_sha256_prices"], "raw-sha")
        self.assertEqual(tags["processed_data_sha256_prices"], "processed-sha")

    def test_raw_source_verification_reuses_hashes_and_rejects_drift(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "data" / "prices.csv"
            raw.parent.mkdir()
            raw.write_bytes(b"value\n1\n")
            manifest = {
                "schema_version": 1,
                "plan": {},
                "sources": {
                    "data/prices.csv": {
                        "path": "data/prices.csv",
                        "size_bytes": raw.stat().st_size,
                        "sha256": hashlib.sha256(raw.read_bytes()).hexdigest(),
                    }
                },
                "datasets": {},
            }
            cache: dict[Path, str] = {}

            self.assertEqual(
                verify_raw_sources(manifest, project_root=root, verified_hashes=cache),
                (raw,),
            )
            with patch("llca.data.versioning.sha256_file") as sha256:
                verify_raw_sources(manifest, project_root=root, verified_hashes=cache)
            sha256.assert_not_called()

            raw.write_bytes(b"value\n2\n")
            with self.assertRaisesRegex(DataVersioningError, "differs from archived run"):
                verify_raw_sources(manifest, project_root=root, verified_hashes={})


if __name__ == "__main__":
    unittest.main()
