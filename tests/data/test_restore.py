import hashlib
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import yaml

from llca.data.restore import restore_data_manifest


class DataRestoreTest(unittest.TestCase):
    def test_existing_verified_data_recreates_archived_pointer(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data" / "prices.csv"
            data.parent.mkdir()
            data.write_bytes(b"value\n1\n")
            digest = hashlib.sha256(data.read_bytes()).hexdigest()
            pointer = {
                "outs": [
                    {
                        "md5": "abc123",
                        "size": data.stat().st_size,
                        "hash": "md5",
                        "path": "prices.csv",
                    }
                ]
            }
            manifest = {
                "sources": {
                    "data/prices.csv": {
                        "path": "data/prices.csv",
                        "sha256": digest,
                        "dvc": {
                            "remote": "archive",
                            "pointer_path": "data/prices.csv.dvc",
                            "pointer": pointer,
                        },
                    }
                }
            }

            restored = restore_data_manifest(manifest, project_root=root)

            self.assertEqual(restored, (data,))
            self.assertEqual(
                yaml.safe_load((root / "data" / "prices.csv.dvc").read_text()), pointer
            )


if __name__ == "__main__":
    unittest.main()
