import unittest
from base64 import b64decode
from pathlib import Path
from tempfile import TemporaryDirectory

from omegaconf import OmegaConf

from llca.training.manifests import (
    build_environment_manifest,
    build_invocation_manifest,
    build_source_snapshot,
    build_training_manifest,
    validate_training_manifest,
)


class TrainingManifestTest(unittest.TestCase):
    def test_excludes_analytics_recovery_and_tracking_runtime(self) -> None:
        config = OmegaConf.create(
            {
                "experiment_name": "example",
                "data": {
                    "cache": {"enabled": True, "directory": ".cache/data"},
                    "selection": {"entity_ids": [1], "csv_chunk_size": 100},
                },
                "preprocessing": {},
                "features": {},
                "masking": {},
                "loss": {},
                "model": {},
                "training": {},
                "split": {},
                "analytics": {"show_plots": True},
                "recovery": {"mode": "off"},
                "mlflow_tracking_uri": "sqlite:///elsewhere.db",
            }
        )

        manifest = build_training_manifest(config)

        self.assertEqual(manifest["schema_version"], 1)
        self.assertNotIn("analytics", manifest)
        self.assertNotIn("recovery", manifest)
        self.assertNotIn("mlflow_tracking_uri", manifest)
        self.assertNotIn("cache", manifest["data"])
        self.assertNotIn("csv_chunk_size", manifest["data"]["selection"])
        self.assertEqual(manifest["data"]["selection"]["entity_ids"], [1])

    def test_rejects_noncanonical_training_manifest_schema(self) -> None:
        with self.assertRaisesRegex(ValueError, "schema_version must be 1"):
            validate_training_manifest({"schema_version": 2})

    def test_invocation_is_sorted_and_preserves_overrides(self) -> None:
        manifest = build_invocation_manifest(
            task_overrides=["training.epochs=10"],
            config_choices={"model": "fmg-ctct-2", "data": "sp500-crsp-compustat"},
        )

        self.assertEqual(manifest["task_overrides"], ["training.epochs=10"])
        self.assertEqual(list(manifest["config_choices"]), ["data", "model"])

    def test_source_snapshot_preserves_exact_bytes(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "module.py").write_bytes(b"print('audit')\r\n")

            snapshot = build_source_snapshot(root)

            self.assertEqual(b64decode(snapshot["files"]["module.py"]), b"print('audit')\r\n")
            self.assertEqual(len(snapshot["source_sha256"]), 64)

    def test_environment_manifest_records_interpreter_packages_and_torch(self) -> None:
        manifest = build_environment_manifest()

        self.assertEqual(manifest["schema_version"], 1)
        self.assertEqual(manifest["python"]["implementation"], "CPython")
        self.assertTrue(any(package["name"] == "torch" for package in manifest["packages"]))
        self.assertIn("accelerator", manifest["torch"])


if __name__ == "__main__":
    unittest.main()
