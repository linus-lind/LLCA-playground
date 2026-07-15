import unittest

from llca.analytics.utils.manifest_compatibility import (
    canonical_data_manifest,
    canonical_training_manifest,
)
from llca.data.versioning import DataVersioningError


def _training_manifest(version: int) -> dict[str, object]:
    return {
        "schema_version": version,
        "experiment_name": "test",
        "data": {},
        "preprocessing": {},
        "features": {},
        "masking": {},
        "loss": {"name": "portfolio"},
        "model": {"name": "model"},
        "training": {"name": "torch"},
        "split": {"name": "single"},
    }


class AnalyticsManifestCompatibilityTest(unittest.TestCase):
    def test_version_only_revision_is_canonicalized_without_mutating_artifact(self) -> None:
        training = _training_manifest(2)
        data = {"schema_version": 2, "plan": {}, "sources": {}, "datasets": {}}

        canonical_training = canonical_training_manifest(training)
        canonical_data = canonical_data_manifest(data)

        self.assertEqual(canonical_training["schema_version"], 1)
        self.assertEqual(canonical_data["schema_version"], 1)
        self.assertEqual(training["schema_version"], 2)
        self.assertEqual(data["schema_version"], 2)

    def test_unknown_revisions_remain_invalid(self) -> None:
        with self.assertRaisesRegex(ValueError, "schema_version"):
            canonical_training_manifest(_training_manifest(3))
        with self.assertRaisesRegex(DataVersioningError, "schema_version"):
            canonical_data_manifest(
                {"schema_version": 3, "plan": {}, "sources": {}, "datasets": {}}
            )


if __name__ == "__main__":
    unittest.main()
