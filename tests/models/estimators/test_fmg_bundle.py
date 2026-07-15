import unittest

from llca.models.estimators.fmg.base import _validate_bundle_format


def _bundle(version: int) -> dict[str, object]:
    return {
        "format_version": version,
        "config": {},
        "feature_columns": [],
        "context_columns": [],
        "model_state_dict": {},
        "feature_scaler": {},
        "context_scaler": {},
        "prediction_kind": "portfolio",
    }


class FmgBundleCompatibilityTest(unittest.TestCase):
    def test_version_only_revision_requires_exact_payload_shape(self) -> None:
        _validate_bundle_format(_bundle(2), "fmg-test")

        incompatible = _bundle(2) | {"unknown": True}
        with self.assertRaisesRegex(ValueError, "unsupported fmg-test bundle format"):
            _validate_bundle_format(incompatible, "fmg-test")

    def test_unknown_bundle_version_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsupported fmg-test bundle format"):
            _validate_bundle_format(_bundle(3), "fmg-test")


if __name__ == "__main__":
    unittest.main()
