import unittest

from llca.models.estimators.fmg.base import _validate_bundle_format


def _bundle(version: int) -> dict[str, object]:
    """A payload shaped like the current inference bundle at the given format version."""
    return {
        "format_version": version,
        "config": {},
        "feature_columns": [],
        "context_columns": [],
        "model_state_dict": {},
        "feature_ewma": {},
        "prediction_kind": "portfolio",
    }


class FmgBundleCompatibilityTest(unittest.TestCase):
    def test_current_revision_is_accepted(self) -> None:
        _validate_bundle_format(_bundle(3), "fmg-test")

    def test_unknown_bundle_version_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsupported fmg-test bundle format"):
            _validate_bundle_format(_bundle(4), "fmg-test")

    def test_pre_ewma_revisions_are_rejected(self) -> None:
        """Format versions 1 and 2 persisted fitted scalers rather than the EWMA normalizer
        state ``_restore`` now requires under 'feature_ewma', so neither can be restored and
        both must be rejected at validation instead of failing later inside restoration."""
        for version in (1, 2):
            with self.assertRaisesRegex(ValueError, "unsupported fmg-test bundle format"):
                _validate_bundle_format(_bundle(version), "fmg-test")


if __name__ == "__main__":
    unittest.main()
