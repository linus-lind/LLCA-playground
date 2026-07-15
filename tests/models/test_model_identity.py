import unittest

from llca.mappers.model.mapper import model_registry
from llca.models import fmg
from llca.models.estimators.fmg import FmgCtct2Estimator


class ModelIdentityTest(unittest.TestCase):
    def test_only_canonical_model_identities_are_registered(self) -> None:
        self.assertEqual(FmgCtct2Estimator._MODEL_NAME, "fmg-ctct-2")
        self.assertEqual(
            fmg.__all__,
            ["FmgClstm", "FmgCtct1", "FmgCtct2", "FmgCtt"],
        )
        self.assertTrue(model_registry.is_registered("fmg-ctct-2"))
        self.assertEqual(
            model_registry.available(),
            ["fmg-clstm", "fmg-ctct-1", "fmg-ctct-2", "fmg-ctt"],
        )


if __name__ == "__main__":
    unittest.main()
