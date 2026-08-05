from __future__ import annotations

import importlib
import sys
import unittest
import warnings


class PyfuncTest(unittest.TestCase):
    def test_import_does_not_request_mlflow_list_input_type_hint(self) -> None:
        sys.modules.pop("llca.models.pyfunc", None)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "error",
                message=r"Type hint used in the model's predict function is not supported.*",
                category=UserWarning,
            )
            module = importlib.import_module("llca.models.pyfunc")

        self.assertTrue(module.Pyfunc._skip_type_hint_validation)


if __name__ == "__main__":
    unittest.main()
