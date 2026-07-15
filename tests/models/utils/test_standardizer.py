import unittest

import torch

from llca.models.utils.standardizer import Standardizer


class StandardizerRestoreTest(unittest.TestCase):
    def test_restores_statistics_on_requested_device(self) -> None:
        state = {
            "mean": torch.tensor([1.0, 2.0]),
            "std": torch.tensor([2.0, 4.0]),
        }

        standardizer = Standardizer.from_state_dict(state, device="cpu")
        transformed = standardizer.transform(torch.tensor([[3.0, 6.0]]))

        self.assertEqual(transformed.device.type, "cpu")
        torch.testing.assert_close(transformed, torch.ones(1, 2))


if __name__ == "__main__":
    unittest.main()
