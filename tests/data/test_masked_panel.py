import unittest

import pandas as pd

from llca.data.modules.masked_panel import MaskedPanel


class MaskedPanelTest(unittest.TestCase):
    def test_rejects_misaligned_axes_and_duplicate_rows(self) -> None:
        index = pd.Index([1, 2], name="row")
        values = pd.DataFrame({"x": [1.0, 2.0]}, index=index)
        observed = pd.DataFrame({"x": [True, True]}, index=index)
        age = pd.DataFrame({"x": [0, 0]}, index=index)
        segment = pd.Series([0, 0], index=index)

        with self.assertRaisesRegex(ValueError, "observed index"):
            MaskedPanel(values, observed.iloc[::-1], age, segment)

        duplicate_index = pd.Index([1, 1], name="row")
        with self.assertRaisesRegex(ValueError, "index must be unique"):
            MaskedPanel(
                values.set_axis(duplicate_index),
                observed.set_axis(duplicate_index),
                age.set_axis(duplicate_index),
                segment.set_axis(duplicate_index),
            )


if __name__ == "__main__":
    unittest.main()
