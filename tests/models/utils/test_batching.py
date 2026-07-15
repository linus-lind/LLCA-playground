import unittest

import pandas as pd

from llca.models.utils.batching import build_batches


class BatchMetadataTest(unittest.TestCase):
    def test_batch_records_dates_observations_and_instrument_union(self) -> None:
        dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
        index = pd.MultiIndex.from_tuples(
            [
                (dates[0], "A"),
                (dates[0], "B"),
                (dates[1], "B"),
                (dates[1], "C"),
                (dates[2], "A"),
            ],
            names=["date", "instrument"],
        )

        batches = build_batches(index, batch_size=2)

        self.assertEqual(len(batches), 2)
        self.assertEqual(batches[0].start_date, dates[0])
        self.assertEqual(batches[0].end_date, dates[1])
        self.assertEqual(len(batches[0].dates), 2)
        self.assertEqual(batches[0].observations, 4)
        self.assertEqual(batches[0].n_max, 3)


if __name__ == "__main__":
    unittest.main()
