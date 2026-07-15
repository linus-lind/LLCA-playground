import unittest
from unittest.mock import patch

from llca.training.modules.training_diagnostics import PanelBatchMetadata
from llca.training.training_tracker import MlflowTrainingTracker


class TrainingTrackerTest(unittest.TestCase):
    @patch("llca.training.training_tracker.MlflowClient")
    def test_logs_manifest_observation_throughput_and_clipping_share(
        self, client_type: object
    ) -> None:
        client = client_type.return_value  # type: ignore[attr-defined]
        tracker = MlflowTrainingTracker("run", batch_log_interval=1)
        metadata = PanelBatchMetadata(
            index=1,
            observations=7,
            start_date="2024-01-02",
            end_date="2024-01-03",
            dates=2,
            entities=4,
        )
        tracker.begin(2, steps_per_epoch=1, batch_manifest=[metadata])
        tracker.on_batch_end(
            0.5,
            grad_norm=2.0,
            clipped=True,
            observations=7,
            diagnostics={"gradient/norm_after_clip": 1.0},
        )

        client.log_table.assert_called_once()
        table = client.log_table.call_args.args[1]
        self.assertEqual(table.loc[0, "start_date"], "2024-01-02")
        entries = client.log_batch.call_args.kwargs["metrics"]
        metrics = {entry.key: entry.value for entry in entries}
        self.assertEqual(metrics["gradient/clipped_step_fraction"], 1.0)
        self.assertEqual(metrics["gradient/norm_after_clip"], 1.0)
        self.assertGreater(metrics["throughput/samples_per_sec"], 0.0)


if __name__ == "__main__":
    unittest.main()
