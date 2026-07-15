import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from mlflow import MlflowClient

from llca.analytics.audit import log_analytics_report
from llca.analytics.reporting import PublicationReport


@patch.dict(os.environ, {"MLFLOW_ALLOW_FILE_STORE": "true"})
class AnalyticsAuditTest(unittest.TestCase):
    def test_report_and_manifest_are_archived_in_dedicated_run(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            report_dir = root / "report"
            report_dir.mkdir()
            (report_dir / "table.csv").write_text("metric,value\nloss,1\n", encoding="utf-8")
            tracking_uri = (root / "mlruns").resolve().as_uri()
            manifest = {
                "schema_version": 1,
                "models": [{"name": "model", "version": 2}],
                "source": {"sha256": "source-sha", "git_commit": "abc"},
            }

            run_id = log_analytics_report(
                manifest,
                PublicationReport(directory=report_dir, artifacts={}),
                tracking_uri=tracking_uri,
                experiment_name="analytics-audit",
            )

            client = MlflowClient(tracking_uri=tracking_uri)
            run = client.get_run(run_id)
            self.assertEqual(run.data.tags["llca.run_kind"], "analytics")
            artifacts = client.list_artifacts(run_id, "analytics")
            self.assertEqual(
                {item.path for item in artifacts}, {"analytics/manifest.json", "analytics/report"}
            )
            self.assertTrue((report_dir / "analytics_manifest.json").is_file())


if __name__ == "__main__":
    unittest.main()
