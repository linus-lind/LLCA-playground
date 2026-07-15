import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from llca.analytics.utils.config import RegisteredModelConfig
from llca.analytics.utils.model_loader import get_registered_model_metadata
from llca.core.artifacts import DATA_MANIFEST_ARTIFACT, TRAINING_MANIFEST_ARTIFACT


def _training_manifest() -> dict[str, object]:
    return {
        "schema_version": 1,
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


class RegisteredModelMetadataTest(unittest.TestCase):
    def test_canonical_artifacts_are_authoritative_without_path_tags(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            training = root / "training_manifest.json"
            data = root / "data_manifest.json"
            training.write_text(json.dumps(_training_manifest()), encoding="utf-8")
            data.write_text(
                json.dumps({"schema_version": 1, "plan": {}, "sources": {}, "datasets": {}}),
                encoding="utf-8",
            )

            client = MagicMock()
            client.get_model_version.return_value = SimpleNamespace(
                run_id="run-1",
                tags={"test_start": "2024-01-01", "test_end": "2024-12-31"},
            )
            client.download_artifacts.side_effect = lambda _run_id, artifact: str(
                training if artifact == TRAINING_MANIFEST_ARTIFACT else data
            )

            config = RegisteredModelConfig(name="model", version=1, label="model-v1")
            with patch("llca.analytics.utils.model_loader.MlflowClient", return_value=client):
                metadata = get_registered_model_metadata(config, "sqlite:///test.db")

        self.assertEqual(metadata.run_id, "run-1")
        self.assertEqual(metadata.data_manifest["schema_version"], 1)
        self.assertEqual(
            [call.args[1] for call in client.download_artifacts.call_args_list],
            [TRAINING_MANIFEST_ARTIFACT, DATA_MANIFEST_ARTIFACT],
        )


if __name__ == "__main__":
    unittest.main()
