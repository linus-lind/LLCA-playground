import sqlite3
import unittest
from contextlib import closing
from pathlib import Path
from tempfile import TemporaryDirectory

from llca.core.experiment_archive import (
    ExperimentArchiveError,
    archive_experiment_store,
    restore_experiment_archive,
    verify_experiment_archive,
)


def _database(path: Path, status: str = "FINISHED") -> None:
    with closing(sqlite3.connect(path)) as connection:
        connection.execute("CREATE TABLE runs (run_uuid TEXT, status TEXT)")
        connection.execute("INSERT INTO runs VALUES ('run-1', ?)", (status,))
        connection.commit()


class ExperimentArchiveTest(unittest.TestCase):
    def test_creates_and_verifies_immutable_snapshot(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            database = root / "mlflow.db"
            artifacts = root / "mlruns"
            artifacts.mkdir()
            (artifacts / "artifact.txt").write_text("model", encoding="utf-8")
            _database(database)

            archive = archive_experiment_store(
                database=database,
                artifacts=artifacts,
                archive_root=root / "archive",
            )

            verify_experiment_archive(archive)
            (archive / "mlruns" / "artifact.txt").write_text("changed", encoding="utf-8")
            with self.assertRaisesRegex(ExperimentArchiveError, "verification failed"):
                verify_experiment_archive(archive)

    def test_restores_verified_snapshot_to_original_paths(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            database = root / "mlflow.db"
            artifacts = root / "mlruns"
            artifacts.mkdir()
            (artifacts / "artifact.txt").write_text("model", encoding="utf-8")
            _database(database)
            archive = archive_experiment_store(
                database=database,
                artifacts=artifacts,
                archive_root=root / "archive",
            )
            database.unlink()
            (artifacts / "artifact.txt").unlink()
            artifacts.rmdir()

            restored_database, restored_artifacts = restore_experiment_archive(archive)

            self.assertEqual(restored_database, database)
            self.assertEqual(
                (restored_artifacts / "artifact.txt").read_text(encoding="utf-8"),
                "model",
            )

    def test_refuses_snapshot_while_run_is_active(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            database = root / "mlflow.db"
            artifacts = root / "mlruns"
            artifacts.mkdir()
            _database(database, "RUNNING")

            with self.assertRaisesRegex(ExperimentArchiveError, "runs are active"):
                archive_experiment_store(
                    database=database,
                    artifacts=artifacts,
                    archive_root=root / "archive",
                )


if __name__ == "__main__":
    unittest.main()
