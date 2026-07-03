# Audit, versioning, and archives

LLCA assigns one non-overlapping responsibility to each persistence layer:

| Layer | Responsibility |
| --- | --- |
| Git | Source code and Hydra source configurations |
| Hydra | Configuration composition, command-line overrides, and short-lived execution logs |
| MLflow | Runs, metrics, checkpoints, registered models, manifests, and analytics reports |
| DVC | Restorable content-addressed raw datasets |

## Training artifacts

Every new cross-validation parent and fold run receives three JSON artifacts:

- `pipeline/training_manifest.json` contains only the resolved data, preprocessing,
  feature, masking, loss, model, training, and split contract. Analytics, recovery policy,
  Hydra internals, and the MLflow location are excluded.
- `pipeline/data_manifest.json` binds logical datasets to raw SHA-256 values, DVC hashes,
  complete DVC pointers, and processed-panel fingerprints and schemas.
- `pipeline/hydra_invocation.json` preserves Hydra group choices and task overrides.
- `pipeline/source_snapshot.json` stores the exact executable package sources as base64,
  so a run remains reconstructable even when it started from a dirty Git worktree.

The training and data manifests are independently fingerprinted in MLflow tags. Registered
model versions point directly to the training manifest of their producing fold. Recovery
checks the artifacts, fingerprints, data provenance, checkpoint, source fingerprint, and
fold boundaries before loading mutable state.

Legacy runs containing only `pipeline/config.json` remain loadable. Their raw DVC hashes
and source fingerprint are still checked, but they do not gain a historical processed-data
manifest retroactively.

## DVC raw-data policy

At the start of a new training run, each unique configured raw file is processed once:

1. `dvc add` updates its pointer.
2. `dvc push` uploads the content to the configured default remote immediately.
3. `dvc status --cloud --json` must confirm synchronization.
4. SHA-256, size, DVC content hash, remote name, and the complete pointer are stored in the
   run's data manifest.

Multiple logical datasets using the same file share one archived source. Derived feature
panels are no longer written to `data/processed` or added to DVC. Their ordered values,
index, column labels, and dtypes are hashed in bounded-memory chunks instead.

The configured DVC remote is currently local. For disaster recovery, point it at a
different physical disk or an object-storage remote. DVC pointer files under `data/` should
be committed with the rest of this migration; the MLflow data manifest nevertheless keeps
each run restorable even if a later pointer changes.

Restore the exact raw inputs of a parent or fold run with:

```powershell
.\.venv\Scripts\python.exe -m llca.data.restore `
  --run-id <MLFLOW_RUN_ID> `
  --tracking-uri sqlite:///mlflow.db
```

Existing files with a different SHA-256 are never replaced unless `--force` is supplied.
After pulling, every restored file is checked against the archived SHA-256.

The old `data/processed/*.parquet` and matching `.dvc` files have been removed from this
workspace. If they reappear from an older checkout, they are migration leftovers and are
not read or updated by current code.

## Analytics audit

Training and analytics use separate Hydra roots:

- `hydra/configs/train.yaml`
- `hydra/configs/analytics.yaml`

An analytics execution stores `analytics_manifest.json` beside its local report and creates
a separate MLflow run in `${experiment_name}-analytics`. The run records exact model names,
versions and run IDs, evaluation dates, common observation count, analytics settings,
source provenance, and a snapshot of every generated report file. Analytics settings are
therefore never part of a new model's training identity.

## Hydra retention

Hydra output directories remain useful while a job is running or when startup fails before
MLflow can persist diagnostics. They are not the durable experiment record and are excluded
from experiment archives. Once successful runs and failures have been diagnosed, they may
be removed with the existing `make clean` task or an operating-system retention policy.

## MLflow archive

The local SQLite database and `mlruns` artifact store are archived independently from DVC:

```powershell
make experiment-backup
```

The target defaults to the sibling directory `../LLCA-audit-archive` and can be overridden
through `LLCA_AUDIT_ARCHIVE_DIR`. Archiving refuses to run while MLflow contains active runs,
uses SQLite's online backup API, copies the artifact store into a new immutable timestamped
directory, and writes SHA-256 and size for every archived file.

Verify any snapshot later with:

```powershell
.\.venv\Scripts\python.exe scripts\archive_experiments.py `
  --verify <ARCHIVE_DIRECTORY>
```

Restore a verified snapshot to its recorded original `mlflow.db` and `mlruns` paths with:

```powershell
.\.venv\Scripts\python.exe scripts\archive_experiments.py `
  --restore <ARCHIVE_DIRECTORY>
```

Existing stores are never replaced unless `--force` is supplied, and an existing database
with active runs is never replaced. Restoration stages both the database and artifacts and
rolls back if replacement fails.

Do not run DVC garbage collection against raw-data versions required by retained MLflow
runs. Do not edit an archive in place; create a new snapshot instead.
