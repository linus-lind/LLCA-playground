# Audit, versioning, and archives

Each persistence layer has one responsibility:

| Layer | Responsibility |
| --- | --- |
| Git | Source code and Hydra source configurations |
| Hydra | Composition, overrides, and short-lived startup/execution logs |
| MLflow | Runs, metrics, checkpoints, registered models, manifests, and reports |
| DVC | Restorable content-addressed raw datasets |
| Preparation cache | Disposable acceleration; never audit evidence |

## Training evidence

Every new training-plan parent and fold receives five JSON artifacts:

- `pipeline/training_manifest.json` contains the semantic resolved data, preprocessing,
  features, masking, loss, model, training, and split contract. Analytics, recovery policy,
  Hydra internals, MLflow location, cache path, and CSV chunk size are excluded.
- `pipeline/data_manifest.json` binds the resolved model-aware data plan to raw SHA-256,
  DVC identities/pointers, and ordered final feature-panel fingerprints and schemas.
- `pipeline/hydra_invocation.json` preserves Hydra group choices and task overrides.
- `pipeline/source_snapshot.json` stores the exact executable package source, including a
  dirty worktree.
- `pipeline/environment_manifest.json` records interpreter, packages, platform,
  PyTorch/CUDA/cuDNN, and accelerators.

Manifest fingerprints are searchable MLflow tags. Registered model versions point to the
producing fold and its immutable test interval. Recovery validates manifests, provenance,
checkpoint, source fingerprint, and fold boundaries before loading mutable state.

Metrics are deliberately task-specific. The tracker logs generic progress/throughput and
only diagnostics returned by the selected objective or estimator. A regression run does
not acquire portfolio variance; a portfolio run does not acquire classification accuracy.

Runs missing any canonical training or data manifest are rejected by recovery and
analytics. This keeps every model-producing run under one auditable contract.

Analytics preserves read access to registered version-1 research models through two
lossless canonicalizations: the version-only manifest revision `2` is validated as the
current structure, and the retired FMG bundle labels `ranking`/`allocation` map to
`portfolio`. Stored MLflow artifacts remain unchanged. Training and recovery accept only
the current canonical contracts.

## DVC raw-data policy

At the start of a new training invocation, each unique physical source selected by the
model's data plan is handled once:

1. `dvc add` updates its pointer.
2. `dvc push` uploads the content to the default remote.
3. `dvc status --cloud --json` confirms synchronization.
4. SHA-256, size, DVC hash, remote, and complete pointer enter the run manifest.

Several logical datasets backed by one CSV share the archived source and ingestion scan.
Derived panels are not written to `data/processed` or stored in DVC; their values, order,
labels, and dtypes are fingerprinted in bounded-memory chunks.

Restore exact inputs for a parent or fold run with:

```powershell
.\.venv\Scripts\python.exe -m llca.data.restore `
  --run-id <MLFLOW_RUN_ID> `
  --tracking-uri sqlite:///mlflow.db
```

Existing data with a different SHA-256 is never replaced unless `--force` is supplied.
Do not run DVC garbage collection against versions referenced by retained MLflow runs.

## Analytics verification

Analytics downloads both training and data manifests before
preparation. Every local raw source must match the archived path, size, and SHA-256. The
same verified SHA identifies a compatible preparation-cache entry, so verification does
not sacrifice deterministic reuse. Different recorded expectations for one local path
are rejected; restore the required version through DVC before evaluating that model.

An analytics execution creates a separate run in `${experiment_name}-analytics` and stores
model names/versions/run IDs, evaluation interval, common coverage, analytics settings,
source/environment provenance, report hashes, and a report snapshot. Analytics settings
never enter model-training identity.

## Hydra retention

Hydra directories are useful when startup fails before MLflow can persist diagnostics.
They are not the durable record. `make clean` may remove them and disposable caches, but
never MLflow stores, checkpoints, DVC data, reports, or archives.

## MLflow archive

Archive the SQLite database and `mlruns` store independently from DVC:

```powershell
make experiment-backup
```

The default destination is `../LLCA-audit-archive`; override it with
`LLCA_AUDIT_ARCHIVE_DIR`. Archiving refuses active runs, uses SQLite online backup, copies
the artifact store into an immutable timestamped directory, and records hashes and sizes.

Verify or restore a snapshot with:

```powershell
.\.venv\Scripts\python.exe scripts\archive_experiments.py --verify <ARCHIVE_DIRECTORY>
.\.venv\Scripts\python.exe scripts\archive_experiments.py --restore <ARCHIVE_DIRECTORY>
```

Existing stores are not replaced without `--force`, and an active database is never
replaced. Do not edit archives in place; create a new snapshot.
