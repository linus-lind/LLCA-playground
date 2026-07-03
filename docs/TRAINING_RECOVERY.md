# Training recovery

Training recovery is an execution concern and is configured through the Hydra
`recovery` group. Model, loss, split, and training values are restored from the selected
run's `pipeline/training_manifest.json`; current YAML values do not alter an interrupted
run. Runs created before the manifest migration remain readable through the legacy
`pipeline/config.json` fallback.

## Commands

List unfinished fold runs without preparing data or starting training:

```powershell
.\.venv\Scripts\python.exe -m llca recovery=list
```

Resume when exactly one candidate is valid:

```powershell
.\.venv\Scripts\python.exe -m llca recovery=auto
```

Select a fold run or its cross-validation parent explicitly:

```powershell
.\.venv\Scripts\python.exe -m llca `
  recovery=explicit `
  recovery.run_id=<MLFLOW_RUN_ID>
```

Source mismatches are rejected by default. A deliberate source migration can be allowed
explicitly, while data and checkpoint compatibility remain mandatory:

```powershell
.\.venv\Scripts\python.exe -m llca `
  recovery=explicit `
  recovery.run_id=<MLFLOW_RUN_ID> `
  recovery.allow_source_mismatch=true
```

The default `recovery=off` always starts a new cross-validation job.

## Safety and lifecycle

A fold is resumable only when its resolved training manifest and `latest.pt` are available.
Recovery validates checkpoint schema, model configuration, optimizer identity, the data
manifest fingerprint, data hashes, source provenance, and fold date boundaries before
applying optimizer or model state. A per-run operating-system lock prevents concurrent
training of the same fold.

Checkpoint replacement is atomic. A process interruption during a save therefore leaves
the previous complete checkpoint in place.

MLflow runs advance through these phases:

```text
prepared -> training -> trained -> model_logged -> registered -> completed
```

Recovery is idempotent across these phases. If training already ended, it restores the best
state and completes only the missing model logging or registry steps. For walk-forward jobs,
completed folds are skipped, the interrupted fold is resumed, and later folds are started
normally under the original parent run.
