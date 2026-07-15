# Training recovery

Recovery is an execution concern configured through the Hydra `recovery` group. Model,
objective, data, split, and training values come from the selected run's immutable training
manifest; current YAML does not alter an interrupted run. Only runs created with the
current manifest, lifecycle-tag, provenance, and checkpoint schemas are resumable.

## Commands

List unfinished fold runs without preparing data or starting training:

```powershell
.\.venv\Scripts\python.exe -m llca recovery=list
```

Resume when exactly one candidate is valid:

```powershell
.\.venv\Scripts\python.exe -m llca recovery=auto
```

Select a fold or its training-plan parent explicitly:

```powershell
.\.venv\Scripts\python.exe -m llca `
  recovery=explicit `
  recovery.run_id=<MLFLOW_RUN_ID>
```

A deliberate source migration can be allowed while data and checkpoint compatibility
remain mandatory:

```powershell
.\.venv\Scripts\python.exe -m llca `
  recovery=explicit `
  recovery.run_id=<MLFLOW_RUN_ID> `
  recovery.allow_source_mismatch=true
```

The default `recovery=off` starts a new execution of the selected split plan.

## Safety and lifecycle

A fold is resumable only when its estimator lifecycle has produced a compatible atomic
checkpoint. The built-in PyTorch lifecycle does so every epoch and captures optimizer,
best state, early-stopping counters, and RNG state. The generic scikit-learn lifecycle is
one-shot and is not resumable unless a concrete plugin explicitly implements incremental
checkpoint recovery.

Recovery validates checkpoint schema, model/training configuration, optimizer identity,
data manifest and hashes, source provenance, and fold boundaries. A per-run operating-
system lock prevents concurrent training of one fold.

MLflow phases are:

```text
prepared -> training -> trained -> model_logged -> registered -> completed
```

The lifecycle is idempotent. If fitting ended, recovery restores the best state and
performs only missing logging or registration. For walk-forward plans, completed folds are
skipped, the selected interrupted fold resumes, and later folds start normally below the
original parent.
