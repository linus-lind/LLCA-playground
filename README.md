# LLCA model-training pipeline

LLCA is a configuration-driven research pipeline for reproducible model training. A model
plugin declares the logical datasets, entity scope, data representation, supported
objectives, and training engines it needs. The pipeline then loads only those inputs,
applies registered preprocessing and feature steps, executes a chronological training
strategy, and stores task-specific evidence in MLflow.

The current article contributes four FMG architectures, but the orchestration is not tied
to this family or to neural networks. PyTorch and scikit-learn-compatible estimators have
separate lifecycle bases; additional data assemblers, objectives, training policies, and
split strategies are registry extensions rather than branches in the application entry
point.

## FMG article models

| Model | Temporal encoder | Cross-sectional operation | Native output |
| --- | --- | --- | --- |
| FMG-CTCT-2 | causal CNN + temporal Transformer | every asset queries every asset | one score per asset |
| FMG-CTCT-1 | causal CNN + temporal Transformer | target queries the full universe | one target allocation |
| FMG-CTT | causal CNN + temporal Transformer | target asset only | one target allocation |
| FMG-CLSTM | causal CNN + LSTM | target asset only | one target allocation |

The target-only CTT and CLSTM capability filters the CSV during ingestion, before numeric
conversion, preprocessing, feature construction, and alignment. CTCT-1 retains the full
input universe because it is cross-sectional, but loads supervision only for its target.
CTCT-2 retains the full universe for both.

## Installation

Python 3.13 is required. Install PyTorch for the intended CPU/CUDA platform before or with
the project dependencies.

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m pre_commit install `
  --config .git-hooks-config.yaml -t pre-commit -t pre-push
```

Raw CRSP/Compustat-derived inputs are not distributed. Configure the DVC remote in
`.dvc/config` and restore the licensed CSV sources below `data/`.

When running an installed wheel outside a source checkout, the current directory is the
runtime workspace for `data/`, MLflow, checkpoints, and reports. Set the process environment
variable `LLCA_PROJECT_ROOT` to choose a different workspace explicitly.

## Training

There is one Hydra application root, `train.yaml`. Reusable group configurations describe
data, transformations, objective, model, training engine, and split strategy; an
`experiment` preset only composes a convenient research combination.

The default is FMG-CTCT-2:

```powershell
.\.venv\Scripts\python.exe -m llca
```

A three-epoch integration run of all article models can use Apple (CRSP PERMNO `14593`) for
the instrument-specific variants:

```powershell
.\.venv\Scripts\python.exe -m llca experiment=fmg-ctct-2 training.epochs=3

.\.venv\Scripts\python.exe -m llca experiment=fmg-ctct-1 `
  model.target.entity_id=14593 training.epochs=3

.\.venv\Scripts\python.exe -m llca experiment=fmg-ctt `
  model.target.entity_id=14593 training.epochs=3

.\.venv\Scripts\python.exe -m llca experiment=fmg-clstm `
  model.target.entity_id=14593 training.epochs=3
```

An objective can be changed independently when the model declares compatibility. For
example, CTCT-2 can learn a return regression instead of a portfolio ranking:

```powershell
.\.venv\Scripts\python.exe -m llca experiment=fmg-ctct-2 `
  loss=mse training.epochs=3
```

That run emits regression diagnostics such as MSE and MAE. Portfolio-only fields such as
variance, turnover, or net exposure are not synthesized or logged. The same rule applies
to classification and future custom objectives.

Every selected raw source is added and pushed to DVC before model preparation. Deterministic
prepared data is cached by source SHA-256, selection, transformations, assembler, and
relevant implementation source. The cache is disposable; MLflow manifests and DVC remain
the audit record. Interrupted checkpoint-capable runs can be resumed as described in
[`docs/TRAINING_RECOVERY.md`](docs/TRAINING_RECOVERY.md).

## Data and extension architecture

A data configuration may declare any number of logical CSV datasets. Each dataset owns its
index mapping, semantic kind, frequency, selected columns, preprocessing chain, and feature
chain. Date-only context, entity panels, single instruments, and event-keyed inputs can
coexist. Physical files shared by several logical views are scanned once per preparation.

The built-in `aligned_panel` assembler creates leakage-safe point-in-time panels for the
FMG models. The `independent` assembler preserves native indices and frequencies for
models that consume ticks, events, text-derived tables, or their own joins. New storage
formats can later be added behind ingestion without changing estimator or execution
contracts.

The complete architecture and extension recipes are in
[`docs/PIPELINE_ARCHITECTURE.md`](docs/PIPELINE_ARCHITECTURE.md). The model-specific adapter
contract is summarized in [`docs/EXTENDING_MODELS.md`](docs/EXTENDING_MODELS.md).

## Held-out analytics

Analytics loads immutable registered model versions, verifies local raw files against each
run's archived SHA-256 data manifest, reconstructs the stored pipeline, and restricts
predictions to the registered test interval. Cross-model comparisons use the intersection
of prediction/target coverage and reject incompatible target values or return conventions.

```powershell
.\.venv\Scripts\python.exe -m llca.analytics `
  analytics.models='[{name: fmg-ctct-2, version: 1, label: FMG-CTCT-2}]' `
  analytics.show_plots=false
```

Publication artifacts are written below `reports/analytics` and archived in a dedicated
MLflow analytics run. Registry version numbers, not mutable aliases, are the analytical
identity.

## Quality gate

```powershell
.\.venv\Scripts\python.exe -m ruff check src tests scripts
.\.venv\Scripts\python.exe -m ruff format --check src tests scripts
.\.venv\Scripts\python.exe -m mypy src scripts
.\.venv\Scripts\python.exe scripts/validate_configs.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

`make check` runs the same gate where GNU Make is available. `make clean` removes only
disposable caches and Hydra logs. Persistence boundaries and restore procedures are
documented in [`docs/AUDIT_AND_VERSIONING.md`](docs/AUDIT_AND_VERSIONING.md).
