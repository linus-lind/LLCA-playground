# Extending models

A model plugin owns architecture-specific choices. It does not own CSV scanning, Hydra
startup, MLflow run orchestration, DVC archival, or chronological plan execution.

## Choose the estimator lifecycle

All models implement `Estimator[DataT]`. Use the narrowest reusable base:

- `TorchTrainableEstimator[BatchT]` for gradient-based models. Implement
  `_build_training_task`, prediction, evaluation metadata, and inference payload hooks.
  The shared trainer provides device selection, precision, optimization, clipping,
  diagnostics, early stopping, atomic checkpoints, exact recovery, and best-state restore.
- `SklearnEstimator[DataT]` for random forests, boosted trees, linear models, and compatible
  libraries. Implement `_fit_backend`, prediction, evaluation metadata, and persistence
  payload hooks. `_fit_backend` returns only metrics meaningful to its task.
- `Estimator[DataT]` directly when a backend needs a genuinely different lifecycle.

Network modules remain ordinary backend code. The four article networks live one per file
under `llca.models.fmg`; their estimator adapters mirror this structure under
`llca.models.estimators.fmg`, and registry bindings mirror it under
`llca.mappers.model.fmg`. Shared family mechanics belong in a `base` module, not in one
concrete architecture. Component names are canonical and aliases are not registered.

## Declare capabilities

Register construction, validation, and `ModelCapabilities` under one canonical name. The
capability resolves:

- every logical `DatasetRequirement`;
- `TARGET` or `UNIVERSE` entity scope per dataset;
- the primary time dataset;
- supported `ObjectiveKind` values;
- supported `TrainingEngine` values;
- the estimator's data assembler (`aligned_panel`, `independent`, or a plugin).

This declaration is what lets target-only models filter rows before preprocessing while a
cross-sectional model keeps its context universe. The application entry point contains no
model-name conditionals.

The model builder receives both the constructed loss and its resolved Hydra configuration.
A classical adapter can therefore translate an objective into a backend criterion without
coupling the core to that library.

## Return typed predictions

`predict` returns `PredictionOutput` with one native semantic kind:

- `regression`: numerical forecasts;
- `binary`: one decision score and optional positive-class probabilities;
- `multiclass`: one score column and optional probability column per class;
- `portfolio`: native portfolio scores or allocations interpreted by the portfolio objective.

`EvaluationSpec` binds the primary calendar and supervision dataset/column. Use the built-in
`panel` or `rows` objective layout, or provide a pure adapter for a different tensor shape.
This keeps analytical evaluation independent of concrete architecture classes.

## Add configuration and tests

Add a model group YAML under `hydra/configs/model/`. Put only architecture and data-role
bindings there. Optimization runtime belongs in `training/`, objective parameters in
`loss/`, and research compositions in `experiment/`.

At minimum test:

- Hydra composition and invalid configurations;
- declared data scope and engine/objective compatibility;
- output shape, semantic prediction kind, and gradients where applicable;
- training and inference on a minimal data object;
- persistence round-trip;
- absence of unrelated objective metrics.

See [`PIPELINE_ARCHITECTURE.md`](PIPELINE_ARCHITECTURE.md) for dataset, transform,
objective, assembler, and split extension points.
