# Pipeline architecture

## Composition and execution

The application is a small orchestrator over registered contracts:

```text
Hydra composition
  -> aggregate validation
  -> model capabilities
  -> model-aware DataPlan
  -> selective ingestion
  -> per-dataset preprocessing
  -> per-dataset features
  -> registered data assembler
  -> registered split strategy
  -> estimator + training policy
  -> objective-specific MLflow evidence
```

The training and analytics applications are two independent Hydra config roots,
`hydra/configs/training/train.yaml` and `hydra/configs/analytics/analytics.yaml`. Each owns
its own `data`, `preprocessing`, `features`, and `masking` groups, so the two applications
can use structurally different datasets; nothing is shared implicitly. Within the training
root, group YAML files are reusable components and experiment presets are thin compositions
that must not duplicate an application root for every model.

## Data contracts

`data.datasets` is an open mapping of logical dataset names. A dataset specifies:

- a CSV path;
- semantic `kind` (`panel`, `context`, `events`, or `table`);
- a descriptive frequency such as `daily`, `tick`, or `point_in_time`;
- canonical-to-raw index and value-column mappings;
- optional auxiliary columns required during preprocessing.

The first configured index role is time. An entity role is optional, and additional index
roles such as event or message identifiers are preserved. The current reader is CSV by
design; storage is isolated in ingestion so Parquet, databases, streams, or feature stores
can later implement the same logical output contract.

`DataRequirements` belongs to the model plugin. It names only datasets the model uses and
sets the entity scope of each one. `build_data_plan` combines those requirements with an
optional user universe selection. The result is pushed into ingestion:

- unused datasets are never opened;
- several logical views of one CSV share one scan;
- target filters are applied in chunks before conversion and transformations;
- global context without an entity index remains global;
- a target excluded by explicit user selection fails before I/O.

Preprocessing and features are independent ordered chains per logical dataset. Registry
entries declare their column dependencies, enabling common validation and useful runtime
errors without central knowledge of transform-specific fields.

## Data assemblers

An assembler converts independently transformed frames into the native `DataT` consumed by
an estimator and splitter.

- `aligned_panel` performs leakage-safe point-in-time alignment, membership masking, age
  construction, and segment construction for time/entity models.
- `independent` retains each frame's native index and frequency. Event, tick, news, or
  model-specific joins can be handled by an estimator or a dedicated registered assembler.

Do not force event data into the daily panel assembler. Register a focused adapter when a
model has a stable reusable join/encoding contract.

## Models and training engines

`Estimator[DataT]` is the backend-neutral contract used by execution, MLflow PyFunc, and
analytics loading. Backend bases isolate implementation details:

- `TorchTrainableEstimator` consumes a `TrainingConfig` and `TrainingTask`;
- `SklearnEstimator` consumes `SklearnTrainingConfig` and a one-shot backend fit;
- custom engines can register another `TrainingPolicy` and estimator lifecycle.

The model capability explicitly lists compatible engines. Invalid combinations therefore
fail during aggregate configuration validation, before data or tracking state changes.

Splitters are generic over the same `DataT`. Built-in single-split and walk-forward
strategies operate on aligned panels. A native-event model can register a splitter for its
own assembler output without changing training execution. Both strategies use the same
parent/fold MLflow lifecycle and immutable fold boundaries.

### Chronological splitting

Both built-in splitters are **end-anchored**: the newest observation is always the final scored
test date, and the scored layout `train | purge | validation | purge | test` is laid out backward
from there. For `N` ordered dates and `block = train + 2·purge + val + test`, the single split's
inclusive scored positions are `test_end = N-1`, `test_start = N-test`, `val_end = N-test-purge-1`,
`val_start = N-test-purge-val`, `train_end = N-test-purge-val-purge-1`, `train_start = N-block`.
The walk-forward splitter anchors its final fold the same way and steps earlier folds backward by
`step_size`, yielding them oldest-first. **Excess observations are dropped from the beginning**, so
adding history never changes which recent dates are scored, and every experiment reports on the
most recent data.

Two knobs are deliberately separated:

- **`purge`** guards against the forward-looking *label* realizing across a boundary, so it equals
  the supervision target's forward horizon (`1` for the current one-step return) — never the
  sequence length or a feature window. Configuration validation rejects a purge below that horizon.
- **`lookback`** is input history prepended to the train/validation slices (and, at evaluation,
  the test slice) so a model can score its first observation. It defaults to the estimator's
  `required_lookback` (`sequence_length + CNN buffer − 1` for temporal FMG models, `0` for
  point-in-time baselines). Because the split is end-anchored, differing lookbacks leave the scored
  dates identical, so a temporal model and a point-in-time baseline sharing the outer geometry are
  compared on exactly the same test window.

### Stock-feature standardization (FMG)

`FmgEstimator.inputs.features` (price/volume-derived stock features, e.g. returns,
volatility, momentum) is causally standardized by `EwmaStandardizer`
(`models/utils/ewma_standardizer.py`) — an exponentially-weighted z-score fit and advanced
independently per `(entity, feature)` pair, never pooled across entities or features. Its
half-life is set by `model.standardization.half_life` (default 126 observations) and is a
configured modeling choice, not a tuned hyperparameter.

`FmgEstimator.inputs.context` (firm characteristics, macro) receives no standardization —
those values enter the model exactly as preprocessing/features produced them, so their
absolute magnitude stays available to the network's context-conditioning pathway (e.g. size
or leverage modulating how a GRN weighs other inputs) instead of being normalized away.

The normalizer is fit once from the training split, then advances continuously and
causally as training, validation, and (from restored state, potentially in a different
process) test rows are windowed — splits are never refit independently, and a row is never
normalized using its own value or any later one. Its full state (per-entity mean/variance,
a bounded per-entity replay buffer, and the fitted half-life) is part of the FMG bundle
(`_inference_payload`/`_restore`), so a reloaded estimator's `predict` continues the same
recursion rather than restarting it.

## Objectives and metrics

Each loss plugin registers an `ObjectiveKind`: portfolio, regression, binary
classification, multiclass classification, or custom. Model capabilities declare which
kinds their native outputs support.

Structured torch objective results expose `diagnostic_metrics()`. The training tracker
logs exactly those names plus backend-generic progress and optimizer diagnostics. A
classical estimator's `_fit_backend` similarly returns its task metrics. The core never
assumes that all objectives have variance, turnover, accuracy, MSE, or any other
task-specific field.

Current examples:

- portfolio: return, variance, costs, exposure, and penalties;
- MSE: mean squared and mean absolute error;
- binary cross-entropy: accuracy, mean positive probability, and positive rate.

Native prediction semantics are also objective-aware. For example, CTCT-2 emits
`portfolio` predictions under the portfolio objective and `regression` predictions under
MSE; analytics will not apply portfolio normalization to regression output.

## MLflow, DVC, manifests, and cache

Before preparation, DVC archives only the physical raw sources selected by the data plan.
The data manifest records source SHA-256/DVC identity, logical bindings, selected entities,
and fingerprints of final feature panels. The training manifest contains semantic resolved
configuration but excludes runtime cache location and CSV chunk size.

The preparation cache covers source SHA-256, selected logical datasets/entities,
preprocessing, features, masking, assembler, and relevant implementation source. It is an
optimization only and may be deleted at any time. Analytics verifies each raw source
against the model run's manifest before using a compatible cached preparation.

MLflow stores parent-plan and fold evidence, source/environment snapshots, checkpoints,
models, and reports. Hydra output directories remain short-lived startup diagnostics; Git
stores source configurations; DVC stores restorable raw bytes.

## Extension checklist

1. Add or reuse logical datasets in a data group.
2. Register focused preprocessing/feature steps only when existing primitives are
   insufficient.
3. Select or register the estimator-native assembler.
4. Implement the model through the appropriate estimator lifecycle.
5. Register its builder, validator, and capabilities.
6. Register an objective kind and structured diagnostics if a new loss is required.
7. Reuse or register a training policy and splitter compatible with the model's `DataT`.
8. Add Hydra composition, failure, persistence, and metric-isolation tests.
9. Add the combination to `scripts/validate_configs.py` when it is a supported preset.
