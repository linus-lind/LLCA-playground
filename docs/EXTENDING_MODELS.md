# Extending the model pipeline

The reusable boundary is an `Estimator`, not a particular neural-network architecture.
A new trainable model normally requires four small additions.

## 1. Implement the network

Create an ordinary `torch.nn.Module`. Keep panel preparation, optimization, tracking, and
serialization outside the network.

## 2. Implement a trainable estimator adapter

Subclass `TrainableEstimator[BatchType]` and implement:

- `_build_training_task(...)`: prepare training-only preprocessing state and return a
  `TrainingTask` containing the model, batches, train step, optional validation step, and
  optional batch metadata;
- `predict(...)`: return a validated `PredictionOutput`;
- `evaluation_spec`: identify the primary calendar, supervision binding, and built-in
  objective tensor layout (`panel` or `rows`). Supply a pure `objective_adapter` there only
  when neither standard layout matches the loss signature;
- `_inference_payload`, `_from_payload`, and `_restore`: persist inference state only.

The shared trainer supplies device resolution, BF16 autocasting, optimizers, clipping,
parameter/update diagnostics, early stopping, tracking, exact resume, and best-state
restoration.

Return `TrainingBatchOutput(loss=..., metrics_factory=...)` from the train step when the
model has custom diagnostics. The factory is invoked only on configured diagnostic steps.
Loss functions may return a scalar tensor or any object with a scalar tensor `.loss`.
Implement `diagnostic_metrics()` when structured scalar components should be logged and
included in analytical objective tables; no inheritance from a concrete loss class is
required.

## 3. Register construction and validation together

Add one mapper module using `model_registry.register(name)` and
`model_registry.register_validator(name)`. Model-specific settings, including output
diagnostics, belong under the model Hydra group. Generic execution settings remain under
the training group.

The objective and optimizer registries follow the same paired builder/validator pattern.
Portfolio objectives declare both `normalization` and `return_type`. Use
`bounded` when raw scores should determine cash usage, net exposure, side balance, and
relative weights subject only to `leverage` as a maximum gross-exposure cap. Use
`market_neutral` for a fixed-gross, zero-net cross-sectional portfolio, and `gross` for a
fixed-gross directional or single-asset portfolio. A supervision target produced by
`log_change` requires `return_type: log`; one produced by `simple_change` requires
`return_type: simple`. The validator checks known target transforms, while unrelated
model features may freely mix both conventions.

## 4. Add a Hydra model configuration

The estimator bundle stores its resolved model configuration. Each trained fold also logs
the complete resolved pipeline configuration to MLflow. Analytics therefore rebuilds the
correct data, feature, target, and objective pipeline for every registered model version,
even when compared models use different architectures or dataset bindings.

For a fair cross-model table, aligned supervision values must be identical on the common
evaluation universe. Incompatible targets are rejected instead of silently displayed in
one comparison. Ranking outputs are normalized through the configured objective's
`normalize_weights` contract. Models that already emit final weights use
`PredictionOutput(kind="allocation")`; analytics evaluates those weights directly without
applying a second normalization.

## Analytics output

Analytics writes publication tables to the configured `analytics.output_dir`. CSV keeps
numeric values for further processing, LaTeX is suitable for paper integration, PDF is a
vector table, and PNG is rendered at `analytics.table_dpi`. Console output contains only
the report directory, never tabular results.
