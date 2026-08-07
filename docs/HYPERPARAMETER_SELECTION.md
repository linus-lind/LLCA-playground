# Hyperparameter selection

Statistical estimators (elastic-net logistic regression, random forest, and future
tree/boosting/linear models) can select their hyperparameters through an **inner
walk-forward cross-validation performed entirely inside the outer training window**. This is
part of fitting a single estimator; it is not a second evaluation system.

## Outer evaluation vs. inner selection

- The **outer** split (`training/split/`) owns the experiment: it produces the
  train/validation/test folds that measure a model's held-out performance. Both the single and
  walk-forward splitters are **end-anchored** — the newest observation is always the final scored
  test date and any excess is dropped from the *beginning* — so every experiment is evaluated on
  the most recent data (see [PIPELINE_ARCHITECTURE](PIPELINE_ARCHITECTURE.md#chronological-splitting)).
- The **inner** cross-validation runs *inside one outer training window* to choose
  hyperparameters, then discards its temporary models. It is end-anchored the same way: the
  newest inner validation window ends on the last date of the outer training data and earlier
  folds step backward. The inner CV never sees the outer validation set, the outer test set, or
  any future observation, so selection is nested by construction.

## What happens during a fit

For each outer training window the estimator does one of two things:

1. **Selection disabled** (`hyperparameter_selection.enabled=false`, the default): fit once on
   the whole training window using the model's baseline hyperparameters (its ordinary
   top-level values).
2. **Selection enabled**: build inner walk-forward folds within the training window; score the
   baseline and every searched candidate on the *same* folds through the configured objective;
   adopt the best candidate only if it beats the baseline under the paired standard-error rule;
   then **refit a fresh estimator with the selected hyperparameters on the entire training
   window**. The inner fold models are never reused.

Each fold is scored independently and the candidate score is the **mean of the fold losses**.
Fold losses are never pooled into one series, so time-dependent objective terms (turnover,
transaction costs) are computed inside each fold and no artificial transition is created across
fold boundaries.

## The CV objective is the deployment objective

Selection does not introduce its own metric. Candidates are scored through the **same loss the
model is finally evaluated with** — the experiment's `loss`. For the single-asset directional
classifiers this is the gross-normalized `PortfolioLoss`, so the inner CV ranks candidates on
exactly the realized directional-portfolio economics (return net of turnover cost), consistent
with how the model is deployed. Any objective following the `PortfolioLoss` tensor contract
works; the framework is objective-agnostic.

## Baseline comparison rule

The baseline and every candidate are scored on identical folds, so the comparison is paired.
With per-fold losses `L_base,i` and `L_cand,i` (lower is better) and improvement
`d_i = L_base,i − L_cand,i`, the best candidate is adopted only when

```
mean(d) > standard_error_margin × se(d),   se(d) = std(d, ddof=1) / sqrt(K)
```

over the `K ≥ 2` folds. A margin of `1.0` (the default) is a one-standard-error rule; `0.0`
adopts on any mean improvement. When the improvement is statistically indistinguishable from
noise — common with weak financial signals — the **baseline is retained**. This is a normal,
frequent outcome, not a failure.

## Search methods

- **`grid`**: deterministically enumerates the Cartesian product of the search space.
- **`random`**: draws a configured number of unique candidates from the seeded generator.

Bayesian optimization is intentionally not implemented: model fits here are cheap and the
objective is noisy, so its sample-efficiency advantage does not apply. The candidate generator
is separate from fold evaluation, so a future method needs no change to the CV loop.

## Configuration

Two pieces compose the configuration:

1. A shared top-level group, `hyperparameter_selection`, holding the inner-CV geometry, search
   method, and adoption margin. Presets: `off` (default) and `walk-forward`.

   ```yaml
   # hydra/configs/training/hyperparameter_selection/walk-forward.yaml
   enabled: true
   search: { method: "grid", n_trials: 64, seed: 37 }
   cv:
     train_size: 1260  # multi-year inner-train window so selection transfers to the full refit
     val_size: 252     # one year of scored inner-validation per fold
     step_size: 378    # dates the block advances between folds (larger => fewer folds)
     purge: 1          # dates dropped between train and validation (>= label horizon)
     lookback: 0       # warmup dates for history-rebuilding models (0 for point-in-time)
     min_folds: 3
   selection: { standard_error_margin: 0.0 }
   ```

   On the 4250-observation single-split training window this end-anchored geometry yields
   `(4250 - (1260 + 1 + 252)) // 378 + 1 = 8` folds.

2. A model-specific `search_space` in the model config. The model's ordinary top-level
   hyperparameters are the baseline; only the parameters listed in `search_space` are searched.

   ```yaml
   # model/elastic-net.yaml — C, l1_ratio, class_weight, fit_intercept above are the baseline;
   # the grid tunes the two statistical axes (13 x 7 = 91 candidates).
   search_space:
     C:        { type: "log_range", low: 1.0e-4, high: 1.0e2, num: 13 }
     l1_ratio: { type: "choice",    values: [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0] }
   ```

   ```yaml
   # model/rf.yaml — the full estimator surface is configurable; n_estimators, bootstrap,
   # class_weight, ccp_alpha, max_samples, and runtime controls are held fixed, and the grid
   # tunes the four axes that most move tree bias/variance (4 x 4 x 2 x 2 = 64 candidates).
   search_space:
     max_depth:         { type: "choice", values: [4, 8, 12, 20] }
     min_samples_leaf:  { type: "choice", values: [5, 20, 50, 100] }
     min_samples_split: { type: "choice", values: [2, 20] }
     max_features:      { type: "choice", values: ["sqrt", 0.5] }
   ```

   Distinguish **configurable** from **tuned**: every meaningful scikit-learn constructor
   argument is exposed in the model YAML, but only the `search_space` axes are searched. Ensemble
   size (`n_estimators`) and pure runtime controls (`n_jobs`, `random_state`, `verbose`,
   `warm_start`) are configurable and deliberately kept out of the grid.

An experiment enables selection by choosing the preset:

```yaml
defaults:
  - override /hyperparameter_selection: "walk-forward"
```

Search-space dimensions are explicit and typed — `choice`, `log_range`, and `int_range` — so
no arbitrary expression is ever evaluated from YAML. `grid` uses each dimension's enumerated
values; `random` samples them (log-uniformly for `log_range`).

### Purge vs. lookback

These are distinct and answer to different horizons.

**`purge`** removes dates between a fold's train and validation windows so the forward-looking
*label* horizon cannot leak across the boundary. Its size is the supervision target's forward
span, **not** the sequence length or any feature window (features are backward-looking and are
computed over the full processed history before splitting, so a feature window reaching back into
an earlier segment is not leakage). The current target is a one-observation close-to-close forward
return, so the shipped purge is `1`; both the outer split and inner CV validators reject a purge
smaller than that derived label horizon, and a future five-step target would simply need `purge:
5` with no splitter change.

**`lookback`** prepends input history to each slice so a model can score its *first* observation.
It equals the model's own required history — `required_lookback`, one less than the full causal
window (`sequence_length + CNN buffer - 1` for the temporal FMG models, `0` for the point-in-time
baselines) — and the outer split derives it automatically from the estimator, so no per-model
constant is hand-maintained. Because both splitters are end-anchored, a larger lookback only
attaches more leading history; it never moves the scored train/validation/test dates, so models
with different history depths are still evaluated on identical windows.

## Failure behavior

Invalid parameter combinations are prevented up front: configuration validation rejects
search-space keys outside a model's declared tunable set, malformed dimensions, and impossible
fold geometry before any data is read. During selection the policy is fail-loud, never
fail-silent — a fold whose objective is non-finite, or a fold with no jointly observed
score and return, aborts the selection with a clear error rather than being dropped or treated
as a valid candidate. If too few folds fit the training window, selection raises rather than
selecting from an under-powered comparison.

## Provenance

When selection runs, the fold's MLflow run records the search method, baseline and selected
parameters, whether the baseline was retained, the baseline and best-candidate mean losses, the
paired improvement and its standard error, the fold and candidate counts, and a full
`hyperparameter_selection.json` artifact. The registered model carries the selected
hyperparameters.

## Adding selection to a new estimator

The central CV algorithm in `llca.training.tuning` is model-agnostic; a new model does not
modify it. A tunable statistical model:

1. reads its hyperparameters through `_tuned(name)` in its `_construct`, so a selected value
   overrides the baseline while fixed settings come from the config;
2. declares its searchable names in `ModelCapabilities.tunable_parameters`;
3. threads the deployment objective and the built `HyperparameterSelection` into the estimator
   in its registry `build` (see `mappers/model/linear.py`);
4. adds a `search_space` and baseline values to its model config.

No central conditional references a model name.
