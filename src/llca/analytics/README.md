# Portfolio analytics

`llca.analytics` is the held-out evaluation and publication pipeline for registered
portfolio-optimisation models. It reconstructs the data and objective contracts archived
with immutable MLflow model versions, evaluates every model on one common test sample,
builds realised portfolios, runs signal and factor analyses, and archives a reproducible
report.

The production implementation is deliberately **portfolio-only**. The shared estimator
contract still reserves the prediction kinds `regression`, `binary`, and `multiclass` as
future integration points, but Analytics does not calculate partial reports for them.
`require_supported_prediction_kind` rejects each of those known kinds explicitly. A new
kind must receive a complete evaluator, comparison policy, reporting layer, tests, and
documentation before that gate is widened.

For the exact exported rows and artifact names, see
[`docs/ANALYTICS_OUTPUTS.md`](../../../docs/ANALYTICS_OUTPUTS.md). The default configuration
is [`hydra/configs/analytics/analytics/default.yaml`](../../../hydra/configs/analytics/analytics/default.yaml).

## Architecture and execution flow

The package is split into sub-packages by analytical responsibility:

- `__main__.py` orchestrates registry resolution, source verification, sequential
  prediction, the common sample, factor preparation, export, and audit logging.
- `inputs/` accesses everything a run consumes: `preparation.py` reconstructs model panels
  and prepares risk-free, FF6, and timing inputs through one Hydra data pipeline;
  `registry.py` loads registered estimators and canonicalises their archived manifests;
  `risk_free.py` aligns the daily risk-free series.
- `evaluation/` turns one aligned portfolio prediction into evidence: `predictions.py`
  dispatches by kind, `portfolio.py` builds the realised path, `signals.py` the signal
  analysis, and `plots.py` the single-model figures.
- `factors/` owns factor and IPCA analysis: `ipca.py` (estimator), `factor_models.py`
  (FF/timing regressions), and `panel.py` (the point-in-time IPCA sample).
- `comparison/` aggregates models onto the common universe (`aggregation.py`, `plots.py`);
  `stats/` is the cross-cutting statistical toolkit (`statistics.py`, `inference.py`).
- `modules/` holds the immutable runtime dataclasses (evaluation config, factor settings,
  registry metadata, and typed results); every Hydra-to-runtime translation and validation
  lives with the training pipeline's in `llca.mappers.analytics`, so analytics code consumes
  typed settings and never parses configuration itself. Analytics depends only on shared
  layers (`core`, `data`, `models`, `pipeline`, `mappers`), never on `llca.training`.
- `reporting/` converts typed results into publication tables and figures. `audit.py`
  records the analytical run independently of the training runs.

One invocation follows this sequence:

```text
compose and validate Analytics Hydra configuration
  -> resolve immutable MLflow model versions and canonical manifests
  -> verify every local raw source against its archived SHA-256
  -> intersect registered test intervals
  -> load one estimator at a time, reconstruct its panels, and predict
  -> reject non-portfolio prediction kinds
  -> intersect prediction rows and valid supervision; verify identical targets
  -> prepare risk-free and factor inputs once through Hydra
  -> evaluate each realised portfolio on the common sample
  -> build comparisons, inference, FF6/IPCA/timing analysis
  -> export tables and figures
  -> write and archive the Analytics manifest and report in a dedicated MLflow run
```

Estimators are deliberately loaded and released sequentially, so comparing several neural
models does not retain all model weights on the accelerator. Factor data are model
independent and are prepared once per Analytics invocation.

## Running Analytics

Use exact registry versions; mutable aliases are not accepted as analytical identity.

```powershell
.\.venv\Scripts\python.exe -m llca.analytics `
  analytics.models='[{name: fmg-ctct-2, version: 1, label: FMG-CTCT-2}]' `
  analytics.show_plots=false
```

Several models may be supplied in `analytics.models`. Labels must be unique. Reports are
written below `analytics.output_dir` and archived in the experiment
`${analytics_experiment_name}`.

## Configuration contract

The Analytics application composes its own `data`, `preprocessing`, `features`, and
`masking` groups in addition to `analytics/default`. Dataset references below
`analytics.risk_free` and `analytics.factor_analysis` name **feature outputs**, not raw CSV
columns. Changing a source or transformation therefore remains a Hydra change rather than
an Analytics-code branch.

The main portfolio settings are:

| Setting | Meaning |
| --- | --- |
| `models` | Exact MLflow model names, integer versions, and unique report labels. |
| `device` | Inference device (`auto`, CPU, or an available accelerator). |
| `annualization_periods` | Periods per year used by annualised statistics. |
| `return_type` | Supervision convention (`simple` or `log`); it must match every stored portfolio objective. |
| `return_realization_lag` | Shared decision-to-realisation lag used by returns, FF factors, risk-free, and IPCA feature creation. |
| `signal_buckets` | Number of ordered score buckets under the selected IC basis. |
| `target_threshold` | Outcome threshold for directional signal statistics. |
| `minimum_acceptable_return` | Annual threshold used by downside and Sortino statistics. |
| `var_levels` | Historical VaR and expected-shortfall confidence levels. |
| `autocorrelation_lags` | Return-autocorrelation lags reported per gross/net return series. |
| `worst_rolling_windows` | Compounded-window lengths for the worst-rolling-period-return statistic. |
| `rolling_window` | Trailing window for portfolio and signal stability reports. |
| `signal_decay_periods` | Same-entity outcome leads paired with the current score. |
| `active_weight_threshold` | Absolute-weight threshold used for active-position, holding-period, and directional diagnostics; smaller allocations are neutral abstentions. |
| `include_initial_trade` | Whether the transition from cash to the first reported weights incurs turnover and costs. |
| `hac_lag` | Newey-West lag; `null` selects the automatic bandwidth. |
| `bootstrap_*` | Stationary-bootstrap resamples, mean block length, and deterministic seed. |
| `test_significance_level` | Confidence-set and interval significance level. |
| `multiple_testing_correction` | Pairwise correction: `none`, `holm`, or `bh`. |
| `evaluation_end` | Optional upper bound inside the common registered test interval. |
| `table_formats` / `plot_formats` | Requested export formats; PNG is the default. |

Each invocation writes to a new comparison directory with a unique run suffix. An older
report can therefore never leak stale formats or skipped factor artifacts into a later MLflow
archive.

`analytics.risk_free.dataset` and `.column` select the prepared daily risk-free feature.
Factor-specific settings are described below. Configuration validation is strict: unknown
Analytics keys, missing feature aliases, inconsistent return lags, leading timing features,
and invalid windows fail before model evaluation.

## Common held-out sample and return conventions

The comparison interval starts at the latest registered `test_start` and ends at the
earliest registered `test_end`, optionally shortened by `analytics.evaluation_end`. Inside
that interval Analytics intersects the models' prediction indices and keeps only rows with
explicitly observed, finite supervision for every model. It then verifies that all models
see the same target values on those rows. Empty intersections, incompatible targets, and
duplicate model labels are errors rather than silently different samples.

Each model is scored on its own native test universe; the shared item sample is the
intersection used only for the cross-model item-level checks — verifying identical target
values and computing the score, return, and position similarity matrices — so those
comparisons are aligned item-by-item across models. Pairwise portfolio statistics
(Diebold-Mariano, Sharpe differences, and the model confidence set) are computed on the
models' realised daily returns over their common dates. A single model's cross-sectional
coverage can therefore differ from another's and from its original test set.

`analytics.return_type` must equal the return type archived with every portfolio objective.
Log supervision is converted to simple asset returns exactly once before portfolio
accounting; compounding, weights times returns, costs, drawdowns, and attribution all use
simple returns. The feature pipeline owns decision-to-realisation alignment through
`analytics.return_realization_lag`; factor loaders do not apply a second shift.

The risk-free reference is the already prepared **daily** series selected by
`analytics.risk_free.{dataset,column}`. It is aligned causally to the realised portfolio
calendar: an observation from the current or a prior date may be carried forward, but a
future value is never backfilled and an unresolved leading date is an error. The daily rate
is not an annual rate and is never divided by `annualization_periods`. Residual cash with
weight `1 - net_exposure` earns that rate; negative residual cash therefore represents
risk-free financing. Gross return reconciles risky long/short contributions plus cash, and
excess net return is the funded portfolio return less the same risk-free rate and costs.
Headline and rolling Sharpe use that exact daily excess-return series. The same
decision-aligned risk-free input is used for IPCA excess returns.

The residual-cash-at-risk-free accounting (net exposure, residual cash weight, risky
contribution, cash contribution, cash-inclusive gross return, and cash-inclusive NAV drift)
is **not** an analytics-only definition: it lives once in `llca.core.portfolio_accounting`
and is consumed by both this evaluation and the training objective (`llca.loss.portfolio`),
so a model is trained to optimise the very same funded return that is reported here. Analytics
reconstructs the funded return from stored scores, realised returns, and its own causally
aligned risk-free series, so the risk-free contribution is applied exactly once — the pipeline
never re-adds cash to a return that already contains it. See [Risk-free funding](../../../docs/RISK_FREE_FUNDING.md).

## Scores, weights, objective accounting, and costs

Portfolio predictions contain one finite scalar **score** per constructible date/entity row.
Analytics reconstructs the registered model's stored portfolio objective and requires its
callable, mask-aware `normalize_weights` contract. Prediction values have no implicit
final-weight semantics: a missing objective or normalisation callback is an error. This
makes the archived objective authoritative for weight construction, leverage, and return
convention and ensures weights are constructed exactly once. A future estimator that emits
final weights needs an explicit adapter contract before Analytics can support it.

The realised date-by-entity weight matrix is the single source for all subsequent results.
Risky-book return is the sum of `weight * simple_asset_return`; funded gross return adds the
risk-free return earned by residual cash weight `1 - net_exposure` (or the financing charge
for negative residual cash). Turnover compares today's target weights with prior holdings
after risky-asset, cash, and portfolio-NAV drift; one-way turnover is half the L1 trade.
Transaction costs use the archived objective's execution fee, bid-ask spread, and slippage,
while borrow cost uses current short exposure. Net return is funded gross return less those
costs. Daily reconciliation checks both long-plus-short-plus-cash equals gross and
gross-minus-cost equals net.

Before inference, Analytics requires every selected portfolio objective to share
`return_type`, normalisation, leverage, execution fee, bid-ask spread, slippage, and borrow
cost. A mismatch fails instead of creating economically incomparable report columns. Risk
aversion, concentration regularisation, and other training preferences may remain
model-specific because they shape the learned scores rather than the realised accounting
rule. Analytics-wide settings control the common initial-trade convention, rolling window,
risk thresholds, and annualisation.

## Portfolio signal analysis

Signal magnitude is not interpreted as a return forecast. Portfolio signal analysis reports
pooled Pearson/Spearman association, direction, magnitude-weighted direction, ordered score
buckets, stability, and same-entity decay.

When at least one eligible date provides finite Pearson and rank association across several
instruments, IC is measured cross-sectionally on each date. The report then summarises the
daily IC series and its information ratio. A single-asset sample—or a nominal panel whose
dates never identify both cross-sectional correlations—uses a trailing through-time
correlation as an explicitly labelled fallback. `SignalEvaluation.ic_basis` is exactly
`cross_sectional` or `rolling_time_series`; it does not fabricate a one-name or degenerate
cross-sectional IC. Undefined dates and windows remain missing.

Directional statistics compare the normalized allocation weight with
`target > target_threshold`; this remains valid when market-neutral normalization centres
otherwise positive raw scores. Allocations whose absolute weight does not exceed
`active_weight_threshold` are neutral abstentions and are excluded from hit rates,
confusion matrices, ROC, and Pesaran-Timmermann inference rather than being counted as
negative predictions. Rank/association diagnostics continue to use the native score. The
headline accuracy is the mean of daily hit rates, so dates receive equal weight; the pooled
item-weighted accuracy is retained separately for transparency. Signal decay pairs a score
at time `t` only with the same entity's outcome at `t + lead`, so values never shift across
instruments and uses the same recorded IC basis. Signal-quality buckets rank scores within
each date under the cross-sectional basis and pool observations through time under the
rolling fallback. The portfolio attribution buckets use the same convention and additionally
show total and active directional observations, score ranges, outcomes, active-position hit
rate, realised weight, and return contribution.

## Inference and significance

Time-series uncertainty uses Newey-West HAC covariance with `analytics.hac_lag`; when unset,
Analytics uses its automatic bandwidth. Bootstrap procedures use the configured stationary
bootstrap and seed. The report includes, where defined:

- a HAC test of mean daily rank IC;
- a HAC directional hit-rate test against 50%;
- Pesaran-Timmermann directional-content for time-series allocations and HAC
  excess-profitability tests; genuine panels use the date-level HAC hit-rate test instead;
- a HAC/bootstrapped test and interval for positive net Sharpe;
- pairwise Diebold-Mariano tests on daily economic loss (`-net_return`);
- pairwise net excess-return Sharpe-difference tests with a paired stationary bootstrap;
- a Hansen-Lunde-Nason model confidence set;
- return, score-rank, and portfolio-position similarity matrices.

Pairwise p-values use the configured multiple-testing correction. A p-value that belongs to
a displayed estimate is rendered as `(*)`, `(**)`, or `(***)` in that value cell at the
10%, 5%, and 1% levels. A standalone p-value row is retained only when no paired statistic
exists.

For the single-asset rolling-IC fallback, consecutive estimates overlap. Its HAC inference
therefore uses at least `rolling_window - 1` lags, and its IC ratio is not annualised or
presented as a daily cross-sectional ICIR.

## Shared factor-input pipeline

Risk-free, Fama-French + Momentum, Conditional Timing, and IPCA inputs are resolved as one
union of logical dataset requirements and prepared once through the Analytics-owned Hydra
pipeline. Native date-only FF and macro feature panels remain date-only; the IPCA reference
sample uses the aligned and membership-masked asset panel.

### Fama-French + Momentum

`factor_analysis.factors.dataset` selects the prepared factor panel and `ff6` selects its
feature aliases. The default factors are Market, Size, Value, Profitability, Investment, and
Momentum; `market` identifies the market alias used by the timing model. FF returns and the
risk-free output are created with the shared realisation lag in the feature configuration
and are not shifted again at runtime.

For each portfolio, the report estimates annualised alpha, factor loadings, coefficient
significance, R-squared, rolling FF exposures, and cumulative alpha. Mean-variance spanning
is tested per portfolio. When at least two portfolios are analysed, the package additionally
reports the kernel-HAC traded-factor **joint zero-alpha J-test** across the complete model
set. This is a robust analogue of the classical GRS question, not the classical iid GRS
F-statistic. For one portfolio the joint restriction is redundant with its displayed alpha
significance and is therefore omitted.

`factor_analysis.rolling_beta_window` is the trailing observation count of each rolling OLS
exposure estimate. It must exceed the intercept plus the configured FF-factor count, so every
window retains at least one residual degree of freedom; the two-column plot title records the
chosen window.

### Conditional Timing

`factor_analysis.timing.instruments` selects arbitrary prepared macro feature aliases.
Levels, simple changes, log changes, returns, or another registered feature transform can
therefore be selected without changing Analytics code. Timing features remain on their
native information dates. `instrument_lag` is applied on that full native calendar before
alignment to portfolio-return dates; configured timing features themselves must not lead or
duplicate this lag.

The conditional specification can include state-dependent alpha, market-by-instrument
terms, and the squared-market timing term. Its table reports annualised conditional alpha,
all selected coefficients with inline significance, and R-squared.

### IPCA reference sample and missing data

IPCA is estimated on an Analytics-owned reference universe, independent of the registered
model's own feature panels. The default is the point-in-time S&P membership cross-section.
This permits a single-asset strategy or a strategy without fundamentals to receive IPCA
alpha as long as the independent reference cross-section has enough usable returns and
characteristics.

The returns and the firm-characteristic instruments are both prepared once by the shared
Hydra pipeline — preprocessing, feature creation, and the membership-masked backward-as-of
alignment in `llca.data.masking` — so the IPCA sample is exactly the aligned panel. Just as
model training aligns every dataset to the model's primary feature dataset, factor analysis
aligns every panel to `factor_analysis.aligning_dataset` (the entity-indexed grid, by
default `asset_returns`); returns and instruments are read from that masked grid, and the
per-instrument observation age it carries drives the `max_age` caps. The
response is the decision-aligned simple open-to-open return from `open[t+1]` to `open[t+2]`,
less the corresponding daily risk-free return. Returns must be newly observed, finite, and
age zero; they are never imputed. Every feature output of the characteristics dataset is
used as an instrument (there is no separate column selection); the set is defined entirely
by `features/analytics-features.yaml`. Fundamental characteristics are aligned backward-as-of
from their availability date, and stale carried values are bounded by the `max_age` caps read
from the aligned panel's observation age.

With the sole `rank_neutral` policy, observed values are ranked cross-sectionally first;
eligible residual missing values are then assigned the neutral rank zero. Rows with no
characteristics or coverage below `min_characteristic_coverage` are excluded. Instruments
with no fundamentals can therefore drop out without discarding the rest of a usable
cross-section. A universe with no viable fundamental cross-section cannot produce IPCA.
When IPCA is enabled, this is a hard contract failure with an actionable error;
configurations that intentionally do not require IPCA must set
`factor_analysis.ipca.enabled: false`.

Zero-variance and globally linearly dependent characteristics are removed explicitly and
disclosed. Dates that remain rank-deficient for the requested factor count are removed,
audited, and followed by a fresh global rank check until the usable sample is stable; the
latent-factor count is never reduced silently.

## Outputs, audit, and reproducibility

The default report contains portfolio overview, objective, signal, performance,
construction, attribution, significance, factor-model, and comparison artifacts. Portfolio,
signal, and directional figures use the same common-axis design for one or several models;
a single model simply contributes one series. Genuine pairwise/model-set comparisons are
omitted unless at least two models are evaluated. Grouped detail tables use the same content
in both cases but remove the redundant `Model` sub-header for a single model. The exact
conditional artifact list is maintained in
[`docs/ANALYTICS_OUTPUTS.md`](../../../docs/ANALYTICS_OUTPUTS.md).

An Analytics run never mutates a training run. Before evaluation it verifies each raw file's
path, size, and SHA-256 against every selected model's archived data manifest. The Analytics
manifest records exact model versions and run IDs, prediction kind, IC basis, the trained
portfolio-accounting contract, the realised residual-cash-at-risk-free funding convention,
the common interval and observations, resolved Analytics and factor-pipeline configuration,
source/environment provenance, raw and feature fingerprints, IPCA diagnostics, and hashes
of every exported report file. The report snapshot and manifest are then archived in a
dedicated MLflow Analytics run.

The preparation cache is only an acceleration layer and is not audit evidence. Bootstrap
randomness is controlled by `bootstrap_seed`; immutable model versions, source hashes,
resolved Hydra configuration, and report hashes provide the reproducibility boundary.

## Extending the module

For another portfolio model:

1. Return `PredictionOutput(kind="portfolio", ...)` with one finite scalar per indexed
   portfolio decision.
2. Bind the primary calendar and supervision dataset/column through `EvaluationSpec`.
3. Store a portfolio objective with a callable `normalize_weights` implementation and test
   that Analytics constructs weights exactly once. A native final-weight output first needs
   a new explicit adapter; it is not inferred from values or column names.
4. Keep model-specific feature preparation in Hydra/model capabilities; do not add a model
   name branch to Analytics.
5. Add common-sample, return-convention, objective-accounting, report, and manifest tests.

For `regression`, `binary`, or `multiclass`, retaining the shared prediction kind is only the
starting dock. Add a separate typed evaluation result and a complete vertical slice for
alignment, metrics, inference, comparisons, tables, figures, audit fields, configuration,
and tests. Do not weaken `require_supported_prediction_kind` until that slice is production
ready.

## Quality gate

Run the complete repository gate before publishing Analytics changes:

```powershell
.\.venv\Scripts\python.exe -m ruff check src tests scripts
.\.venv\Scripts\python.exe -m ruff format --check src tests scripts
.\.venv\Scripts\python.exe -m mypy src scripts
.\.venv\Scripts\python.exe scripts/validate_configs.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Focused Analytics tests can be run with:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests/analytics -v
```
