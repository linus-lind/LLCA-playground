# Analytics report — output reference

This is the artifact-level reference for `llca.analytics.reporting`. For architecture,
configuration, data contracts, and extension guidance, see the
[Analytics module README](../src/llca/analytics/README.md).

## Scope and common conventions

The production report supports `PredictionOutput(kind="portfolio")` only. The shared model
contract reserves `regression`, `binary`, and `multiclass`, but Analytics rejects those
known kinds explicitly and produces no partial metrics, tables, or figures for them.

Every model column uses the same held-out item sample: the intersection of registered test
dates, prediction indices, and explicitly observed finite supervision. Analytics also
verifies that target values agree across models. Percentages are stored as fractions of 1.

Unless stated otherwise:

- annualisation uses `analytics.annualization_periods`;
- log target returns are converted once to simple returns before portfolio accounting;
- the risk-free feature selected by `analytics.risk_free.{dataset,column}` is already a
  daily return; residual cash `1 - net_exposure` earns it, and the same rate is deducted
  from funded net return to obtain excess return;
- risk-free calendar alignment is causal: current or prior observations may be used, never
  future observations; an unresolved leading value is an error;
- headline and rolling Sharpe statistics use daily excess net returns;
- `(HAC)` denotes Newey-West heteroskedasticity/autocorrelation-consistent covariance with
  `analytics.hac_lag`, or the automatic bandwidth when it is `null`;
- bootstrap results use `analytics.bootstrap_resamples`,
  `analytics.bootstrap_block_length`, and `analytics.bootstrap_seed`;
- pairwise p-values use `analytics.multiple_testing_correction` (`none`, `holm`, or `bh`);
- a paired p-value is rendered in its estimate cell as `(*)`, `(**)`, or `(***)` at the
  10%, 5%, and 1% levels; a p-value row remains only when no paired estimate exists.

Configured table formats are written for every table and configured plot formats for every
figure. PNG is the default for both. Output availability is conditional: a factor table is
omitted when that factor estimate is unavailable, and the consolidated statistical
`model_comparison` figure requires at least two evaluated models. All other report figures
use the same design for one and several models.

## Core tables

### `model_overview`

- **Registry version** — exact immutable MLflow model version.
- **Output** — validated prediction kind; currently always `portfolio`.
- **Observations** — aligned item rows in the common held-out sample.
- **Dates** — distinct common evaluation dates.
- **Test start / end** — first and last date of the common comparison interval.

### `objective_metrics`

The archived training objective is reconstructed and evaluated again on the common test
sample. Only diagnostics returned by that objective are shown. For the portfolio objective
these can include:

- **Objective loss** — complete portfolio loss on the test tensors.
- **Mean return / Return variance** — per-period portfolio return and variance seen by the
  objective.
- **L1 turnover / Mean cost** — objective turnover and transaction-plus-borrow cost.
- **Gross / Net / Long / Short exposure** — mean objective-side exposures.
- **Concentration (HHI)** — mean sum of squared weights.

Penalty diagnostics emitted by a particular objective remain in the evaluation and audit
record even when the compact publication table does not display them.

### `signal_metrics`

Portfolio-score quality on identical item rows:

- **Pearson / Spearman correlation** — pooled score–outcome association.
- **Mean Pearson IC / Mean rank IC (recorded basis)** — cross-sectional correlation across
  instruments on each identifiable date, averaged through time. If no date identifies both
  correlations, Analytics uses the labelled rolling time-series fallback.
- **IC basis** — stored in the typed evaluation as exactly `cross_sectional` for multi-asset
  dates or `rolling_time_series` for a single-asset strategy.
- **Rank ICIR / Annualized rank ICIR** — mean rank IC divided by its sample standard
  deviation; square-root-of-time annualisation is reported only for the cross-sectional
  daily-IC basis.
- **Directional accuracy** — mean of the daily fractions for which an active normalized
  `allocation_weight > 0` agrees with `outcome > analytics.target_threshold`; dates receive
  equal weight. Weights with absolute value at or below `analytics.active_weight_threshold`
  are neutral abstentions and do not enter directional denominators.
- **Pooled directional accuracy** — the item-weighted fraction among active allocations,
  retained in the typed metrics for transparency when daily cross-section sizes differ.
- **Magnitude-weighted accuracy** — mean of each date's hit rate weighted within that date
  by absolute outcome; dates still receive equal weight in the headline.
- **ROC AUC** — directional discrimination of the normalized allocation when both outcome
  directions are present.
- **Top-minus-bottom outcome** — mean outcome in the highest ordered score bucket less the
  lowest.
- **Bucket monotonicity** — Spearman association between bucket order and bucket mean
  outcome.

Stars on **Mean rank IC (recorded basis)** represent the one-sided HAC test that mean IC is
positive. Rolling time-series fallback inference uses at least `rolling_window - 1` HAC lags;
its overlapping IC ratio is not annualised.
Stars on **Directional accuracy** represent the one-sided HAC test against 50%.
Single-asset or wholly degenerate panel IC is never presented as a cross-sectional
correlation: it uses the configured trailing window through time, and degenerate windows
remain undefined.

### `portfolio_performance`

Metrics use the realised net portfolio after the archived objective's execution, spread,
slippage, and borrow costs. Analytics requires those accounting settings, normalisation, and
leverage to be identical across all models in one report:

- **Net total return** — compounded product of daily net returns minus 1.
- **Net CAGR** — geometric annualisation over the observed number of periods.
- **Net annualized return** — arithmetic daily mean times periods per year.
- **Net annualized volatility** — daily standard deviation times square root of periods.
- **Net Sharpe ratio** — mean daily excess net return divided by its standard deviation,
  annualised by square root of periods.
- **Net Sortino ratio** — return above the daily equivalent of
  `analytics.minimum_acceptable_return`, divided by downside deviation.
- **Net Calmar ratio** — CAGR divided by absolute maximum drawdown.
- **Net maximum drawdown** — minimum compounded wealth relative to its running high-water
  mark.
- **Net expected shortfall 95/99%** — empirical mean loss beyond the matching historical
  VaR threshold.
- **Net skewness / excess kurtosis** — bias-corrected third and fourth shape statistics.
- **Net profit factor** — sum of positive net returns divided by absolute sum of negative
  net returns.

Stars on **Net Sharpe ratio** test positive mean daily excess net return. The reported daily
and rolling Sharpe paths use the same daily risk-free convention.

### `portfolio_construction`

- **Annualized cost drag** — mean daily total cost times annualisation periods.
- **Mean one-way turnover** — mean of one half the L1 drift-adjusted trade.
- **Annualized L1 turnover** — mean daily L1 trade times annualisation periods.
- **Mean gross / net / long / short exposure** — time averages of the realised weight
  matrix.
- **Mean effective positions** — mean inverse HHI.
- **Maximum absolute weight** — largest absolute weight in the sample.
- **Average holding period** — mean contiguous periods above
  `analytics.active_weight_threshold`.
- **Annualized long / short contribution** — mean daily additive contribution of each side
  times annualisation periods.

Turnover compares current target weights with prior holdings after asset and portfolio-NAV
drift. `analytics.include_initial_trade` controls whether the initial move out of cash is
included.

## Statistical and factor tables

### `statistical_significance`

This table merges predictive-content statistics with factor-model additional statistics.
It contains only estimates not already shown in a metric table:

- **Pesaran-Timmermann statistic** — one-sided test of independence between predicted and
  realised direction for the time-series allocation fallback. Genuine panels omit pooled PT
  because same-date observations are cross-sectionally dependent and use the date-level HAC
  hit-rate test instead.
- **Excess profitability (HAC)** — one-sided test that the daily direction–return covariance
  is positive.
- **Mean-variance spanning statistic (HAC)** — per-portfolio Huberman–Kandel joint test of
  `alpha = 0` and factor-loading sum `= 1` against FF6 benchmark returns.
- **Joint zero-alpha J-statistic (HAC)** — when at least two portfolios are analysed, the
  kernel-HAC traded-factor system test that all portfolio alphas are jointly zero. It is
  omitted for one portfolio because that single restriction duplicates the alpha
  significance already shown in the FF6 factor table.

The joint zero-alpha result is the HAC analogue of the economic question commonly associated
with Gibbons–Ross–Shanken. It is a chi-squared J-test, **not** the classical iid GRS
F-statistic.

### `factor_alpha_ff6`

Fama-French + Momentum time-series results: annualised alpha, Market, Size, Value,
Profitability, Investment, and Momentum loadings, and R-squared. Alpha and loading
significance appears inline.

### `factor_alpha_ipca`

Annualised alpha, loadings on the configured latent IPCA factors, and R-squared. The factor
sample is estimated independently of the registered models. If IPCA is enabled, panel
preparation and estimation are required; insufficient data or rank fails the run instead of
silently producing a report without this table.

### `factor_alpha_timing`

Conditional alpha, unconditional factor exposures, configured state-dependent alpha and
market interactions, optional squared-market timing term, and R-squared. Timing instruments
are lagged on their complete native calendar before alignment to portfolio dates.

### Factor-input and IPCA conventions

Risk-free, FF6, timing, and IPCA inputs are prepared once from the Analytics-owned Hydra
`data`, `preprocessing`, `features`, and `masking` groups. References select named feature
outputs, so timing levels, changes, or returns can be exchanged without code changes.
`analytics.return_realization_lag` is applied during FF/risk-free/return feature creation;
those values are not shifted again at runtime. Timing uses the separate
`factor_analysis.timing.instrument_lag`.

The default IPCA response is the decision-aligned simple open-to-open return from
`open[t+1]` to `open[t+2]`, less the aligned daily risk-free return. Returns must be observed,
finite, fresh, and age zero. Fundamentals are backward-as-of aligned from availability dates
and expire at configured ages.

Under `rank_neutral`, observed characteristics are ranked cross-sectionally before residual
missing values in otherwise eligible rows are set to neutral rank zero. All-missing and
low-coverage rows leave the estimation sample; one instrument without fundamentals does not
discard the rest of the cross-section. Complete-case sensitivity, dropped instruments,
staleness, imputation counts, source hashes, and feature fingerprints are stored in the
manifest.

## Detail tables

Detail-table content and styling are identical for one and several evaluated models. With
several models, columns use the grouped `(Statistic, Model)` header; with one model, the
redundant `Model` level is removed and only the statistic labels remain.

### `yearly_returns`

Calendar-year compounded **gross return** and **net return**, grouped by model.

### `side_attribution`

Total and mean daily additive contribution of the long book, short book, residual cash,
transaction costs, and borrow costs, grouped by model.

### `signal_bucket_analysis`

For each score bucket and model, ranked within date for cross-sectional signals and pooled
through time for the rolling single-asset fallback:

- score range plus total and active directional observation counts;
- mean score and mean asset return;
- active-position directional hit rate;
- mean realised weight;
- total and mean daily return contribution.

## Figures

### Every evaluated model count

- `portfolio_comparison` — common-axis cumulative net return, drawdown, rolling excess
  Sharpe/volatility, tail loss, turnover, and short exposure. One model produces one series;
  several models are overlaid.
- `signal_comparison` — common-axis IC, ICIR, directional quality, and decay, again with one
  series or several overlays.
- `confusion_roc` — directional up/down confusion matrices and ROC curves for active
  normalized portfolio allocations; neutral abstentions are excluded, and this is
  allocation-direction analysis, not classifier evaluation.
- `rolling_factor_betas` — one panel per portfolio model in a two-column layout; the title
  reports `factor_analysis.rolling_beta_window`.
- `cumulative_factor_alpha` — one shared overlay of every portfolio model's FF6 abnormal
  return path.

### At least two evaluated models

- `model_comparison` — consolidated strict-lower-triangle matrices and model confidence-set
  summary.

`model_comparison` can include:

- corrected Diebold-Mariano p-values on daily economic loss (`-net_return`);
- corrected paired-stationary-bootstrap net excess-return Sharpe-difference p-values;
- Pearson net-return correlation;
- Spearman score-rank correlation;
- mean daily cosine similarity of portfolio weights;
- pairwise FF6 alpha-difference p-values;
- Hansen-Lunde-Nason confidence-set membership, p-value, and mean economic loss.

Matrix colours use theoretical ranges rather than sample minima and maxima: p-values use
0–1 and correlations/overlap use -1–1. Only the strict lower triangle is displayed because
the matrices are symmetric.

## Audit artifacts

Every report directory contains the exported tables/figures and
`analytics_manifest.json`. The same snapshot is archived in a dedicated MLflow Analytics
run. The manifest binds:

- exact model names, versions, source run IDs, prediction kind, IC basis, trained
  portfolio-accounting contract, and common sample;
- the realized funding convention (`residual_cash_at_risk_free`);
- resolved Analytics and factor-pipeline configuration;
- verified source identities and feature-panel fingerprints;
- IPCA sample and estimation diagnostics;
- source and environment provenance;
- relative path, size, and SHA-256 for every report file.

Training runs and model artifacts remain immutable. Preparation caches and Hydra output
directories are not audit evidence.
