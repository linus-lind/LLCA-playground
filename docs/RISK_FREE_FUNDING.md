# Risk-Free Funding of Residual Cash

The portfolio objective and analytics evaluation both treat capital not invested in risky
assets as earning the risk-free rate. This is the `residual_cash_at_risk_free` funding
convention. It is defined **once** and applied **exactly once per date** on both the training
and evaluation sides.

## Convention

For risky asset weights `w_t = (w_{1,t}, …, w_{N,t})` on date `t`:

- **Net risky exposure:** `n_t = Σ_i w_{i,t}`
- **Residual cash weight:** `c_t = 1 − n_t` (positive when underinvested, negative under net
  leverage; never clamped to `[0, 1]`)
- **Gross portfolio return** (before trading/borrow costs):

  ```
  r_gross_t = Σ_i w_{i,t} · r_{i,t+1} + (1 − Σ_i w_{i,t}) · rf_{t+1}
            = rf_{t+1} + Σ_i w_{i,t} · (r_{i,t+1} − rf_{t+1})
  ```

`r_{i,t+1}` is the realised simple risky return over the holding period `t → t+1`, and
`rf_{t+1}` is the simple risk-free return over the **same** period.

### Interpretation by book

| Book | `Σ w_i` | Residual cash `c` | Effect |
|------|---------|-------------------|--------|
| Long-only, partially invested | 0.7 | 0.3 | 30% earns `rf` |
| Fully invested | 1 | 0 | `rf` contributes nothing |
| Market neutral | 0 | 1 | earns `rf + Σ w_i r_i` on full collateral |
| Net-leveraged long | 1.5 | −0.5 | pays `rf` on the borrowed 0.5 |
| Single-asset long (`w=+1`) | 1 | 0 | no cash |
| Single-asset short (`w=−1`) | −1 | 2 | 2 units of cash earn `rf` |

Residual cash funding (`rf` on `1 − Σ w_i`) is a **balance-sheet** quantity and is distinct
from the explicit **short borrow cost**, which is still charged on short exposure `Σ max(−w, 0)`.
A short both earns `rf` on its residual cash and pays the borrow cost; these are not merged.

## Single source of truth

`src/llca/core/portfolio_accounting.py` holds the canonical primitives — `net_exposure`,
`residual_cash_weight`, `risky_return`, `cash_return_contribution`, `gross_return`,
`portfolio_nav_growth`, and `drifted_weights` — as pure Torch functions on `[D, N]` weights /
returns and a per-date `[D]` risk-free vector. `core` depends on neither `loss` nor `analytics`,
so both consume it without a cycle:

- **Training** — `PortfolioLoss.forward` (`src/llca/loss/portfolio.py`) computes the
  cash-inclusive `gross_return` as its realised return and uses `drifted_weights` for
  drift-adjusted turnover.
- **Evaluation** — `build_portfolio_evaluation`
  (`src/llca/analytics/evaluation/portfolio.py`) reconstructs gross return, residual cash, and
  drifted turnover from the same primitives; it keeps only its own reporting layer (Sharpe,
  attribution, drawdowns, factor/IPCA excess returns), which legitimately still needs `rf`.

Because both sides call the same functions, they cannot diverge on the accounting. Analytics
reconstructs the funded return from stored scores + realised returns + its own risk-free
series, so it never re-adds cash to a return that already contains it.

## NAV drift (turnover)

Turnover compares each date's target weights against the previous date's holdings drifted one
period forward. The drift renormalises by **total** portfolio wealth including residual cash
growing at `rf`:

```
w_drift_i = w_i (1 + r_i) / (1 + r_gross),   r_gross as above (cash-inclusive)
```

Using a zero-return-cash denominator here would make turnover and costs inconsistent with the
return; both training and analytics use the cash-inclusive NAV via `drifted_weights` /
`portfolio_nav_growth`. Training fills the first date of each batch with the within-batch mean
turnover; analytics charges the first trade of the whole history only when
`include_initial_trade` is set. These first-date boundary policies differ by context; the
per-period drift math is identical.

## Ordering of returns and costs

`net_return = gross_return − costs`, where `costs = (execution_fee + bid_ask_spread + slippage)
· turnover + borrow_cost · short_exposure`. Risk-free funding enters the **gross** return
(before costs) on both sides; costs are then subtracted consistently. Excess net return
subtracts `rf` again from the *net* return for Sharpe-style reporting only — that is an
excess-return transform, not a second funding step.

## Data, units, and horizon

- **Source:** the Fama-French daily file `data/fama-french/fama-french.csv`, `rf` column,
  already used by analytics. Training loads it as a date-only dataset named `risk_free`
  (`hydra/configs/training/data/sp500-crsp-compustat.yaml`), preprocessed with a
  trading-calendar filter and de-duplication.
- **Units:** the `rf` column is a **decimal daily** simple return (e.g. `0.00020` = 2 bps),
  the same convention as the asset returns — no percent/annualised rescaling is applied. Tests
  guard against a 100× percent-vs-decimal error.
- **Horizon:** the risk-free feature is `passthrough(rf, shift: -1)`
  (`hydra/configs/training/features/fmg-features.yaml`), so date `t` carries `rf_{t+1}` — the
  rate over the same `t → t+1` period as the `fwd_return` label (also `shift: -1`). Deterministic
  tests pin this pairing so an off-by-one is caught.

## Configuration

A model trained with the portfolio objective binds the rate on its config:

```yaml
risk_free:
  dataset: "risk_free"
  column: "rf"
```

Validation (pre-I/O) requires this binding whenever `loss.name == portfolio` for FMG and
single-asset tabular models, and data-requirement resolution adds the dataset so it survives
planning into the training split. Models that do not evaluate the objective (equal-weight,
inverse-volatility, random-long-short baselines) and pointwise objectives (MSE, BCE) neither
require nor load it. Missing rate resolves to zero only at the low-level accounting API for
isolated tests; the training pipeline fails loudly if a portfolio model's rate cannot be
resolved on a scored date, rather than silently dropping cash funding.

## Provenance

Portfolio training runs tag `llca.funding_convention = residual_cash_at_risk_free` plus the
`llca.risk_free_dataset` / `llca.risk_free_column` used, and the analytics audit manifest
records the same convention and per-model risk-free binding, so a reloaded run is
self-describing.
