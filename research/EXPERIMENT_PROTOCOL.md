# Empirical protocol for the research article

Freeze this document before producing the final result tables. The pipeline guarantees
technical reproducibility, but it cannot decide scientific design choices without
changing the estimand.

## Decisions to pre-register

- Paper hypothesis and primary metric.
- Licensed data vintage and coverage dates.
- Target PERMNO for FMG-CTCT-1, FMG-CTT, and FMG-CLSTM, with an ex-ante selection rule.
- Primary split (`single_split` or `walk_forward`) and all date counts.
- Hyperparameter search space, selection metric, and validation-only selection procedure.
- Final random seeds and whether results are reported per seed or as a distribution.
- Transaction-cost and borrow-cost assumptions.
- Portfolio normalization and all penalty coefficients.
- Baselines and ablations, including equal-weight, long-only, linear, and architecture
  ablations where relevant to the hypothesis.
- Treatment of delistings, survivorship, corporate actions, and unavailable borrow data.
- Statistical uncertainty procedure and multiple-comparison correction.

## Frozen model matrix

| Run | Hydra root | Required override | Scientific comparison |
| --- | --- | --- | --- |
| FMG-CTCT-2 | `train.yaml` | none | full cross-sectional allocation |
| FMG-CTCT-1 | `train-fmg-ctct-1.yaml` | `model.target.entity_id` | value of cross-sectional context for one asset |
| FMG-CTT | `train-fmg-ctt.yaml` | `model.target.entity_id` | temporal Transformer without cross-section |
| FMG-CLSTM | `train-fmg-clstm.yaml` | `model.target.entity_id` | recurrent versus attention-based temporal encoding |

All single-asset comparisons must use the same target, raw-data versions, target return,
split, costs, seed set, and evaluation interval. Architectural comparisons should change
only the model group unless the deviation is declared in advance.

## Run acceptance checklist

1. The Git commit and working-tree state are recorded.
2. Every raw source has a committed `.dvc` pointer and is synchronized to the configured
   remote.
3. Ruff, Ruff format, strict Mypy, and the complete test suite pass.
4. The Hydra invocation and resolved training manifest match this frozen protocol.
5. Parent and fold MLflow runs are `completed`, and every fold has a registered immutable
   model version.
6. Training, data, invocation, source, and environment manifests exist for each retained
   run.
7. Analytics references exact registry versions, uses only registered test windows, and
   produces a dedicated analytics MLflow run.
8. The MLflow store is archived and verified; the DVC remote is backed up independently.

## Result ledger

Record the final immutable identities here rather than copying mutable filenames.

| Model | Git commit | MLflow run | Registry version | Data-manifest SHA-256 | Seed |
| --- | --- | --- | --- | --- | --- |
| FMG-CTCT-2 | | | | | |
| FMG-CTCT-1 | | | | | |
| FMG-CTT | | | | | |
| FMG-CLSTM | | | | | |

## Remaining methodological caveats

The current repository does not infer or certify data-license compliance, economic
identification, benchmark adequacy, statistical power, or robustness across target assets.
Those claims belong in the article and must be supported by the frozen empirical design
and reported results.
