# Customer Next Purchase Recommendation

[![CI](https://github.com/parisaMSTFV/next-purchase-recommendation/actions/workflows/ci.yml/badge.svg)](https://github.com/parisaMSTFV/next-purchase-recommendation/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12-3776AB)](https://www.python.org/)
[![Included data](https://img.shields.io/badge/included%20data-100%25%20synthetic-0F766E)](DATA_PROVENANCE.md)

A relevant recommendation sent at the wrong time is still a poor decision. This project separates
purchase timing from category relevance, then combines them only when building a capacity-ranked
review queue.

| Decision | Executed synthetic evidence | Operational output |
|---|---:|---|
| **When** is a purchase likely? | Readiness AP `0.771` vs `0.658` cadence baseline | 30-day readiness score |
| **What** category is relevant? | Top-3 hit rate `83.7%` vs `72.8%` last-category baseline | Three ranked categories |
| **Who** enters limited capacity? | 599 of 3,999 eligible customers | `Priority`, `Review`, or `Monitor` |

These evaluation results come only from the committed synthetic benchmark. They are not production
performance or evidence that contacting a customer creates incremental behavior.

![Ranked recommendation quality](reports/figures/top_k_performance.png)

## Quick start with supplied transactions

```bash
python -m pip install -e ".[dev]"
next-purchase score --transactions examples/supplied_transactions_v1.csv \
  --provenance examples/supplied_provenance_v1.json --score-date 2025-06-01 \
  --output-root artifacts/supplied-example
```

Inspect `artifacts/supplied-example/reports/recommendations.csv`, `activation_summary.csv`, and
`run_metadata.json`. The example uses fictional identifiers and the outputs are ignored by Git.

## Two modes, separate evidence

| Mode | Input | Output | Evaluation boundary |
|---|---|---|---|
| `run` | Internally generated synthetic history | Fitted models, temporal metrics, and queue | Reported performance applies only to synthetic truth |
| `score` | Versioned caller-supplied transactions and provenance | Transparent baseline queue and metadata | No fitted model, future label, synthetic truth, uplift, or performance metric |

## Business question

For each eligible customer:

1. Is a purchase likely within the next 30 days?
2. Which three categories are the strongest candidates for the next purchase?
3. With limited contact capacity, who should be reviewed first?

The project keeps category relevance separate from purchase timing. This avoids
sending a relevant message at the wrong point in the customer's purchase cycle.

## Decision flow

```mermaid
flowchart TD
    A["Past transactions"] --> B["Point-in-time snapshots"]
    B --> C["30-day purchase readiness"]
    B --> D["Next-category ranking"]
    C --> E["Capacity-ranked activation queue"]
    D --> E
```

Every feature uses orders strictly before its score date. The readiness outcome
looks 30 days forward. The next-category outcome uses the first purchase within
60 days.

## Synthetic benchmark results

The committed run uses 4,000 simulated customers, 91,154 transactions, 91,339
historical score-date snapshots, and seed `42`.

| Untouched test measure | Selected model | Baseline |
|---|---:|---:|
| Next-category Top-1 hit rate | **54.2%** | 52.2% last category |
| Next-category Top-3 hit rate | **83.7%** | 72.8% last category |
| Mean reciprocal rank | **0.707** | 0.666 last category |
| Category Log Loss | **1.308** | 1.537 last category |
| Readiness Average Precision | **0.771** | 0.658 cadence rule |
| Readiness ROC-AUC | **0.709** | 0.610 cadence rule |
| Readiness Recall at top 20% | **28.5%** | 23.6% cadence rule |
| Readiness Lift at top 20% | **1.42x** | 1.18x cadence rule |

The selected next-category model improves Top-3 hit rate by 10.9 percentage
points over repeating the customer's last category.

### Model selection

The pipeline compares:

- training-set category popularity;
- a smoothed repeat-last-category rule;
- purchase-cycle progress for timing;
- logistic regression;
- histogram gradient boosting.

Histogram gradient boosting won both pre-defined validation selection rules.
For next category, its Top-3 result was nearly tied with logistic regression:
83.94% versus 83.92%. The more complex model was selected because the
[analysis plan](docs/analysis_plan.md) fixes Top-3 hit rate as the primary
criterion before the test period is evaluated.

![Category model comparison](reports/figures/category_model_comparison.png)

![Readiness model comparison](reports/figures/readiness_model_comparison.png)

Category-level test coverage remains between 81.9% and 86.2% for Top-3 recall,
so the portfolio result is not produced by one dominant category alone.

![Category recall](reports/figures/category_recall.png)

## From prediction to an activation queue

The final score is a transparent prioritization heuristic:

```text
Action score =
P(purchase within 30 days)
× P(recommended category is next | purchase within 60 days)
× historical median category margin
```

The highest-ranked 15% of eligible customers enter `Priority`; the next 25%
enter `Review`; the rest remain in `Monitor`. The latest synthetic scoring run
contains 3,999 eligible customers and places 599 in Priority.

![Activation queue](reports/figures/activation_queue.png)

This score is not an uplift estimate. It ranks likely, relevant, and
historically valuable opportunities; a randomized holdout is still required to
measure whether a message, offer, or channel creates incremental behavior.

## Evaluation design

```mermaid
flowchart TD
    A["Train: before 2025-01-01"] --> B["Validation: Jan–Jun 2025"]
    B --> C["Select one model per task"]
    C --> D["Refit on train + validation"]
    D --> E["Test: from 2025-07-01"]
```

- Model selection uses validation only.
- The untouched test period is evaluated once after refitting.
- Category quality uses Top-K metrics, mean reciprocal rank, Log Loss, and
  multiclass Brier score.
- Readiness uses Average Precision, ROC-AUC, Brier score, and capacity-based
  Recall and Lift.
- A leakage test changes future transactions and confirms that historical
  features remain identical.

## Supplied-input contract

The `score` mode accepts one completed-order CSV at schema version `1.0` plus a required provenance
manifest. It validates identifiers, dates, numeric fields, category coverage, schema version,
pseudonymization, and authorization metadata. Only transactions strictly before `score_date` enter
the queue; later rows are counted and ignored.

This mode uses a cadence readiness rule and a smoothed last-category rule. It does not silently
apply a model fitted on the synthetic generator to external customers. The full field definitions,
outputs, security boundary, and example command are in the
[supplied-input contract](docs/supplied_input_contract_v1.md).

## Repository structure

```text
src/next_purchase/       simulation, features, models, evaluation, decisions
tests/                   leakage, metrics, queue, simulation, and pipeline tests
docs/                    analysis plan, metric dictionary, model card, interview guide
schemas/                 versioned supplied-input contract
examples/                fictional transaction and provenance fixture
reports/                 reproducible metrics, tables, figures, and decision note
scripts/                 public-file sensitive-content check
.github/workflows/       CI for Python 3.11 and 3.12
```

Generated row-level transactions and model snapshots are written to
`data/generated/` and excluded from Git. Only compact aggregate reports and a
30-row synthetic recommendation sample are committed.

## Reproduce the synthetic benchmark

Python 3.11 or later is required.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
make run
make check
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

The complete pipeline regenerates all report tables and figures.

## Documentation

- [Analysis plan](docs/analysis_plan.md)
- [Metric dictionary](docs/metric_dictionary.md)
- [Model card](docs/model_card.md)
- [Interview guide](docs/interview_guide.md)
- [Data provenance](DATA_PROVENANCE.md)
- [Supplied-input contract v1.0](docs/supplied_input_contract_v1.md)
- [Reproducible run summary](reports/run_summary.md)
- [Decision note](reports/decision_note.md)
- [Recommendation sample](reports/recommendations_sample.csv)

## Limitations

- Customers need at least three historical purchases; a separate cold-start
  policy is required for newer customers.
- Synthetic behavior simplifies availability, prices, exposure history,
  returns, cancellations, consent, and channel constraints.
- The category model is conditional on a purchase occurring within 60 days.
- Historical category margin is not customer-level incremental value.
- Production use requires point-in-time data controls, drift and calibration
  monitoring, inventory and eligibility rules, and controlled experiments.
- Supplied-input mode is a transparent reference policy, not a substitute for
  training and temporally evaluating models on an approved local dataset.

## License

MIT
