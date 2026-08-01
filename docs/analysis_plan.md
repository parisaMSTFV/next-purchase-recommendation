# Analysis Plan

## Decision

For each eligible customer, decide:

1. whether the customer is likely to purchase within 30 days;
2. which three categories are most plausible for the next purchase; and
3. where the customer should sit in a capacity-limited activation queue.

The output can inform category-level content, landing pages, and audience
prioritization. It does not determine an offer or estimate incremental response.

## Unit of analysis

One row represents one customer at one historical score date. Features use only
orders strictly before that date.

## Eligibility

- At least three historical successful purchases.
- The last observed purchase is no more than 365 days before scoring.
- The full 60-day outcome window is observable for labeled snapshots.

Customers without enough history require a separate cold-start policy.

## Outcomes

### Purchase readiness

`purchase_within_30d = 1` when the next purchase occurs from day 0 through day
30 after scoring.

### Next category

`next_category` is the category of the first purchase from day 0 through day 60
after scoring. Rows without a purchase in that window remain in the readiness
task but are excluded from category-model training and evaluation.

## Evaluation design

The split follows score date:

- Train: before 2025-01-01
- Validation: 2025-01-01 through 2025-06-30
- Test: from 2025-07-01

Model selection uses validation only. The chosen model is refit on train plus
validation and evaluated once on the untouched test period.

## Candidates

Category task:

- training-set popularity baseline;
- smoothed repeat-last-category baseline;
- multinomial logistic regression;
- histogram gradient boosting.

Readiness task:

- purchase-cycle progress rule;
- logistic regression;
- histogram gradient boosting.

## Primary metrics

- Category: Top-3 hit rate, with Log Loss as a probability-quality control.
- Readiness: Average Precision, with Brier score as a calibration control.

Top-1/Top-2 hit rate, mean reciprocal rank, ROC-AUC, Recall and Lift at the top
20% are secondary measures.

## Model-selection rules

- Select the category model with the highest validation Top-3 hit rate; break a
  tie using lower Log Loss.
- Select the readiness model with the highest validation Average Precision;
  break a tie using lower Brier score.

## Decision rule

The demonstration action score is:

```text
P(purchase within 30 days)
× P(recommended category is next | purchase within 60 days)
× historical median contribution margin for that category
```

This is a capacity-ranking heuristic. It is not expected incremental profit.
The top 15% becomes `Priority`, the next 25% becomes `Review`, and the remainder
becomes `Monitor`.

## Failure and review conditions

- Category Top-3 quality does not improve on the last-category baseline.
- Readiness Average Precision does not improve on the cadence baseline.
- One category has materially worse coverage than the others.
- Performance falls across later score months.
- The Priority queue becomes concentrated in a category without sufficient
  inventory, eligible content, or channel capacity.
