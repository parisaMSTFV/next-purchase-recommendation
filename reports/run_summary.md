# Reproducible Run Summary

This report was generated from the committed pipeline using seed
`42`.

## Data and evaluation

- Synthetic customers: 4,000
- Synthetic transactions: 91,154
- Historical score-date snapshots: 91,339
- Category-labeled test snapshots: 10,015
- Untouched test period starts: 2025-07-01

## Selected models

- Next-category model: `hist_gradient_boosting`
- 30-day readiness model: `hist_gradient_boosting`

## Test performance

| Measure | Result |
|---|---:|
| Category Top-1 hit rate | 54.2% |
| Category Top-3 hit rate | 83.7% |
| Mean reciprocal rank | 0.707 |
| Category log loss | 1.308 |
| Readiness average precision | 0.771 |
| Readiness ROC-AUC | 0.709 |
| Readiness Brier score | 0.213 |
| Readiness recall in top 20% | 28.5% |
| Readiness lift in top 20% | 1.42x |

These figures validate the synthetic pipeline. They are not estimates of
production performance or incremental campaign impact.
