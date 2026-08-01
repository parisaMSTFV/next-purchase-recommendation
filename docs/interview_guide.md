# Interview Guide

## One-minute explanation

This project separates two decisions that are often mixed together: whether a
customer is ready to purchase soon and what category is most likely to come
next. Historical score-date snapshots prevent future transactions from entering
features. Models are selected on a later validation period and evaluated on an
untouched test period. The final queue combines readiness, category probability,
and a transparent margin proxy, but it is presented as prioritization rather
than incremental impact.

## Why not use one model?

A category model trained only on customers who purchase soon answers a
conditional relevance question. A readiness model answers whether contact timing
is appropriate. Keeping the tasks separate makes both evaluation and activation
logic easier to audit.

## Why Top-3?

Many activation surfaces can display more than one category. Top-3 hit rate
tests whether the model creates a useful shortlist, while Log Loss checks whether
the probabilities are trustworthy enough for ranking.

## How is leakage prevented?

Each feature row uses transactions strictly before its score date. The label
comes from the first later purchase. The test period is later than train and
validation, and a dedicated test confirms that changing future transactions
does not change a historical scoring row.

## Why is the action score not uplift?

The score estimates likelihood and historical value, not the probability that a
specific treatment changes behavior. Incrementality requires randomized
experiments or a valid treatment-effect design.

## What would change in production?

Add exposure history, product availability, prices, returns, cancellations,
channel eligibility, and consent. Use a feature store or equivalent
point-in-time controls, monitor calibration and category coverage, and test each
activation policy against a holdout.
