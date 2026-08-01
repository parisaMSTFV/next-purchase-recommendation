# Model Card

## Intended use

Rank eligible ecommerce customers by near-term purchase readiness and suggest
up to three plausible next categories. The output is intended for analyst review
and controlled CRM or onsite experiments.

## Not intended for

- claiming that a contact caused a purchase;
- setting individual prices or credit decisions;
- choosing a channel without consent and frequency rules;
- recommending unavailable or ineligible products;
- serving customers with fewer than three historical purchases without a
  separate cold-start policy.

## Data

The repository generates all transactions from scratch. Latent shopping
preferences, category transitions, seasonality, purchase cadence, discounts,
order value, and margin are simulated. The model does not receive the latent
variables.

## Features

Point-in-time features cover:

- recency, frequency, monetary value, and purchase cadence;
- category counts, shares, and recency;
- last and favorite category;
- discount and weekend behavior;
- acquisition channel, region, lifecycle state, and calendar context.

## Outcomes

- binary purchase within 30 days;
- next category when a purchase occurs within 60 days.

## Evaluation

Score-date splits keep the test period later than training and validation.
Every transaction on or after the score date is excluded from features.

## Limitations

- Synthetic behavior is less complex than production behavior.
- Availability, price changes, exposure history, returns, and cancellations are
  simplified or absent.
- The conditional category model does not predict whether a purchase will occur.
- The readiness model does not select the best treatment.
- The action score uses historical category margin and can be wrong for a
  specific customer or future order.

## Production controls

- data contracts and point-in-time feature validation;
- cold-start and low-history routing;
- monthly drift and calibration monitoring;
- category-level coverage checks;
- consent, frequency, inventory, and eligibility filters;
- randomized holdouts for incremental impact.
