# Data Provenance

All customer, transaction, category, order-value, margin, channel, region, and
date records in this repository are generated from scratch by
`src/next_purchase/simulation.py`.

The generator creates:

- latent customer category preferences;
- changing purchase cadences;
- imperfect last-category repetition and cross-category transitions;
- category seasonality;
- synthetic discounts, order values, and contribution margins.

The latent variables are not provided to the models. They exist only to produce
learnable but noisy behavior for pipeline validation.

No employer dataset, schema, query, credential, customer identifier, business
threshold, or production result is included. Reported model metrics describe the
synthetic run only.
