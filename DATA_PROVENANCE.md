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

`examples/supplied_transactions_v1.csv` is a separate, hand-authored fictional fixture used to
test the versioned supplied-input path. Its `FIX-` identifiers do not refer to people, accounts,
orders, or products. The accompanying provenance manifest identifies it as a synthetic fixture.

Caller-supplied files are read locally and are not copied into the output directory. The generated
recommendation queue retains the caller's pseudonymous customer IDs, so supplied inputs and outputs
must remain outside Git unless their license, authorization, and privacy controls are independently verified.
