# Supplied transaction contract v1.0

The `score` command creates a review queue from a caller-provided transaction history. It does not
train or evaluate the synthetic benchmark models.

## Provenance manifest

A JSON manifest is required beside the transaction file:

| Field | Rule |
|---|---|
| `schema_version` | Must equal `1.0` |
| `source_name` | Non-blank source label chosen by the caller |
| `source_type` | Non-blank type such as `public`, `synthetic_fixture`, or `approved_internal` |
| `license_or_authorization` | Non-blank license or authorization basis |
| `identifiers_pseudonymized` | Must be `true` |
| `contains_direct_identifiers` | Must be `false` |

The command records this manifest and a SHA-256 checksum in `run_metadata.json`. It does not copy
the source CSV. The output queue retains the supplied pseudonymous customer IDs, so its directory
must still be protected and must not be committed.

## Transaction CSV

The machine-readable contract is in [`schemas/supplied_transactions_v1.json`](../schemas/supplied_transactions_v1.json).

| Column | Rule |
|---|---|
| `order_id` | Unique, non-blank pseudonymous order key |
| `customer_id` | Non-blank pseudonymous customer key |
| `order_date` | Valid ISO-8601 date |
| `category` | Non-blank category label; at least three historical categories are required |
| `order_value` | Finite number greater than or equal to zero |
| `contribution_margin` | Finite number; negative values are allowed |
| `used_discount` | Integer `0` or `1` |
| `acquisition_channel` | Non-blank label |
| `region` | Non-blank label |

Extra columns are ignored. Formula-prefixed identifiers are rejected. Only transactions strictly
before `score_date` enter features, category distributions, and expected margins; later rows are
counted in metadata and ignored.

## Policy and outputs

```bash
next-purchase score \
  --transactions path/to/transactions.csv \
  --provenance path/to/provenance.json \
  --score-date 2025-06-01 \
  --output-root artifacts/my-run
```

Supplied-input mode uses two transparent reference rules:

- purchase readiness: progress through each customer's observed purchase cadence;
- category relevance: a smoothed repeat-last-category baseline.

It writes:

- `reports/recommendations.csv` with the ranked customer queue;
- `reports/activation_summary.csv` with aggregate tier/category counts;
- `reports/run_metadata.json` with schema, provenance, checksum, row counts, and policy identity;
- `reports/run_summary.md` with the evaluation boundary;
- `reports/figures/activation_queue.png` with aggregate queue composition.

No fitted model, future outcome, synthetic truth, Top-K score, readiness accuracy, uplift, or
business-impact estimate is produced for supplied input. Those claims require a separately designed
temporal evaluation or randomized experiment.
