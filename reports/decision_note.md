# Decision Note

## Recommended use

Use the 30-day readiness score to control contact capacity, then use the
ranked category probabilities to choose the most relevant category or landing
page. The current demonstration assigns the top 15% of eligible customers to a
`Priority` queue and the next 25% to `Review`.

The selected category model places the observed next category in its first
three suggestions for 83.7% of test snapshots. The
readiness model captures 28.5% of 30-day
purchasers in the top 20% of its test ranking.

At the latest synthetic scoring date, the Priority queue contains
599 customers. Its recommended-category mix is: Sports 27.0%, Home 26.2%, Electronics 22.5%, Fashion 18.7%, Beauty 5.5%.

## Guardrails

- Treat the action score as a capacity-ranking heuristic.
- Do not interpret recommendation probability as incremental response.
- Test channel, offer, and creative choices through a randomized holdout.
- Monitor Top-K quality, readiness calibration, category coverage, and score
  drift by scoring month.
- Apply contact-frequency, eligibility, inventory, and consent rules after
  model scoring.
