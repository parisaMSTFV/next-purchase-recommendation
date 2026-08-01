import unittest

import pandas as pd

from next_purchase.config import ProjectConfig
from next_purchase.features import build_scoring_snapshot
from next_purchase.simulation import generate_transactions


class FeatureTests(unittest.TestCase):
    def test_scoring_features_ignore_future_transactions(self) -> None:
        config = ProjectConfig(n_customers=50, seed=11)
        transactions = generate_transactions(config)
        score_date = pd.Timestamp("2024-08-01")

        original = build_scoring_snapshot(transactions, score_date, config)
        modified = transactions.copy()
        future_mask = modified["order_date"] >= score_date
        modified.loc[future_mask, "category"] = "Electronics"
        modified.loc[future_mask, "order_value"] = 99_999.0
        modified.loc[future_mask, "contribution_margin"] = 50_000.0

        rescored = build_scoring_snapshot(modified, score_date, config)
        pd.testing.assert_frame_equal(original, rescored)

    def test_scoring_uses_only_orders_strictly_before_score_date(self) -> None:
        config = ProjectConfig(n_customers=40, seed=17)
        transactions = generate_transactions(config)
        score_date = pd.Timestamp("2024-09-01")
        customer_id = transactions.iloc[0]["customer_id"]

        eligible_history = transactions.loc[
            (transactions["customer_id"] == customer_id)
            & (transactions["order_date"] < score_date)
        ]
        if len(eligible_history) < config.min_history_orders:
            self.skipTest("Selected synthetic customer has insufficient history.")

        scored = build_scoring_snapshot(transactions, score_date, config)
        customer_row = scored.loc[scored["customer_id"] == customer_id].iloc[0]
        self.assertEqual(customer_row["orders_lifetime"], len(eligible_history))


if __name__ == "__main__":
    unittest.main()
