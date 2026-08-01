import unittest

import numpy as np
import pandas as pd

from next_purchase.decision import activation_summary, build_activation_queue


class DecisionTests(unittest.TestCase):
    def test_activation_queue_is_ranked_and_capacity_limited(self) -> None:
        n_rows = 100
        scoring = pd.DataFrame(
            {
                "customer_id": [f"C{index:03d}" for index in range(n_rows)],
                "score_date": pd.Timestamp("2025-11-01"),
            }
        )
        probabilities = np.tile(np.array([[0.7, 0.2, 0.1]]), (n_rows, 1))
        readiness = np.linspace(0.99, 0.10, n_rows)
        categories = ("Beauty", "Home", "Sports")

        queue = build_activation_queue(
            scoring,
            probabilities,
            readiness,
            categories,
            expected_margin_by_category={"Beauty": 12.0, "Home": 18.0, "Sports": 15.0},
        )

        self.assertTrue(queue["action_score"].is_monotonic_decreasing)
        self.assertEqual((queue["capacity_tier"] == "Priority").sum(), 15)
        self.assertEqual((queue["capacity_tier"] == "Review").sum(), 25)
        self.assertEqual(queue.iloc[0]["recommended_category"], "Beauty")

        summary = activation_summary(queue)
        self.assertEqual(summary["customers"].sum(), n_rows)


if __name__ == "__main__":
    unittest.main()
