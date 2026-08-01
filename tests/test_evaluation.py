import unittest

import numpy as np
import pandas as pd

from next_purchase.evaluation import (
    binary_metrics,
    category_metrics,
    mean_reciprocal_rank,
    top_k_hit_rate,
)


class EvaluationTests(unittest.TestCase):
    def test_ranked_metrics_match_known_example(self) -> None:
        classes = ["A", "B", "C"]
        observed = pd.Series(["A", "B", "C"])
        probabilities = np.array(
            [
                [0.70, 0.20, 0.10],
                [0.55, 0.35, 0.10],
                [0.45, 0.20, 0.35],
            ]
        )

        self.assertAlmostEqual(
            top_k_hit_rate(observed, probabilities, classes, 1),
            1 / 3,
        )
        self.assertEqual(top_k_hit_rate(observed, probabilities, classes, 2), 1.0)
        self.assertAlmostEqual(
            mean_reciprocal_rank(observed, probabilities, classes),
            2 / 3,
        )

        metrics = category_metrics(observed, probabilities, classes)
        self.assertEqual(metrics["top_3_hit_rate"], 1.0)
        self.assertGreater(metrics["log_loss"], 0.0)

    def test_binary_metrics_reward_a_useful_ranking(self) -> None:
        observed = np.array([1, 0, 1, 0, 1, 0, 0, 0, 1, 0])
        useful = np.array([0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.15, 0.05, 0.6, 0.4])
        reversed_scores = 1.0 - useful

        useful_metrics = binary_metrics(observed, useful)
        reversed_metrics = binary_metrics(observed, reversed_scores)

        self.assertGreater(
            useful_metrics["average_precision"],
            reversed_metrics["average_precision"],
        )
        self.assertGreater(useful_metrics["roc_auc"], reversed_metrics["roc_auc"])


if __name__ == "__main__":
    unittest.main()
