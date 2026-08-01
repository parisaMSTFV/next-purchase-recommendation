import unittest

import pandas as pd

from next_purchase.config import ProjectConfig
from next_purchase.simulation import generate_transactions


class SimulationTests(unittest.TestCase):
    def test_simulation_is_reproducible_and_valid(self) -> None:
        config = ProjectConfig(n_customers=30, seed=7)
        first = generate_transactions(config)
        second = generate_transactions(config)

        pd.testing.assert_frame_equal(first, second)
        self.assertTrue(set(first["category"]).issubset(config.categories))
        self.assertTrue((first["order_value"] > 0).all())
        self.assertTrue((first["contribution_margin"] >= 0).all())
        self.assertTrue(first["order_id"].is_unique)


if __name__ == "__main__":
    unittest.main()
