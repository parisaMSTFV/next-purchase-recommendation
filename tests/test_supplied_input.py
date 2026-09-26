import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from next_purchase.supplied_input import (
    SCHEMA_VERSION,
    SuppliedInputError,
    load_provenance,
    load_transactions,
    score_supplied_transactions,
)


class SuppliedInputTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(__file__).resolve().parents[1]
        self.transactions = self.root / "examples" / "supplied_transactions_v1.csv"
        self.provenance = self.root / "examples" / "supplied_provenance_v1.json"

    def test_fixture_matches_versioned_contract(self) -> None:
        frame = load_transactions(self.transactions)
        provenance = load_provenance(self.provenance)
        schema = json.loads((self.root / "schemas" / "supplied_transactions_v1.json").read_text())
        self.assertEqual(provenance["schema_version"], SCHEMA_VERSION)
        self.assertEqual(schema["schema_version"], SCHEMA_VERSION)
        self.assertEqual(list(frame.columns), schema["required_columns"])
        self.assertEqual(len(frame), 32)

    def test_provenance_rejects_direct_identifiers(self) -> None:
        raw = json.loads(self.provenance.read_text())
        raw["identifiers_pseudonymized"] = False
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "provenance.json"
            path.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaisesRegex(SuppliedInputError, "must be pseudonymized"):
                load_provenance(path)

    def test_duplicate_order_ids_are_rejected(self) -> None:
        frame = pd.read_csv(self.transactions)
        frame.loc[1, "order_id"] = frame.loc[0, "order_id"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "transactions.csv"
            frame.to_csv(path, index=False)
            with self.assertRaisesRegex(SuppliedInputError, "order_id must be unique"):
                load_transactions(path)

    def test_order_timestamp_is_not_silently_truncated(self) -> None:
        frame = pd.read_csv(self.transactions)
        frame.loc[0, "order_date"] = "2025-01-01T23:00:00Z"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "transactions.csv"
            frame.to_csv(path, index=False)
            with self.assertRaisesRegex(SuppliedInputError, "without a time or timezone"):
                load_transactions(path)

    def test_score_date_requires_a_date_without_time(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(SuppliedInputError, "without a time or timezone"):
                score_supplied_transactions(
                    self.transactions,
                    self.provenance,
                    "2025-06-01T12:00:00",
                    Path(directory),
                )

    def test_supplied_scoring_writes_outputs_without_evaluation_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory) / "output"
            metadata = score_supplied_transactions(
                self.transactions,
                self.provenance,
                "2025-06-01",
                output_root,
            )
            required = [
                "reports/recommendations.csv",
                "reports/activation_summary.csv",
                "reports/run_metadata.json",
                "reports/run_summary.md",
                "reports/figures/activation_queue.png",
            ]
            self.assertTrue(all((output_root / path).exists() for path in required))
            self.assertEqual(metadata["mode"], "supplied_input_scoring")
            self.assertEqual(metadata["input_summary"]["eligible_customers"], 8)
            self.assertNotIn("category_test", metadata)
            self.assertNotIn("readiness_test", metadata)
            self.assertIn("No fitted model", metadata["evaluation_boundary"])

    def test_rows_on_or_after_score_date_cannot_change_queue(self) -> None:
        frame = pd.read_csv(self.transactions)
        future = frame.iloc[[0]].copy()
        future["order_id"] = "FIX-FUTURE"
        future["order_date"] = "2025-07-01"
        future["category"] = "Future Only"
        future["contribution_margin"] = 999999
        modified = pd.concat([frame, future], ignore_index=True)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            modified_path = root / "modified.csv"
            modified.to_csv(modified_path, index=False)
            score_supplied_transactions(
                self.transactions,
                self.provenance,
                "2025-06-01",
                root / "base",
            )
            changed = score_supplied_transactions(
                modified_path,
                self.provenance,
                "2025-06-01",
                root / "changed",
            )
            base_queue = pd.read_csv(root / "base" / "reports" / "recommendations.csv")
            changed_queue = pd.read_csv(root / "changed" / "reports" / "recommendations.csv")
            pd.testing.assert_frame_equal(base_queue, changed_queue)
            self.assertEqual(changed["input_summary"]["ignored_on_or_after_score_date"], 1)


if __name__ == "__main__":
    unittest.main()
