import json
import tempfile
import unittest
from pathlib import Path

from next_purchase.cli import build_parser
from next_purchase.config import ProjectConfig
from next_purchase.pipeline import run_pipeline


class PipelineTests(unittest.TestCase):
    def test_run_defaults_to_ignored_local_directory(self) -> None:
        args = build_parser().parse_args(["run"])
        self.assertEqual(args.project_root, "local-runs/latest")

    def test_small_pipeline_writes_reproducible_outputs(self) -> None:
        config = ProjectConfig(n_customers=90, seed=23)
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            result = run_pipeline(root, config)

            self.assertIn(
                result["selected_category_model"],
                {"logistic_regression", "hist_gradient_boosting"},
            )
            self.assertIn(
                result["selected_readiness_model"],
                {"logistic_regression", "hist_gradient_boosting"},
            )
            self.assertTrue((root / "reports" / "metrics.json").exists())
            self.assertTrue((root / "reports" / "decision_note.md").exists())
            self.assertTrue((root / "reports" / "figures" / "top_k_performance.png").exists())

            metrics = json.loads((root / "reports" / "metrics.json").read_text(encoding="utf-8"))
            self.assertLessEqual(metrics["category_test"]["top_3_hit_rate"], 1.0)
            self.assertGreaterEqual(metrics["category_test"]["top_3_hit_rate"], 0.0)
            self.assertLessEqual(metrics["readiness_test"]["average_precision"], 1.0)
            self.assertGreaterEqual(metrics["readiness_test"]["average_precision"], 0.0)


if __name__ == "__main__":
    unittest.main()
