"""Command-line interface for the case study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from next_purchase.config import ProjectConfig
from next_purchase.pipeline import run_pipeline
from next_purchase.supplied_input import score_supplied_transactions


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the next-purchase benchmark or score validated supplied transactions."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser(
        "run", help="run synthetic generation, training, evaluation, and scoring"
    )
    run.add_argument(
        "--project-root",
        default=".",
        help="Repository root where data and reports should be written.",
    )
    run.add_argument(
        "--customers",
        type=int,
        default=ProjectConfig.n_customers,
        help="Number of synthetic customers.",
    )
    run.add_argument(
        "--seed",
        type=int,
        default=ProjectConfig.seed,
        help="Random seed.",
    )
    score = subparsers.add_parser(
        "score", help="score a versioned supplied transaction history without model evaluation"
    )
    score.add_argument("--transactions", type=Path, required=True)
    score.add_argument("--provenance", type=Path, required=True)
    score.add_argument("--score-date", required=True)
    score.add_argument("--output-root", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "score":
        result = score_supplied_transactions(
            transactions_path=args.transactions,
            provenance_path=args.provenance,
            score_date=args.score_date,
            output_root=args.output_root,
        )
    else:
        config = ProjectConfig(n_customers=args.customers, seed=args.seed)
        result = run_pipeline(Path(args.project_root), config)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
