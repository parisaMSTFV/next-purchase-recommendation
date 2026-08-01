"""Command-line interface for the case study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from next_purchase.config import ProjectConfig
from next_purchase.pipeline import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the synthetic next-purchase recommendation case study."
    )
    parser.add_argument(
        "command",
        choices=["run"],
        help="Run data generation, training, evaluation, and scoring.",
    )
    parser.add_argument(
        "--project-root",
        default=".",
        help="Repository root where data and reports should be written.",
    )
    parser.add_argument(
        "--customers",
        type=int,
        default=ProjectConfig.n_customers,
        help="Number of synthetic customers.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=ProjectConfig.seed,
        help="Random seed.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = ProjectConfig(n_customers=args.customers, seed=args.seed)
    result = run_pipeline(Path(args.project_root), config)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
