"""Configuration shared across the reproducible pipeline."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProjectConfig:
    """Parameters for simulation, snapshot creation, and evaluation."""

    seed: int = 42
    n_customers: int = 4_000
    start_date: str = "2023-01-01"
    end_date: str = "2025-12-31"
    snapshot_start: str = "2024-01-01"
    snapshot_end: str = "2025-10-01"
    snapshot_frequency_days: int = 28
    validation_start: str = "2025-01-01"
    test_start: str = "2025-07-01"
    scoring_date: str = "2025-11-01"
    min_history_orders: int = 3
    target_horizon_days: int = 60
    readiness_horizon_days: int = 30
    categories: tuple[str, ...] = (
        "Beauty",
        "Electronics",
        "Fashion",
        "Grocery",
        "Home",
        "Sports",
    )
