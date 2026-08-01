"""Generate a fully synthetic ecommerce purchase history."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from next_purchase.config import ProjectConfig

CATEGORY_VALUE = {
    "Beauty": 55.0,
    "Electronics": 145.0,
    "Fashion": 75.0,
    "Grocery": 38.0,
    "Home": 95.0,
    "Sports": 88.0,
}

CATEGORY_MARGIN_RATE = {
    "Beauty": 0.24,
    "Electronics": 0.11,
    "Fashion": 0.22,
    "Grocery": 0.14,
    "Home": 0.19,
    "Sports": 0.20,
}


def _mean_gap(shopper_type: str, progress: float) -> float:
    """Return a latent mean purchase gap that can change over time."""

    if shopper_type == "frequent":
        return 25.0
    if shopper_type == "regular":
        return 43.0
    if shopper_type == "occasional":
        return 75.0
    if shopper_type == "growing":
        return 68.0 - 35.0 * progress
    return 34.0 + 55.0 * progress


def _seasonal_weights(categories: tuple[str, ...], date: pd.Timestamp) -> np.ndarray:
    weights = np.ones(len(categories), dtype=float)
    month = date.month
    boosts = {
        1: {"Sports": 0.45},
        2: {"Beauty": 0.35},
        3: {"Beauty": 0.30, "Home": 0.20},
        4: {"Home": 0.35},
        6: {"Sports": 0.30},
        9: {"Fashion": 0.40},
        11: {"Electronics": 0.55, "Fashion": 0.20},
        12: {"Electronics": 0.45, "Home": 0.20},
    }
    for category, boost in boosts.get(month, {}).items():
        weights[categories.index(category)] += boost
    return weights


def _transition_weights(categories: tuple[str, ...], last_category: str | None) -> np.ndarray:
    weights = np.ones(len(categories), dtype=float)
    if last_category is None:
        return weights

    weights[categories.index(last_category)] += 1.4
    cross_sell = {
        "Beauty": "Fashion",
        "Electronics": "Home",
        "Fashion": "Beauty",
        "Grocery": "Home",
        "Home": "Electronics",
        "Sports": "Fashion",
    }
    weights[categories.index(cross_sell[last_category])] += 0.55
    return weights


def generate_transactions(config: ProjectConfig) -> pd.DataFrame:
    """Create customer transactions with learnable but imperfect behavior patterns."""

    rng = np.random.default_rng(config.seed)
    categories = config.categories
    start = pd.Timestamp(config.start_date)
    end = pd.Timestamp(config.end_date)
    total_days = (end - start).days

    shopper_types = np.array(["frequent", "regular", "occasional", "growing", "cooling"])
    shopper_probabilities = np.array([0.17, 0.34, 0.23, 0.14, 0.12])
    channels = np.array(["Organic", "Paid Search", "Referral", "CRM"])
    regions = np.array(["Central", "North", "South", "West"])

    records: list[dict[str, object]] = []
    order_number = 1

    for customer_number in range(1, config.n_customers + 1):
        customer_id = f"C{customer_number:05d}"
        shopper_type = str(rng.choice(shopper_types, p=shopper_probabilities))
        acquisition_channel = str(rng.choice(channels, p=[0.38, 0.27, 0.16, 0.19]))
        region = str(rng.choice(regions, p=[0.34, 0.23, 0.22, 0.21]))

        preference = rng.dirichlet(np.full(len(categories), 1.1))
        dominant_index = int(rng.integers(0, len(categories)))
        preference[dominant_index] += rng.uniform(0.45, 0.95)
        preference = preference / preference.sum()

        customer_value_multiplier = float(rng.lognormal(mean=0.0, sigma=0.28))
        current_date = start + pd.Timedelta(days=int(rng.integers(0, 150)))
        last_category: str | None = None

        while current_date <= end:
            progress = min(max((current_date - start).days / total_days, 0.0), 1.0)
            mean_gap = _mean_gap(shopper_type, progress)
            gap = max(4, int(round(rng.gamma(shape=3.2, scale=mean_gap / 3.2))))
            current_date += pd.Timedelta(days=gap)
            if current_date > end:
                break

            seasonal = _seasonal_weights(categories, current_date)
            transition = _transition_weights(categories, last_category)
            probabilities = preference * seasonal * transition
            probabilities = probabilities / probabilities.sum()
            category = str(rng.choice(categories, p=probabilities))

            base_value = CATEGORY_VALUE[category]
            order_value = base_value * customer_value_multiplier * rng.lognormal(0.0, 0.34)
            promotion_pressure = 0.10 + 0.08 * (current_date.month in {3, 9, 11})
            discount_probability = min(
                0.62,
                promotion_pressure
                + 0.12 * (acquisition_channel == "Paid Search")
                + 0.08 * (shopper_type == "growing"),
            )
            used_discount = bool(rng.random() < discount_probability)
            discount_rate = float(rng.uniform(0.05, 0.20)) if used_discount else 0.0
            net_value = order_value * (1.0 - discount_rate)

            margin_rate = CATEGORY_MARGIN_RATE[category]
            margin_noise = rng.normal(1.0, 0.07)
            contribution_margin = max(0.0, net_value * margin_rate * margin_noise)

            records.append(
                {
                    "order_id": f"O{order_number:07d}",
                    "customer_id": customer_id,
                    "order_date": current_date.normalize(),
                    "category": category,
                    "order_value": round(net_value, 2),
                    "contribution_margin": round(contribution_margin, 2),
                    "used_discount": int(used_discount),
                    "acquisition_channel": acquisition_channel,
                    "region": region,
                }
            )
            order_number += 1
            last_category = category

            if shopper_type == "cooling" and progress > 0.55:
                stop_probability = 0.015 + 0.06 * (progress - 0.55)
                if rng.random() < stop_probability:
                    break

    transactions = pd.DataFrame.from_records(records)
    if transactions.empty:
        raise RuntimeError("Synthetic generator produced no transactions.")

    transactions["order_date"] = pd.to_datetime(transactions["order_date"])
    transactions = transactions.sort_values(["customer_id", "order_date", "order_id"])
    transactions = transactions.reset_index(drop=True)

    if not math.isclose(float(transactions["used_discount"].mean()), 0.0):
        return transactions
    raise RuntimeError("Synthetic generator produced a degenerate discount feature.")
