"""Convert model probabilities into a reviewable activation queue."""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_activation_queue(
    scoring_frame: pd.DataFrame,
    category_probabilities: np.ndarray,
    readiness_probability: np.ndarray,
    categories: tuple[str, ...],
    expected_margin_by_category: dict[str, float],
) -> pd.DataFrame:
    """Create a capacity-ranked queue without claiming incremental campaign impact."""

    ranking = np.argsort(-category_probabilities, axis=1)
    category_array = np.asarray(categories)
    top_categories = category_array[ranking[:, :3]]

    recommended_index = ranking[:, 0]
    recommendation_probability = category_probabilities[
        np.arange(len(scoring_frame)), recommended_index
    ]
    recommended_category = category_array[recommended_index]
    expected_margin = np.array(
        [expected_margin_by_category[str(category)] for category in recommended_category],
        dtype=float,
    )
    action_score = readiness_probability * recommendation_probability * expected_margin

    queue = pd.DataFrame(
        {
            "customer_id": scoring_frame["customer_id"].to_numpy(),
            "score_date": scoring_frame["score_date"].to_numpy(),
            "recommended_category": recommended_category,
            "category_probability": recommendation_probability,
            "purchase_readiness_30d": readiness_probability,
            "expected_category_margin": expected_margin,
            "action_score": action_score,
            "second_category": top_categories[:, 1],
            "third_category": top_categories[:, 2],
        }
    )

    queue = queue.sort_values(
        ["action_score", "purchase_readiness_30d"],
        ascending=False,
    ).reset_index(drop=True)
    percentile = (np.arange(len(queue)) + 1) / len(queue)
    queue["capacity_tier"] = np.select(
        [percentile <= 0.15, percentile <= 0.40],
        ["Priority", "Review"],
        default="Monitor",
    )
    queue["queue_rank"] = np.arange(1, len(queue) + 1)
    return queue


def activation_summary(queue: pd.DataFrame) -> pd.DataFrame:
    """Summarize the decision queue without exposing all scored rows."""

    summary = (
        queue.groupby(["capacity_tier", "recommended_category"], observed=True)
        .agg(
            customers=("customer_id", "size"),
            average_readiness=("purchase_readiness_30d", "mean"),
            average_category_probability=("category_probability", "mean"),
            average_action_score=("action_score", "mean"),
        )
        .reset_index()
    )
    tier_order = pd.CategoricalDtype(
        categories=["Priority", "Review", "Monitor"],
        ordered=True,
    )
    summary["capacity_tier"] = summary["capacity_tier"].astype(tier_order)
    return summary.sort_values(["capacity_tier", "customers"], ascending=[True, False])
