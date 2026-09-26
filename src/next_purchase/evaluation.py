"""Evaluation metrics for ranked category recommendations and readiness."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)


def _true_class_indices(
    y_true: pd.Series | np.ndarray,
    class_order: list[str] | tuple[str, ...],
) -> np.ndarray:
    lookup = {label: index for index, label in enumerate(class_order)}
    return np.array([lookup[str(label)] for label in y_true], dtype=int)


def top_k_hit_rate(
    y_true: pd.Series | np.ndarray,
    probabilities: np.ndarray,
    class_order: list[str] | tuple[str, ...],
    k: int,
) -> float:
    true_indices = _true_class_indices(y_true, class_order)
    top_k = np.argsort(-probabilities, axis=1, kind="stable")[:, :k]
    return float(np.mean(np.any(top_k == true_indices[:, None], axis=1)))


def mean_reciprocal_rank(
    y_true: pd.Series | np.ndarray,
    probabilities: np.ndarray,
    class_order: list[str] | tuple[str, ...],
) -> float:
    true_indices = _true_class_indices(y_true, class_order)
    ranking = np.argsort(-probabilities, axis=1, kind="stable")
    ranks = np.argmax(ranking == true_indices[:, None], axis=1) + 1
    return float(np.mean(1.0 / ranks))


def multiclass_brier_score(
    y_true: pd.Series | np.ndarray,
    probabilities: np.ndarray,
    class_order: list[str] | tuple[str, ...],
) -> float:
    true_indices = _true_class_indices(y_true, class_order)
    observed = np.eye(len(class_order))[true_indices]
    return float(np.mean(np.sum((probabilities - observed) ** 2, axis=1)))


def category_metrics(
    y_true: pd.Series | np.ndarray,
    probabilities: np.ndarray,
    class_order: list[str] | tuple[str, ...],
) -> dict[str, float]:
    """Return ranking and probability-quality measures."""

    true_indices = _true_class_indices(y_true, class_order)
    predicted_indices = probabilities.argmax(axis=1)
    return {
        "top_1_hit_rate": float(np.mean(predicted_indices == true_indices)),
        "top_2_hit_rate": top_k_hit_rate(y_true, probabilities, class_order, 2),
        "top_3_hit_rate": top_k_hit_rate(y_true, probabilities, class_order, 3),
        "mean_reciprocal_rank": mean_reciprocal_rank(y_true, probabilities, class_order),
        "log_loss": float(log_loss(y_true, probabilities, labels=list(class_order))),
        "multiclass_brier_score": multiclass_brier_score(y_true, probabilities, class_order),
    }


def per_category_recall(
    y_true: pd.Series | np.ndarray,
    probabilities: np.ndarray,
    class_order: list[str] | tuple[str, ...],
) -> pd.DataFrame:
    true_values = np.asarray(y_true, dtype=str)
    predicted = np.asarray(class_order)[probabilities.argmax(axis=1)]
    rows = []
    for category in class_order:
        mask = true_values == category
        rows.append(
            {
                "category": category,
                "support": int(mask.sum()),
                "top_1_recall": float(np.mean(predicted[mask] == category))
                if mask.any()
                else np.nan,
                "top_3_recall": top_k_hit_rate(
                    true_values[mask],
                    probabilities[mask],
                    class_order,
                    3,
                )
                if mask.any()
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


def binary_metrics(y_true: pd.Series | np.ndarray, probability: np.ndarray) -> dict[str, float]:
    """Return discrimination, calibration, and capacity metrics."""

    observed = np.asarray(y_true, dtype=int)
    probability = np.asarray(probability, dtype=float)
    order = np.argsort(-probability, kind="stable")
    top_20_count = max(1, int(np.ceil(0.20 * len(observed))))
    positives = observed.sum()
    recall_top_20 = float(observed[order[:top_20_count]].sum() / positives) if positives else 0.0
    base_rate = float(observed.mean())
    precision_top_20 = float(observed[order[:top_20_count]].mean())
    lift_top_20 = precision_top_20 / base_rate if base_rate else 0.0

    return {
        "average_precision": float(average_precision_score(observed, probability)),
        "roc_auc": float(roc_auc_score(observed, probability)),
        "brier_score": float(brier_score_loss(observed, probability)),
        "log_loss": float(log_loss(observed, probability, labels=[0, 1])),
        "recall_at_top_20pct": recall_top_20,
        "lift_at_top_20pct": lift_top_20,
    }
