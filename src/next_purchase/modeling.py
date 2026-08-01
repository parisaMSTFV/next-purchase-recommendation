"""Model definitions and probability alignment helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from next_purchase.features import CATEGORICAL_FEATURES


MODEL_NAMES = ("logistic_regression", "hist_gradient_boosting")


def _preprocessor(feature_names: list[str]) -> ColumnTransformer:
    categorical = [name for name in CATEGORICAL_FEATURES if name in feature_names]
    numeric = [name for name in feature_names if name not in categorical]

    numeric_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            (
                "one_hot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric),
            ("categorical", categorical_pipeline, categorical),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_classifier(
    model_name: str,
    feature_names: list[str],
    seed: int,
    binary: bool,
) -> Pipeline:
    """Build a reproducible classification pipeline."""

    if model_name == "logistic_regression":
        classifier: Any = LogisticRegression(
            max_iter=700,
            C=0.8,
            class_weight="balanced" if binary else None,
            random_state=seed,
        )
    elif model_name == "hist_gradient_boosting":
        classifier = HistGradientBoostingClassifier(
            learning_rate=0.07,
            max_iter=180,
            max_leaf_nodes=19,
            min_samples_leaf=30,
            l2_regularization=0.25,
            class_weight="balanced" if binary else None,
            random_state=seed,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

    return Pipeline(
        steps=[
            ("preprocess", _preprocessor(feature_names)),
            ("classifier", classifier),
        ]
    )


def aligned_probabilities(
    model: Pipeline,
    features: pd.DataFrame,
    class_order: list[str] | tuple[str, ...] | list[int],
) -> np.ndarray:
    """Return probabilities with columns aligned to an explicit class order."""

    probabilities = model.predict_proba(features)
    model_classes = list(model.named_steps["classifier"].classes_)
    aligned = np.zeros((len(features), len(class_order)), dtype=float)
    for destination, label in enumerate(class_order):
        if label in model_classes:
            source = model_classes.index(label)
            aligned[:, destination] = probabilities[:, source]
    row_sums = aligned.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0.0] = 1.0
    return aligned / row_sums


def popularity_probabilities(
    y_train: pd.Series,
    n_rows: int,
    class_order: list[str] | tuple[str, ...],
) -> np.ndarray:
    """Repeat the training-set category distribution for every row."""

    distribution = y_train.value_counts(normalize=True)
    row = np.array([float(distribution.get(label, 0.0)) for label in class_order])
    row = row / row.sum()
    return np.tile(row, (n_rows, 1))


def last_category_probabilities(
    last_category: pd.Series,
    y_train: pd.Series,
    class_order: list[str] | tuple[str, ...],
) -> np.ndarray:
    """Create a smoothed repeat-the-last-category baseline."""

    base = popularity_probabilities(y_train, len(last_category), class_order)
    result = 0.35 * base
    class_lookup = {label: index for index, label in enumerate(class_order)}
    for row_index, label in enumerate(last_category.astype(str)):
        if label in class_lookup:
            result[row_index, class_lookup[label]] += 0.65
    return result / result.sum(axis=1, keepdims=True)


def cadence_rule_probabilities(frame: pd.DataFrame) -> np.ndarray:
    """Map purchase-cycle progress to a bounded 30-day readiness probability."""

    progress = frame["cadence_progress"].to_numpy(dtype=float)
    momentum = frame["order_momentum"].to_numpy(dtype=float)
    linear = 2.1 * (progress - 0.75) + 0.25 * np.clip(momentum - 1.0, -2.0, 2.0)
    return 1.0 / (1.0 + np.exp(-linear))
