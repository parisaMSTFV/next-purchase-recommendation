"""End-to-end reproducible training, evaluation, and recommendation pipeline."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pandas as pd

from next_purchase.config import ProjectConfig
from next_purchase.decision import activation_summary, build_activation_queue
from next_purchase.evaluation import (
    binary_metrics,
    category_metrics,
    per_category_recall,
)
from next_purchase.features import (
    build_labeled_snapshots,
    build_scoring_snapshot,
    feature_columns,
    split_by_time,
)
from next_purchase.modeling import (
    MODEL_NAMES,
    aligned_probabilities,
    build_classifier,
    cadence_rule_probabilities,
    last_category_probabilities,
    popularity_probabilities,
)
from next_purchase.reporting import (
    plot_activation_mix,
    plot_category_model_comparison,
    plot_category_recall,
    plot_readiness_comparison,
    plot_top_k_performance,
    write_decision_note,
    write_json,
    write_run_summary,
)
from next_purchase.simulation import generate_transactions


def _category_candidates(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    features: list[str],
    config: ProjectConfig,
) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    train_category = train.loc[train["next_category"].notna()].copy()
    validation_category = validation.loc[validation["next_category"].notna()].copy()
    categories = list(config.categories)

    probabilities = {
        "popularity_baseline": popularity_probabilities(
            train_category["next_category"],
            len(validation_category),
            categories,
        ),
        "last_category_baseline": last_category_probabilities(
            validation_category["last_category"],
            train_category["next_category"],
            categories,
        ),
    }
    for model_name in MODEL_NAMES:
        model = build_classifier(
            model_name,
            feature_names=features,
            seed=config.seed,
            binary=False,
        )
        model.fit(train_category[features], train_category["next_category"])
        probabilities[model_name] = aligned_probabilities(
            model,
            validation_category[features],
            categories,
        )

    metrics = {
        candidate: category_metrics(
            validation_category["next_category"],
            candidate_probability,
            categories,
        )
        for candidate, candidate_probability in probabilities.items()
    }
    comparison = pd.DataFrame(
        [{"candidate": name, **values} for name, values in metrics.items()]
    )
    return comparison, metrics


def _readiness_candidates(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    features: list[str],
    config: ProjectConfig,
) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    probabilities = {
        "cadence_rule_baseline": cadence_rule_probabilities(validation),
    }
    for model_name in MODEL_NAMES:
        model = build_classifier(
            model_name,
            feature_names=features,
            seed=config.seed,
            binary=True,
        )
        model.fit(train[features], train["purchase_within_30d"])
        probabilities[model_name] = model.predict_proba(validation[features])[:, 1]

    metrics = {
        candidate: binary_metrics(
            validation["purchase_within_30d"],
            candidate_probability,
        )
        for candidate, candidate_probability in probabilities.items()
    }
    comparison = pd.DataFrame(
        [{"candidate": name, **values} for name, values in metrics.items()]
    )
    return comparison, metrics


def _select_category_model(comparison: pd.DataFrame) -> str:
    candidates = comparison.loc[comparison["candidate"].isin(MODEL_NAMES)].copy()
    candidates = candidates.sort_values(
        ["top_3_hit_rate", "log_loss"],
        ascending=[False, True],
    )
    return str(candidates.iloc[0]["candidate"])


def _select_readiness_model(comparison: pd.DataFrame) -> str:
    candidates = comparison.loc[comparison["candidate"].isin(MODEL_NAMES)].copy()
    candidates = candidates.sort_values(
        ["average_precision", "brier_score"],
        ascending=[False, True],
    )
    return str(candidates.iloc[0]["candidate"])


def run_pipeline(
    project_root: str | Path,
    config: ProjectConfig | None = None,
) -> dict[str, object]:
    """Run simulation, point-in-time training, evaluation, and final scoring."""

    config = config or ProjectConfig()
    project_root = Path(project_root)
    generated_dir = project_root / "data" / "generated"
    reports_dir = project_root / "reports"
    figures_dir = reports_dir / "figures"
    for directory in (generated_dir, reports_dir, figures_dir):
        directory.mkdir(parents=True, exist_ok=True)

    transactions = generate_transactions(config)
    transactions.to_csv(generated_dir / "transactions.csv", index=False)

    snapshots = build_labeled_snapshots(transactions, config)
    snapshots.to_csv(generated_dir / "historical_snapshots.csv", index=False)
    train, validation, test = split_by_time(snapshots, config)
    features = feature_columns(snapshots)

    category_comparison, category_validation_metrics = _category_candidates(
        train,
        validation,
        features,
        config,
    )
    readiness_comparison, readiness_validation_metrics = _readiness_candidates(
        train,
        validation,
        features,
        config,
    )
    selected_category = _select_category_model(category_comparison)
    selected_readiness = _select_readiness_model(readiness_comparison)

    train_validation = pd.concat([train, validation], ignore_index=True)
    train_validation_category = train_validation.loc[
        train_validation["next_category"].notna()
    ].copy()
    test_category = test.loc[test["next_category"].notna()].copy()
    categories = list(config.categories)

    category_model = build_classifier(
        selected_category,
        feature_names=features,
        seed=config.seed,
        binary=False,
    )
    category_model.fit(
        train_validation_category[features],
        train_validation_category["next_category"],
    )
    category_test_probability = aligned_probabilities(
        category_model,
        test_category[features],
        categories,
    )
    category_test_metrics = category_metrics(
        test_category["next_category"],
        category_test_probability,
        categories,
    )
    category_recall = per_category_recall(
        test_category["next_category"],
        category_test_probability,
        categories,
    )

    readiness_model = build_classifier(
        selected_readiness,
        feature_names=features,
        seed=config.seed,
        binary=True,
    )
    readiness_model.fit(
        train_validation[features],
        train_validation["purchase_within_30d"],
    )
    readiness_test_probability = readiness_model.predict_proba(test[features])[:, 1]
    readiness_test_metrics = binary_metrics(
        test["purchase_within_30d"],
        readiness_test_probability,
    )

    scoring_frame = build_scoring_snapshot(
        transactions,
        config.scoring_date,
        config,
    )
    category_scoring_probability = aligned_probabilities(
        category_model,
        scoring_frame[features],
        categories,
    )
    readiness_scoring_probability = readiness_model.predict_proba(
        scoring_frame[features]
    )[:, 1]
    historical_margin = (
        transactions.loc[transactions["order_date"] < pd.Timestamp(config.scoring_date)]
        .groupby("category")["contribution_margin"]
        .median()
        .to_dict()
    )
    queue = build_activation_queue(
        scoring_frame,
        category_scoring_probability,
        readiness_scoring_probability,
        config.categories,
        expected_margin_by_category={
            category: float(historical_margin[category]) for category in config.categories
        },
    )
    queue_summary = activation_summary(queue)

    popularity_test_probability = popularity_probabilities(
        train_validation_category["next_category"],
        len(test_category),
        categories,
    )
    last_category_test_probability = last_category_probabilities(
        test_category["last_category"],
        train_validation_category["next_category"],
        categories,
    )
    test_plot_metrics = {
        "Popularity baseline": category_metrics(
            test_category["next_category"],
            popularity_test_probability,
            categories,
        ),
        "Last-category baseline": category_metrics(
            test_category["next_category"],
            last_category_test_probability,
            categories,
        ),
        selected_category.replace("_", " ").title(): category_test_metrics,
    }

    category_comparison_to_write = category_comparison.copy()
    readiness_comparison_to_write = readiness_comparison.copy()
    category_comparison_to_write["task"] = "next_category"
    readiness_comparison_to_write["task"] = "purchase_readiness_30d"
    model_comparison = pd.concat(
        [category_comparison_to_write, readiness_comparison_to_write],
        ignore_index=True,
        sort=False,
    )
    model_comparison.to_csv(reports_dir / "model_comparison.csv", index=False)
    category_comparison.to_csv(
        reports_dir / "category_model_comparison.csv",
        index=False,
    )
    readiness_comparison.to_csv(
        reports_dir / "readiness_model_comparison.csv",
        index=False,
    )
    category_recall.to_csv(reports_dir / "per_category_recall.csv", index=False)
    queue.head(30).to_csv(reports_dir / "recommendations_sample.csv", index=False)
    queue_summary.to_csv(reports_dir / "activation_summary.csv", index=False)

    metrics: dict[str, object] = {
        "selected_category_model": selected_category,
        "selected_readiness_model": selected_readiness,
        "category_validation": category_validation_metrics[selected_category],
        "readiness_validation": readiness_validation_metrics[selected_readiness],
        "category_test": category_test_metrics,
        "readiness_test": readiness_test_metrics,
    }
    dataset_summary: dict[str, int | float | str] = {
        "seed": config.seed,
        "customers": int(transactions["customer_id"].nunique()),
        "transactions": int(len(transactions)),
        "snapshots": int(len(snapshots)),
        "category_test_rows": int(len(test_category)),
        "test_start": config.test_start,
        "scoring_customers": int(len(scoring_frame)),
    }
    write_json(
        {
            "config": asdict(config),
            "dataset": dataset_summary,
            **metrics,
        },
        reports_dir / "metrics.json",
    )
    write_run_summary(
        reports_dir / "run_summary.md",
        metrics,
        dataset_summary,
    )
    write_decision_note(
        reports_dir / "decision_note.md",
        metrics,
        queue,
    )

    category_plot = category_comparison.copy()
    category_plot["top_3_hit_rate"] = category_plot["top_3_hit_rate"] * 100
    plot_category_model_comparison(
        category_plot,
        figures_dir / "category_model_comparison.png",
    )
    plot_top_k_performance(
        test_plot_metrics,
        figures_dir / "top_k_performance.png",
    )
    plot_readiness_comparison(
        readiness_comparison,
        figures_dir / "readiness_model_comparison.png",
    )
    plot_category_recall(
        category_recall,
        figures_dir / "category_recall.png",
    )
    plot_activation_mix(
        queue_summary,
        figures_dir / "activation_queue.png",
    )

    return {
        "metrics": metrics,
        "dataset": dataset_summary,
        "selected_category_model": selected_category,
        "selected_readiness_model": selected_readiness,
    }
