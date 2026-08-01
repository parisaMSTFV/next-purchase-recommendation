"""Write compact, reproducible tables, charts, and decision notes."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLORS = {
    "baseline": "#94A3B8",
    "model": "#0F766E",
    "accent": "#C2410C",
    "dark": "#1E293B",
}


def _format_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#E2E8F0", linewidth=0.8)
    ax.set_axisbelow(True)


def plot_category_model_comparison(
    comparison: pd.DataFrame,
    output_path: Path,
) -> None:
    ordered = comparison.sort_values("top_3_hit_rate")
    colors = [
        COLORS["baseline"] if "baseline" in name else COLORS["model"]
        for name in ordered["candidate"]
    ]
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    bars = ax.barh(ordered["candidate"], ordered["top_3_hit_rate"], color=colors)
    ax.bar_label(bars, fmt="%.1f%%", padding=5)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Validation Top-3 hit rate")
    ax.set_title("Next-category model comparison")
    _format_axes(ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def plot_top_k_performance(
    metrics_by_candidate: dict[str, dict[str, float]],
    output_path: Path,
) -> None:
    names = list(metrics_by_candidate)
    k_labels = ["Top-1", "Top-2", "Top-3"]
    keys = ["top_1_hit_rate", "top_2_hit_rate", "top_3_hit_rate"]
    positions = np.arange(len(k_labels))
    width = 0.8 / len(names)

    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    palette = ["#CBD5E1", "#94A3B8", COLORS["model"]]
    for index, name in enumerate(names):
        values = [metrics_by_candidate[name][key] * 100 for key in keys]
        offset = (index - (len(names) - 1) / 2) * width
        ax.bar(
            positions + offset,
            values,
            width=width,
            label=name,
            color=palette[index] if index < len(palette) else COLORS["accent"],
        )
    ax.set_xticks(positions, k_labels)
    ax.set_ylabel("Test hit rate")
    ax.set_title("Ranked recommendation quality")
    ax.legend(frameon=False)
    _format_axes(ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def plot_readiness_comparison(
    comparison: pd.DataFrame,
    output_path: Path,
) -> None:
    ordered = comparison.sort_values("average_precision")
    values = ordered["average_precision"] * 100
    colors = [
        COLORS["baseline"] if "baseline" in name else COLORS["accent"]
        for name in ordered["candidate"]
    ]
    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    bars = ax.barh(ordered["candidate"], values, color=colors)
    ax.bar_label(bars, fmt="%.1f%%", padding=5)
    ax.set_xlabel("Validation average precision")
    ax.set_title("30-day purchase-readiness model comparison")
    _format_axes(ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def plot_category_recall(recall: pd.DataFrame, output_path: Path) -> None:
    ordered = recall.sort_values("top_3_recall")
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    ax.barh(
        ordered["category"],
        ordered["top_3_recall"] * 100,
        color=COLORS["model"],
    )
    ax.set_xlabel("Test Top-3 recall")
    ax.set_title("Recommendation coverage by category")
    _format_axes(ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def plot_activation_mix(summary: pd.DataFrame, output_path: Path) -> None:
    pivot = summary.pivot(
        index="recommended_category",
        columns="capacity_tier",
        values="customers",
    ).fillna(0)
    column_order = [name for name in ["Priority", "Review", "Monitor"] if name in pivot]
    pivot = pivot[column_order]
    colors = [COLORS["accent"], COLORS["model"], COLORS["baseline"]][: len(column_order)]

    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    pivot.plot(kind="bar", stacked=True, ax=ax, color=colors)
    ax.set_xlabel("")
    ax.set_ylabel("Eligible customers")
    ax.set_title("Activation queue by recommended category")
    ax.legend(title="Capacity tier", frameon=False)
    ax.tick_params(axis="x", rotation=25)
    _format_axes(ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def write_json(data: dict[str, object], output_path: Path) -> None:
    output_path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def write_run_summary(
    output_path: Path,
    metrics: dict[str, object],
    dataset_summary: dict[str, int | float | str],
) -> None:
    category = metrics["category_test"]
    readiness = metrics["readiness_test"]
    text = f"""# Reproducible Run Summary

This report was generated from the committed pipeline using seed
`{dataset_summary["seed"]}`.

## Data and evaluation

- Synthetic customers: {dataset_summary["customers"]:,}
- Synthetic transactions: {dataset_summary["transactions"]:,}
- Historical score-date snapshots: {dataset_summary["snapshots"]:,}
- Category-labeled test snapshots: {dataset_summary["category_test_rows"]:,}
- Untouched test period starts: {dataset_summary["test_start"]}

## Selected models

- Next-category model: `{metrics["selected_category_model"]}`
- 30-day readiness model: `{metrics["selected_readiness_model"]}`

## Test performance

| Measure | Result |
|---|---:|
| Category Top-1 hit rate | {category["top_1_hit_rate"]:.1%} |
| Category Top-3 hit rate | {category["top_3_hit_rate"]:.1%} |
| Mean reciprocal rank | {category["mean_reciprocal_rank"]:.3f} |
| Category log loss | {category["log_loss"]:.3f} |
| Readiness average precision | {readiness["average_precision"]:.3f} |
| Readiness ROC-AUC | {readiness["roc_auc"]:.3f} |
| Readiness Brier score | {readiness["brier_score"]:.3f} |
| Readiness recall in top 20% | {readiness["recall_at_top_20pct"]:.1%} |
| Readiness lift in top 20% | {readiness["lift_at_top_20pct"]:.2f}x |

These figures validate the synthetic pipeline. They are not estimates of
production performance or incremental campaign impact.
"""
    output_path.write_text(text, encoding="utf-8")


def write_decision_note(
    output_path: Path,
    metrics: dict[str, object],
    queue: pd.DataFrame,
) -> None:
    category = metrics["category_test"]
    readiness = metrics["readiness_test"]
    priority = queue.loc[queue["capacity_tier"] == "Priority"]
    category_mix = (
        priority["recommended_category"]
        .value_counts(normalize=True)
        .mul(100)
        .round(1)
        .to_dict()
    )
    mix_text = ", ".join(
        f"{category_name} {share:.1f}%"
        for category_name, share in category_mix.items()
    )
    text = f"""# Decision Note

## Recommended use

Use the 30-day readiness score to control contact capacity, then use the
ranked category probabilities to choose the most relevant category or landing
page. The current demonstration assigns the top 15% of eligible customers to a
`Priority` queue and the next 25% to `Review`.

The selected category model places the observed next category in its first
three suggestions for {category["top_3_hit_rate"]:.1%} of test snapshots. The
readiness model captures {readiness["recall_at_top_20pct"]:.1%} of 30-day
purchasers in the top 20% of its test ranking.

At the latest synthetic scoring date, the Priority queue contains
{len(priority):,} customers. Its recommended-category mix is: {mix_text}.

## Guardrails

- Treat the action score as a capacity-ranking heuristic.
- Do not interpret recommendation probability as incremental response.
- Test channel, offer, and creative choices through a randomized holdout.
- Monitor Top-K quality, readiness calibration, category coverage, and score
  drift by scoring month.
- Apply contact-frequency, eligibility, inventory, and consent rules after
  model scoring.
"""
    output_path.write_text(text, encoding="utf-8")
