"""Versioned supplied-input scoring without synthetic-truth evaluation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from next_purchase.config import ProjectConfig
from next_purchase.decision import activation_summary, build_activation_queue
from next_purchase.features import build_scoring_snapshot
from next_purchase.modeling import cadence_rule_probabilities, last_category_probabilities
from next_purchase.reporting import plot_activation_mix, write_json

SCHEMA_VERSION = "1.0"
TRANSACTION_COLUMNS = (
    "order_id",
    "customer_id",
    "order_date",
    "category",
    "order_value",
    "contribution_margin",
    "used_discount",
    "acquisition_channel",
    "region",
)
PROVENANCE_FIELDS = (
    "schema_version",
    "source_name",
    "source_type",
    "license_or_authorization",
    "identifiers_pseudonymized",
    "contains_direct_identifiers",
)


class SuppliedInputError(ValueError):
    """Raised when supplied transactions or provenance violate the input contract."""


@dataclass(frozen=True)
class SuppliedInputSummary:
    input_rows: int
    historical_rows: int
    ignored_on_or_after_score_date: int
    customers: int
    eligible_customers: int
    categories: int
    checks_passed: int


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_provenance(path: Path) -> dict[str, Any]:
    """Load the required provenance manifest and enforce its safety assertions."""
    if not path.is_file():
        raise SuppliedInputError(f"Provenance file does not exist: {path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise SuppliedInputError("Provenance file must be valid UTF-8 JSON") from error
    missing = set(PROVENANCE_FIELDS).difference(raw)
    if missing:
        raise SuppliedInputError(f"Missing provenance fields: {sorted(missing)}")
    if raw["schema_version"] != SCHEMA_VERSION:
        raise SuppliedInputError(
            f"Unsupported schema_version: {raw['schema_version']!r}; expected {SCHEMA_VERSION!r}"
        )
    for field in ("source_name", "source_type", "license_or_authorization"):
        if not isinstance(raw[field], str) or not raw[field].strip():
            raise SuppliedInputError(f"Provenance field {field!r} must be a non-blank string")
    if raw["identifiers_pseudonymized"] is not True:
        raise SuppliedInputError("Supplied customer identifiers must be pseudonymized")
    if raw["contains_direct_identifiers"] is not False:
        raise SuppliedInputError("Supplied input must not contain direct identifiers")
    return {field: raw[field] for field in PROVENANCE_FIELDS}


def _clean_identifier(series: pd.Series, field: str) -> pd.Series:
    clean = series.astype("string").str.strip()
    if clean.isna().any() or clean.eq("").any():
        raise SuppliedInputError(f"{field} must not contain null or blank values")
    if clean.str.startswith(("=", "+", "-", "@")).any():
        raise SuppliedInputError(f"{field} contains a spreadsheet-formula prefix")
    return clean


def load_transactions(path: Path) -> pd.DataFrame:
    """Load and validate version 1.0 transaction rows."""
    if not path.is_file():
        raise SuppliedInputError(f"Transaction file does not exist: {path}")
    try:
        frame = pd.read_csv(path)
    except (pd.errors.EmptyDataError, pd.errors.ParserError) as error:
        raise SuppliedInputError(
            "Transaction input must be a readable CSV with a header"
        ) from error
    missing = set(TRANSACTION_COLUMNS).difference(frame.columns)
    if missing:
        raise SuppliedInputError(f"Missing transaction columns: {sorted(missing)}")
    if frame.empty:
        raise SuppliedInputError("Transaction input must not be empty")

    clean = frame[list(TRANSACTION_COLUMNS)].copy()
    if clean.isna().any().any():
        raise SuppliedInputError("Required transaction fields must not contain nulls")
    for field in ("order_id", "customer_id", "category", "acquisition_channel", "region"):
        clean[field] = _clean_identifier(clean[field], field)
    if clean["order_id"].duplicated().any():
        raise SuppliedInputError("order_id must be unique")
    if clean["category"].str.casefold().nunique() != clean["category"].nunique():
        raise SuppliedInputError("Category names must be unique after case folding")

    parsed_dates = pd.to_datetime(clean["order_date"], errors="coerce", utc=True)
    if parsed_dates.isna().any():
        raise SuppliedInputError("order_date must contain valid ISO-8601 dates")
    clean["order_date"] = parsed_dates.dt.tz_convert(None).dt.normalize()
    for field in ("order_value", "contribution_margin", "used_discount"):
        clean[field] = pd.to_numeric(clean[field], errors="coerce")
        if clean[field].isna().any() or not np.isfinite(clean[field].to_numpy(dtype=float)).all():
            raise SuppliedInputError(f"{field} must contain finite numeric values")
    if (clean["order_value"] < 0).any():
        raise SuppliedInputError("order_value must be non-negative")
    if not clean["used_discount"].isin([0, 1]).all():
        raise SuppliedInputError("used_discount must contain only 0 or 1")
    clean["used_discount"] = clean["used_discount"].astype(int)
    return clean.sort_values(["customer_id", "order_date", "order_id"], ignore_index=True)


def score_supplied_transactions(
    transactions_path: Path,
    provenance_path: Path,
    score_date: str | pd.Timestamp,
    output_root: Path,
) -> dict[str, Any]:
    """Create a transparent queue from supplied history without fitted-model evaluation."""
    provenance = load_provenance(provenance_path)
    transactions = load_transactions(transactions_path)
    score_timestamp = pd.Timestamp(score_date).normalize()
    history = transactions.loc[transactions["order_date"] < score_timestamp].copy()
    if history.empty:
        raise SuppliedInputError("No transactions occur before score_date")
    categories = tuple(sorted(history["category"].unique()))
    if len(categories) < 3:
        raise SuppliedInputError(
            "At least three historical categories are required for Top-3 output"
        )

    config = replace(
        ProjectConfig(),
        categories=categories,
        scoring_date=score_timestamp.strftime("%Y-%m-%d"),
    )
    scoring_frame = build_scoring_snapshot(history, score_timestamp, config)
    category_probabilities = last_category_probabilities(
        scoring_frame["last_category"],
        history["category"],
        categories,
    )
    readiness_probability = cadence_rule_probabilities(scoring_frame)
    expected_margin = history.groupby("category")["contribution_margin"].median().to_dict()
    queue = build_activation_queue(
        scoring_frame,
        category_probabilities,
        readiness_probability,
        categories,
        expected_margin_by_category={
            category: float(expected_margin[category]) for category in categories
        },
    )
    summary = activation_summary(queue)

    reports_dir = output_root / "reports"
    figures_dir = reports_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    queue.to_csv(reports_dir / "recommendations.csv", index=False)
    summary.to_csv(reports_dir / "activation_summary.csv", index=False)
    plot_activation_mix(summary, figures_dir / "activation_queue.png")

    input_summary = SuppliedInputSummary(
        input_rows=len(transactions),
        historical_rows=len(history),
        ignored_on_or_after_score_date=len(transactions) - len(history),
        customers=int(transactions["customer_id"].nunique()),
        eligible_customers=len(scoring_frame),
        categories=len(categories),
        checks_passed=12,
    )
    metadata: dict[str, Any] = {
        "mode": "supplied_input_scoring",
        "schema_version": SCHEMA_VERSION,
        "score_date": score_timestamp.strftime("%Y-%m-%d"),
        "input_sha256": _sha256(transactions_path),
        "provenance": provenance,
        "input_summary": asdict(input_summary),
        "policy": {
            "category_relevance": "smoothed_last_category_baseline",
            "purchase_readiness": "cadence_rule_baseline",
            "priority_share": 0.15,
            "review_share": 0.25,
        },
        "evaluation_boundary": (
            "No fitted model, synthetic truth, future label, uplift, or performance metric is "
            "used or reported in supplied-input mode."
        ),
    }
    write_json(metadata, reports_dir / "run_metadata.json")
    _write_supplied_summary(metadata, reports_dir / "run_summary.md")
    return metadata


def _write_supplied_summary(metadata: dict[str, Any], output_path: Path) -> None:
    summary = metadata["input_summary"]
    provenance = metadata["provenance"]
    text = f"""# Supplied-input scoring summary

- Schema version: `{metadata["schema_version"]}`
- Source: {provenance["source_name"]} ({provenance["source_type"]})
- Score date: {metadata["score_date"]}
- Historical transactions used: {summary["historical_rows"]:,}
- Rows on or after the score date ignored: {summary["ignored_on_or_after_score_date"]:,}
- Eligible customers: {summary["eligible_customers"]:,}
- Historical categories: {summary["categories"]:,}
- Input SHA-256: `{metadata["input_sha256"]}`

This run uses transparent cadence and smoothed last-category baselines. It does not fit or
evaluate a model, access synthetic truth, estimate uplift, or claim production performance.
"""
    output_path.write_text(text, encoding="utf-8")
