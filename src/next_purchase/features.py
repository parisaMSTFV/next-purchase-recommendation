"""Create point-in-time customer snapshots without future information."""

from __future__ import annotations

import numpy as np
import pandas as pd

from next_purchase.config import ProjectConfig

IDENTIFIER_COLUMNS = ["customer_id", "score_date"]
LABEL_COLUMNS = [
    "purchase_within_30d",
    "next_category",
    "days_to_next_purchase",
]
CATEGORICAL_FEATURES = [
    "acquisition_channel",
    "region",
    "lifecycle_segment",
    "last_category",
    "favorite_category",
]


def _safe_mean_gap(order_dates: pd.Series) -> tuple[float, float]:
    gaps = order_dates.sort_values().diff().dt.days.dropna()
    if gaps.empty:
        return 90.0, 0.0
    recent_gaps = gaps.tail(8)
    return float(recent_gaps.median()), float(recent_gaps.std(ddof=0))


def _lifecycle_segment(
    tenure_days: int,
    total_orders: int,
    orders_90d: int,
    recency_days: int,
    average_gap: float,
) -> str:
    if tenure_days <= 150 and total_orders <= 4:
        return "New"
    if recency_days > max(60.0, 1.5 * average_gap):
        return "At Risk"
    if orders_90d >= 3:
        return "Loyal"
    return "Growing"


def _feature_row(
    customer_id: str,
    history: pd.DataFrame,
    score_date: pd.Timestamp,
    categories: tuple[str, ...],
) -> dict[str, object]:
    history = history.sort_values("order_date")
    last_order = history.iloc[-1]
    window_start = score_date - pd.Timedelta(days=365)
    history_365 = history.loc[history["order_date"] >= window_start]
    history_90 = history.loc[history["order_date"] >= score_date - pd.Timedelta(days=90)]
    history_30 = history.loc[history["order_date"] >= score_date - pd.Timedelta(days=30)]

    average_gap, gap_std = _safe_mean_gap(history["order_date"])
    recency_days = int((score_date - last_order["order_date"]).days)
    tenure_days = int((score_date - history.iloc[0]["order_date"]).days)
    category_counts = history_365["category"].value_counts()
    favorite_category = str(category_counts.index[0])
    favorite_share = float(category_counts.iloc[0] / len(history_365))

    row: dict[str, object] = {
        "customer_id": customer_id,
        "score_date": score_date,
        "acquisition_channel": str(last_order["acquisition_channel"]),
        "region": str(last_order["region"]),
        "lifecycle_segment": _lifecycle_segment(
            tenure_days=tenure_days,
            total_orders=len(history),
            orders_90d=len(history_90),
            recency_days=recency_days,
            average_gap=average_gap,
        ),
        "last_category": str(last_order["category"]),
        "favorite_category": favorite_category,
        "customer_tenure_days": tenure_days,
        "recency_days": recency_days,
        "orders_lifetime": len(history),
        "orders_365d": len(history_365),
        "orders_90d": len(history_90),
        "orders_30d": len(history_30),
        "order_momentum": float(len(history_90) / max(len(history_365) / 4.0, 0.25)),
        "average_order_value_365d": float(history_365["order_value"].mean()),
        "total_margin_365d": float(history_365["contribution_margin"].sum()),
        "discount_share_365d": float(history_365["used_discount"].mean()),
        "weekend_share_365d": float(
            (history_365["order_date"].dt.dayofweek >= 5).mean()
        ),
        "category_diversity_365d": int(history_365["category"].nunique()),
        "favorite_category_share_365d": favorite_share,
        "average_gap_days": average_gap,
        "gap_variability_days": gap_std,
        "cadence_progress": float(recency_days / max(average_gap, 7.0)),
        "last_order_value": float(last_order["order_value"]),
        "last_order_margin": float(last_order["contribution_margin"]),
        "score_month": int(score_date.month),
        "score_quarter": int(score_date.quarter),
    }

    for category in categories:
        category_history = history_365.loc[history_365["category"] == category]
        feature_key = category.lower()
        count = len(category_history)
        row[f"{feature_key}_orders_365d"] = count
        row[f"{feature_key}_share_365d"] = float(count / len(history_365))
        if count:
            category_recency = int(
                (score_date - category_history.iloc[-1]["order_date"]).days
            )
        else:
            category_recency = 400
        row[f"{feature_key}_recency_days"] = category_recency

    return row


def _validate_transactions(transactions: pd.DataFrame) -> pd.DataFrame:
    required = {
        "customer_id",
        "order_date",
        "category",
        "order_value",
        "contribution_margin",
        "used_discount",
        "acquisition_channel",
        "region",
    }
    missing = required.difference(transactions.columns)
    if missing:
        raise ValueError(f"Missing transaction columns: {sorted(missing)}")

    clean = transactions.copy()
    clean["order_date"] = pd.to_datetime(clean["order_date"]).dt.normalize()
    return clean.sort_values(["customer_id", "order_date"]).reset_index(drop=True)


def build_labeled_snapshots(
    transactions: pd.DataFrame,
    config: ProjectConfig,
) -> pd.DataFrame:
    """Build historical score-date rows and labels observable after each score date."""

    clean = _validate_transactions(transactions)
    data_end = clean["order_date"].max()
    score_dates = pd.date_range(
        config.snapshot_start,
        config.snapshot_end,
        freq=f"{config.snapshot_frequency_days}D",
    )
    score_dates = [
        date
        for date in score_dates
        if date + pd.Timedelta(days=config.target_horizon_days) <= data_end
    ]

    rows: list[dict[str, object]] = []
    for customer_id, customer_orders in clean.groupby("customer_id", sort=False):
        customer_orders = customer_orders.sort_values("order_date").reset_index(drop=True)
        dates = customer_orders["order_date"].to_numpy(dtype="datetime64[ns]")

        for score_date in score_dates:
            history_end = int(np.searchsorted(dates, np.datetime64(score_date), side="left"))
            if history_end < config.min_history_orders:
                continue

            history = customer_orders.iloc[:history_end]
            recency_days = int((score_date - history.iloc[-1]["order_date"]).days)
            if recency_days > 365:
                continue

            row = _feature_row(
                customer_id=customer_id,
                history=history,
                score_date=score_date,
                categories=config.categories,
            )

            if history_end < len(customer_orders):
                next_order = customer_orders.iloc[history_end]
                days_to_next = int((next_order["order_date"] - score_date).days)
            else:
                next_order = None
                days_to_next = config.target_horizon_days + 1

            row["purchase_within_30d"] = int(
                0 <= days_to_next <= config.readiness_horizon_days
            )
            row["next_category"] = (
                str(next_order["category"])
                if next_order is not None
                and 0 <= days_to_next <= config.target_horizon_days
                else None
            )
            row["days_to_next_purchase"] = (
                days_to_next if 0 <= days_to_next <= config.target_horizon_days else np.nan
            )
            rows.append(row)

    snapshots = pd.DataFrame.from_records(rows)
    if snapshots.empty:
        raise RuntimeError("No eligible historical snapshots were created.")
    snapshots["score_date"] = pd.to_datetime(snapshots["score_date"])
    return snapshots.sort_values(["score_date", "customer_id"]).reset_index(drop=True)


def build_scoring_snapshot(
    transactions: pd.DataFrame,
    score_date: str | pd.Timestamp,
    config: ProjectConfig,
) -> pd.DataFrame:
    """Create one unlabeled row per eligible customer at a production-like score date."""

    clean = _validate_transactions(transactions)
    score_timestamp = pd.Timestamp(score_date).normalize()
    rows: list[dict[str, object]] = []

    for customer_id, customer_orders in clean.groupby("customer_id", sort=False):
        customer_orders = customer_orders.loc[
            customer_orders["order_date"] < score_timestamp
        ].sort_values("order_date")
        if len(customer_orders) < config.min_history_orders:
            continue
        if (score_timestamp - customer_orders.iloc[-1]["order_date"]).days > 365:
            continue
        rows.append(
            _feature_row(
                customer_id=customer_id,
                history=customer_orders,
                score_date=score_timestamp,
                categories=config.categories,
            )
        )

    scoring = pd.DataFrame.from_records(rows)
    if scoring.empty:
        raise RuntimeError("No eligible customers were available for scoring.")
    return scoring.sort_values("customer_id").reset_index(drop=True)


def feature_columns(frame: pd.DataFrame) -> list[str]:
    """Return model feature names while excluding identifiers and outcomes."""

    excluded = set(IDENTIFIER_COLUMNS + LABEL_COLUMNS)
    return [column for column in frame.columns if column not in excluded]


def split_by_time(
    snapshots: pd.DataFrame,
    config: ProjectConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create chronological train, validation, and untouched test partitions."""

    validation_start = pd.Timestamp(config.validation_start)
    test_start = pd.Timestamp(config.test_start)
    train = snapshots.loc[snapshots["score_date"] < validation_start].copy()
    validation = snapshots.loc[
        (snapshots["score_date"] >= validation_start)
        & (snapshots["score_date"] < test_start)
    ].copy()
    test = snapshots.loc[snapshots["score_date"] >= test_start].copy()

    if min(len(train), len(validation), len(test)) == 0:
        raise RuntimeError("At least one chronological split is empty.")
    return train, validation, test
