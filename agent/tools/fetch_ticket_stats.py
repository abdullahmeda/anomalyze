"""Tool to fetch ticket statistics for a given date with baseline comparison."""

import json
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

DATASET_PATH = Path(__file__).parent.parent.parent / "dataset" / "data" / "full_dataset.csv"
BASELINE_DAYS = 7


def _load_dataset() -> pd.DataFrame:
    """Load the full dataset with parsed timestamps."""
    return pd.read_csv(DATASET_PATH, parse_dates=["timestamp"])


def _get_distribution(df: pd.DataFrame, column: str, top_n: int = 10) -> dict:
    """Get percentage distribution for a column."""
    counts = df[column].value_counts(normalize=True) * 100
    return counts.head(top_n).round(1).to_dict()


def _get_tag_distribution(df: pd.DataFrame, top_n: int = 10) -> dict:
    """Get combined tag distribution across all tag columns."""
    tag_cols = [c for c in df.columns if c.startswith("tag_")]
    all_tags = df[tag_cols].values.flatten()
    all_tags = pd.Series([t for t in all_tags if pd.notna(t)])
    counts = all_tags.value_counts(normalize=True) * 100
    return counts.head(top_n).round(1).to_dict()


def fetch_ticket_stats(date: str) -> str:
    """
    Retrieve ticket statistics for a specific date with baseline comparison.

    Args:
        date: The date to analyze in YYYY-MM-DD format (e.g., "2023-10-05")

    Returns:
        JSON string containing:
        - ticket_count: Number of tickets on this date
        - baseline_avg: Average daily tickets over previous 7 days
        - volume_change_pct: Percentage change vs baseline
        - priority_distribution: Current day priority breakdown with baseline comparison
        - type_distribution: Current day type breakdown with baseline comparison
        - tag_distribution: Top tags with baseline comparison
        - queue_distribution: Top queues with baseline comparison
    """
    df = _load_dataset()

    target_date = datetime.strptime(date, "%Y-%m-%d").date()
    df["date"] = df["timestamp"].dt.date

    # Current day data
    current = df[df["date"] == target_date]

    # Baseline: previous 7 days
    baseline_start = target_date - timedelta(days=BASELINE_DAYS)
    baseline_end = target_date - timedelta(days=1)
    baseline = df[(df["date"] >= baseline_start) & (df["date"] <= baseline_end)]

    # Volume stats
    current_count = len(current)
    baseline_avg = len(baseline) / BASELINE_DAYS if BASELINE_DAYS > 0 else 0
    volume_change = ((current_count - baseline_avg) / baseline_avg * 100) if baseline_avg > 0 else 0

    # Build comparison stats
    def compare(column: str, top_n: int = 5) -> dict:
        current_dist = _get_distribution(current, column, top_n)
        baseline_dist = _get_distribution(baseline, column, top_n)
        return {
            "current": current_dist,
            "baseline": baseline_dist,
        }

    # Tag comparison
    current_tags = _get_tag_distribution(current, top_n=10)
    baseline_tags = _get_tag_distribution(baseline, top_n=10)

    stats = {
        "date": date,
        "ticket_count": current_count,
        "baseline_avg": round(baseline_avg, 1),
        "volume_change_pct": round(volume_change, 1),
        "priority_distribution": compare("priority"),
        "type_distribution": compare("type"),
        "queue_distribution": compare("queue", top_n=5),
        "tag_distribution": {
            "current": current_tags,
            "baseline": baseline_tags,
        },
    }

    return json.dumps(stats, indent=2)

