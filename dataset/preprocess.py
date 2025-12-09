"""
Prepare Customer Support Tickets Dataset for Anomaly Detection

Transforms the raw HuggingFace dataset into a timestamped, anomaly-labeled dataset.

Outputs:
- dataset/data/full_dataset.csv: Raw tickets with timestamps and anomaly labels
- dataset/data/anomaly_metadata.json: Ground truth summary
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_dataset

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

RANDOM_SEED = 42
SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPT_DIR / "data"

START_DATE = datetime(2023, 1, 1)
END_DATE = datetime(2023, 12, 31, 23, 59, 59)
TRAIN_END = datetime(2023, 9, 30, 23, 59, 59)

WEEKLY_PATTERN = {0: 1.15, 1: 1.05, 2: 1.00, 3: 1.00, 4: 0.95, 5: 0.45, 6: 0.40}

HOURLY_WEIGHTS = np.array([
    0.02, 0.01, 0.01, 0.01, 0.02, 0.03, 0.04, 0.05, 0.07,
    0.09, 0.10, 0.09, 0.06, 0.06, 0.08, 0.09, 0.08, 0.06,
    0.04, 0.03, 0.02, 0.02, 0.01, 0.01,
])
HOURLY_WEIGHTS = HOURLY_WEIGHTS / HOURLY_WEIGHTS.sum()

ANOMALY = {
    "label": "spike_volume",
    "type": "spike",
    "date": datetime(2023, 10, 5),
    "volume_multiplier": 1.5,
    "description": "1.5x volume spike with characteristic skew (simulates major outage/bug)",
}

ANOMALY_CHARACTERISTICS = {
    "priority": {"critical": 5.0, "high": 3.0, "medium": 1.0, "low": 0.3, "very_low": 0.2},
    "type": {"Incident": 4.0, "Problem": 2.0, "Request": 0.5, "Change": 0.3},
    "tags": {"Bug": 4.0, "Outage": 5.0, "Disruption": 4.0, "Technical": 3.0, "Security": 3.0, "Network": 2.5, "Performance": 2.0, "Crash": 4.0},
    "queue": {"Technical Support": 3.0, "IT Support": 2.5, "Product Support": 1.5},
}


# ═══════════════════════════════════════════════════════════════════════════════
# PIPELINE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════


def load_and_clean() -> pd.DataFrame:
    """Load dataset from HuggingFace and clean it."""
    dataset = load_dataset("Tobi-Bueck/customer-support-tickets")["train"]
    df = dataset.to_pandas()
    df = df.dropna(subset=["body"])
    df = df.drop(columns=["answer", "version", "tag_5", "tag_6", "tag_7", "tag_8"], errors="ignore")
    df = df.reset_index(drop=True)
    df.insert(0, "ticket_id", range(len(df)))
    return df


def compute_anomaly_scores(df: pd.DataFrame) -> np.ndarray:
    """Compute anomaly likelihood score for each ticket based on characteristics."""
    scores = np.ones(len(df))

    for priority, weight in ANOMALY_CHARACTERISTICS["priority"].items():
        scores[df["priority"] == priority] *= weight

    for ticket_type, weight in ANOMALY_CHARACTERISTICS["type"].items():
        scores[df["type"] == ticket_type] *= weight

    for tag, weight in ANOMALY_CHARACTERISTICS["tags"].items():
        for col in ["tag_1", "tag_2", "tag_3", "tag_4"]:
            scores[df[col] == tag] *= weight

    for queue, weight in ANOMALY_CHARACTERISTICS["queue"].items():
        scores[df["queue"] == queue] *= weight

    return scores


def generate_daily_volumes(total_tickets: int, rng: np.random.Generator) -> dict:
    """Generate daily ticket volumes with weekly seasonality and spike anomaly."""
    anomaly_date = ANOMALY["date"].date()
    raw_volumes = {}
    current = START_DATE

    while current <= END_DATE:
        volume = WEEKLY_PATTERN[current.weekday()] * rng.uniform(0.85, 1.15)
        if current.date() == anomaly_date:
            volume *= ANOMALY["volume_multiplier"]
        raw_volumes[current.date()] = volume
        current += timedelta(days=1)

    scale = total_tickets / sum(raw_volumes.values())
    daily_volumes = {d: max(1, int(v * scale)) for d, v in raw_volumes.items()}

    # Adjust to exactly match total
    diff = total_tickets - sum(daily_volumes.values())
    dates = [d for d in daily_volumes if d != anomaly_date]
    for _ in range(abs(diff)):
        date = rng.choice(dates)
        daily_volumes[date] += 1 if diff > 0 else (-1 if daily_volumes[date] > 1 else 0)

    return daily_volumes


def assign_timestamps(df: pd.DataFrame, daily_volumes: dict, anomaly_scores: np.ndarray, rng: np.random.Generator) -> pd.DataFrame:
    """Assign timestamps with characteristic-based selection for anomaly day."""
    anomaly_date = ANOMALY["date"].date()
    anomaly_volume = daily_volumes[anomaly_date]

    # Select anomaly day tickets using weighted sampling
    probabilities = anomaly_scores / anomaly_scores.sum()
    anomaly_indices = set(rng.choice(len(df), size=anomaly_volume, replace=False, p=probabilities))
    normal_indices = [i for i in range(len(df)) if i not in anomaly_indices]
    rng.shuffle(normal_indices)

    def make_timestamps(date, count):
        timestamps = []
        for hour, n in enumerate(rng.multinomial(count, HOURLY_WEIGHTS)):
            for _ in range(n):
                timestamps.append(datetime(date.year, date.month, date.day, hour,
                                           int(rng.integers(0, 60)), int(rng.integers(0, 60)), int(rng.integers(0, 1000000))))
        return sorted(timestamps)

    # Generate all timestamps
    anomaly_timestamps = make_timestamps(anomaly_date, anomaly_volume)
    normal_timestamps = []
    for date, volume in sorted((d, v) for d, v in daily_volumes.items() if d != anomaly_date):
        normal_timestamps.extend(make_timestamps(date, volume))

    # Map indices to timestamps
    timestamp_map = {idx: ts for idx, ts in zip(anomaly_indices, anomaly_timestamps)}
    timestamp_map.update({idx: ts for idx, ts in zip(normal_indices, normal_timestamps)})

    df = df.copy()
    df["timestamp"] = df["ticket_id"].map(timestamp_map)
    return df.sort_values("timestamp").reset_index(drop=True)


def label_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """Add anomaly labels to tickets on the spike day."""
    df = df.copy()
    is_anomaly = df["timestamp"].dt.date == ANOMALY["date"].date()
    df["anomaly_label"] = None
    df.loc[is_anomaly, "anomaly_label"] = ANOMALY["label"]
    df["is_anomaly"] = is_anomaly
    return df


def add_split_marker(df: pd.DataFrame) -> pd.DataFrame:
    """Add train/test split marker based on timestamp."""
    df = df.copy()
    df["split"] = df["timestamp"].apply(lambda x: "train" if x <= TRAIN_END else "test")
    return df


def generate_metadata(df: pd.DataFrame) -> dict:
    """Generate metadata JSON."""
    train_df, test_df = df[df["split"] == "train"], df[df["split"] == "test"]
    anomaly_df, normal_df = df[df["is_anomaly"]], df[~df["is_anomaly"]]

    def dist(subset, col):
        return (subset[col].value_counts(normalize=True) * 100).round(1).to_dict()

    return {
        "dataset_info": {
            "source": "Tobi-Bueck/customer-support-tickets",
            "total_tickets": len(df), "train_tickets": len(train_df), "test_tickets": len(test_df),
            "date_range": [START_DATE.strftime("%Y-%m-%d"), END_DATE.strftime("%Y-%m-%d")],
            "train_range": [START_DATE.strftime("%Y-%m-%d"), TRAIN_END.strftime("%Y-%m-%d")],
            "test_range": [(TRAIN_END + timedelta(days=1)).strftime("%Y-%m-%d"), END_DATE.strftime("%Y-%m-%d")],
        },
        "anomaly": {
            "label": ANOMALY["label"], "type": ANOMALY["type"],
            "date": ANOMALY["date"].strftime("%Y-%m-%d"), "volume_multiplier": ANOMALY["volume_multiplier"],
            "affected_tickets": len(anomaly_df), "description": ANOMALY["description"],
            "characteristics": {"priority_distribution": dist(anomaly_df, "priority"), "type_distribution": dist(anomaly_df, "type"), "top_tags": dist(anomaly_df, "tag_1")},
        },
        "baseline": {"priority_distribution": dist(normal_df, "priority"), "type_distribution": dist(normal_df, "type"), "top_tags": dist(normal_df, "tag_1")},
        "statistics": {"normal_tickets": len(normal_df), "anomaly_tickets": len(anomaly_df), "anomaly_percentage": round(len(anomaly_df) / len(df) * 100, 2)},
        "columns": list(df.columns), "random_seed": RANDOM_SEED,
    }


def export_datasets(df: pd.DataFrame, metadata: dict):
    """Export full dataset and metadata JSON."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_DIR / "full_dataset.csv", index=False)
    with open(OUTPUT_DIR / "anomaly_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    rng = np.random.default_rng(RANDOM_SEED)

    # Pipeline
    df = load_and_clean()
    anomaly_scores = compute_anomaly_scores(df)
    daily_volumes = generate_daily_volumes(len(df), rng)
    df = assign_timestamps(df, daily_volumes, anomaly_scores, rng)
    df = label_anomalies(df)
    df = add_split_marker(df)
    metadata = generate_metadata(df)
    export_datasets(df, metadata)

    # Report
    anomaly_df = df[df["is_anomaly"]]
    normal_df = df[~df["is_anomaly"]]
    anomaly_high = (anomaly_df["priority"].isin(["high", "critical"])).mean() * 100
    normal_high = (normal_df["priority"].isin(["high", "critical"])).mean() * 100

    logger.info(f"\nDataset Preparation Complete")
    logger.info(f"  Total: {len(df):,} tickets | Train: {len(df[df['split']=='train']):,} | Test: {len(df[df['split']=='test']):,}")
    logger.info(f"  Anomaly: {ANOMALY['date'].strftime('%Y-%m-%d')} ({len(anomaly_df)} tickets, {ANOMALY['volume_multiplier']}× volume)")
    logger.info(f"  High/Critical: {anomaly_high:.0f}% anomaly vs {normal_high:.0f}% baseline")
    logger.info(f"  Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
