"""
Prepare Customer Support Tickets Dataset for Anomaly Detection

This script transforms the raw HuggingFace dataset into a timestamped,
anomaly-labeled dataset ready for hybrid anomaly detection experiments.

Outputs:
- dataset/data/full_dataset.csv: Raw tickets with timestamps and anomaly labels
- dataset/data/anomaly_metadata.json: Ground truth summary

Note: Train/test aggregation is handled by ml/features.py
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from datasets import load_dataset
import json
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

RANDOM_SEED = 42
# Use path relative to this script's location
SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPT_DIR / "data"

# Time range
START_DATE = datetime(2023, 1, 1)
END_DATE = datetime(2023, 12, 31, 23, 59, 59)
TRAIN_END = datetime(2023, 9, 30, 23, 59, 59)  # 9 months train, 3 months test

# Weekly pattern (day-of-week multipliers, 0=Monday)
WEEKLY_PATTERN = {
    0: 1.15,  # Monday (weekend backlog)
    1: 1.05,  # Tuesday
    2: 1.00,  # Wednesday
    3: 1.00,  # Thursday
    4: 0.95,  # Friday
    5: 0.45,  # Saturday
    6: 0.40,  # Sunday
}

# Hourly distribution weights
HOURLY_WEIGHTS = np.array([
    0.02, 0.01, 0.01, 0.01, 0.02, 0.03,  # 00:00-05:59 (night)
    0.04, 0.05, 0.07,                     # 06:00-08:59 (early morning)
    0.09, 0.10, 0.09,                     # 09:00-11:59 (peak morning)
    0.06, 0.06,                           # 12:00-13:59 (lunch)
    0.08, 0.09, 0.08, 0.06,               # 14:00-17:59 (afternoon)
    0.04, 0.03, 0.02, 0.02, 0.01, 0.01,   # 18:00-23:59 (evening)
])
HOURLY_WEIGHTS = HOURLY_WEIGHTS / HOURLY_WEIGHTS.sum()  # Normalize

# Anomaly definition
ANOMALY = {
    "label": "spike_volume",
    "type": "spike",
    "date": datetime(2023, 10, 5),  # Single day (Thursday)
    "volume_multiplier": 3.0,
    "description": "3x volume spike with characteristic skew (simulates major outage/bug)",
}

# Anomaly characteristics - tickets with these traits are more likely on anomaly day
ANOMALY_CHARACTERISTICS = {
    # Priority weights (higher = more likely to appear on anomaly day)
    "priority": {
        "critical": 5.0,
        "high": 3.0,
        "medium": 1.0,
        "low": 0.3,
        "very_low": 0.2,
    },
    # Type weights
    "type": {
        "Incident": 4.0,
        "Problem": 2.0,
        "Request": 0.5,
        "Change": 0.3,
    },
    # Tags that indicate outage/bug (checked across tag_1 to tag_4)
    "tags": {
        "Bug": 4.0,
        "Outage": 5.0,
        "Disruption": 4.0,
        "Technical": 3.0,
        "Security": 3.0,
        "Network": 2.5,
        "Performance": 2.0,
        "Crash": 4.0,
    },
    # Queue weights
    "queue": {
        "Technical Support": 3.0,
        "IT Support": 2.5,
        "Product Support": 1.5,
    },
}

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1: LOAD & CLEAN
# ═══════════════════════════════════════════════════════════════════════════════


def load_and_clean() -> pd.DataFrame:
    """Load dataset from HuggingFace and clean it."""
    print("Loading dataset from HuggingFace...")
    dataset = load_dataset("Tobi-Bueck/customer-support-tickets")["train"]
    df = dataset.to_pandas()
    print(f"  Loaded {len(df)} rows")

    # Drop rows with null body
    before = len(df)
    df = df.dropna(subset=["body"])
    print(f"  Dropped {before - len(df)} rows with null body")

    # Drop unnecessary columns
    cols_to_drop = ["answer", "version", "tag_5", "tag_6", "tag_7", "tag_8"]
    df = df.drop(columns=cols_to_drop, errors="ignore")
    print(f"  Dropped columns: {cols_to_drop}")

    # Reset index and create ticket_id
    df = df.reset_index(drop=True)
    df.insert(0, "ticket_id", range(len(df)))

    print(f"  Final dataset: {len(df)} rows, {len(df.columns)} columns")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2: COMPUTE ANOMALY SCORES
# ═══════════════════════════════════════════════════════════════════════════════


def compute_anomaly_scores(df: pd.DataFrame) -> np.ndarray:
    """Compute anomaly likelihood score for each ticket based on characteristics."""
    scores = np.ones(len(df))

    # Priority score
    priority_weights = ANOMALY_CHARACTERISTICS["priority"]
    for priority, weight in priority_weights.items():
        mask = df["priority"] == priority
        scores[mask] *= weight

    # Type score
    type_weights = ANOMALY_CHARACTERISTICS["type"]
    for ticket_type, weight in type_weights.items():
        mask = df["type"] == ticket_type
        scores[mask] *= weight

    # Tag scores (check all tag columns)
    tag_weights = ANOMALY_CHARACTERISTICS["tags"]
    tag_cols = ["tag_1", "tag_2", "tag_3", "tag_4"]
    for tag, weight in tag_weights.items():
        for col in tag_cols:
            mask = df[col] == tag
            scores[mask] *= weight

    # Queue score
    queue_weights = ANOMALY_CHARACTERISTICS["queue"]
    for queue, weight in queue_weights.items():
        mask = df["queue"] == queue
        scores[mask] *= weight

    return scores


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3: GENERATE TIMESTAMPS
# ═══════════════════════════════════════════════════════════════════════════════


def generate_daily_volumes(total_tickets: int, rng: np.random.Generator) -> dict:
    """Generate daily ticket volumes with weekly seasonality and spike anomaly."""
    daily_volumes = {}
    current = START_DATE
    anomaly_date = ANOMALY["date"].date()

    # First pass: generate raw volumes with weekly pattern + noise
    raw_volumes = {}
    while current <= END_DATE:
        dow = current.weekday()
        base = WEEKLY_PATTERN[dow]
        noise = rng.uniform(0.85, 1.15)
        volume = base * noise

        # Apply spike multiplier on anomaly date
        if current.date() == anomaly_date:
            volume *= ANOMALY["volume_multiplier"]

        raw_volumes[current.date()] = volume
        current += timedelta(days=1)

    # Normalize to sum to total_tickets
    total_raw = sum(raw_volumes.values())
    scale = total_tickets / total_raw

    for date, vol in raw_volumes.items():
        daily_volumes[date] = max(1, int(vol * scale))

    # Adjust to exactly match total_tickets
    current_total = sum(daily_volumes.values())
    diff = total_tickets - current_total
    dates = [d for d in daily_volumes.keys() if d != anomaly_date]

    for _ in range(abs(diff)):
        date = rng.choice(dates)
        if diff > 0:
            daily_volumes[date] += 1
        elif daily_volumes[date] > 1:
            daily_volumes[date] -= 1

    return daily_volumes


def assign_timestamps(
    df: pd.DataFrame,
    daily_volumes: dict,
    anomaly_scores: np.ndarray,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Assign timestamps with characteristic-based selection for anomaly day."""
    anomaly_date = ANOMALY["date"].date()
    anomaly_volume = daily_volumes[anomaly_date]

    # Separate anomaly day and normal days
    normal_volumes = {d: v for d, v in daily_volumes.items() if d != anomaly_date}

    # Select tickets for anomaly day using weighted sampling
    probabilities = anomaly_scores / anomaly_scores.sum()
    anomaly_indices = rng.choice(
        len(df), size=anomaly_volume, replace=False, p=probabilities
    )
    anomaly_indices_set = set(anomaly_indices)

    # Remaining tickets for normal days
    normal_indices = [i for i in range(len(df)) if i not in anomaly_indices_set]
    rng.shuffle(normal_indices)

    # Generate timestamps for anomaly day
    anomaly_timestamps = []
    hour_counts = rng.multinomial(anomaly_volume, HOURLY_WEIGHTS)
    for hour, count in enumerate(hour_counts):
        for _ in range(count):
            minute = int(rng.integers(0, 60))
            second = int(rng.integers(0, 60))
            microsecond = int(rng.integers(0, 1000000))
            ts = datetime(
                anomaly_date.year, anomaly_date.month, anomaly_date.day,
                hour, minute, second, microsecond
            )
            anomaly_timestamps.append(ts)
    anomaly_timestamps.sort()

    # Generate timestamps for normal days
    normal_timestamps = []
    for date, volume in sorted(normal_volumes.items()):
        hour_counts = rng.multinomial(volume, HOURLY_WEIGHTS)
        for hour, count in enumerate(hour_counts):
            for _ in range(count):
                minute = int(rng.integers(0, 60))
                second = int(rng.integers(0, 60))
                microsecond = int(rng.integers(0, 1000000))
                ts = datetime(
                    date.year, date.month, date.day,
                    hour, minute, second, microsecond
                )
                normal_timestamps.append(ts)
    normal_timestamps.sort()

    # Create timestamp mapping
    timestamp_map = {}

    # Map anomaly tickets
    for idx, ts in zip(anomaly_indices, anomaly_timestamps):
        timestamp_map[idx] = ts

    # Map normal tickets
    for idx, ts in zip(normal_indices, normal_timestamps):
        timestamp_map[idx] = ts

    # Assign timestamps and sort
    df = df.copy()
    df["timestamp"] = df["ticket_id"].map(timestamp_map)
    df = df.sort_values("timestamp").reset_index(drop=True)

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4: LABEL ANOMALIES
# ═══════════════════════════════════════════════════════════════════════════════


def label_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """Add anomaly labels to tickets on the spike day."""
    df = df.copy()
    anomaly_date = ANOMALY["date"].date()

    is_anomaly = df["timestamp"].dt.date == anomaly_date

    df["anomaly_label"] = None
    df.loc[is_anomaly, "anomaly_label"] = ANOMALY["label"]
    df["is_anomaly"] = is_anomaly

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 5: ADD SPLIT MARKER
# ═══════════════════════════════════════════════════════════════════════════════


def add_split_marker(df: pd.DataFrame) -> pd.DataFrame:
    """Add train/test split marker based on timestamp."""
    df = df.copy()
    df["split"] = df["timestamp"].apply(lambda x: "train" if x <= TRAIN_END else "test")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 6: EXPORT
# ═══════════════════════════════════════════════════════════════════════════════


def generate_metadata(df: pd.DataFrame) -> dict:
    """Generate metadata JSON."""
    train_df = df[df["split"] == "train"]
    test_df = df[df["split"] == "test"]
    anomaly_df = df[df["is_anomaly"]]
    normal_df = df[~df["is_anomaly"]]
    anomaly_count = len(anomaly_df)

    # Compute characteristic comparisons
    def get_distribution(subset, col):
        return (subset[col].value_counts(normalize=True) * 100).round(1).to_dict()

    return {
        "dataset_info": {
            "source": "Tobi-Bueck/customer-support-tickets",
            "total_tickets": int(len(df)),
            "train_tickets": int(len(train_df)),
            "test_tickets": int(len(test_df)),
            "date_range": [START_DATE.strftime("%Y-%m-%d"), END_DATE.strftime("%Y-%m-%d")],
            "train_range": [START_DATE.strftime("%Y-%m-%d"), TRAIN_END.strftime("%Y-%m-%d")],
            "test_range": [(TRAIN_END + timedelta(days=1)).strftime("%Y-%m-%d"), END_DATE.strftime("%Y-%m-%d")],
        },
        "anomaly": {
            "label": ANOMALY["label"],
            "type": ANOMALY["type"],
            "date": ANOMALY["date"].strftime("%Y-%m-%d"),
            "volume_multiplier": ANOMALY["volume_multiplier"],
            "affected_tickets": int(anomaly_count),
            "description": ANOMALY["description"],
            "characteristics": {
                "priority_distribution": get_distribution(anomaly_df, "priority"),
                "type_distribution": get_distribution(anomaly_df, "type"),
                "top_tags": get_distribution(anomaly_df, "tag_1"),
            },
        },
        "baseline": {
            "priority_distribution": get_distribution(normal_df, "priority"),
            "type_distribution": get_distribution(normal_df, "type"),
            "top_tags": get_distribution(normal_df, "tag_1"),
        },
        "statistics": {
            "normal_tickets": int(len(normal_df)),
            "anomaly_tickets": int(anomaly_count),
            "anomaly_percentage": round(anomaly_count / len(df) * 100, 2),
        },
        "columns": list(df.columns),
        "random_seed": RANDOM_SEED,
    }


def export_datasets(df: pd.DataFrame, metadata: dict):
    """Export full dataset and metadata JSON."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Raw full dataset
    full_path = OUTPUT_DIR / "full_dataset.csv"
    df.to_csv(full_path, index=False)
    print(f"  Saved {full_path} ({len(df)} rows)")

    # Metadata
    meta_path = OUTPUT_DIR / "anomaly_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Saved {meta_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════


def main():
    print("=" * 60)
    print("DATASET PREPARATION PIPELINE")
    print("=" * 60)

    rng = np.random.default_rng(RANDOM_SEED)

    # Step 1: Load and clean
    print("\n[STEP 1] Loading and cleaning dataset...")
    df = load_and_clean()

    # Step 2: Compute anomaly scores
    print("\n[STEP 2] Computing anomaly scores...")
    anomaly_scores = compute_anomaly_scores(df)
    print(f"  Score range: {anomaly_scores.min():.2f} to {anomaly_scores.max():.2f}")
    print(f"  Mean score: {anomaly_scores.mean():.2f}")

    # Step 3: Generate timestamps
    print("\n[STEP 3] Generating timestamps...")
    daily_volumes = generate_daily_volumes(len(df), rng)
    anomaly_date = ANOMALY["date"].date()
    print(f"  Anomaly day ({anomaly_date}): {daily_volumes[anomaly_date]} tickets")
    print(f"  Normal day avg: ~{sum(daily_volumes.values()) // 365} tickets")

    df = assign_timestamps(df, daily_volumes, anomaly_scores, rng)
    print(f"  Timestamps: {df['timestamp'].min()} to {df['timestamp'].max()}")

    # Step 4: Label anomalies
    print("\n[STEP 4] Labeling anomalies...")
    df = label_anomalies(df)
    print(f"  Anomaly tickets: {df['is_anomaly'].sum()}")

    # Step 5: Add split marker
    print("\n[STEP 5] Adding split marker...")
    df = add_split_marker(df)

    # Show characteristic comparison
    anomaly_df = df[df["is_anomaly"]]
    normal_df = df[~df["is_anomaly"]]

    print("\n[CHARACTERISTIC COMPARISON]")
    print("  Priority (high+critical):")
    anomaly_high = (anomaly_df["priority"].isin(["high", "critical"])).mean() * 100
    normal_high = (normal_df["priority"].isin(["high", "critical"])).mean() * 100
    print(f"    Anomaly day: {anomaly_high:.1f}%")
    print(f"    Normal days: {normal_high:.1f}%")

    print("  Type (Incident):")
    anomaly_incident = (anomaly_df["type"] == "Incident").mean() * 100
    normal_incident = (normal_df["type"] == "Incident").mean() * 100
    print(f"    Anomaly day: {anomaly_incident:.1f}%")
    print(f"    Normal days: {normal_incident:.1f}%")

    # Step 6: Export
    print("\n[STEP 6] Exporting datasets...")
    metadata = generate_metadata(df)
    export_datasets(df, metadata)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total tickets: {len(df)}")
    print(f"Train tickets: {len(df[df['split'] == 'train'])} (Jan 1 - Sep 30)")
    print(f"Test tickets: {len(df[df['split'] == 'test'])} (Oct 1 - Dec 31)")
    print(f"Anomaly: {ANOMALY['date'].strftime('%Y-%m-%d')} ({df['is_anomaly'].sum()} tickets)")
    print(f"  - Volume: {ANOMALY['volume_multiplier']}× normal")
    print(f"  - High/Critical priority: {anomaly_high:.1f}% (vs {normal_high:.1f}% baseline)")
    print(f"  - Incident type: {anomaly_incident:.1f}% (vs {normal_incident:.1f}% baseline)")
    print("\nDone!")


if __name__ == "__main__":
    main()
