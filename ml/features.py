"""
Feature Engineering and Data Preparation for Anomaly Detection

Reads full_dataset.csv and creates aggregated train/test splits for time-series anomaly detection.

Outputs:
- ml/data/train.csv: Daily aggregated time-series (Jan-Sep 2023, normal only)
- ml/data/test.csv: Daily aggregated time-series (Oct-Dec 2023, contains anomaly)
"""

import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
DATASET_PATH = PROJECT_ROOT / "dataset" / "data" / "full_dataset.csv"
OUTPUT_DIR = Path(__file__).parent / "data"
TRAIN_END = datetime(2023, 9, 30, 23, 59, 59)


def aggregate_to_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate raw tickets to daily time-series."""
    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    daily = df.set_index("timestamp").resample("D").size().reset_index()
    daily.columns = ["timestamp", "count"]
    return daily


def create_train_test_splits(daily_df: pd.DataFrame, train_end: datetime) -> tuple:
    """Split daily time-series into train and test sets."""
    train_df = daily_df[daily_df["timestamp"] <= train_end].copy()
    test_df = daily_df[daily_df["timestamp"] > train_end].copy()
    return train_df, test_df


def add_time_series_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add time-series features for tree-based models.

    Adds:
    - Date components: day_of_week, is_weekend, month, day_of_month
    - Lags: lag_1, lag_7, lag_14
    - Rolling stats: rolling_mean_7d, rolling_std_7d
    """
    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Date components
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["is_weekend"] = df["timestamp"].dt.dayofweek >= 5
    df["month"] = df["timestamp"].dt.month
    df["day_of_month"] = df["timestamp"].dt.day
    df["day_of_year"] = df["timestamp"].dt.dayofyear

    # Lags
    df["lag_1"] = df["count"].shift(1)
    df["lag_7"] = df["count"].shift(7)
    df["lag_14"] = df["count"].shift(14)

    # Rolling stats (shift by 1 to avoid data leakage)
    df["rolling_mean_7d"] = df["count"].shift(1).rolling(window=7).mean()
    df["rolling_std_7d"] = df["count"].shift(1).rolling(window=7).std()

    # Drop NaNs created by lags/rolling
    df = df.dropna()

    return df


def export_splits(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """Export train and test splits to CSV files."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(OUTPUT_DIR / "train.csv", index=False)
    test_df.to_csv(OUTPUT_DIR / "test.csv", index=False)


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if not DATASET_PATH.exists():
        logger.error(f"Dataset not found: {DATASET_PATH}. Run 'make dataset' first.")
        return

    df = pd.read_csv(DATASET_PATH, parse_dates=["timestamp"])
    daily_df = aggregate_to_daily(df)
    train_df, test_df = create_train_test_splits(daily_df, TRAIN_END)
    export_splits(train_df, test_df)

    logger.info(f"\nFeature Preparation Complete")
    logger.info(f"  Train: {len(train_df)} days ({train_df['count'].sum():,} tickets) | {train_df['timestamp'].min().date()} to {train_df['timestamp'].max().date()}")
    logger.info(f"  Test: {len(test_df)} days ({test_df['count'].sum():,} tickets) | {test_df['timestamp'].min().date()} to {test_df['timestamp'].max().date()}")
    logger.info(f"  Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
