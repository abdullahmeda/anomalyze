"""
Feature Engineering and Data Preparation for Anomaly Detection

This script reads the raw full_dataset.csv and creates aggregated train/test splits
suitable for time-series anomaly detection.

Outputs:
- ml/data/train.csv: Daily aggregated time-series (Jan-Sep 2023, normal only)
- ml/data/test.csv: Daily aggregated time-series (Oct-Dec 2023, contains anomaly)

Each output CSV contains:
- timestamp: Daily timestamp
- count: Number of tickets on that day
"""

import pandas as pd
from pathlib import Path
from datetime import datetime

# Configuration
DATASET_PATH = Path("dataset/data/full_dataset.csv")
OUTPUT_DIR = Path("ml/data")
TRAIN_END = datetime(2023, 9, 30, 23, 59, 59)


def aggregate_to_daily(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate raw tickets to daily time-series.
    
    Args:
        df: Raw tickets dataframe with timestamp column
    
    Returns:
        Daily aggregated dataframe with timestamp and count columns
    """
    df = df.copy()
    
    # Ensure timestamp is datetime
    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    
    # Resample to daily counts
    daily = df.set_index("timestamp").resample("D").size().reset_index()
    daily.columns = ["timestamp", "count"]
    
    return daily


def create_train_test_splits(daily_df: pd.DataFrame, train_end: datetime) -> tuple:
    """
    Split daily time-series into train and test sets.
    
    Args:
        daily_df: Daily aggregated dataframe
        train_end: Cutoff datetime for train/test split
    
    Returns:
        Tuple of (train_df, test_df)
    """
    train_df = daily_df[daily_df["timestamp"] <= train_end].copy()
    test_df = daily_df[daily_df["timestamp"] > train_end].copy()
    
    return train_df, test_df


def export_splits(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """
    Export train and test splits to CSV files.
    
    Args:
        train_df: Training dataframe
        test_df: Testing dataframe
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    train_path = OUTPUT_DIR / "train.csv"
    test_path = OUTPUT_DIR / "test.csv"
    
    train_df.to_csv(train_path, index=False)
    print(f"  Saved {train_path} ({len(train_df)} days)")
    
    test_df.to_csv(test_path, index=False)
    print(f"  Saved {test_path} ({len(test_df)} days)")


def main():
    print("=" * 60)
    print("FEATURE ENGINEERING & DATA PREPARATION")
    print("=" * 60)
    
    # Step 1: Load raw dataset
    print("\n[STEP 1] Loading raw dataset...")
    if not DATASET_PATH.exists():
        print(f"  ERROR: {DATASET_PATH} not found!")
        print("  Please run 'make dataset' first to generate the raw dataset.")
        return
    
    df = pd.read_csv(DATASET_PATH, parse_dates=["timestamp"])
    print(f"  Loaded {len(df):,} tickets")
    print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    # Step 2: Aggregate to daily time-series
    print("\n[STEP 2] Aggregating to daily time-series...")
    daily_df = aggregate_to_daily(df)
    print(f"  Created {len(daily_df)} daily records")
    print(f"  Average daily volume: {daily_df['count'].mean():.1f} tickets/day")
    print(f"  Max daily volume: {daily_df['count'].max()} tickets")
    
    # Step 3: Create train/test splits
    print("\n[STEP 3] Creating train/test splits...")
    train_df, test_df = create_train_test_splits(daily_df, TRAIN_END)
    print(f"  Train: {len(train_df)} days ({train_df['count'].sum():,} tickets)")
    print(f"  Test: {len(test_df)} days ({test_df['count'].sum():,} tickets)")
    
    # Step 4: Export
    print("\n[STEP 4] Exporting datasets...")
    export_splits(train_df, test_df)
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Train period: {train_df['timestamp'].min().date()} to {train_df['timestamp'].max().date()}")
    print(f"Test period: {test_df['timestamp'].min().date()} to {test_df['timestamp'].max().date()}")
    print(f"\nOutput files saved to: {OUTPUT_DIR.absolute()}")
    print("\nDone!")


if __name__ == "__main__":
    main()

