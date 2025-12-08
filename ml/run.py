"""
Unified Anomaly Detection Runner

Run anomaly detection models with a consistent interface.

Usage:
    python3 -m ml.run --model prophet
    python3 -m ml.run --model arima
    python3 -m ml.run --model prophet --interval 0.95
"""

import argparse
import pandas as pd
from pathlib import Path
from datetime import datetime

from ml.models import AnomalyProphet, AnomalyARIMA
from ml.utils import evaluate, plot_results


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / "data"
TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"
OUTPUT_DIR = SCRIPT_DIR / "plots"

# Ground truth for evaluation
ANOMALY_DATE = datetime(2023, 10, 5).date()

# Available models
MODELS = {
    "prophet": AnomalyProphet,
    "arima": AnomalyARIMA,
}


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description="Run anomaly detection on ticket volume data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 -m ml.run --model prophet
  python3 -m ml.run --model arima
  python3 -m ml.run --model prophet --interval 0.95
        """,
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        required=True,
        choices=list(MODELS.keys()),
        help="Anomaly detection model to use",
    )
    parser.add_argument(
        "--interval", "-i",
        type=float,
        default=0.99,
        help="Confidence interval width (default: 0.99)",
    )
    args = parser.parse_args()

    model_name = args.model
    interval_width = args.interval
    ModelClass = MODELS[model_name]

    print("=" * 60)
    print(f"{ModelClass.name.upper()} ANOMALY DETECTION")
    print("=" * 60)

    # Step 1: Load data
    print("\n[STEP 1] Loading data...")
    if not TRAIN_PATH.exists() or not TEST_PATH.exists():
        print("  ERROR: Data files not found!")
        print("  Please run 'make prepare' first to generate train/test splits.")
        return

    train_df = pd.read_csv(TRAIN_PATH, parse_dates=["timestamp"])
    test_df = pd.read_csv(TEST_PATH, parse_dates=["timestamp"])
    print(f"  Train: {len(train_df)} days ({train_df['count'].sum():,} tickets)")
    print(f"  Test: {len(test_df)} days ({test_df['count'].sum():,} tickets)")

    # Step 2: Train model
    print(f"\n[STEP 2] Training {ModelClass.name} model...")
    detector = ModelClass(interval_width=interval_width)
    detector.fit(train_df)
    print(f"  Model trained with {interval_width*100:.0f}% confidence interval")

    # Step 3: Generate forecast
    print("\n[STEP 3] Generating forecast for test period...")
    forecast = detector.predict(test_df)
    print(f"  Forecast range: {forecast['ds'].min().date()} to {forecast['ds'].max().date()}")

    # Step 4: Detect anomalies
    print("\n[STEP 4] Detecting anomalies...")
    anomalies = detector.detect_anomalies(test_df)
    print(f"  Detected {len(anomalies)} anomaly day(s)")

    if not anomalies.empty:
        for _, row in anomalies.iterrows():
            date = row["timestamp"].date() if hasattr(row["timestamp"], "date") else row["timestamp"]
            actual = int(row["actual"])
            expected = int(row["yhat"])
            upper = int(row["yhat_upper"])
            print(f"    - {date}: {actual} tickets (expected: {expected}, upper bound: {upper})")

    # Step 5: Evaluate
    print("\n[STEP 5] Evaluating against ground truth...")
    detected_dates = [
        ts.date() if hasattr(ts, "date") else ts
        for ts in anomalies["timestamp"]
    ]
    metrics = evaluate(detected_dates, ANOMALY_DATE)

    print(f"  Ground truth anomaly: {ANOMALY_DATE}")
    print(f"  True Positives: {metrics['true_positives']}")
    print(f"  False Positives: {metrics['false_positives']}")
    print(f"  False Negatives: {metrics['false_negatives']}")
    print(f"  Precision: {metrics['precision']:.2%}")
    print(f"  Recall: {metrics['recall']:.2%}")
    print(f"  F1 Score: {metrics['f1']:.2%}")

    # Step 6: Visualize
    print("\n[STEP 6] Generating visualization...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_path = OUTPUT_DIR / f"{model_name}_anomaly_detection.png"
    plot_results(
        train_df, test_df, forecast, anomalies, plot_path,
        model_name=ModelClass.name, interval_width=interval_width
    )

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    if metrics["recall"] == 1.0:
        print("✓ Successfully detected the anomaly on October 5, 2023!")
    else:
        print("✗ Missed the anomaly on October 5, 2023")

    if metrics["precision"] == 1.0:
        print("✓ No false positives!")
    else:
        print(f"✗ {metrics['false_positives']} false positive(s) detected")

    print(f"\nF1 Score: {metrics['f1']:.2%}")
    print("\nDone!")


if __name__ == "__main__":
    main()

