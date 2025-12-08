"""
Unified Anomaly Detection Runner

Run anomaly detection models with a consistent interface.

Usage:
    python3 -m ml.run --model prophet
    python3 -m ml.run --model arima
    python3 -m ml.run --model prophet --interval 0.95
"""

import argparse
import logging
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

ANOMALY_DATE = datetime(2023, 10, 5).date()

MODELS = {
    "prophet": AnomalyProphet,
    "arima": AnomalyARIMA,
}

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(description="Run anomaly detection on ticket volume data.")
    parser.add_argument("--model", "-m", required=True, choices=list(MODELS.keys()), help="Model to use")
    parser.add_argument("--interval", "-i", type=float, default=0.99, help="Confidence interval (default: 0.99)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    ModelClass = MODELS[args.model]

    # Load data
    if not TRAIN_PATH.exists() or not TEST_PATH.exists():
        logger.error("Data files not found. Run 'make prepare' first.")
        return

    train_df = pd.read_csv(TRAIN_PATH, parse_dates=["timestamp"])
    test_df = pd.read_csv(TEST_PATH, parse_dates=["timestamp"])

    # Train and detect
    detector = ModelClass(interval_width=args.interval)
    detector.fit(train_df)
    forecast = detector.predict(test_df)
    anomalies = detector.detect_anomalies(test_df)

    # Evaluate
    detected_dates = [ts.date() if hasattr(ts, "date") else ts for ts in anomalies["timestamp"]]
    metrics = evaluate(detected_dates, ANOMALY_DATE)

    # Visualize
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_path = OUTPUT_DIR / f"{args.model}_anomaly_detection.png"
    plot_results(train_df, test_df, forecast, anomalies, plot_path, ModelClass.name, args.interval)

    # Report results
    logger.info(f"\n{ModelClass.name} Anomaly Detection ({args.interval*100:.0f}% CI)")
    logger.info(f"  Data: {len(train_df)} train days, {len(test_df)} test days")
    logger.info(f"  Detected: {len(anomalies)} anomaly(s) | Ground truth: {ANOMALY_DATE}")
    for _, row in anomalies.iterrows():
        date = row["timestamp"].date() if hasattr(row["timestamp"], "date") else row["timestamp"]
        logger.info(f"    → {date}: {int(row['actual'])} tickets (expected {int(row['yhat'])}, bound {int(row['yhat_upper'])})")
    logger.info(f"  Metrics: P={metrics['precision']:.0%} R={metrics['recall']:.0%} F1={metrics['f1']:.0%}")
    logger.info(f"  Plot: {plot_path}")


if __name__ == "__main__":
    main()
